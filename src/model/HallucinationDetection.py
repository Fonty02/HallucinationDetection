import os
import json
import torch
from tqdm import tqdm
import src.model.utils as ut
from src.data.SimpleQADataset import SimpleQADataset
from src.data.HaluBenchDataset import HaluBenchDataset
from src.data.BeliefBankDataset import BeliefBankDataset
from src.model.InspectOutputContext import InspectOutputContext
from src.model.prompts import PROMPT_QA as prompt
from src.model.gemini_autorater import GeminiAutorater


class HallucinationDetection:
    # -------------
    # Constants
    # -------------
    MAX_NEW_TOKENS = 100
    DEFAULT_DATASET = "belief_bank"
    CACHE_DIR_NAME = "activation_cache"
    ACTIVATION_TARGET = ["hidden", "mlp", "attn"]

    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.gemini_autorater = None

    def load_dataset(self, dataset_name=DEFAULT_DATASET, use_local=False, belief_bank_data_type="facts"):
        print("--" * 50)
        print(f"Loading dataset {dataset_name}")
        print("--" * 50)

        if dataset_name == "simpleqa":
            self.dataset = SimpleQADataset(use_local=use_local)
        elif dataset_name == "halu_bench":
            self.dataset = HaluBenchDataset(use_local=use_local)
        elif dataset_name == "belief_bank":
            self.dataset = BeliefBankDataset(
                project_root=self.project_dir,
                data_type=belief_bank_data_type,
                recreate_ids=True
            )
        else:
            raise ValueError(f"Dataset {dataset_name} not supported.")
        self.dataset_name = dataset_name

    def load_llm(self, llm_name, use_local=False, dtype=torch.bfloat16,
                 use_device_map=True, use_flash_attn=False, quantization=True):
        print("--" * 50)
        print(f"Loading LLM {llm_name}")
        print("Using 4-bit quantization" if quantization else f"Using full precision ({dtype})")
        print("--" * 50)

        self.llm_name = llm_name
        self.tokenizer = ut.load_tokenizer(llm_name, local=use_local)
        self.tokenizer.padding_side = "left"          # importante per generazione batched
        self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = ut.create_bnb_config() if quantization else None
        self.llm = ut.load_llm(llm_name, bnb_config, local=use_local, dtype=dtype,
                               use_device_map=use_device_map, use_flash_attention=use_flash_attn)

        # Auto-detect numero layer
        num_layers = getattr(self.llm.config, "num_hidden_layers", 32)
        self.TARGET_LAYERS = list(range(num_layers))
        print(f"Detected {num_layers} layers")

    @torch.no_grad()
    def save_model_activations(self,
                               llm_name,
                               data_name=DEFAULT_DATASET,
                               use_local=False,
                               dtype=torch.bfloat16,
                               use_device_map=True,
                               use_flash_attn=False,
                               use_gemini_autorater=False,
                               gemini_model="gemini-2.0-flash-lite",
                               max_samples=None,
                               quantization=True,
                               gpu_batch_size=64,           # <- scegli tu (16-64 tipico su A100/H100)
                               belief_bank_data_type="facts"):

        self.load_dataset(dataset_name=data_name, use_local=use_local,
                          belief_bank_data_type=belief_bank_data_type)
        self.load_llm(llm_name, use_local=use_local, dtype=dtype,
                      use_device_map=use_device_map, use_flash_attn=use_flash_attn,
                      quantization=quantization)

        # Gemini autorater (opzionale)
        if use_gemini_autorater:
            try:
                self.gemini_autorater = GeminiAutorater(model_name=gemini_model)
                print("Gemini autorater ready")
            except Exception as e:
                print(f"Gemini failed: {e} → using substring fallback")
                self.gemini_autorater = None

        self._create_folders_if_not_exists()
        self.max_samples = max_samples
        self.save_activations_batched(gpu_batch_size=gpu_batch_size)

    # ========================
    # VERSIONE BATCHED OTTIMIZZATA
    # ========================
    def save_activations_batched(self, gpu_batch_size=64):
        module_names = [
            f"model.layers.{i}" for i in self.TARGET_LAYERS
        ] + [
            f"model.layers.{i}.self_attn" for i in self.TARGET_LAYERS
        ] + [
            f"model.layers.{i}.mlp" for i in self.TARGET_LAYERS
        ]

        num_samples = min(self.max_samples, len(self.dataset)) if self.max_samples else len(self.dataset)
        print(f"Processing {num_samples} samples in batches of {gpu_batch_size}")

        hallucination_labels = []

        for start_idx in tqdm(range(0, num_samples, gpu_batch_size), desc="GPU Batches"):
            end_idx = min(start_idx + gpu_batch_size, num_samples)
            batch_questions = []
            batch_answers = []
            batch_instance_ids = []
            batch_prompts = []

            # Preparazione batch
            for idx in range(start_idx, end_idx):
                q, a, iid = self.dataset[idx]
                batch_questions.append(q)
                batch_answers.append(a)
                batch_instance_ids.append(iid)
                batch_prompts.append(prompt.format(question=q))

            # Tokenizzazione con padding a sinistra (necessario per generazione)
            encodings = self.tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt"
            ).to("cuda")

            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"]

            # ==== GENERAZIONE + CATTURA ATTIVAZIONI ====
            # IMPORTANTE: save_generation=False → niente file enormi!
            with InspectOutputContext(
                self.llm,
                module_names,
                save_generation=False,      # ← disabilitato
                save_dir=None
            ) as inspect:

                outputs = self.llm.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.MAX_NEW_TOKENS,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=False         # ← disattivato (risparmi GB)
                )

            generated_ids = outputs.sequences
            prompt_len = input_ids.shape[1]

            # ==== ELABORAZIONE RISULTATI SINGOLI ====
            for i in range(generated_ids.shape[0]):
                instance_id = batch_instance_ids[i]
                question = batch_questions[i]
                gold_answer = batch_answers[i]

                # Testo generato
                gen_ids = generated_ids[i, prompt_len:]
                generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

                # Label hallucination
                if self.gemini_autorater:
                    eval_res = self.gemini_autorater.evaluate(question, gold_answer, generated_text)
                    is_halluc = int(eval_res["is_hallucination"])
                else:
                    is_halluc = int(gold_answer.lower().strip() not in generated_text.lower())

                label_info = {
                    "instance_id": instance_id,
                    "question": question,
                    "gold_answer": gold_answer,
                    "generated_answer": generated_text,
                    "is_hallucination": is_halluc,
                    "evaluation_method": "gemini" if self.gemini_autorater else "substring"
                }
                hallucination_labels.append(label_info)

                # Salva generazione (solo testo, niente logits enormi)
                ut.save_generation_output(
                    generated_text,
                    batch_prompts[i],
                    instance_id,
                    self.generation_save_dir
                )

                # ==== SALVATAGGIO ATTIVAZIONI (solo ultimo token) ====
                for module_name, activation_tensor in inspect.catcher.items():
                    # activation_tensor: [batch, seq_len, dim]
                    last_token_act = activation_tensor[i, -1].float().cpu()

                    layer_idx = int(module_name.split(".")[2])
                    save_name = f"layer{layer_idx}-id{instance_id}.pt"

                    if "mlp" in module_name:
                        save_dir = self.mlp_save_dir
                    elif "self_attn" in module_name:
                        save_dir = self.attn_save_dir
                    else:
                        save_dir = self.hidden_save_dir

                    torch.save(last_token_act, os.path.join(save_dir, save_name))

            # Pulizia GPU
            del encodings, input_ids, attention_mask, outputs, generated_ids
            torch.cuda.empty_cache()

            # Salva label ogni 5 batch (o alla fine)
            if (start_idx // gpu_batch_size + 1) % 5 == 0 or end_idx == num_samples:
                lbl_path = os.path.join(self.generation_save_dir, "hallucination_labels.json")
                with open(lbl_path, "w") as f:
                    json.dump(hallucination_labels, f, indent=4)
                print(f"Intermediate labels saved ({len(hallucination_labels)} samples)")

        # Salva label finali
        lbl_path = os.path.join(self.generation_save_dir, "hallucination_labels.json")
        with open(lbl_path, "w") as f:
            json.dump(hallucination_labels, f, indent=4)
        print(f"Final labels → {lbl_path}")

        # Combina file .pt per layer
        print("Combining activations per layer...")
        self.combine_activations()
        print("Done!")

    # ========================
    # Combine + utility (invariati)
    # ========================
    def combine_activations(self):
        results_dir = os.path.join(self.project_dir, self.CACHE_DIR_NAME)
        model_name = self.llm_name.split("/")[-1]

        for target in self.ACTIVATION_TARGET:
            act_dir = os.path.join(results_dir, model_name, self.dataset_name, f"activation_{target}")
            files = [f for f in os.listdir(act_dir) if f.count("-") == 1 and f.endswith(".pt")]

            layer_to_files = {l: [] for l in self.TARGET_LAYERS}
            for f in files:
                layer_id, instance_id = ut.parse_layer_id_and_instance_id(f)
                layer_to_files[layer_id].append((f, instance_id))

            for layer_id, items in layer_to_files.items():
                items = sorted(items, key=lambda x: x[1])
                activations = []
                instance_ids = []
                paths = []

                for fname, iid in items:
                    path = os.path.join(act_dir, fname)
                    activations.append(torch.load(path, map_location="cpu"))
                    instance_ids.append(iid)
                    paths.append(path)

                stacked = torch.stack(activations)
                torch.save(stacked, os.path.join(act_dir, f"layer{layer_id}_activations.pt"))
                json.dump(instance_ids, open(os.path.join(act_dir, f"layer{layer_id}_instance_ids.json"), "w"))

                for p in paths:
                    os.remove(p)

    def _create_folders_if_not_exists(self):
        model_name = self.llm_name.split("/")[-1]
        base = os.path.join(self.project_dir, self.CACHE_DIR_NAME, model_name, self.dataset_name)

        self.hidden_save_dir = os.path.join(base, "activation_hidden")
        self.mlp_save_dir     = os.path.join(base, "activation_mlp")
        self.attn_save_dir    = os.path.join(base, "activation_attn")
        self.generation_save_dir = os.path.join(base, "generations")
        self.logits_save_dir  = os.path.join(base, "logits")   # (non più usato)

        for d in [self.hidden_save_dir, self.mlp_save_dir, self.attn_save_dir,
                  self.generation_save_dir, self.logits_save_dir]:
            os.makedirs(d, exist_ok=True)