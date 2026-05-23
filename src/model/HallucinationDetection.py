import os
import json
import torch
from tqdm import tqdm
import src.model.utils as ut
from src.data.BeliefBankDataset import BeliefBankDataset
from src.data.HaluEvalDataset import HaluEvalDataset
from src.model.InspectOutputContext import InspectOutputContext
from src.model.prompts import PROMPT_QA, PROMPT_HALU


class HallucinationDetection:
    TARGET_LAYERS = list(range(0, 32))
    MAX_NEW_TOKENS = 50
    DEFAULT_DATASET = "belief_bank"
    CACHE_DIR_NAME = "activation_cache"
    ACTIVATION_TARGET = ["hidden", "mlp", "attn"]

    def __init__(self, project_dir):
        self.project_dir = project_dir

    
    def load_dataset(self, dataset_name=DEFAULT_DATASET, use_local=False, belief_bank_data_type="facts"):
        print("--"*50)
        print(f"Loading dataset {dataset_name}")
        print("--"*50)
        
       
        if dataset_name == "halu_eval":
            self.dataset_name = dataset_name
            self.dataset = HaluEvalDataset(use_local=use_local)
        elif dataset_name == "belief_bank":
            self.dataset_name = f"{dataset_name}_{belief_bank_data_type}"
            self.dataset = BeliefBankDataset(
                project_root=self.project_dir,
                data_type=belief_bank_data_type,
                recreate_ids=True
            )
            print(f"BeliefBank loaded with data_type='{belief_bank_data_type}'")
        else:
            raise ValueError(
                f"Dataset {dataset_name} not supported. Available: 'belief_bank', 'halu_eval'"
            )


    def load_llm(
        self,
        llm_name,
        use_local=False,
        dtype=torch.bfloat16,
        use_device_map=True,
        use_flash_attn=False,
        quantization=True,
        device="cuda:2",
    ):
        print("--"*50)
        print(f"Loading LLM {llm_name}")
        if quantization:
            print("Using 4-bit quantization")
        else:
            print(f"Using full precision ({dtype})")
        print("--"*50)
        self.llm_name = llm_name
        self.tokenizer = ut.load_tokenizer(llm_name, local=use_local)
        bnb_config = ut.create_bnb_config() if quantization else None
        if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(torch.device(device))
        self.llm = ut.load_llm(
            llm_name,
            bnb_config,
            local=use_local,
            dtype=dtype,
            use_device_map=use_device_map,
            use_flash_attention=use_flash_attn,
            device=device,
        )
        self.device = device
        print("\n\nQUANTIZATION\n\n:", quantization)

        if hasattr(self.llm.config, 'num_hidden_layers'):
            num_layers = self.llm.config.num_hidden_layers
            self.TARGET_LAYERS = list(range(0, num_layers))
            print(f"Detected {num_layers} layers in model")

        print("--"*50)


    @torch.no_grad()
    def save_model_activations(
        self,
        llm_name,
        data_name=DEFAULT_DATASET,
        use_local=False,
        dtype=torch.bfloat16,
        use_device_map=True,
        use_flash_attn=False,
        max_samples=None,
        quantization=False,
        belief_bank_data_type="facts",
        device="cuda:2",
    ):
        """Save LLM activations for the dataset.
        
        Args:
            max_samples: Number of samples to process (None = all samples)
            quantization: If True, use 4-bit quantization; if False, use full precision
            belief_bank_data_type: For BeliefBank only - "facts" or "constraints"
        """
        self.load_dataset(dataset_name=data_name, use_local=use_local, belief_bank_data_type=belief_bank_data_type)
        self.max_samples = max_samples

        if data_name == "belief_bank":
            self.prompt_template = PROMPT_QA
        elif data_name == "halu_eval":
            self.prompt_template = PROMPT_HALU
        else:
            raise ValueError(
                f"Dataset {data_name} not supported. Available: 'belief_bank', 'halu_eval'"
            )

        self.load_llm(
            llm_name,
            use_local=use_local,
            dtype=dtype,
            use_device_map=use_device_map,
            use_flash_attn=use_flash_attn,
            quantization=quantization,
            device=device,
        )
        
        print("--"*50)
        print("Hallucination Detection - Saving LLM's activations")
        print("--"*50)
        
        print("\n0. Prepare folders")
        self._create_folders_if_not_exists()
    
        print(f"\n1. Saving {self.llm_name} activations for layers {self.TARGET_LAYERS}")
        self.save_activations()
        
        print("--"*50)

    
    def save_activations(self):
        module_names = []
        module_names += [f'model.layers.{idx}' for idx in self.TARGET_LAYERS]
        module_names += [f'model.layers.{idx}.self_attn' for idx in self.TARGET_LAYERS]
        module_names += [f'model.layers.{idx}.mlp' for idx in self.TARGET_LAYERS]

        hallucination_labels = []

        num_samples = min(self.max_samples, len(self.dataset)) if self.max_samples else len(self.dataset)
        print(f"Processing {num_samples} samples out of {len(self.dataset)} total")

        BATCH_SIZE = 1
        num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Processing in {num_batches} batches of {BATCH_SIZE} samples")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, num_samples)
            print(f"\nProcessing batch {batch_idx+1}/{num_batches}: samples {start_idx} to {end_idx-1}")
            
            for idx in tqdm(range(start_idx, end_idx), desc=f"Batch {batch_idx+1}/{num_batches}"):
                question, answer, instance_id = self.dataset[idx]
                model_input = self.prompt_template.format(question=question)
                tokens = self.tokenizer(model_input, return_tensors="pt")
                # Keep inputs on the same device as the embedding layer (matters with device_map="auto")
                if hasattr(self.llm, "get_input_embeddings") and self.llm.get_input_embeddings() is not None:
                    input_device = self.llm.get_input_embeddings().weight.device
                else:
                    input_device = next(self.llm.parameters()).device
                attention_mask = tokens["attention_mask"].to(input_device) if "attention_mask" in tokens else None

                with InspectOutputContext(self.llm, module_names, save_generation=True, save_dir=self.generation_save_dir) as inspect:
                    output = self.llm.generate(
                        input_ids=tokens["input_ids"].to(input_device),
                        max_new_tokens=self.MAX_NEW_TOKENS,
                        attention_mask=attention_mask,
                        do_sample=False,
                        top_p=0.95,
                        temperature=0.1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=False
                    )
                    
                    generated_ids = output.sequences[0][tokens["input_ids"].shape[1]:]
                    generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    
                    is_hallucination = (
                        answer.lower().strip() not in generated_text.lower().strip()
                    )

                    label_info = {
                        "instance_id": instance_id,
                        "question": question,
                        "gold_answer": answer,
                        "generated_answer": generated_text,
                        "is_hallucination": int(is_hallucination),
                        "evaluation_method": "substring_match_case_insensitive",
                    }

                    hallucination_labels.append(label_info)

                    ut.save_generation_output(generated_text, model_input, instance_id, self.generation_save_dir)

                for module, ac in inspect.catcher.items():
                    # ac: [batch_size, sequence_length, hidden_dim]
                    ac_last = ac[0, -1].float().cpu()
                    layer_idx = int(module.split(".")[2])

                    save_name = f"layer{layer_idx}-id{instance_id}.pt"
                    if "mlp" in module:
                        save_path = os.path.join(self.mlp_save_dir, save_name)
                    elif "self_attn" in module:
                        save_path = os.path.join(self.attn_save_dir, save_name)
                    else:
                        save_path = os.path.join(self.hidden_save_dir, save_name)

                    torch.save(ac_last, save_path)
                    del ac_last

                del tokens, output, generated_ids
                if attention_mask is not None:
                    del attention_mask
                torch.cuda.empty_cache()
                import gc
                gc.collect()

            labels_path = os.path.join(self.generation_save_dir, "hallucination_labels.json")
            with open(labels_path, 'w') as f:
                json.dump(hallucination_labels, f, indent=4)
            print(f"Saved intermediate labels after batch {batch_idx+1}")

        labels_path = os.path.join(self.generation_save_dir, "hallucination_labels.json")
        with open(labels_path, 'w') as f:
            json.dump(hallucination_labels, f, indent=4)
        print(f"\nSaved hallucination labels to: {labels_path}")

        self.combine_activations()


    def combine_activations(self):
        results_dir = os.path.join(self.project_dir, self.CACHE_DIR_NAME)
        model_name = self.llm_name.split("/")[-1]

        for aa in self.ACTIVATION_TARGET:
            act_dir = os.path.join(results_dir, model_name, self.dataset_name, f"activation_{aa}")

            act_files = list(os.listdir(act_dir))
            act_files = [f for f in act_files if len(f.split("-")) == 2]

            act_files_layer_idx_instance_idx = [
                [act_f, ut.parse_layer_id_and_instance_id(os.path.basename(act_f))]
                for act_f in act_files
            ]

            layer_group_files = {lid: [] for lid in self.TARGET_LAYERS}
            for act_f, (layer_id, instance_id) in act_files_layer_idx_instance_idx:
                layer_group_files[layer_id].append([act_f, instance_id])

            for layer_id in self.TARGET_LAYERS:
                layer_group_files[layer_id] = sorted(layer_group_files[layer_id], key=lambda x: x[1])

                acts = []
                loaded_paths = []
                instance_ids = []
                for idx, (act_f, instance_id) in enumerate(layer_group_files[layer_id]):
                    path_to_load = os.path.join(act_dir, act_f)
                    acts.append(torch.load(path_to_load))
                    loaded_paths.append(path_to_load)
                    instance_ids.append(instance_id)

                acts = torch.stack(acts)
                save_path = os.path.join(act_dir, f"layer{layer_id}_activations.pt")
                torch.save(acts, save_path)

                ids_save_path = os.path.join(act_dir, f"layer{layer_id}_instance_ids.json")
                json.dump(instance_ids, open(ids_save_path, "w"), indent=4)

                for p in loaded_paths:
                    os.remove(p)


    def _create_folders_if_not_exists(self):
        model_name = self.llm_name.split("/")[-1]

        results_dir = os.path.join(self.project_dir, self.CACHE_DIR_NAME)

        self.hidden_save_dir = os.path.join(results_dir, model_name, self.dataset_name, "activation_hidden")
        self.mlp_save_dir = os.path.join(results_dir, model_name, self.dataset_name, "activation_mlp")
        self.attn_save_dir = os.path.join(results_dir, model_name, self.dataset_name, "activation_attn")

        self.generation_save_dir = os.path.join(results_dir, model_name, self.dataset_name, "generations")
        self.logits_save_dir = os.path.join(results_dir, model_name, self.dataset_name, "logits")
        
        for sd in [self.hidden_save_dir, self.mlp_save_dir, self.attn_save_dir, self.generation_save_dir, self.logits_save_dir]:
            print(f"Creating directory: {sd}")
            if not os.path.exists(sd):
                os.makedirs(sd)

        print("\n\n")
 
