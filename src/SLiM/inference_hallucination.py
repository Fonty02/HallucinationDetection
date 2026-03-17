"""
Inference e valutazione per SLiM Hallucination Reduction.

Esegue esperimenti di tipo:
- Baseline: modello senza SLiM
- SLiM: modello con SLiM (same-dataset)
- CrossDataset: modello con SLiM trainato su un dataset, valutato su un altro
- CrossModel: SLiM trainato su un altro LLM, trasferito via aligner Procrustes

Salva i risultati in SLiMExperiments.csv e i dettagli per campione in InferenceResults/.

Uso:
    python -m src.SLiM.inference_hallucination \
        --model_name Qwen/Qwen2.5-7B \
        --dataset belief_bank_facts \
        --slim_checkpoint SteeringVectors/SLiM/.../slim_xxx.pth \
        --top_k 10 \
        --state_value 1.0 \
        --device cuda:2
"""

import argparse
import csv
import json
import os
import pickle
import sys
import time
import traceback
import warnings
from glob import glob

import numpy as np
import torch
from tqdm import tqdm
from peft import prepare_model_for_kbit_training

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.utils import create_bnb_config, load_llm, load_tokenizer
from src.model.prompts import PROMPT_QA, PROMPT_HALU
from src.SLiM.model_general import GeneralSLiMedNet, StateBlock, LowRankLinear
from src.truthx.truthx_model import LLMArchitectureDetector


# =============================================================================
# ALIGNER UTILITIES (CrossModel Procrustes)
# =============================================================================


class MLPAlignerWrapper:
    """
    Wrapper per un aligner MLP salvato come checkpoint PyTorch (.pt),
    con interfaccia .predict(X_numpy) compatibile con sklearn.
    """

    def __init__(self, checkpoint_path, device="cpu"):
        from src.truthx.MLP import MLP

        ckpt = torch.load(checkpoint_path, map_location=device)
        input_size = ckpt["input_size"]
        output_size = ckpt["output_size"]
        state_dict = ckpt["model_state_dict"]
        hidden_size = state_dict["fc1.weight"].shape[0]
        self.model = MLP(input_size, hidden_size, output_size)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def predict(self, x):
        with torch.no_grad():
            x_t = torch.from_numpy(x).float().to(self.device)
            out = self.model(x_t)
        return out.cpu().numpy()


def has_bidirectional_directional_aligners(aligner_dir):
    """
    True se nella cartella sono presenti entrambe le direzioni esplicite:
    - target_to_trainer
    - trainer_to_target
    """
    directional_filenames = {
        "target_to_trainer": [
            "procrustes_target_to_trainer.pkl",
            "ridge_regressor_target_to_trainer.pkl",
        ],
        "trainer_to_target": [
            "procrustes_trainer_to_target.pkl",
            "ridge_regressor_trainer_to_target.pkl",
        ],
    }

    for _, names in directional_filenames.items():
        found = any(os.path.exists(os.path.join(aligner_dir, n)) for n in names)
        if not found:
            return False
    return True


def load_aligner(
    aligner_dir,
    device="cpu",
    preferred_direction="trainer_to_target",
    strict_direction=False,
):
    """
    Carica un aligner dalla directory data.
    Supporta:
      - procrustes_*_to_*.pkl
      - ridge_regressor_*_to_*.pkl
      - procrustes.pkl / ridge_regressor.pkl (legacy, trainer->target)
      - mlp_regressor.pt
    """
    if preferred_direction not in {"target_to_trainer", "trainer_to_target"}:
        raise ValueError(f"Unsupported preferred_direction: {preferred_direction}")

    directional_paths = {
        "target_to_trainer": [
            os.path.join(aligner_dir, "procrustes_target_to_trainer.pkl"),
            os.path.join(aligner_dir, "ridge_regressor_target_to_trainer.pkl"),
        ],
        "trainer_to_target": [
            os.path.join(aligner_dir, "procrustes_trainer_to_target.pkl"),
            os.path.join(aligner_dir, "ridge_regressor_trainer_to_target.pkl"),
        ],
    }

    legacy_paths = [
        os.path.join(aligner_dir, "procrustes.pkl"),
        os.path.join(aligner_dir, "ridge_regressor.pkl"),
    ]
    mlp_path = os.path.join(aligner_dir, "mlp_regressor.pt")

    for model_path in directional_paths[preferred_direction]:
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                aligner = pickle.load(f)
            print(f"  ✓ Aligner caricato ({preferred_direction}): {model_path}")
            return aligner

    if strict_direction:
        print(f"  ✗ Aligner direzionale '{preferred_direction}' non trovato in: {aligner_dir}")
        return None

    if preferred_direction == "trainer_to_target":
        for model_path in legacy_paths:
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    aligner = pickle.load(f)
                print(f"  ✓ Aligner legacy caricato (trainer_to_target): {model_path}")
                return aligner

    if os.path.exists(mlp_path):
        aligner = MLPAlignerWrapper(mlp_path, device=device)
        print(f"  ✓ Aligner MLP caricato: {mlp_path}")
        return aligner

    print(f"  ✗ Nessun aligner trovato in: {aligner_dir}")
    print(
        "    (cercati procrustes_*.pkl, ridge_regressor_*.pkl, "
        "procrustes.pkl, ridge_regressor.pkl, mlp_regressor.pt)"
    )
    return None


def load_crossmodel_aligners(project_root, experiment_id, aligner_cache_dir, device):
    """
    Carica pre/post aligner per esperimenti CrossModel.
    Modalità preferita: singola cartella per esperimento con entrambe le direzioni.
    Fallback legacy: due cartelle separate.
    """
    if not os.path.isabs(aligner_cache_dir):
        aligner_cache_dir = os.path.join(project_root, aligner_cache_dir)

    print(f"  Aligner cache directory: {aligner_cache_dir}")
    single_aligner_dir = os.path.join(aligner_cache_dir, experiment_id)
    print(f"  Candidate single aligner dir: {single_aligner_dir}")

    if has_bidirectional_directional_aligners(single_aligner_dir):
        print("  ✓ Rilevato aligner bidirezionale nella stessa cartella.")
        pre_aligner = load_aligner(
            single_aligner_dir,
            device=device,
            preferred_direction="target_to_trainer",
            strict_direction=True,
        )
        post_aligner = load_aligner(
            single_aligner_dir,
            device=device,
            preferred_direction="trainer_to_target",
            strict_direction=True,
        )
        return pre_aligner, post_aligner

    print("  ! Aligner bidirezionale non trovato, fallback legacy a due cartelle.")
    exp_parts = experiment_id.split("_to_")
    if len(exp_parts) != 2:
        raise FileNotFoundError(
            f"Cannot parse experiment_id '{experiment_id}' for aligner names"
        )

    trainer_name = exp_parts[0]
    target_and_dataset = exp_parts[1].split("_")
    target_name = target_and_dataset[0]
    dataset_suffix = "_".join(target_and_dataset[1:])

    post_aligner_name = experiment_id
    pre_aligner_name = f"{target_name}_to_{trainer_name}_{dataset_suffix}"

    pre_aligner_dir = os.path.join(aligner_cache_dir, pre_aligner_name)
    post_aligner_dir = os.path.join(aligner_cache_dir, post_aligner_name)

    print(f"  Legacy pre-aligner dir:  {pre_aligner_dir}")
    print(f"  Legacy post-aligner dir: {post_aligner_dir}")

    pre_aligner = load_aligner(
        pre_aligner_dir,
        device=device,
        preferred_direction="trainer_to_target",
        strict_direction=False,
    )
    post_aligner = load_aligner(
        post_aligner_dir,
        device=device,
        preferred_direction="trainer_to_target",
        strict_direction=False,
    )
    return pre_aligner, post_aligner


def model_alias(model_name):
    lower = (model_name or "").lower()
    if "gemma" in lower:
        return "Gemma"
    if "llama" in lower:
        return "Llama"
    return (model_name or "Unknown").replace("/", "_")


def dataset_alias(dataset_name):
    mapping = {
        "belief_bank_facts": "BBF",
        "belief_bank_constraints": "BBC",
        "halu_eval": "HE",
    }
    return mapping.get(dataset_name, dataset_name)


def infer_crossmodel_experiment_id(trainer_model_name, target_model_name, dataset_name):
    return (
        f"{model_alias(trainer_model_name)}_to_"
        f"{model_alias(target_model_name)}_"
        f"{dataset_alias(dataset_name)}_PROCRUSTES"
    )


def infer_slim_dims(slim_state_dict, slim_args):
    """
    Infer hidden_size e rank del trainer SLiM dal checkpoint.
    """
    if "SLiM_scale.up.weight" in slim_state_dict:
        trainer_hidden = int(slim_state_dict["SLiM_scale.up.weight"].shape[0])
    elif "SLiM_shift.up.weight" in slim_state_dict:
        trainer_hidden = int(slim_state_dict["SLiM_shift.up.weight"].shape[0])
    else:
        raise ValueError("Impossibile inferire hidden size trainer dal checkpoint SLiM.")

    if "SLiM_scale.down.weight" in slim_state_dict:
        slim_rank = int(slim_state_dict["SLiM_scale.down.weight"].shape[0])
    else:
        slim_rank = int(slim_args.get("slim_rank", 32))

    return trainer_hidden, slim_rank


class CrossModelSLiMWrapper:
    """
    Wrapper SLiM per CrossModel:
    - pre-align: target -> trainer
    - modulazione SLiM nello spazio trainer
    - post-align: trainer -> target
    """

    def __init__(self, base_model, checkpoint_data, pre_aligner, post_aligner, alpha=1.0):
        self.base_model = base_model
        self.pre_aligner = pre_aligner
        self.post_aligner = post_aligner
        self.alpha = float(alpha)

        slim_args = checkpoint_data.get("args", {})
        self.target_layer = int(
            checkpoint_data.get("target_layer", slim_args.get("target_layer", 0))
        )
        self.state_dim = int(slim_args.get("state_dim", 1))
        self.current_state_embed = None

        slim_state_dict = checkpoint_data.get("slim_state_dict", {})
        if not slim_state_dict:
            raise ValueError("Checkpoint SLiM senza 'slim_state_dict'.")

        self.trainer_hidden_size, self.slim_rank = infer_slim_dims(slim_state_dict, slim_args)
        self.state_proj = StateBlock(self.state_dim, self.trainer_hidden_size)
        self.SLiM_scale = LowRankLinear(
            self.trainer_hidden_size, self.trainer_hidden_size, rank=self.slim_rank
        )
        self.SLiM_shift = LowRankLinear(
            self.trainer_hidden_size, self.trainer_hidden_size, rank=self.slim_rank
        )

        self._load_slim_modules_from_checkpoint(slim_state_dict)
        self._freeze_slim_modules()

        self.arch_name, self.arch_config = LLMArchitectureDetector.detect_architecture(base_model)
        layers = LLMArchitectureDetector.get_layers(base_model, self.arch_config)
        self.num_layers = len(layers)

        if self.target_layer < 0 or self.target_layer >= self.num_layers:
            raise ValueError(
                f"target_layer={self.target_layer} fuori range [0, {self.num_layers - 1}]"
            )

        self.hooks = [layers[self.target_layer].register_forward_hook(self._create_hook())]

    def _load_slim_modules_from_checkpoint(self, slim_state_dict):
        def pick(prefix):
            return {
                k[len(prefix):]: v
                for k, v in slim_state_dict.items()
                if k.startswith(prefix)
            }

        state_proj_sd = pick("state_proj.")
        scale_sd = pick("SLiM_scale.")
        shift_sd = pick("SLiM_shift.")

        if not state_proj_sd or not scale_sd or not shift_sd:
            raise ValueError("Checkpoint SLiM incompleto: state_proj/SLiM_scale/SLiM_shift mancanti.")

        self.state_proj.load_state_dict(state_proj_sd, strict=True)
        self.SLiM_scale.load_state_dict(scale_sd, strict=True)
        self.SLiM_shift.load_state_dict(shift_sd, strict=True)

    def _freeze_slim_modules(self):
        self.state_proj.eval()
        self.SLiM_scale.eval()
        self.SLiM_shift.eval()

        for module in (self.state_proj, self.SLiM_scale, self.SLiM_shift):
            for p in module.parameters():
                p.requires_grad = False

    def to(self, device):
        self.state_proj = self.state_proj.to(device)
        self.SLiM_scale = self.SLiM_scale.to(device)
        self.SLiM_shift = self.SLiM_shift.to(device)
        return self

    def eval(self):
        self.base_model.eval()
        self._freeze_slim_modules()
        return self

    def _create_hook(self):
        def hook_fn(module, input, output):
            if self.current_state_embed is None:
                return output

            hidden = output[0] if isinstance(output, tuple) else output
            bsz, seq_len, target_hidden_size = hidden.shape

            # 1) Target -> Trainer
            x = hidden.detach().contiguous().view(-1, target_hidden_size).float()

            if hasattr(self.pre_aligner, "n_features_in_"):
                if int(self.pre_aligner.n_features_in_) != target_hidden_size:
                    raise ValueError(
                        "Dimension mismatch pre-aligner: "
                        f"expects {int(self.pre_aligner.n_features_in_)}, got {target_hidden_size}"
                    )

            x_np = x.cpu().numpy()
            x_aligned_np = self.pre_aligner.predict(x_np)
            x_aligned = torch.from_numpy(x_aligned_np).to(hidden.device)
            x_aligned = x_aligned.view(bsz, seq_len, -1).to(hidden.dtype)

            if x_aligned.shape[-1] != self.trainer_hidden_size:
                raise ValueError(
                    "Dimension mismatch after pre-aligner: "
                    f"expected trainer hidden {self.trainer_hidden_size}, got {x_aligned.shape[-1]}"
                )

            # 2) SLiM in trainer space
            state_input = self.current_state_embed.to(hidden.device).float()
            projected_state = self.state_proj(state_input)
            scale = torch.tanh(self.SLiM_scale(projected_state)).to(hidden.dtype)
            shift = torch.tanh(self.SLiM_shift(projected_state)).to(hidden.dtype)
            steered_aligned = x_aligned + self.alpha * (x_aligned * scale + shift)

            # 3) Trainer -> Target
            y = steered_aligned.contiguous().view(-1, self.trainer_hidden_size).float()
            if hasattr(self.post_aligner, "n_features_in_"):
                if int(self.post_aligner.n_features_in_) != self.trainer_hidden_size:
                    raise ValueError(
                        "Dimension mismatch post-aligner: "
                        f"expects {int(self.post_aligner.n_features_in_)}, "
                        f"got {self.trainer_hidden_size}"
                    )

            y_np = y.cpu().numpy()
            y_back_np = self.post_aligner.predict(y_np)
            y_back = torch.from_numpy(y_back_np).to(hidden.device)
            y_back = y_back.view(bsz, seq_len, -1).to(hidden.dtype)

            if y_back.shape[-1] != target_hidden_size:
                raise ValueError(
                    "Dimension mismatch after post-aligner: "
                    f"expected target hidden {target_hidden_size}, got {y_back.shape[-1]}"
                )

            if isinstance(output, tuple):
                return (y_back,) + output[1:]
            return y_back

        return hook_fn

    def generate(
        self,
        input_ids: torch.Tensor,
        state_tensor: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        **generate_kwargs,
    ):
        if state_tensor is None:
            self.current_state_embed = None
        else:
            self.current_state_embed = state_tensor.unsqueeze(1)

        return self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# =============================================================================
# CSV Columns
# =============================================================================

CSV_COLUMNS = [
    "experiment_id",
    "type",
    "model",
    "dataset_eval",
    "dataset_train",
    "num_samples_evaluated",
    "num_hallucinations",
    "hallucination_rate",
    "slim_checkpoint",
    "num_pairs_train",
    "state_value",
    "slim_layer",
    "num_layers_total",
    "slim_training_time_seconds",
    "slim_inference_time_seconds",
    "slim_trainable_params",
    "learning_rate",
    "batch_size",
    "num_epochs",
    "highlight",
]


def save_to_csv(csv_path: str, result: dict):
    """Append un risultato al CSV. Crea il file con header se non esiste."""
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()

        # Filtra solo le colonne previste
        row = {k: result.get(k, None) for k in CSV_COLUMNS}
        writer.writerow(row)

    print(f"[CSV] Risultato salvato in: {csv_path}")


def load_dataset_for_eval(dataset_name: str, project_root: str):
    """
    Carica il dataset per la valutazione (non paired, tutto il dataset).

    Returns:
        dataset: Dataset object con __getitem__ che ritorna (fact, label, instance_id)
        valid_indices: Lista di indici da valutare
        is_halu_eval: Bool, True se HaluEval
    """
    is_halu_eval = (dataset_name == "halu_eval")

    if is_halu_eval:
        from src.data.HaluEvalDataset import HaluEvalDataset
        # label=0 = risposte corrette (per valutare se il modello genera la corretta)
        dataset = HaluEvalDataset(label=0, use_local=False)
    elif dataset_name in ("belief_bank_facts", "belief_bank_constraints"):
        from src.data.BeliefBankDataset import BeliefBankDataset
        data_type = dataset_name.replace("belief_bank_", "")
        dataset = BeliefBankDataset(
            project_root=project_root,
            model_type="demo",
            recreate_ids=True,
            data_type=data_type,
        )
    else:
        raise ValueError(f"Dataset sconosciuto: {dataset_name}")

    valid_indices = list(range(len(dataset)))

    return dataset, valid_indices, is_halu_eval


def evaluate_hallucination(
    generated_text: str,
    expected_label: str,
    is_halu_eval: bool,
) -> bool:
    """
    Determina se una risposta generata è un'allucinazione.

    Per BeliefBank: verifica se "yes"/"no" è contenuto nella risposta.
    Per HaluEval: contenimento bidirezionale tra risposta generata e attesa.

    Returns:
        True se è un'allucinazione
    """
    gen_lower = generated_text.lower().strip()
    exp_lower = expected_label.lower().strip()

    if is_halu_eval:
        return (exp_lower not in gen_lower) and (gen_lower not in exp_lower)
    else:
        return exp_lower not in gen_lower


def load_training_config(project_root: str, model_name: str, dataset_name: str):
    """
    Carica la config di training e ricava gli instance_id da escludere,
    come in TruthX, per evitare leakage train/val durante la valutazione.

    Returns:
        (excluded_original_indices, training_config)
    """
    model_name_safe = model_name.replace("/", "_")
    config_dir = os.path.join(project_root, "ConfigTraining")

    config_pattern = os.path.join(
        config_dir, f"{model_name_safe}_{dataset_name}_pairs*.json"
    )
    config_matches = glob(config_pattern)

    config_path = None
    if config_matches:
        config_path = max(config_matches, key=os.path.getmtime)
        print(f"Training config trovata: {config_path}")
    else:
        legacy_config = os.path.join(config_dir, f"{model_name_safe}_{dataset_name}.json")
        if os.path.exists(legacy_config):
            config_path = legacy_config
            print(f"Training config legacy trovata: {config_path}")
        else:
            print("Warning: training config non trovata, nessun filtro train/val applicato")
            return None, None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            training_config = json.load(f)

        total_pairs = 0
        for key in ("total_pairs", "num_pairs", "total_training_pairs"):
            value = training_config.get(key)
            if value is not None:
                try:
                    total_pairs = int(value)
                    break
                except (TypeError, ValueError):
                    continue

        # Fallback per config che salvano solo instance_ids sintetici (pair_id*2 / pair_id*2+1)
        if total_pairs <= 0:
            synthetic_ids = []
            for key in ("train_instance_ids", "val_instance_ids"):
                for iid in training_config.get(key, []):
                    try:
                        iid_int = int(iid)
                    except (TypeError, ValueError):
                        continue
                    if iid_int >= 0:
                        synthetic_ids.append(iid_int)
            if synthetic_ids:
                total_pairs = (max(synthetic_ids) // 2) + 1

        if total_pairs <= 0:
            print("Warning: total_pairs non valido in training config, nessun filtro train/val applicato")
            return None, training_config

        if dataset_name == "halu_eval":
            from src.data.HaluEvalDataset import HaluEvalDataset

            try:
                ds_halu = HaluEvalDataset(label=0, use_local=False)
            except Exception:
                ds_halu = HaluEvalDataset(label=0, use_local=True)

            n_pairs = min(total_pairs, len(ds_halu))
            excluded_original_indices = set()
            for i in range(n_pairs):
                _, _, instance_id = ds_halu[i]
                excluded_original_indices.add(int(instance_id))
            print(
                "Esclusione HaluEval train/val: "
                f"{len(excluded_original_indices)} instance_id "
                f"(pairs={n_pairs})"
            )
            return excluded_original_indices, training_config

        from src.data.BeliefBankDataset import BeliefBankDataset

        bb_data_type = "facts"
        if dataset_name == "belief_bank_constraints":
            bb_data_type = "constraints"

        full_dataset = BeliefBankDataset(
            project_root=project_root,
            model_type="demo",
            recreate_ids=True,
            data_type=bb_data_type,
        )

        total_samples = len(full_dataset)
        excluded_original_indices = set()
        if dataset_name == "belief_bank_constraints":
            # Nel training constraints le coppie sono consecutive: (2*i, 2*i+1)
            for i in range(total_pairs):
                left = 2 * i
                right = left + 1
                if left < total_samples:
                    excluded_original_indices.add(left)
                if right < total_samples:
                    excluded_original_indices.add(right)
            print(
                "Esclusione BeliefBank constraints train/val: "
                f"{len(excluded_original_indices)} instance_id "
                f"(pairs={total_pairs}, schema=2*i/2*i+1)"
            )
        else:
            half = total_samples // 2
            n_pairs = min(total_pairs, half)
            for i in range(n_pairs):
                excluded_original_indices.add(i)
                idx_neg = i + half
                if idx_neg < total_samples:
                    excluded_original_indices.add(idx_neg)
            print(
                "Esclusione BeliefBank facts train/val: "
                f"{len(excluded_original_indices)} instance_id "
                f"(pairs={n_pairs}, half={half})"
            )
        return excluded_original_indices, training_config
    except Exception as e:
        print(f"Warning: errore nel caricamento training config: {e}")
        return None, None


def detect_cache_structure_type(
    project_root: str,
    cache_dir_name: str,
    model_name: str,
    dataset_name: str,
) -> str:
    """
    Rileva automaticamente se la struttura cache è:
    - new: activation_attn/hallucinated|not_hallucinated
    - old: generations/hallucination_labels.json
    """
    model_name_short = model_name.split("/")[-1]
    results_dir = os.path.join(project_root, cache_dir_name)
    base_path = os.path.join(results_dir, model_name_short, dataset_name, "activation_attn")
    hallucinated_path = os.path.join(base_path, "hallucinated")

    if os.path.isdir(hallucinated_path):
        return "new"
    return "old"


def load_labels_from_old_structure(
    project_root: str,
    cache_dir_name: str,
    model_name: str,
    dataset_name: str,
) -> list:
    """Carica labels da generations/hallucination_labels.json."""
    model_name_short = model_name.split("/")[-1]
    results_dir = os.path.join(project_root, cache_dir_name)
    gen_dir = os.path.join(results_dir, model_name_short, dataset_name, "generations")
    labels_path = os.path.join(gen_dir, "hallucination_labels.json")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"hallucination_labels.json non trovato: {labels_path}")

    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_labels_from_new_structure(
    project_root: str,
    cache_dir_name: str,
    model_name: str,
    dataset_name: str,
) -> list:
    """Carica labels da activation_attn/hallucinated|not_hallucinated."""
    model_name_short = model_name.split("/")[-1]
    results_dir = os.path.join(project_root, cache_dir_name)
    base_path = os.path.join(results_dir, model_name_short, dataset_name, "activation_attn")
    hallucinated_path = os.path.join(base_path, "hallucinated")
    not_hallucinated_path = os.path.join(base_path, "not_hallucinated")

    hall_ids_path = os.path.join(hallucinated_path, "layer0_instance_ids.json")
    not_hall_ids_path = os.path.join(not_hallucinated_path, "layer0_instance_ids.json")

    if not os.path.exists(hall_ids_path):
        raise FileNotFoundError(f"File instance_ids non trovato: {hall_ids_path}")
    if not os.path.exists(not_hall_ids_path):
        raise FileNotFoundError(f"File instance_ids non trovato: {not_hall_ids_path}")

    with open(hall_ids_path, "r", encoding="utf-8") as f:
        hallucinated_ids = json.load(f)
    with open(not_hall_ids_path, "r", encoding="utf-8") as f:
        not_hallucinated_ids = json.load(f)

    ids_concat = np.array(hallucinated_ids + not_hallucinated_ids)
    labels_concat = np.concatenate(
        [
            np.ones(len(hallucinated_ids), dtype=int),
            np.zeros(len(not_hallucinated_ids), dtype=int),
        ]
    )

    sort_indices = np.argsort(ids_concat)
    ids_sorted = ids_concat[sort_indices]
    labels_sorted = labels_concat[sort_indices]

    labels_data = []
    for instance_id, label in zip(ids_sorted, labels_sorted):
        labels_data.append(
            {
                "instance_id": int(instance_id),
                "is_hallucination": int(label),
            }
        )

    return labels_data


def evaluate_baseline_from_cache(
    experiment_id: str,
    model_name: str,
    dataset_name: str,
    dataset_train: str,
    num_samples: int,
    project_root: str,
    output_csv: str,
):
    """
    Baseline senza inferenza: recupera le allucinazioni dalla cache activation_cache.
    """
    print("[Baseline] Recupero metriche da cache (nessun caricamento modello)")

    cache_dir_name = "activation_cache"
    structure_type = detect_cache_structure_type(
        project_root, cache_dir_name, model_name, dataset_name
    )

    if structure_type == "new":
        labels_data = load_labels_from_new_structure(
            project_root, cache_dir_name, model_name, dataset_name
        )
    else:
        labels_data = load_labels_from_old_structure(
            project_root, cache_dir_name, model_name, dataset_name
        )

    excluded_ids, _ = load_training_config(project_root, model_name, dataset_name)
    if excluded_ids is not None:
        before = len(labels_data)
        labels_data = [
            item for item in labels_data if item.get("instance_id") not in excluded_ids
        ]
        print(
            f"[Baseline] Filtrati {before - len(labels_data)} campioni train/val; "
            f"rimasti {len(labels_data)}"
        )

    if num_samples > 0:
        labels_data = labels_data[:num_samples]

    total_evaluated = len(labels_data)
    n_hallucinations = sum(item.get("is_hallucination", 0) for item in labels_data)
    rate = n_hallucinations / total_evaluated if total_evaluated > 0 else 0.0

    print(
        f"[Baseline] Valutati {total_evaluated} campioni, "
        f"{n_hallucinations} allucinazioni ({rate*100:.2f}%)"
    )

    result = {
        "experiment_id": experiment_id,
        "type": "Baseline",
        "model": model_name,
        "dataset_eval": dataset_name,
        "dataset_train": dataset_train,
        "num_samples_evaluated": total_evaluated,
        "num_hallucinations": n_hallucinations,
        "hallucination_rate": rate,
        "slim_checkpoint": None,
        "num_pairs_train": None,
        "state_value": None,
        "slim_layer": None,
        "num_layers_total": None,
        "slim_training_time_seconds": None,
        "slim_inference_time_seconds": None,
        "slim_trainable_params": None,
        "learning_rate": None,
        "batch_size": None,
        "num_epochs": None,
        "highlight": None,
    }

    save_to_csv(output_csv, result)
    return result


def run_inference(
    experiment_id: str,
    experiment_type: str,
    model_name: str,
    dataset_name: str,
    dataset_train: str,
    slim_checkpoint: str,
    state_value: float,
    num_samples: int,
    project_root: str,
    device: str,
    output_csv: str,
    **kwargs,
):
    """
    Esegue un singolo esperimento di inferenza.

    Args:
        experiment_id: ID univoco dell'esperimento
        experiment_type: "Baseline", "SLiM", "CrossDataset" o "CrossModel"
        model_name: Nome modello HuggingFace
        dataset_name: Dataset di valutazione
        dataset_train: Dataset di training (per SLiM/CrossDataset/CrossModel)
        slim_checkpoint: Path al checkpoint SLiM (None per baseline)
        state_value: Valore dello stato (1.0 = truthful steering)
        num_samples: Numero di campioni da valutare (-1 = tutti)
        project_root: Root del progetto
        device: Device target
        output_csv: Path al CSV di output

    Returns:
        Dict con i risultati
    """
    print(f"\n{'='*60}")
    print(f"  Esperimento: {experiment_id}")
    print(f"  Tipo: {experiment_type}")
    print(f"  Modello: {model_name}")
    print(f"  Dataset eval: {dataset_name}")
    print(f"  Dataset train: {dataset_train}")
    print(f"  Checkpoint: {slim_checkpoint}")
    print(f"  State: {state_value}")
    print(f"{'='*60}\n")

    # Baseline: recupero da cache, senza caricare il modello
    if experiment_type == "Baseline":
        return evaluate_baseline_from_cache(
            experiment_id=experiment_id,
            model_name=model_name,
            dataset_name=dataset_name,
            dataset_train=dataset_train,
            num_samples=num_samples,
            project_root=project_root,
            output_csv=output_csv,
        )

    # 1. Carica tokenizer e modello
    print("[1/4] Caricamento modello...")
    tokenizer = load_tokenizer(model_name)
    bnb_config = create_bnb_config()
    base_model = load_llm(model_name, bnb_config, device=device)
    base_model = prepare_model_for_kbit_training(base_model)

    for param in base_model.parameters():
        param.requires_grad = False

    # 2. Setup SLiM (se non baseline)
    slim_model = None
    checkpoint_data = {}
    training_time = None
    trainable_params = None
    lr = None
    batch_size = None
    num_epochs_train = None
    num_pairs_train = None
    target_layer = None

    use_slim = experiment_type != "Baseline" and slim_checkpoint is not None

    if use_slim:
        print("[2/4] Caricamento SLiM...")

        # Carica checkpoint
        checkpoint_data = torch.load(slim_checkpoint, map_location="cpu")
        slim_args = checkpoint_data.get("args", {})

        training_time = checkpoint_data.get("training_time_seconds")
        trainable_params = slim_args.get("trainable_params")
        lr = slim_args.get("lr")
        batch_size = slim_args.get("batch_size")
        num_epochs_train = slim_args.get("epochs")
        num_pairs_train = slim_args.get("num_pairs")

        if experiment_type == "CrossModel":
            print("[2/4] Setup CrossModel con aligner Procrustes...")
            aligner_cache_dir = kwargs.get("aligner_cache", "alignment_procustes_SLiM")
            pre_aligner, post_aligner = load_crossmodel_aligners(
                project_root=project_root,
                experiment_id=experiment_id,
                aligner_cache_dir=aligner_cache_dir,
                device=device,
            )
            if pre_aligner is None:
                raise FileNotFoundError(
                    f"Pre-aligner non trovato per esperimento {experiment_id} "
                    f"in cache '{aligner_cache_dir}'"
                )
            if post_aligner is None:
                raise FileNotFoundError(
                    f"Post-aligner non trovato per esperimento {experiment_id} "
                    f"in cache '{aligner_cache_dir}'"
                )

            slim_model = CrossModelSLiMWrapper(
                base_model=base_model,
                checkpoint_data=checkpoint_data,
                pre_aligner=pre_aligner,
                post_aligner=post_aligner,
                alpha=kwargs.get("alpha", 1.0),
            )
            slim_model.to(device).eval()
            target_layer = slim_model.target_layer

            print(f"  CrossModel target layer (da checkpoint trainer): {target_layer}")
            print(f"  Trainer hidden size (checkpoint): {slim_model.trainer_hidden_size}")
            print(f"  SLiM alpha (steering strength): {slim_model.alpha}")
        else:
            state_dim = slim_args.get("state_dim", 1)
            target_layer = checkpoint_data.get(
                "target_layer",
                slim_args.get("target_layer", 0)
            )
            slim_rank = slim_args.get("slim_rank", 32)

            # Crea SLiMedNet e carica pesi
            slim_model = GeneralSLiMedNet(
                model=base_model,
                state_embed_dim=state_dim,
                target_layer=target_layer,
                slim_rank=slim_rank,
            )

            # Carica solo i pesi SLiM
            slim_state_dict = checkpoint_data.get("slim_state_dict", {})
            missing, unexpected = slim_model.load_state_dict(slim_state_dict, strict=False)

            # Sposta moduli SLiM su device
            slim_model.state_proj = slim_model.state_proj.to(device)
            slim_model.SLiM_scale = slim_model.SLiM_scale.to(device)
            slim_model.SLiM_shift = slim_model.SLiM_shift.to(device)

            print(f"  Pesi SLiM caricati. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            print(f"  SLiM applicato al layer: {target_layer}")

            slim_model.eval()

            # Set steering strength if provided via alpha kwarg
            if hasattr(slim_model, "alpha"):
                slim_model.alpha = kwargs.get("alpha", 1.0)
                print(f"  SLiM alpha (steering strength): {slim_model.alpha}")
    else:
        print("[2/4] Baseline - nessun SLiM")

    # 3. Carica dataset per valutazione
    print("[3/4] Caricamento dataset di valutazione...")
    dataset, valid_indices, is_halu_eval = load_dataset_for_eval(dataset_name, project_root)

    excluded_ids, _ = load_training_config(project_root, model_name, dataset_name)
    if excluded_ids is not None:
        before = len(valid_indices)
        valid_indices = [
            idx for idx in valid_indices if dataset[idx][2] not in excluded_ids
        ]
        print(
            f"Filtrati {before - len(valid_indices)} campioni train/val; "
            f"rimasti {len(valid_indices)}"
        )

    if num_samples > 0:
        valid_indices = valid_indices[:num_samples]

    if is_halu_eval:
        prompt_template = PROMPT_HALU
        max_new_tokens = 100
        print(f"  Modalità HaluEval ({len(valid_indices)} campioni)")
    else:
        prompt_template = PROMPT_QA
        max_new_tokens = 5
        print(f"  Modalità BeliefBank ({len(valid_indices)} campioni)")

    # 4. Loop di inferenza
    print("[4/4] Inferenza...")
    n_hallucinations = 0
    total_evaluated = 0
    inference_results = []

    # Prepara il tensore di stato
    state_tensor_batch = None
    state_tensor_0_batch = None
    if use_slim:
        state_tensor_batch = torch.FloatTensor([[state_value]]).to(device)

    inference_start_time = time.time()

    for idx in tqdm(valid_indices, desc=f"{experiment_id}"):
        fact, label, instance_id = dataset[idx]
        question = prompt_template.format(question=fact)

        inputs = tokenizer(question, return_tensors="pt").to(device)

        with torch.no_grad():
            if use_slim and slim_model is not None:
                outputs = slim_model.generate(
                    input_ids=inputs["input_ids"],
                    state_tensor=state_tensor_batch,
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    )
            else:
                outputs = base_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        is_hallucinated = evaluate_hallucination(generated_text, label, is_halu_eval)

        if is_hallucinated:
            n_hallucinations += 1
        total_evaluated += 1

        # Debug primi 5 campioni
        if total_evaluated <= 5:
            print(f"\n{'─'*60}")
            print(f"  [{total_evaluated}/5] Prompt: {fact[:100]}...")
            print(f"  Generato: {generated_text}")
            print(f"  Atteso:   {label}")
            print(f"  Allucinazione: {'SÌ ❌' if is_hallucinated else 'NO ✅'}")
            print(f"{'─'*60}")

        inference_results.append({
            "instance_id": instance_id,
            "prompt": fact,
            "generated_answer": generated_text,
            "expected_answer": label,
            "is_hallucination": int(is_hallucinated),
        })

    inference_time = time.time() - inference_start_time
    rate = n_hallucinations / total_evaluated if total_evaluated > 0 else 0

    print(f"\n{'='*40}")
    print(f"  Risultati: {experiment_id}")
    print(f"  Valutati: {total_evaluated}")
    print(f"  Allucinazioni: {n_hallucinations} ({rate*100:.2f}%)")
    print(f"  Tempo: {inference_time:.2f}s")
    print(f"{'='*40}")

    # Salva dettagli per campione in JSON
    results_dir = os.path.join(project_root, "InferenceResults", "SLiM")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{experiment_id}_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(inference_results, f, indent=2, ensure_ascii=False)
    print(f"Dettagli salvati: {results_path}")

    # Costruisci result dict
    result = {
        "experiment_id": experiment_id,
        "type": experiment_type,
        "model": model_name,
        "dataset_eval": dataset_name,
        "dataset_train": dataset_train,
        "num_samples_evaluated": total_evaluated,
        "num_hallucinations": n_hallucinations,
        "hallucination_rate": rate,
        "slim_checkpoint": slim_checkpoint,
        "num_pairs_train": num_pairs_train,
        "state_value": state_value,
        "slim_layer": target_layer,
        "num_layers_total": slim_model.num_layers if slim_model else None,
        "slim_training_time_seconds": training_time,
        "slim_inference_time_seconds": inference_time,
        "slim_trainable_params": trainable_params,
        "learning_rate": lr,
        "batch_size": batch_size,
        "num_epochs": num_epochs_train,
        "highlight": None,
    }

    # Salva nel CSV
    save_to_csv(output_csv, result)

    # Cleanup
    if slim_model:
        slim_model.remove_hooks()
        del slim_model
    del base_model
    torch.cuda.empty_cache()

    return result


# =============================================================================
# PREDEFINED EXPERIMENTS (stile TruthX)
# =============================================================================
# slim_checkpoint_dir: percorso relativo alla directory del checkpoint SLiM.
#   Il runner risolve automaticamente al _best.pth più recente nella cartella.
#   Usare None per esperimenti Baseline.

EXPERIMENTS = {
    "Gemma_to_Llama_BBF_PROCRUSTES": {
        "type": "CrossModel",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "dataset_eval": "belief_bank_facts",
        "dataset_train": "belief_bank_facts",
        "slim_checkpoint_dir": "SteeringVectors/SLiM/google_gemma-2-9b-it/belief_bank_facts",
        "aligner_cache": "alignment_procustes_SLiM",
        "state_value": 1.0,
    },
    "Gemma_to_Llama_BBC_PROCRUSTES": {
        "type": "CrossModel",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "dataset_eval": "belief_bank_constraints",
        "dataset_train": "belief_bank_constraints",
        "slim_checkpoint_dir": "SteeringVectors/SLiM/google_gemma-2-9b-it/belief_bank_constraints",
        "aligner_cache": "alignment_procustes_SLiM",
        "state_value": 1.0,
    },
    "Gemma_to_Llama_HE_PROCRUSTES": {
        "type": "CrossModel",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "dataset_eval": "halu_eval",
        "dataset_train": "halu_eval",
        "slim_checkpoint_dir": "SteeringVectors/SLiM/google_gemma-2-9b-it/halu_eval",
        "aligner_cache": "alignment_procustes_SLiM",
        "state_value": 1.0,
    },
    "Llama_to_Gemma_BBF_PROCRUSTES": {
        "type": "CrossModel",
        "model": "google/gemma-2-9b-it",
        "dataset_eval": "belief_bank_facts",
        "dataset_train": "belief_bank_facts",
        "slim_checkpoint_dir": "SteeringVectors/SLiM/meta-llama_Llama-3.1-8B-Instruct/belief_bank_facts",
        "aligner_cache": "alignment_procustes_SLiM",
        "state_value": 1.0,
    },
    "Llama_to_Gemma_BBC_PROCRUSTES": {
        "type": "CrossModel",
        "model": "google/gemma-2-9b-it",
        "dataset_eval": "belief_bank_constraints",
        "dataset_train": "belief_bank_constraints",
        "slim_checkpoint_dir": "SteeringVectors/SLiM/meta-llama_Llama-3.1-8B-Instruct/belief_bank_constraints",
        "aligner_cache": "alignment_procustes_SLiM",
        "state_value": 1.0,
    },
    "Llama_to_Gemma_HE_PROCRUSTES": {
        "type": "CrossModel",
        "model": "google/gemma-2-9b-it",
        "dataset_eval": "halu_eval",
        "dataset_train": "halu_eval",
        "slim_checkpoint_dir": "SteeringVectors/SLiM/meta-llama_Llama-3.1-8B-Instruct/halu_eval",
        "aligner_cache": "alignment_procustes_SLiM",
        "state_value": 1.0,
    },
}

# =============================================================================
# EXPERIMENTS DEFINITION
# =============================================================================

def build_experiments(
    models: list,
    datasets: list,
    checkpoint_dir: str,
    state_value: float = 1.0,
) -> dict:
    """
    Costruisce il dizionario esperimenti per tutti i modelli e dataset.

    Genera automaticamente:
    1. Baseline per ogni (model, dataset)
    2. SLiM per ogni (model, dataset) — same dataset train/eval
    3. CrossDataset per ogni (model, dataset_train ≠ dataset_eval)

    Il target layer viene letto dal checkpoint.

    Args:
        models: Lista nomi modelli
        datasets: Lista nomi dataset
        checkpoint_dir: Directory base dei checkpoint SLiM
        state_value: Valore dello stato per il steering

    Returns:
        Dict {experiment_id: config_dict}
    """
    experiments = {}

    for model_name in models:
        model_safe = model_name.replace("/", "_")

        for ds in datasets:
            # --- Baseline ---
            exp_id = f"Baseline_{model_safe}_{ds}"
            experiments[exp_id] = {
                "type": "Baseline",
                "model": model_name,
                "dataset_eval": ds,
                "dataset_train": None,
                "slim_checkpoint": None,
                "state_value": state_value,
            }

            # --- SLiM (same dataset) ---
            ckpt_dir = os.path.join(checkpoint_dir, model_safe, ds)
            ckpt = find_latest_checkpoint(ckpt_dir)

            if ckpt:
                exp_id = f"SLiM_{model_safe}_{ds}"
                experiments[exp_id] = {
                    "type": "SLiM",
                    "model": model_name,
                    "dataset_eval": ds,
                    "dataset_train": ds,
                    "slim_checkpoint": ckpt,
                    "state_value": state_value,
                }

            # --- CrossDataset ---
            for ds_train in datasets:
                if ds_train == ds:
                    continue

                ckpt_dir = os.path.join(checkpoint_dir, model_safe, ds_train)
                ckpt = find_latest_checkpoint(ckpt_dir)

                if ckpt:
                    exp_id = f"CrossDS_{model_safe}_train{ds_train}_eval{ds}"
                    experiments[exp_id] = {
                        "type": "CrossDataset",
                        "model": model_name,
                        "dataset_eval": ds,
                        "dataset_train": ds_train,
                        "slim_checkpoint": ckpt,
                        "state_value": state_value,
                    }

    return experiments


def find_best_checkpoint(directory: str) -> str:
    """
    Trova il checkpoint migliore in una directory.

    Priorità:
    1. *_best.pth  (salvato da train_slim con early stopping)
    2. Qualsiasi .pth più recente come fallback

    Returns:
        Percorso assoluto al checkpoint, o None se la directory non esiste
        o non contiene checkpoint.
    """
    if not os.path.isdir(directory):
        return None

    all_pth = [f for f in os.listdir(directory) if f.endswith(".pth")]
    if not all_pth:
        return None

    # Preferisci _best.pth
    best = [f for f in all_pth if f.endswith("_best.pth")]
    if best:
        # Se ci sono più _best.pth (run diverse), prendi il più recente
        best.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)
        return os.path.join(directory, best[0])

    # Fallback: qualsiasi .pth più recente
    all_pth.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)
    return os.path.join(directory, all_pth[0])


def find_latest_checkpoint(directory: str) -> str:
    """Alias mantenuto per compatibilità con build_experiments()."""
    return find_best_checkpoint(directory)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SLiM Hallucination Reduction - Inference")

    # Modalità singolo esperimento
    parser.add_argument("--model_name", type=str, default=None, help="Modello da valutare")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset di valutazione")
    parser.add_argument("--slim_checkpoint", type=str, default=None, help="Path checkpoint SLiM")
    parser.add_argument("--dataset_train", type=str, default=None, help="Dataset di training (per CrossDataset)")
    parser.add_argument("--state_value", type=float, default=1.0, help="Valore dello stato")
    parser.add_argument("--alpha", type=float, default=1.0, help="SLiM steering strength (0.0=no steering, 1.0=full)")
    parser.add_argument(
        "--aligner_cache",
        type=str,
        default="alignment_procustes_SLiM",
        help="Cache aligner per CrossModel (default: alignment_procustes_SLiM)",
    )
    parser.add_argument("--experiment_type", type=str, default="SLiM",
                        choices=["Baseline", "SLiM", "CrossDataset","CrossModel"])

    # Modalità batch (tutti gli esperimenti)
    parser.add_argument("--run_all", action="store_true", help="Esegui tutti gli esperimenti", default=False)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Lista modelli per modalità batch")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Lista dataset per modalità batch")

    # Opzioni comuni
    parser.add_argument("--num_samples", type=int, default=-1, help="Campioni da valutare (-1 = tutti)")
    parser.add_argument("--device", type=str, default="cuda:2", help="Device")
    parser.add_argument("--project_root", type=str, default=".", help="Root progetto")
    parser.add_argument("--output_csv", type=str, default="SLiMExperiments.csv", help="File CSV output")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory base checkpoint (default: SteeringVectors/SLiM)")
    parser.add_argument("--failure_log", type=str, default="slim_log.txt", help="Log fallimenti")
    parser.add_argument(
        "--use_predefined_experiments",
        action="store_true",
        help="In modalità --run_all usa il dizionario EXPERIMENTS predefinito",
    )

    args = parser.parse_args()
    if isinstance(args.device, str) and args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(args.device))
    project_root = os.path.abspath(args.project_root)
    output_csv = os.path.join(project_root, args.output_csv) if not os.path.isabs(args.output_csv) else args.output_csv
    failure_log = os.path.join(project_root, args.failure_log) if not os.path.isabs(args.failure_log) else args.failure_log
    checkpoint_dir = args.checkpoint_dir or os.path.join(project_root, "SteeringVectors", "SLiM")

    if args.run_all or (not args.model_name and not args.dataset):
        # ============================
        # Modalità batch
        # ============================
        if args.use_predefined_experiments or args.models is None:
            experiments = EXPERIMENTS
            print(f"Uso esperimenti predefiniti da EXPERIMENTS ({len(experiments)} totali)")
        else:
            models = args.models
            datasets = args.datasets or [
                "belief_bank_facts",
                "belief_bank_constraints",
                "halu_eval",
            ]
            experiments = build_experiments(
                models=models,
                datasets=datasets,
                checkpoint_dir=checkpoint_dir,
                state_value=args.state_value,
            )

        print(f"Trovati {len(experiments)} esperimenti da eseguire.\n")

        completed = 0
        failed = 0
        skipped = 0

        for exp_id, exp_config in experiments.items():
            try:
                # Risolvi slim_checkpoint_dir → file _best.pth
                slim_ckpt = exp_config.get("slim_checkpoint")  # path file esplicito (legacy)
                ckpt_dir = exp_config.get("slim_checkpoint_dir")  # directory (nuovo stile)

                if slim_ckpt is None and ckpt_dir is not None:
                    # Risolvi percorso relativo rispetto al project_root
                    if not os.path.isabs(ckpt_dir):
                        ckpt_dir = os.path.join(project_root, ckpt_dir)
                    slim_ckpt = find_best_checkpoint(ckpt_dir)
                    if slim_ckpt is None and exp_config["type"] != "Baseline":
                        print(
                            f"[SKIP] {exp_id}: nessun checkpoint trovato in {ckpt_dir}"
                        )
                        skipped += 1
                        continue
                elif slim_ckpt and not os.path.isabs(slim_ckpt):
                    slim_ckpt = os.path.join(project_root, slim_ckpt)

                run_inference(
                    experiment_id=exp_id,
                    experiment_type=exp_config["type"],
                    model_name=exp_config["model"],
                    dataset_name=exp_config["dataset_eval"],
                    dataset_train=exp_config.get("dataset_train"),
                    slim_checkpoint=slim_ckpt,
                    state_value=exp_config.get("state_value", 1.0),
                    num_samples=args.num_samples,
                    project_root=project_root,
                    device=args.device,
                    output_csv=output_csv,
                    alpha=args.alpha,
                    aligner_cache=exp_config.get("aligner_cache", args.aligner_cache),
                )
                completed += 1
            except Exception as e:
                tb = traceback.format_exc()
                failed += 1
                print(f"\n[ERRORE] {exp_id}: {e}")
                with open(failure_log, "a") as f:
                    f.write(f"\n{'='*60}\n{exp_id}\n{tb}\n")
                continue

        print(f"\nCompletati: {completed} | Saltati: {skipped} | Falliti: {failed}")

    else:
        # ============================
        # Modalità singolo esperimento
        # ============================
        if not args.model_name or not args.dataset:
            parser.error("--model_name e --dataset sono richiesti in modalità singola")

        model_safe = args.model_name.replace("/", "_")
        ds_train = args.dataset_train or args.dataset
        exp_id = f"{args.experiment_type}_{model_safe}_{args.dataset}"

        run_inference(
            experiment_id=exp_id,
            experiment_type=args.experiment_type,
            model_name=args.model_name,
            dataset_name=args.dataset,
            dataset_train=ds_train,
            slim_checkpoint=args.slim_checkpoint,
            state_value=args.state_value,
            num_samples=args.num_samples,
            project_root=project_root,
            device=args.device,
            output_csv=output_csv,
            alpha=args.alpha,
            aligner_cache=args.aligner_cache,
        )


if __name__ == "__main__":
    main()
