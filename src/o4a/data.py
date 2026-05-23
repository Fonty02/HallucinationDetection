"""Data loading, balancing, splitting — shared across all methods."""

import json
import os
import random

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .config import (
    CACHE_DIR_NAME, ROOT_DIR, SEED,
    TRAIN_SPLIT, ALIGNMENT_SPLIT, PROBER_VAL_SPLIT,
    MODEL_ALIASES,
)


# ==================================================================
# PERFORMANCE OPTIMIZATION HELPERS
# ==================================================================

def apply_performance_optimizations() -> dict:
    """
    Apply PyTorch performance optimizations based on environment variables.
    Returns a dict of applied settings for logging.
    """
    settings = {}

    # cuDNN benchmark mode - faster but non-deterministic
    cudnn_benchmark = os.environ.get("O4A_CUDNN_BENCHMARK", "false").lower() == "true"
    if cudnn_benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        settings["cudnn_benchmark"] = True
    else:
        settings["cudnn_benchmark"] = False

    # TF32 for Ampere+ GPUs (faster matrix ops)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        settings["tf32"] = True

    # torch.compile settings (PyTorch 2.0+)
    compile_model = os.environ.get("O4A_COMPILE_MODEL", "false").lower() == "true"
    settings["compile_model"] = compile_model

    # Mixed precision settings
    use_amp = os.environ.get("O4A_USE_AMP", "false").lower() == "true"
    settings["use_amp"] = use_amp

    return settings


def get_dataloader_kwargs() -> dict:
    """
    Get optimized DataLoader kwargs based on environment variables.
    For high RAM/CPU utilization with parallel data loading.
    """
    num_workers = int(os.environ.get("O4A_NUM_WORKERS", "0"))
    pin_memory = os.environ.get("O4A_PIN_MEMORY", "false").lower() == "true"
    prefetch_factor = int(os.environ.get("O4A_PREFETCH_FACTOR", "2"))

    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory and torch.cuda.is_available(),
    }

    # prefetch_factor only valid when num_workers > 0
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = True  # Keep workers alive between batches

    return kwargs


# ==================================================================
# REPRODUCIBILITY
# ==================================================================

def set_seed(seed: int = SEED, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Only set deterministic mode if not using cudnn_benchmark optimization
    cudnn_benchmark = os.environ.get("O4A_CUDNN_BENCHMARK", "false").lower() == "true"
    if deterministic and not cudnn_benchmark:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_generator(seed: int = SEED) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ==================================================================
# BALANCING UTILITIES
# ==================================================================

def get_balanced_indices(y: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Deterministic undersampling to the minority class size."""
    rng = np.random.RandomState(seed)
    unique_classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    selected = []
    for cls in unique_classes:
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) > min_count:
            sampled = rng.choice(cls_idx, size=min_count, replace=False)
            selected.extend(sampled)
        else:
            selected.extend(cls_idx)
    return np.sort(np.array(selected))


# ==================================================================
# DATA MANAGER
# ==================================================================

class DataManager:
    _cache_root = os.path.join(ROOT_DIR, CACHE_DIR_NAME)

    @classmethod
    def _resolve_model(cls, model: str) -> str:
        return MODEL_ALIASES.get(model, model)

    @classmethod
    def _activation_dir(cls, model: str, dataset: str, layer_type: str) -> str:
        model_dir = cls._resolve_model(model)
        return os.path.join(cls._cache_root, model_dir, dataset, f"activation_{layer_type}")

    @classmethod
    def detect_structure(cls, model: str, dataset: str, layer_type: str) -> str:
        """Return 'new' if hallucinated/not_hallucinated sub-dirs exist, else 'old'."""
        d = cls._activation_dir(model, dataset, layer_type)
        return "new" if os.path.isdir(os.path.join(d, "hallucinated")) else "old"

    # --- stats (for concordant sampling via instance ids) ---

    @classmethod
    def get_stats(cls, model: str, dataset: str, layer_type: str = "attn") -> dict:
        structure = cls.detect_structure(model, dataset, layer_type)
        if structure == "new":
            return cls._stats_new(model, dataset, layer_type)
        return cls._stats_old(model, dataset)

    @classmethod
    def _stats_old(cls, model: str, dataset: str) -> dict:
        model_dir = cls._resolve_model(model)
        p = os.path.join(cls._cache_root, model_dir, dataset, "generations", "hallucination_labels.json")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        hall = [it["instance_id"] for it in data if it["is_hallucination"]]
        not_hall = [it["instance_id"] for it in data if not it["is_hallucination"]]
        return {"total": len(data), "hallucinated_ids": hall, "not_hallucinated_ids": not_hall}

    @classmethod
    def _stats_new(cls, model: str, dataset: str, layer_type: str = "attn") -> dict:
        base = cls._activation_dir(model, dataset, layer_type)
        with open(os.path.join(base, "hallucinated", "layer0_instance_ids.json"), "r") as f:
            hall = json.load(f)
        with open(os.path.join(base, "not_hallucinated", "layer0_instance_ids.json"), "r") as f:
            not_hall = json.load(f)
        return {"total": len(hall) + len(not_hall), "hallucinated_ids": hall, "not_hallucinated_ids": not_hall}

    # --- single-layer loading ---

    @classmethod
    def load_activations_and_labels(cls, model: str, dataset: str, layer: int, layer_type: str):
        """Return (X, y, instance_ids) for one layer."""
        structure = cls.detect_structure(model, dataset, layer_type)
        base = cls._activation_dir(model, dataset, layer_type)

        if structure == "new":
            hall_act = torch.load(os.path.join(base, "hallucinated", f"layer{layer}_activations.pt"), map_location="cpu")
            not_act = torch.load(os.path.join(base, "not_hallucinated", f"layer{layer}_activations.pt"), map_location="cpu")
            with open(os.path.join(base, "hallucinated", f"layer{layer}_instance_ids.json")) as f:
                hall_ids = json.load(f)
            with open(os.path.join(base, "not_hallucinated", f"layer{layer}_instance_ids.json")) as f:
                not_ids = json.load(f)
            h = hall_act.cpu().float().numpy() if isinstance(hall_act, torch.Tensor) else hall_act.astype(np.float32)
            n = not_act.cpu().float().numpy() if isinstance(not_act, torch.Tensor) else not_act.astype(np.float32)
            X = np.vstack([h, n])
            y = np.concatenate([np.ones(h.shape[0], dtype=int), np.zeros(n.shape[0], dtype=int)])
            ids = np.array(hall_ids + not_ids)
            order = np.argsort(ids)
            return X[order], y[order], ids[order]
        else:
            act = torch.load(os.path.join(base, f"layer{layer}_activations.pt"), map_location="cpu")
            X = act.cpu().float().numpy() if isinstance(act, torch.Tensor) else act.astype(np.float32)
            model_dir = cls._resolve_model(model)
            lp = os.path.join(cls._cache_root, model_dir, dataset, "generations", "hallucination_labels.json")
            with open(lp, "r") as f:
                labels = json.load(f)
            y = np.array([it["is_hallucination"] for it in labels], dtype=int)
            return X, y, np.arange(len(y))

    # --- multi-layer concatenation ---

    @classmethod
    def load_concatenated_layers(cls, model: str, dataset: str, layer_indices: list, layer_type: str):
        """Load & concatenate activations for the requested layers. Returns (X, y)."""
        combined, common_y = [], None
        for idx in layer_indices:
            X_l, y_l, _ = cls.load_activations_and_labels(model, dataset, idx, layer_type)
            if common_y is None:
                common_y = y_l
            elif not np.array_equal(common_y, y_l):
                raise ValueError(f"Label mismatch at layer {idx} for {model}")
            combined.append(X_l)
        return np.concatenate(combined, axis=1), common_y


# ==================================================================
# CONCORDANT SAMPLING (for alignment-based methods)
# ==================================================================

def get_concordant_indices_and_undersample(stats_a: dict, stats_b: dict, seed: int = SEED):
    """Find samples where both models agree on the label, then undersample to balance."""
    hall_a = set(stats_a["hallucinated_ids"])
    hall_b = set(stats_b["hallucinated_ids"])
    all_a = set(stats_a["hallucinated_ids"] + stats_a["not_hallucinated_ids"])
    all_b = set(stats_b["hallucinated_ids"] + stats_b["not_hallucinated_ids"])
    common = sorted(all_a & all_b)
    if not common:
        raise ValueError("No common instance ids between models")

    y1 = np.array([1 if i in hall_a else 0 for i in common], dtype=np.int8)
    y2 = np.array([1 if i in hall_b else 0 for i in common], dtype=np.int8)
    mask = y1 == y2
    conc_ids = np.array(common)[mask]
    conc_labels = y1[mask]

    n_hall = int((conc_labels == 1).sum())
    n_non = int((conc_labels == 0).sum())
    mc = min(n_hall, n_non)
    if mc == 0:
        raise ValueError("Not enough concordant samples to balance")

    rng = np.random.RandomState(seed)
    h_sampled = rng.choice(conc_ids[conc_labels == 1], size=mc, replace=False)
    n_sampled = rng.choice(conc_ids[conc_labels == 0], size=mc, replace=False)
    bal_ids = np.concatenate([h_sampled, n_sampled])
    bal_labels = np.concatenate([np.ones(mc, dtype=np.int8), np.zeros(mc, dtype=np.int8)])
    perm = rng.permutation(len(bal_ids))
    return bal_ids[perm], bal_labels[perm]


def get_undersampled_indices_per_model(stats: dict, seed: int = SEED):
    """Build label array from stats, then undersample."""
    hall_set = set(stats["hallucinated_ids"])
    y = np.array([1 if i in hall_set else 0 for i in range(stats["total"])], dtype=np.int8)
    idx = get_balanced_indices(y, seed)
    return idx, y[idx]


# ==================================================================
# SHARED DATA PREPARATION
# ==================================================================

def prepare_shared_data(experiment: dict, layer_type: str):
    """
    Prepare ALL data splits ONCE for a given (experiment, layer_type).
    Returns a dict consumed by every method (same data → fair comparison).

    Keys in the returned dict
    -------------------------
    trainer / tester : dict with X_train, X_test, y_train, y_test, X_test_raw
    prober_split     : dict with train_idx/val_idx for shared prober split
    alignment        : dict with X_trainer_train/val, X_tester_train/val, scaler_trainer, scaler_tester
    trainer_name / tester_name / layer_type / dataset
    """
    set_seed(SEED)

    dataset = experiment["dataset"]
    trainer_name = experiment["trainer"]
    tester_name = experiment["tester"]
    trainer_layers = experiment["trainer_layers"][layer_type]
    tester_layers = experiment["tester_layers"][layer_type]

    # ---- load full activations ----
    X_trainer_full, _ = DataManager.load_concatenated_layers(
        trainer_name, dataset, trainer_layers, layer_type
    )
    X_tester_full, _ = DataManager.load_concatenated_layers(
        tester_name, dataset, tester_layers, layer_type
    )

    # ---- stats for concordant alignment ----
    stats_trainer = DataManager.get_stats(trainer_name, dataset)
    stats_tester = DataManager.get_stats(tester_name, dataset)
    align_indices, _ = get_concordant_indices_and_undersample(stats_trainer, stats_tester, SEED)

    rng_align = np.random.RandomState(SEED)
    perm_align = rng_align.permutation(len(align_indices))
    split_a = int(ALIGNMENT_SPLIT * len(align_indices))
    align_train_local = perm_align[:split_a]
    align_val_local = perm_align[split_a:]

    X_align_trainer = X_trainer_full[align_indices]
    X_align_tester = X_tester_full[align_indices]

    X_align_trainer_train = X_align_trainer[align_train_local]
    X_align_trainer_val = X_align_trainer[align_val_local]
    X_align_tester_train = X_align_tester[align_train_local]
    X_align_tester_val = X_align_tester[align_val_local]

    # ---- per-model balanced indices for probing ----
    idx_trainer_bal, y_trainer_bal = get_undersampled_indices_per_model(stats_trainer, SEED)
    idx_tester_bal, y_tester_bal = get_undersampled_indices_per_model(stats_tester, SEED)

    X_trainer_bal = X_trainer_full[idx_trainer_bal]
    X_tester_bal = X_tester_full[idx_tester_bal]

    # train/test split
    rng_t = np.random.RandomState(SEED)
    rng_s = np.random.RandomState(SEED + 1)

    perm_t = rng_t.permutation(len(X_trainer_bal))
    perm_s = rng_s.permutation(len(X_tester_bal))

    sp_t = int(TRAIN_SPLIT * len(X_trainer_bal))
    sp_s = int(TRAIN_SPLIT * len(X_tester_bal))

    X_trainer_train_raw = X_trainer_bal[perm_t[:sp_t]]
    X_trainer_test_raw = X_trainer_bal[perm_t[sp_t:]]
    y_trainer_train = y_trainer_bal[perm_t[:sp_t]]
    y_trainer_test = y_trainer_bal[perm_t[sp_t:]]

    X_tester_train_raw = X_tester_bal[perm_s[:sp_s]]
    X_tester_test_raw = X_tester_bal[perm_s[sp_s:]]
    y_tester_train = y_tester_bal[perm_s[:sp_s]]
    y_tester_test = y_tester_bal[perm_s[sp_s:]]

    # ---- prober split (shared across methods) ----
    rng_p = np.random.RandomState(SEED)
    perm_p = rng_p.permutation(len(X_trainer_train_raw))
    v_p = int(PROBER_VAL_SPLIT * len(X_trainer_train_raw))
    prober_val_idx = perm_p[:v_p]
    prober_train_idx = perm_p[v_p:]

    # ---- scalers ----
    scaler_trainer = StandardScaler().fit(X_trainer_train_raw)
    scaler_tester = StandardScaler().fit(X_tester_train_raw)
    scaler_align_trainer = StandardScaler().fit(X_align_trainer_train)
    scaler_align_tester = StandardScaler().fit(X_align_tester_train)

    return {
        "trainer_name": trainer_name,
        "tester_name": tester_name,
        "dataset": dataset,
        "layer_type": layer_type,
        "trainer": {
            "X_train": scaler_trainer.transform(X_trainer_train_raw),
            "X_test": scaler_trainer.transform(X_trainer_test_raw),
            "X_train_raw": X_trainer_train_raw,
            "X_test_raw": X_trainer_test_raw,
            "y_train": y_trainer_train,
            "y_test": y_trainer_test,
            "scaler": scaler_trainer,
        },
        "tester": {
            "X_train": scaler_tester.transform(X_tester_train_raw),
            "X_test": scaler_tester.transform(X_tester_test_raw),
            "X_train_raw": X_tester_train_raw,
            "X_test_raw": X_tester_test_raw,
            "y_train": y_tester_train,
            "y_test": y_tester_test,
            "scaler": scaler_tester,
        },
        "prober_split": {
            "train_idx": prober_train_idx,
            "val_idx": prober_val_idx,
            "val_ratio": PROBER_VAL_SPLIT,
        },
        "alignment": {
            "X_trainer_train": scaler_align_trainer.transform(X_align_trainer_train),
            "X_trainer_val": scaler_align_trainer.transform(X_align_trainer_val),
            "X_tester_train": scaler_align_tester.transform(X_align_tester_train),
            "X_tester_val": scaler_align_tester.transform(X_align_tester_val),
            "scaler_trainer": scaler_align_trainer,
            "scaler_tester": scaler_align_tester,
        },
    }
