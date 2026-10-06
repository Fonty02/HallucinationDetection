"""Data loading, balancing, splitting — shared across all methods."""

import json
import os
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    CACHE_DIR_NAME, DEVICE, ROOT_DIR, SEED,
    TRAIN_SPLIT, ALIGNMENT_SPLIT,
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
        """Load & concatenate activations for the requested layers. Returns (X, y, ids)."""
        combined, common_y, common_ids = [], None, None
        for idx in layer_indices:
            X_l, y_l, ids_l = cls.load_activations_and_labels(model, dataset, idx, layer_type)
            if common_y is None:
                common_y = y_l
                common_ids = ids_l
            elif not np.array_equal(common_y, y_l):
                raise ValueError(f"Label mismatch at layer {idx} for {model}")
            elif not np.array_equal(common_ids, ids_l):
                raise ValueError(f"Instance id mismatch at layer {idx} for {model}")
            combined.append(X_l)
        return np.concatenate(combined, axis=1), common_y, common_ids


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

def split_and_balance(y_all: np.ndarray, split_seed: int, test_seed: int, balance_seed: int):
    """
    Stratified train/val/test split (70/15/15) preserving class ratios, then
    undersample only the training portion to the minority class.

    Returns (train_pos, train_bal_pos, val_pos, test_pos) as positional indices into y_all.
    """
    all_idx = np.arange(len(y_all))
    train_pos, valtest_pos = train_test_split(
        all_idx, test_size=1.0 - TRAIN_SPLIT, random_state=split_seed, stratify=y_all,
    )
    val_pos, test_pos = train_test_split(
        valtest_pos, test_size=0.5, random_state=test_seed, stratify=y_all[valtest_pos],
    )

    rng_bal = np.random.RandomState(balance_seed)
    y_train_part = y_all[train_pos]
    unique, counts = np.unique(y_train_part, return_counts=True)
    min_count = counts.min()
    selected = []
    for cls in unique:
        cls_pos = train_pos[np.where(y_train_part == cls)[0]]
        if len(cls_pos) > min_count:
            selected.extend(rng_bal.choice(cls_pos, size=min_count, replace=False))
        else:
            selected.extend(cls_pos)
    train_bal_pos = np.sort(np.array(selected))
    return train_pos, train_bal_pos, val_pos, test_pos


def labels_from_stats(stats: dict) -> np.ndarray:
    """Full label array (natural distribution) indexed by instance id."""
    hall_set = set(stats["hallucinated_ids"])
    return np.array([1 if i in hall_set else 0 for i in range(stats["total"])], dtype=np.int8)


def load_stratified_test_set(
    model_name: str,
    dataset_name: str,
    layer_indices: list[int],
    layer_type: str,
    test_size: float = 0.15,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a stratified test split from an activation dataset (used as cross-domain target)."""
    X_full, _, _ = DataManager.load_concatenated_layers(model_name, dataset_name, layer_indices, layer_type)
    stats = DataManager.get_stats(model_name, dataset_name, layer_type=layer_type)
    y = labels_from_stats(stats)
    all_idx = np.arange(stats["total"])
    _, test_idx = train_test_split(all_idx, test_size=test_size, random_state=seed, stratify=y)
    return X_full[test_idx], y[test_idx]


def prepare_single_model_data(model: str, dataset: str, layer_indices: list[int], layer_type: str) -> dict:
    """
    Single-LLM counterpart of prepare_shared_data: the same splits/scaling used
    for the trainer side (balanced train, imbalanced val, imbalanced test), so
    single-LLM prober results are directly comparable to trainer metrics.
    """
    set_seed(SEED)
    X_full, _, _ = DataManager.load_concatenated_layers(model, dataset, layer_indices, layer_type)
    y_all = labels_from_stats(DataManager.get_stats(model, dataset))

    _, train_bal_pos, val_pos, test_pos = split_and_balance(
        y_all, split_seed=SEED + 20, test_seed=SEED + 24, balance_seed=SEED + 22,
    )

    X_train_raw = X_full[train_bal_pos]
    X_val_raw = X_full[val_pos]
    X_test_raw = X_full[test_pos]
    scaler = StandardScaler().fit(X_train_raw)

    return {
        "model": model,
        "dataset": dataset,
        "layer_type": layer_type,
        "X_train": scaler.transform(X_train_raw),
        "X_val": scaler.transform(X_val_raw),
        "X_test": scaler.transform(X_test_raw),
        "X_test_raw": X_test_raw,
        "y_train": y_all[train_bal_pos],
        "y_val": y_all[val_pos],
        "y_test": y_all[test_pos],
        "scaler": scaler,
    }

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

    # ---- load full activations (with instance ids for alignment mapping) ----
    X_trainer_full, _, trainer_ids = DataManager.load_concatenated_layers(
        trainer_name, dataset, trainer_layers, layer_type
    )
    X_tester_full, _, tester_ids = DataManager.load_concatenated_layers(
        tester_name, dataset, tester_layers, layer_type
    )

    # ---- stats for concordant alignment (from full dataset) ----
    stats_trainer = DataManager.get_stats(trainer_name, dataset)
    stats_tester = DataManager.get_stats(tester_name, dataset)
    align_ids, _ = get_concordant_indices_and_undersample(stats_trainer, stats_tester, SEED)

    # ---- per-model splits for probing ----
    # Build full label arrays from stats (preserves natural distribution)
    y_trainer_all = labels_from_stats(stats_trainer)
    y_tester_all = labels_from_stats(stats_tester)

    # Stratified train/val/test split (70/15/15), then balance only the training portion
    train_t_pos, train_t_bal_pos, val_t_pos, test_t_pos = split_and_balance(
        y_trainer_all, split_seed=SEED + 20, test_seed=SEED + 24, balance_seed=SEED + 22,
    )
    train_s_pos, train_s_bal_pos, val_s_pos, test_s_pos = split_and_balance(
        y_tester_all, split_seed=SEED + 21, test_seed=SEED + 25, balance_seed=SEED + 23,
    )

    # ---- concordant alignment (constrained to training split only) ----
    # Filter alignment ids to only samples present in both training splits
    trainer_train_ids = set(str(id_val) for id_val in trainer_ids[train_t_pos])
    tester_train_ids = set(str(id_val) for id_val in tester_ids[train_s_pos])
    common_train_ids = trainer_train_ids & tester_train_ids
    align_ids_train = np.array([id_val for id_val in align_ids if str(id_val) in common_train_ids])

    if len(align_ids_train) == 0:
        raise ValueError("No concordant alignment samples in the training split. "
                         f"Trainer train ids: {len(trainer_train_ids)}, "
                         f"Tester train ids: {len(tester_train_ids)}, "
                         f"Common: {len(common_train_ids)}")

    # Map instance ids → positional indices for both models
    trainer_id_to_pos = {str(id_val): pos for pos, id_val in enumerate(trainer_ids)}
    tester_id_to_pos = {str(id_val): pos for pos, id_val in enumerate(tester_ids)}
    align_pos_trainer = np.array([trainer_id_to_pos[str(id_val)] for id_val in align_ids_train])
    align_pos_tester = np.array([tester_id_to_pos[str(id_val)] for id_val in align_ids_train])

    # Use unique seed for alignment split (independent from probing splits)
    rng_align = np.random.RandomState(SEED + 10)
    perm_align = rng_align.permutation(len(align_ids_train))
    split_a = int(ALIGNMENT_SPLIT * len(align_ids_train))
    align_train_local = perm_align[:split_a]
    align_val_local = perm_align[split_a:]

    X_align_trainer = X_trainer_full[align_pos_trainer]
    X_align_tester = X_tester_full[align_pos_tester]

    X_align_trainer_train = X_align_trainer[align_train_local]
    X_align_trainer_val = X_align_trainer[align_val_local]
    X_align_tester_train = X_align_tester[align_train_local]
    X_align_tester_val = X_align_tester[align_val_local]

    # ---- Extract activations — balanced train, imbalanced val, imbalanced test ----
    X_trainer_train_raw = X_trainer_full[train_t_bal_pos]
    X_trainer_val_raw = X_trainer_full[val_t_pos]
    X_trainer_test_raw = X_trainer_full[test_t_pos]
    y_trainer_train = y_trainer_all[train_t_bal_pos]
    y_trainer_val = y_trainer_all[val_t_pos]
    y_trainer_test = y_trainer_all[test_t_pos]

    X_tester_train_raw = X_tester_full[train_s_bal_pos]
    X_tester_val_raw = X_tester_full[val_s_pos]
    X_tester_test_raw = X_tester_full[test_s_pos]
    y_tester_train = y_tester_all[train_s_bal_pos]
    y_tester_val = y_tester_all[val_s_pos]
    y_tester_test = y_tester_all[test_s_pos]

    # ---- prober split (shared across methods) ----
    # train_idx → all of balanced train; val_idx → all of imbalanced val
    prober_train_idx = np.arange(len(X_trainer_train_raw))
    prober_val_idx = np.arange(len(X_trainer_val_raw))

    # ---- scalers (fit on balanced train only) ----
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
            "X_val": scaler_trainer.transform(X_trainer_val_raw),
            "X_test": scaler_trainer.transform(X_trainer_test_raw),
            "X_train_raw": X_trainer_train_raw,
            "X_val_raw": X_trainer_val_raw,
            "X_test_raw": X_trainer_test_raw,
            "y_train": y_trainer_train,
            "y_val": y_trainer_val,
            "y_test": y_trainer_test,
            "scaler": scaler_trainer,
        },
        "tester": {
            "X_train": scaler_tester.transform(X_tester_train_raw),
            "X_val": scaler_tester.transform(X_tester_val_raw),
            "X_test": scaler_tester.transform(X_tester_test_raw),
            "X_train_raw": X_tester_train_raw,
            "X_val_raw": X_tester_val_raw,
            "X_test_raw": X_tester_test_raw,
            "y_train": y_tester_train,
            "y_val": y_tester_val,
            "y_test": y_tester_test,
            "scaler": scaler_tester,
        },
        "prober_split": {
            "train_idx": prober_train_idx,
            "val_idx": prober_val_idx,
            "val_ratio": 1.0 - TRAIN_SPLIT,  # imbalanced val
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
