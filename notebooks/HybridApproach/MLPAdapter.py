"""Hybrid AdapterMLP refactored for modularity, reproducibility, and reusable paths."""

import gc
import json
import os
import random
import traceback
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CACHE_DIR_NAME = "activation_cache"
    ALIGNMENT_MODEL_DIR = os.path.join(BASE_DIR, "alignment_models")
    RESULTS_DIR = os.path.join(BASE_DIR, "results_metrics")
    PLOTS_DIR = os.path.join(BASE_DIR, "alignment_plots")
    CONFUSION_DIR = os.path.join(BASE_DIR, "confusion_matrices")

    SEED = 42
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    LAYER_TYPES = ["attn", "mlp", "hidden"]

    # Configurazione identica al Notebook

    LAYER_CONFIG = {
        "belief_bank_constraints": {
            "Llama-3.1-8B-Instruct": {
                "attn": [5, 8, 12],
                "mlp": [13, 14, 15],
                "hidden": [13, 14, 15]
            },
            "gemma-2-9b-it": {
                "attn": [23, 27, 33],
                "mlp": [24, 25, 26],
                "hidden": [23, 24, 27]
            },
            "save_dir": "LLama_Gemma_BBC"
        },
        "belief_bank_facts": {
            "Llama-3.1-8B-Instruct": {
                "attn": [8, 13, 14],
                "mlp": [21, 14, 15],
                "hidden": [16, 14, 15]
            },
            "gemma-2-9b-it": {
                "attn": [21, 27, 24],
                "mlp": [22, 25, 27],
                "hidden": [23, 26, 34]
            },
            "save_dir": "LLama_Gemma_BBF"
        },
        "halu_eval": {
            "Llama-3.1-8B-Instruct": {
                "attn": [14, 15, 16],
                "mlp": [13, 14, 15],
                "hidden": [16, 14, 15]
            },
            "gemma-2-9b-it": {
                "attn": [21, 27, 26],
                "mlp": [24, 23, 28],
                "hidden": [19, 24, 28]
            },
            "save_dir": "LLama_Gemma_HE"
        }
    }
    

    """
    LAYER_CONFIG = {
        "belief_bank_facts": {
            "Qwen2.5-7B": {
                "attn": [14, 15, 17],
                "mlp": [14, 23, 25],
                "hidden": [15, 16, 17]
            },
            "Falcon3-7B-Base": {
                "attn": [18, 19, 26],
                "mlp": [18, 19, 20],
                "hidden": [17, 18, 21]
            },
            "save_dir": "Qwen_Falcon_BBF_MIAO"
        }
    }
    """ 
    ALIGNMENT_CONFIG = {
        "hidden_dim": 128,
        "dropout": 0.5,
        "learning_rate": 1e-3,
        "weight_decay": 0.1,
        "batch_size": 32,
        "max_epochs": 1000,
        "early_stopping_patience": 50,
        "early_stopping_min_delta": 1e-4,
        "gradient_clip_max_norm": 1.0,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "loss_alpha": 0.01,
        "loss_beta": 1.0
    }

    PROBE_CONFIG = {
        "type": "LogisticRegression",
        "max_iter": 1000,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "n_jobs": -1
    }


# ==================================================================
# Utilities
# ==================================================================

def set_seed(seed: int = Config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_generator(seed: int = Config.SEED) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def get_balanced_indices(y: np.ndarray, seed: int = Config.SEED) -> np.ndarray:
    rng = np.random.RandomState(seed)
    unique_classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    selected_indices = []
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        if len(cls_indices) > min_count:
            sampled = rng.choice(cls_indices, size=min_count, replace=False)
            selected_indices.extend(sampled)
        else:
            selected_indices.extend(cls_indices)

    return np.sort(np.array(selected_indices))


# ==================================================================
# Data management helpers
# ==================================================================

class DataManager:
    _cache_root = os.path.join(Config.ROOT_DIR, Config.CACHE_DIR_NAME)

    @classmethod
    def _make_activation_dir(cls, model_name: str, dataset_name: str, layer_type: str) -> str:
        return os.path.join(cls._cache_root, model_name, dataset_name, f"activation_{layer_type}")

    @classmethod
    def detect_structure_type(cls, model_name: str, dataset_name: str, layer_type: str) -> str:
        layer_dir = cls._make_activation_dir(model_name, dataset_name, layer_type)
        hallucinated_path = os.path.join(layer_dir, "hallucinated")
        return "new" if os.path.isdir(hallucinated_path) else "old"

    @classmethod
    def stats_per_json(cls, model_name: str, dataset_name: str) -> Dict[str, Any]:
        file_path = os.path.join(cls._cache_root, model_name, dataset_name, "generations", "hallucination_labels.json")
        with open(file_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        hallucinated = [item["instance_id"] for item in data if item["is_hallucination"]]
        not_hallucinated = [item["instance_id"] for item in data if not item["is_hallucination"]]
        total = len(data)
        percent = (len(hallucinated) / total) * 100 if total > 0 else 0.0
        return {
            "total": total,
            "hallucinations": len(hallucinated),
            "percent_hallucinations": percent,
            "hallucinated_ids": hallucinated,
            "not_hallucinated_ids": not_hallucinated,
            "model_name": model_name,
            "dataset_name": dataset_name
        }

    @classmethod
    def stats_from_new_structure(cls, model_name: str, dataset_name: str) -> Dict[str, Any]:
        attn_dir = cls._make_activation_dir(model_name, dataset_name, "attn")
        hall_ids_path = os.path.join(attn_dir, "hallucinated", "layer0_instance_ids.json")
        not_hall_ids_path = os.path.join(attn_dir, "not_hallucinated", "layer0_instance_ids.json")

        with open(hall_ids_path, "r", encoding="utf-8") as fp:
            hallucinated_ids = json.load(fp)
        with open(not_hall_ids_path, "r", encoding="utf-8") as fp:
            not_hallucinated_ids = json.load(fp)

        total = len(hallucinated_ids) + len(not_hallucinated_ids)
        percent = (len(hallucinated_ids) / total) * 100 if total > 0 else 0.0
        return {
            "total": total,
            "hallucinations": len(hallucinated_ids),
            "percent_hallucinations": percent,
            "hallucinated_ids": hallucinated_ids,
            "not_hallucinated_ids": not_hallucinated_ids,
            "model_name": model_name,
            "dataset_name": dataset_name
        }

    @classmethod
    def get_stats(cls, model_name: str, dataset_name: str) -> Dict[str, Any]:
        structure = cls.detect_structure_type(model_name, dataset_name, "attn")
        if structure == "new":
            return cls.stats_from_new_structure(model_name, dataset_name)
        return cls.stats_per_json(model_name, dataset_name)

    @classmethod
    def load_activations_and_labels(
        cls, model_name: str, dataset_name: str, layer: int, layer_type: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        structure = cls.detect_structure_type(model_name, dataset_name, layer_type)
        layer_dir = cls._make_activation_dir(model_name, dataset_name, layer_type)

        if structure == "new":
            hall_act_path = os.path.join(layer_dir, "hallucinated", f"layer{layer}_activations.pt")
            hall_ids_path = os.path.join(layer_dir, "hallucinated", f"layer{layer}_instance_ids.json")
            not_hall_act_path = os.path.join(layer_dir, "not_hallucinated", f"layer{layer}_activations.pt")
            not_hall_ids_path = os.path.join(layer_dir, "not_hallucinated", f"layer{layer}_instance_ids.json")

            hall_activations = torch.load(hall_act_path, map_location=Config.DEVICE)
            not_hallactivations = torch.load(not_hall_act_path, map_location=Config.DEVICE)

            with open(hall_ids_path, "r", encoding="utf-8") as fp:
                hall_ids = json.load(fp)
            with open(not_hall_ids_path, "r", encoding="utf-8") as fp:
                not_hall_ids = json.load(fp)

            hall_arr = hall_activations.cpu().numpy().astype(np.float32)
            not_hall_arr = not_hallactivations.cpu().numpy().astype(np.float32)

            X_concat = np.vstack([hall_arr, not_hall_arr])
            y_concat = np.concatenate([
                np.ones(hall_arr.shape[0], dtype=int),
                np.zeros(not_hall_arr.shape[0], dtype=int)
            ])
            ids_concat = np.array(hall_ids + not_hall_ids)

            sort_idx = np.argsort(ids_concat)
            return X_concat[sort_idx], y_concat[sort_idx], ids_concat[sort_idx]

        activations_path = os.path.join(layer_dir, f"layer{layer}_activations.pt")
        activations = torch.load(activations_path, map_location=Config.DEVICE)
        X = activations.cpu().numpy().astype(np.float32)

        labels_path = os.path.join(cls._cache_root, model_name, dataset_name, "generations", "hallucination_labels.json")
        with open(labels_path, "r", encoding="utf-8") as fp:
            labels = json.load(fp)
        y = np.array([item["is_hallucination"] for item in labels], dtype=int)
        instance_ids = np.arange(len(y))
        return X, y, instance_ids

    @classmethod
    def load_concatenated_layers(
        cls, model_name: str, dataset_name: str, layer_indices: List[int], layer_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        combined = []
        common_y = None
        for layer_idx in layer_indices:
            try:
                X_layer, y_layer, _ = cls.load_activations_and_labels(model_name, dataset_name, layer_idx, layer_type)
                if common_y is None:
                    common_y = y_layer
                elif not np.array_equal(common_y, y_layer):
                    raise ValueError(f"Label mismatch at layer {layer_idx} for {model_name}")
                combined.append(X_layer)
            except FileNotFoundError:
                print(f"Layer {layer_idx} not found for {model_name}/{layer_type}; skipping")
            except Exception as exc:
                raise RuntimeError(f"Failed to load layer {layer_idx} for {model_name}/{layer_type}") from exc
        if not combined:
            raise FileNotFoundError(f"No layers loaded for {model_name} {layer_type}")
        X_stacked = np.concatenate(combined, axis=1)
        return X_stacked, common_y


# ==================================================================
# Sampling helpers
# ==================================================================

def get_concordant_indices_and_undersample(
    stats_model1: Dict[str, Any], stats_model2: Dict[str, Any], seed: int = Config.SEED
) -> Tuple[np.ndarray, np.ndarray]:
    hall_1 = set(stats_model1["hallucinated_ids"])
    hall_2 = set(stats_model2["hallucinated_ids"])
    all_1 = set(stats_model1["hallucinated_ids"] + stats_model1.get("not_hallucinated_ids", []))
    all_2 = set(stats_model2["hallucinated_ids"] + stats_model2.get("not_hallucinated_ids", []))

    common_ids = sorted(all_1.intersection(all_2))
    if not common_ids:
        raise ValueError("No common instance ids found between the two models")

    y1 = np.array([1 if idx in hall_1 else 0 for idx in common_ids], dtype=np.int8)
    y2 = np.array([1 if idx in hall_2 else 0 for idx in common_ids], dtype=np.int8)
    concordant_mask = y1 == y2
    concordant_ids = np.array(common_ids)[concordant_mask]
    concordant_labels = y1[concordant_mask]

    n_hall = int((concordant_labels == 1).sum())
    n_non_hall = int((concordant_labels == 0).sum())
    min_count = min(n_hall, n_non_hall)
    if min_count == 0:
        raise ValueError("Not enough concordant samples to balance")

    rng = np.random.RandomState(seed)
    hall_sampled = rng.choice(concordant_ids[concordant_labels == 1], size=min_count, replace=False)
    non_hall_sampled = rng.choice(concordant_ids[concordant_labels == 0], size=min_count, replace=False)

    balanced_indices = np.concatenate([hall_sampled, non_hall_sampled])
    balanced_labels = np.concatenate([
        np.ones(min_count, dtype=np.int8),
        np.zeros(min_count, dtype=np.int8)
    ])
    shuffled = rng.permutation(len(balanced_indices))
    return balanced_indices[shuffled], balanced_labels[shuffled]


def get_undersampled_indices_per_model(model_stats: Dict[str, Any], seed: int = Config.SEED) -> Tuple[np.ndarray, np.ndarray]:
    total = model_stats["total"]
    hall_set = set(model_stats["hallucinated_ids"])
    y = np.array([1 if idx in hall_set else 0 for idx in range(total)], dtype=np.int8)
    balanced_idx = get_balanced_indices(y, seed)
    return balanced_idx, y[balanced_idx]


# ==================================================================
# Dataset + Model Definitions
# ==================================================================

class AlignmentDataset(Dataset):
    def __init__(self, source: torch.Tensor, target: torch.Tensor):
        self.source = source
        self.target = target

    def __len__(self) -> int:
        return self.source.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.source[index], self.target[index]


class AlignmentNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, output_dim, bias=False) if input_dim != output_dim else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )
        nn.init.zeros_(self.net[-2].weight)
        if self.net[-2].bias is not None:
            nn.init.zeros_(self.net[-2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = self.input_proj(x)
        return projection + self.net(projection)


class MixedLoss(nn.Module):
    def __init__(self, alpha: float = 0.01, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_mse = self.mse(pred, target)
        cosine_sim = F.cosine_similarity(pred, target, dim=1).mean()
        return self.alpha * loss_mse + self.beta * (1.0 - cosine_sim)


# ==================================================================
# Alignment training + evaluation
# ==================================================================

def train_alignment_network(
    X_source_train: np.ndarray,
    X_target_train: np.ndarray,
    X_source_val: np.ndarray,
    X_target_val: np.ndarray,
    config: Dict[str, Any],
    layer_type: str,
    teacher_name: str,
    student_name: str
) -> Tuple[AlignmentNetwork, Dict[str, Any]]:
    set_seed(Config.SEED)
    model = AlignmentNetwork(
        input_dim=X_source_train.shape[1],
        output_dim=X_target_train.shape[1],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"]
    ).to(Config.DEVICE)

    criterion = MixedLoss(alpha=config["loss_alpha"], beta=config["loss_beta"])
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = getattr(optim.lr_scheduler, config["scheduler"])(optimizer, T_max=config["max_epochs"])

    train_dataset = AlignmentDataset(torch.tensor(X_source_train, dtype=torch.float32), torch.tensor(X_target_train, dtype=torch.float32))
    val_dataset = AlignmentDataset(torch.tensor(X_source_val, dtype=torch.float32), torch.tensor(X_target_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, generator=get_generator(Config.SEED))
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    best_state = None
    best_loss = float("inf")
    patience = 0

    for epoch in range(config["max_epochs"]):
        model.train()
        for x_src, x_tgt in train_loader:
            x_src, x_tgt = x_src.to(Config.DEVICE), x_tgt.to(Config.DEVICE)
            optimizer.zero_grad()
            preds = model(x_src)
            loss = criterion(preds, x_tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_max_norm"])
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_src, x_tgt in val_loader:
                x_src, x_tgt = x_src.to(Config.DEVICE), x_tgt.to(Config.DEVICE)
                val_loss += criterion(model(x_src), x_tgt).item()
        val_loss /= len(val_loader)

        if val_loss < best_loss - config["early_stopping_min_delta"]:
            best_loss = val_loss
            patience = 0
            best_state = model.state_dict().copy()
        else:
            patience += 1
            if patience >= config["early_stopping_patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(Config.ALIGNMENT_MODEL_DIR, exist_ok=True)
    model_path = os.path.join(Config.ALIGNMENT_MODEL_DIR, f"alignment_{layer_type}_{student_name}_to_{teacher_name}.pt")
    torch.save(model.state_dict(), model_path)

    return model, {
        "input_dim": X_source_train.shape[1],
        "output_dim": X_target_train.shape[1],
        "best_val_loss": best_loss,
        "epochs_trained": epoch + 1,
        "model_path": model_path,
        "config": config
    }


def run_hybrid_experiment_pipeline(
    layer_data: Dict[str, Any],
    teacher_name: str,
    student_name: str,
    layer_type: str,
    model_a_name: str,
    model_b_name: str
) -> Dict[str, Any]:
    teacher_key = "model_a" if teacher_name == model_a_name else "model_b"
    student_key = "model_b" if teacher_key == "model_a" else "model_a"

    teacher_data = layer_data[teacher_key]
    student_data = layer_data[student_key]
    alignment_data = layer_data["alignment"]

    scalers = {
        "model_a": "scaler_a",
        "model_b": "scaler_b"
    }
    student_scaler_key = scalers[student_key]

    probe = LogisticRegression(
        max_iter=Config.PROBE_CONFIG["max_iter"],
        class_weight=Config.PROBE_CONFIG["class_weight"],
        solver=Config.PROBE_CONFIG["solver"],
        n_jobs=Config.PROBE_CONFIG["n_jobs"],
        random_state=Config.SEED
    )
    probe.fit(teacher_data["X_train"], teacher_data["y_train"])

    y_pred_t = probe.predict(teacher_data["X_test"])
    y_proba_t = probe.predict_proba(teacher_data["X_test"])[:, 1]

    metrics_teacher = {
        "accuracy": accuracy_score(teacher_data["y_test"], y_pred_t),
        "precision": precision_score(teacher_data["y_test"], y_pred_t),
        "recall": recall_score(teacher_data["y_test"], y_pred_t),
        "f1": f1_score(teacher_data["y_test"], y_pred_t),
        "auroc": roc_auc_score(teacher_data["y_test"], y_proba_t),
        "confusion_matrix": confusion_matrix(teacher_data["y_test"], y_pred_t).tolist()
    }

    alignment_model, alignment_info = train_alignment_network(
        alignment_data[f"X_{student_key[-1]}_train"],
        alignment_data[f"X_{teacher_key[-1]}_train"],
        alignment_data[f"X_{student_key[-1]}_val"],
        alignment_data[f"X_{teacher_key[-1]}_val"],
        Config.ALIGNMENT_CONFIG,
        layer_type,
        teacher_name,
        student_name
    )

    student_scaled = alignment_data[student_scaler_key].transform(student_data["X_test_raw"])
    with torch.no_grad():
        projected = alignment_model(torch.tensor(student_scaled, dtype=torch.float32, device=Config.DEVICE)).cpu().numpy()

    y_pred_c = probe.predict(projected)
    y_proba_c = probe.predict_proba(projected)[:, 1]

    metrics_student = {
        "accuracy": accuracy_score(student_data["y_test"], y_pred_c),
        "precision": precision_score(student_data["y_test"], y_pred_c),
        "recall": recall_score(student_data["y_test"], y_pred_c),
        "f1": f1_score(student_data["y_test"], y_pred_c),
        "auroc": roc_auc_score(student_data["y_test"], y_proba_c),
        "confusion_matrix": confusion_matrix(student_data["y_test"], y_pred_c).tolist()
    }

    return {
        "type": layer_type,
        "teacher_name": teacher_name,
        "student_name": student_name,
        "teacher": metrics_teacher,
        "student_on_teacher": metrics_student,
        "alignment_model": alignment_info,
        "probe_config": Config.PROBE_CONFIG
    }


def plot_confusion_matrix(cm: np.ndarray, layer_type: str, label: str) -> None:
    os.makedirs(Config.CONFUSION_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Non-Hall", "Hall"],
        yticklabels=["Non-Hall", "Hall"]
    )
    ax.set_title(f"Confusion Matrix - {label} - {layer_type.upper()}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    filename = os.path.join(Config.CONFUSION_DIR, f"cm_{label}_{layer_type}.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ==================================================================
# Data preparation
# ==================================================================

def build_data_splits(
    dataset_name: str,
    model_a_name: str,
    model_b_name: str,
    d_config: Dict[str, Any],
    config: Config
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    stats_a = DataManager.get_stats(model_a_name, dataset_name)
    stats_b = DataManager.get_stats(model_b_name, dataset_name)

    alignment_indices, _ = get_concordant_indices_and_undersample(stats_a, stats_b, seed=config.SEED)

    rng = np.random.RandomState(config.SEED)
    shuffled_alignment = rng.permutation(len(alignment_indices))
    split_align = int(0.7 * len(alignment_indices))
    alignment_train_local_idx = shuffled_alignment[:split_align]
    alignment_val_local_idx = shuffled_alignment[split_align:]

    a_balanced_idx, a_balanced_labels = get_undersampled_indices_per_model(stats_a, config.SEED)
    b_balanced_idx, b_balanced_labels = get_undersampled_indices_per_model(stats_b, config.SEED)

    rng_a = np.random.RandomState(config.SEED)
    rng_b = np.random.RandomState(config.SEED + 1)
    shuffled_a = rng_a.permutation(len(a_balanced_idx))
    shuffled_b = rng_b.permutation(len(b_balanced_idx))
    split_a = int(0.7 * len(a_balanced_idx))
    split_b = int(0.7 * len(b_balanced_idx))

    a_train_local = shuffled_a[:split_a]
    a_test_local = shuffled_a[split_a:]
    b_train_local = shuffled_b[:split_b]
    b_test_local = shuffled_b[split_b:]

    metadata = {
        "alignment_train": len(alignment_train_local_idx),
        "alignment_val": len(alignment_val_local_idx),
        "model_a_train": len(a_train_local),
        "model_a_test": len(a_test_local),
        "model_b_train": len(b_train_local),
        "model_b_test": len(b_test_local)
    }

    data_splits: Dict[str, Any] = {}
    for layer_type in config.LAYER_TYPES:
        gc.collect()
        torch.cuda.empty_cache()

        layer_indices_a = d_config[model_a_name][layer_type]
        layer_indices_b = d_config[model_b_name][layer_type]

        X_a_full, _ = DataManager.load_concatenated_layers(model_a_name, dataset_name, layer_indices_a, layer_type)
        X_b_full, _ = DataManager.load_concatenated_layers(model_b_name, dataset_name, layer_indices_b, layer_type)

        X_align_a = X_a_full[alignment_indices]
        X_align_b = X_b_full[alignment_indices]

        X_align_a_train = X_align_a[alignment_train_local_idx]
        X_align_a_val = X_align_a[alignment_val_local_idx]
        X_align_b_train = X_align_b[alignment_train_local_idx]
        X_align_b_val = X_align_b[alignment_val_local_idx]

        X_a_balanced = X_a_full[a_balanced_idx]
        X_b_balanced = X_b_full[b_balanced_idx]

        X_a_train = X_a_balanced[a_train_local]
        X_a_test = X_a_balanced[a_test_local]
        y_a_train = a_balanced_labels[a_train_local]
        y_a_test = a_balanced_labels[a_test_local]

        X_b_train = X_b_balanced[b_train_local]
        X_b_test = X_b_balanced[b_test_local]
        y_b_train = b_balanced_labels[b_train_local]
        y_b_test = b_balanced_labels[b_test_local]

        del X_a_full, X_b_full, X_a_balanced, X_b_balanced
        gc.collect()

        scaler_align_a = StandardScaler()
        scaler_align_b = StandardScaler()
        scaler_a = StandardScaler()
        scaler_b = StandardScaler()

        X_align_a_train_norm = scaler_align_a.fit_transform(X_align_a_train)
        X_align_b_train_norm = scaler_align_b.fit_transform(X_align_b_train)
        X_align_a_val_norm = scaler_align_a.transform(X_align_a_val)
        X_align_b_val_norm = scaler_align_b.transform(X_align_b_val)

        X_a_train_norm = scaler_a.fit_transform(X_a_train)
        X_a_test_norm = scaler_a.transform(X_a_test)
        X_b_train_norm = scaler_b.fit_transform(X_b_train)
        X_b_test_norm = scaler_b.transform(X_b_test)

        data_splits[layer_type] = {
            "alignment": {
                "X_a_train": X_align_a_train_norm,
                "X_b_train": X_align_b_train_norm,
                "X_a_val": X_align_a_val_norm,
                "X_b_val": X_align_b_val_norm,
                "scaler_a": scaler_align_a,
                "scaler_b": scaler_align_b,
            },
            "model_a": {
                "X_train": X_a_train_norm,
                "X_test": X_a_test_norm,
                "y_train": y_a_train,
                "y_test": y_a_test,
                "X_test_raw": X_a_test
            },
            "model_b": {
                "X_train": X_b_train_norm,
                "X_test": X_b_test_norm,
                "y_train": y_b_train,
                "y_test": y_b_test,
                "X_test_raw": X_b_test
            }
        }

    return data_splits, metadata


# ==================================================================
# Experiment orchestrator + logging
# ==================================================================

def run_experiments(data_splits: Dict[str, Any], model_a_name: str, model_b_name: str, config: Config) -> List[Dict[str, Any]]:
    scenarios = [
        {"teacher": model_a_name, "student": model_b_name},
        {"teacher": model_b_name, "student": model_a_name}
    ]
    all_results = []

    for scenario in scenarios:
        print("\n" + "=" * 60)
        print(f"SCENARIO: {scenario['teacher']} → {scenario['student']}")
        print("=" * 60)

        scenario_results = []
        for layer_type in config.LAYER_TYPES:
            print(f"\nProcessing layer type: {layer_type.upper()}")
            try:
                result = run_hybrid_experiment_pipeline(data_splits[layer_type], scenario['teacher'], scenario['student'], layer_type, model_a_name, model_b_name)
                scenario_results.append(result)

                plot_confusion_matrix(np.array(result['teacher']['confusion_matrix']), layer_type, f"teacher_{scenario['teacher']}")
                plot_confusion_matrix(np.array(result['student_on_teacher']['confusion_matrix']), layer_type, f"{scenario['student']}_on_{scenario['teacher']}")
            except Exception as exc:
                print(f"Error while processing {layer_type}: {exc}")
                traceback.print_exc()
        all_results.append({"scenario": f"{scenario['teacher']} → {scenario['student']}", "results": scenario_results})

    return all_results


def save_metrics(all_results: List[Dict[str, Any]], metadata: Dict[str, int], save_dir: str, config: Config) -> None:
    os.makedirs(save_dir, exist_ok=True)
    metrics_file = os.path.join(save_dir, "hybrid_adapter_logreg_results.json")

    output = []
    for scenario_data in all_results:
        scenario_results = []
        for res in scenario_data['results']:
            align_cfg = res['alignment_model']['config']
            scenario_results.append({
                "layer_type": res['type'],
                "teacher_model": res['teacher_name'],
                "student_model": res['student_name'],
                "data_info": {
                    "alignment_samples_train": metadata['alignment_train'],
                    "alignment_samples_val": metadata['alignment_val'],
                    "model_a_train": metadata['model_a_train'],
                    "model_a_test": metadata['model_a_test'],
                    "model_b_train": metadata['model_b_train'],
                    "model_b_test": metadata['model_b_test'],
                    "concordant_undersampling_for_alignment": True,
                    "separate_undersampling_per_model": True
                },
                "alignment_model_info": {
                    "architecture": "AlignmentNetwork",
                    "input_dim": res['alignment_model']['input_dim'],
                    "output_dim": res['alignment_model']['output_dim'],
                    "hidden_dim": align_cfg['hidden_dim'],
                    "dropout": align_cfg['dropout'],
                    "activation": "GELU",
                    "normalization": "LayerNorm",
                    "residual_connection": True,
                    "initialization": "zero_init"
                },
                "training_hyperparameters": {
                    "optimizer": align_cfg['optimizer'],
                    "learning_rate": align_cfg['learning_rate'],
                    "weight_decay": align_cfg['weight_decay'],
                    "batch_size": align_cfg['batch_size'],
                    "max_epochs": align_cfg['max_epochs'],
                    "scheduler": align_cfg['scheduler'],
                    "gradient_clip_max_norm": align_cfg['gradient_clip_max_norm'],
                    "early_stopping_patience": align_cfg['early_stopping_patience'],
                    "early_stopping_min_delta": align_cfg['early_stopping_min_delta']
                },
                "loss_function": {
                    "type": "MixedLoss",
                    "mse_weight": align_cfg['loss_alpha'],
                    "cosine_weight": align_cfg['loss_beta']
                },
                "training_results": {
                    "alignment_network": {
                        "best_val_loss": round(res['alignment_model']['best_val_loss'], 6),
                        "epochs_trained": res['alignment_model']['epochs_trained'],
                        "model_saved_path": res['alignment_model']['model_path']
                    }
                },
                "teacher_probe": res['probe_config'],
                "metrics": {
                    "teacher": {
                        "accuracy": round(res['teacher']['accuracy'], 4),
                        "precision": round(res['teacher']['precision'], 4),
                        "recall": round(res['teacher']['recall'], 4),
                        "f1_score": round(res['teacher']['f1'], 4),
                        "auroc": round(res['teacher']['auroc'], 4),
                        "confusion_matrix": {
                            "TN": int(res['teacher']['confusion_matrix'][0][0]),
                            "FP": int(res['teacher']['confusion_matrix'][0][1]),
                            "FN": int(res['teacher']['confusion_matrix'][1][0]),
                            "TP": int(res['teacher']['confusion_matrix'][1][1])
                        }
                    },
                    "student_on_teacher": {
                        "accuracy": round(res['student_on_teacher']['accuracy'], 4),
                        "precision": round(res['student_on_teacher']['precision'], 4),
                        "recall": round(res['student_on_teacher']['recall'], 4),
                        "f1_score": round(res['student_on_teacher']['f1'], 4),
                        "auroc": round(res['student_on_teacher']['auroc'], 4),
                        "confusion_matrix": {
                            "TN": int(res['student_on_teacher']['confusion_matrix'][0][0]),
                            "FP": int(res['student_on_teacher']['confusion_matrix'][0][1]),
                            "FN": int(res['student_on_teacher']['confusion_matrix'][1][0]),
                            "TP": int(res['student_on_teacher']['confusion_matrix'][1][1])
                        }
                    }
                }
            })
        output.append({"scenario": scenario_data['scenario'], "results": scenario_results})

    with open(metrics_file, "w", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2)

    print(f"✓ Results persisted at: {metrics_file}")


def plot_alignment_pca_hybrid(data_splits: Dict[str, Any], config: Config, save_dir: str, model_a_name: str, model_b_name: str, dataset_name: str, layer_types: List[str] = None) -> None:
    if layer_types is None:
        layer_types = config.LAYER_TYPES
    scenarios = [
        {"teacher": model_a_name, "student": model_b_name},
        {"teacher": model_b_name, "student": model_a_name}
    ]
    model_key = {model_a_name: "model_a", model_b_name: "model_b"}
    suffix = {"model_a": "a", "model_b": "b"}

    num_rows = len(layer_types) * len(scenarios)
    fig, axes = plt.subplots(num_rows, 3, figsize=(18, max(6, num_rows * 2.5)))
    axes = axes.reshape(num_rows, 3)

    for layer_idx, layer_type in enumerate(layer_types):
        for scenario_idx, scenario in enumerate(scenarios):
            row_idx = layer_idx * len(scenarios) + scenario_idx
            teacher = scenario["teacher"]
            student = scenario["student"]
            checkpoint = os.path.join(config.ALIGNMENT_MODEL_DIR, f"alignment_{layer_type}_{student}_to_{teacher}.pt")

            if not os.path.exists(checkpoint):
                print(f"Missing checkpoint: {checkpoint}")
                for col_idx in range(3):
                    axes[row_idx, col_idx].set_axis_off()
                continue

            teacher_key = model_key[teacher]
            student_key = model_key[student]
            teacher_suffix = suffix[teacher_key]
            student_suffix = suffix[student_key]

            teacher_features = data_splits[layer_type]["alignment"][f"X_{teacher_suffix}_train"]
            student_features = data_splits[layer_type]["alignment"][f"X_{student_suffix}_train"]

            alignment_model = AlignmentNetwork(
                input_dim=student_features.shape[1],
                output_dim=teacher_features.shape[1],
                hidden_dim=config.ALIGNMENT_CONFIG['hidden_dim'],
                dropout=config.ALIGNMENT_CONFIG['dropout']
            ).to(Config.DEVICE)
            alignment_model.load_state_dict(torch.load(checkpoint, map_location=Config.DEVICE))
            alignment_model.eval()

            with torch.no_grad():
                student_tensor = torch.tensor(student_features, dtype=torch.float32, device=Config.DEVICE)
                student_after = alignment_model(student_tensor).cpu().numpy()

            projection_teacher = PCA(n_components=2, random_state=Config.SEED).fit_transform(teacher_features)
            ax = axes[row_idx, 0]
            ax.scatter(projection_teacher[:, 0], projection_teacher[:, 1], s=22, alpha=0.65, color="#2ca02c")
            ax.set_title(f"{layer_type.upper()} — {teacher} (Trainer)")
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
            ax.grid(True, alpha=0.3)

            projection_student = PCA(n_components=2, random_state=Config.SEED).fit_transform(student_features)
            ax = axes[row_idx, 1]
            ax.scatter(projection_student[:, 0], projection_student[:, 1], s=22, alpha=0.65, color="#ff7f0e")
            ax.set_title(f"{layer_type.upper()} — {student} (Tester pre-align)")
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
            ax.grid(True, alpha=0.3)

            combined = np.vstack([teacher_features, student_after])
            projection_combined = PCA(n_components=2, random_state=Config.SEED).fit_transform(combined)
            teacher_combined = projection_combined[: teacher_features.shape[0]]
            student_combined = projection_combined[teacher_features.shape[0]:]
            ax = axes[row_idx, 2]
            ax.scatter(teacher_combined[:, 0], teacher_combined[:, 1], s=22, alpha=0.65, color="#2ca02c", label="Trainer")
            ax.scatter(student_combined[:, 0], student_combined[:, 1], s=22, alpha=0.65, color="#ff7f0e", label="Tester post-align")
            ax.set_title(f"{layer_type.upper()} — Trainer vs Tester post-align")
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=10)

    fig.tight_layout(rect=[0.15, 0, 1, 0.95])
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    filename = os.path.join(save_dir, f"alignment_hybrid_{model_a_name}_{model_b_name}_{dataset_name}.pdf")
    fig.savefig(filename, dpi=200, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"✓ Saved alignment PCA plot at: {filename}")


# ==================================================================
# Entrypoint
# ==================================================================

def main() -> None:
    set_seed(Config.SEED)
    print("=== Starting Hybrid Adapter Analysis ===")
    print(f"Root Dir: {os.path.abspath(Config.ROOT_DIR)}")
    
    for d_name, d_config in Config.LAYER_CONFIG.items():
        save_dir_name = d_config.get("save_dir", f"results_{d_name}")
        save_dir = os.path.join(Config.BASE_DIR, save_dir_name)
        os.makedirs(save_dir, exist_ok=True)
        
        models = [k for k in d_config.keys() if k != "save_dir"]
        if len(models) < 2:
            print(f"Need at least 2 models for dataset {d_name}. Found: {models}")
            continue
            
        model_A, model_B = models[0], models[1]
        print(f"\n{'='*60}")
        print(f"Dataset: {d_name} | Models: {model_A} vs {model_B}")
        print(f"{'='*60}")
        
        try:
            stats_a = DataManager.get_stats(model_A, d_name)
            stats_b = DataManager.get_stats(model_B, d_name)
            
            print("FASE 1: Caricamento statistiche modelli")
            print(f"  {model_A}: {stats_a['total']} totali, {stats_a['hallucinations']} allucinazioni")
            print(f"  {model_B}: {stats_b['total']} totali, {stats_b['hallucinations']} allucinazioni")
            
            data_splits, metadata = build_data_splits(d_name, model_A, model_B, d_config, Config)
            
            print("\nFASE 2: Esecuzione esperimenti")
            all_results = run_experiments(data_splits, model_A, model_B, Config)
            
            print("\nFASE 3: Salvataggio risultati")
            save_metrics(all_results, metadata, save_dir, Config)
            
            print("\nFASE 4: Visualizzazioni")
            plot_alignment_pca_hybrid(data_splits, Config, save_dir, model_A, model_B, d_name)
            
        except Exception as e:
            print(f"Error processing {d_name}: {e}")
            continue


if __name__ == "__main__":
    main()



