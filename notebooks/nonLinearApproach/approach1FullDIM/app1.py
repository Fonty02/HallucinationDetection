
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
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    CACHE_DIR_NAME = "activation_cache"
    ALIGNMENT_MODEL_DIR = os.path.join(BASE_DIR, "alignment_models")
    PROBER_MODEL_DIR = os.path.join(BASE_DIR, "prober_models")
    RESULTS_DIR = os.path.join(BASE_DIR, "results_metrics")
    PLOTS_DIR = os.path.join(BASE_DIR, "alignment_plots")
    CONFUSION_DIR = os.path.join(BASE_DIR, "confusion_matrices")

    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LAYER_TYPES = ["attn", "mlp", "hidden"]



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
        "loss_alpha": 0.01,  # MSE weight
        "loss_beta": 1.0     # Cosine weight
    }

    PROBER_CONFIG = {
        "type": "MLPProber",
        "hidden_dim": 64,
        "dropout": 0.5,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "batch_size": 64,
        "max_epochs": 200,
        "early_stopping_patience": 30,
        "early_stopping_min_delta": 1e-4,
        "gradient_clip_max_norm": 1.0,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "loss_function": "BCEWithLogitsLoss",
        "use_class_weights": True
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
        raise ValueError("No common instance_ids found between models.")

    y1 = np.array([1 if idx in hall_1 else 0 for idx in common_ids], dtype=np.int8)
    y2 = np.array([1 if idx in hall_2 else 0 for idx in common_ids], dtype=np.int8)
    concordant_mask = y1 == y2
    concordant_ids = np.array(common_ids)[concordant_mask]
    concordant_labels = y1[concordant_mask]

    n_hall = int((concordant_labels == 1).sum())
    n_non_hall = int((concordant_labels == 0).sum())
    min_count = min(n_hall, n_non_hall)
    
    print(f"    - Hallucinated (concordant): {n_hall}")
    print(f"    - Non-hallucinated (concordant): {n_non_hall}")

    if min_count == 0:
        raise ValueError("Cannot undersample: one class has 0 samples in concordant set.")

    rng = np.random.RandomState(seed)
    hall_sampled = rng.choice(concordant_ids[concordant_labels == 1], size=min_count, replace=False)
    non_hall_sampled = rng.choice(concordant_ids[concordant_labels == 0], size=min_count, replace=False)

    balanced_indices = np.concatenate([hall_sampled, non_hall_sampled])
    balanced_labels = np.concatenate([
        np.ones(min_count, dtype=np.int8),
        np.zeros(min_count, dtype=np.int8)
    ])
    
    shuffle_idx = rng.permutation(len(balanced_indices))
    balanced_indices = balanced_indices[shuffle_idx]
    balanced_labels = balanced_labels[shuffle_idx]
    
    print(f"  After undersampling: {len(balanced_indices)} balanced samples ({min_count} per class)")
    
    return balanced_indices, balanced_labels


def get_undersampled_indices_per_model(model_stats: Dict[str, Any], seed: int = Config.SEED) -> Tuple[np.ndarray, np.ndarray]:
    total = model_stats["total"]
    hall_set = set(model_stats["hallucinated_ids"])
    y = np.array([1 if idx in hall_set else 0 for idx in range(total)], dtype=np.int8)
    balanced_idx = get_balanced_indices(y, seed)
    balanced_labels = y[balanced_idx]
    return balanced_idx, balanced_labels


# ==================================================================
# Dataset + Model Definitions
# ==================================================================

class AlignmentDataset(Dataset):
    def __init__(self, x_source: torch.Tensor, x_target: torch.Tensor):
        self.x_source = x_source
        self.x_target = x_target

    def __len__(self) -> int:
        return self.x_source.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x_source[idx], self.x_target[idx]


class ClassificationDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AlignmentNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        if input_dim != output_dim:
            self.input_proj = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.input_proj = nn.Identity()

        self.net = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )
        self._init_zero()

    def _init_zero(self):
        nn.init.zeros_(self.net[-2].weight)
        if self.net[-2].bias is not None:
            nn.init.zeros_(self.net[-2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_base = self.input_proj(x)
        return x_base + self.net(x_base)


class MLPProber(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)
    
    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return (torch.sigmoid(logits) > 0.5).long()
    
    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


class MixedLoss(nn.Module):
    def __init__(self, alpha: float = 0.01, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_mse = self.mse(pred, target)
        cosine_sim = F.cosine_similarity(pred, target, dim=1).mean()
        loss_cosine = 1 - cosine_sim
        return self.alpha * loss_mse + self.beta * loss_cosine


# ==================================================================
# Training Functions
# ==================================================================

def train_mlp_prober(X_train, y_train, X_val, y_val, input_dim, device, prober_config):
    set_seed(Config.SEED)
    prober = MLPProber(
        input_dim=input_dim, 
        hidden_dim=prober_config['hidden_dim'], 
        dropout=prober_config['dropout']
    ).to(device)
    
    if prober_config['use_class_weights']:
        n_pos = y_train.sum().item() if isinstance(y_train, torch.Tensor) else y_train.sum()
        n_neg = len(y_train) - n_pos
        if n_pos > 0:
            pos_weight = torch.tensor([n_neg / n_pos]).to(device)
        else:
            pos_weight = torch.tensor([1.0]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.AdamW(
        prober.parameters(), 
        lr=prober_config['learning_rate'], 
        weight_decay=prober_config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=prober_config['max_epochs'])
    
    train_dataset = ClassificationDataset(X_train, y_train)
    val_dataset = ClassificationDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=prober_config['batch_size'], 
                             shuffle=True, num_workers=0, generator=get_generator(Config.SEED))
    val_loader = DataLoader(val_dataset, batch_size=prober_config['batch_size'], 
                           shuffle=False, num_workers=0)
    
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    epochs_trained = 0
    
    for epoch in range(prober_config['max_epochs']):
        prober.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = prober(X_batch)
            loss = criterion(logits, y_batch.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                prober.parameters(), 
                max_norm=prober_config['gradient_clip_max_norm']
            )
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        prober.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = prober.predict(X_batch)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        val_f1 = f1_score(all_labels, all_preds)
        val_acc = accuracy_score(all_labels, all_preds)
        
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1:3d}/{prober_config['max_epochs']} | Train Loss: {avg_train_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc + prober_config['early_stopping_min_delta']:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = prober.state_dict().copy()
            epochs_trained = epoch + 1
        else:
            patience_counter += 1
        
        if patience_counter >= prober_config['early_stopping_patience']:
            print(f"   Early stopping at epoch {epoch+1}. Best Val ACC: {best_val_acc:.4f}")
            break
    
    if epochs_trained == 0:
        epochs_trained = prober_config['max_epochs']
    
    if best_model_state is not None:
        prober.load_state_dict(best_model_state)
    
    return prober, best_val_acc, epochs_trained


def train_alignment_network(X_source_train, X_target_train, X_source_val, X_target_val, 
                           config, layer_type, teacher_name, student_name):
    set_seed(Config.SEED)
    
    input_dim = X_source_train.shape[1]
    output_dim = X_target_train.shape[1]
    
    X_src_train_t = torch.tensor(X_source_train, dtype=torch.float32).to(Config.DEVICE)
    X_tgt_train_t = torch.tensor(X_target_train, dtype=torch.float32).to(Config.DEVICE)
    X_src_val_t = torch.tensor(X_source_val, dtype=torch.float32).to(Config.DEVICE)
    X_tgt_val_t = torch.tensor(X_target_val, dtype=torch.float32).to(Config.DEVICE)
    
    train_dataset = AlignmentDataset(X_src_train_t, X_tgt_train_t)
    val_dataset = AlignmentDataset(X_src_val_t, X_tgt_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, generator=get_generator(Config.SEED))
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    model = AlignmentNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    ).to(Config.DEVICE)
    
    criterion = MixedLoss(alpha=config['loss_alpha'], beta=config['loss_beta'])
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['max_epochs'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    epochs_trained = 0
    
    for epoch in range(config['max_epochs']):
        model.train()
        train_loss = 0.0
        for x_src, x_tgt in train_loader:
            optimizer.zero_grad()
            pred = model(x_src)
            loss = criterion(pred, x_tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_max_norm'])
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_src, x_tgt in val_loader:
                pred = model(x_src)
                val_loss += criterion(pred, x_tgt).item()
        
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 50 == 0:
            print(f"   Epoch {epoch+1:3d}/{config['max_epochs']} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss - config['early_stopping_min_delta']:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            epochs_trained = epoch + 1
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"   Early stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.6f}")
                break
    
    if epochs_trained == 0:
        epochs_trained = config['max_epochs']
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    os.makedirs(Config.ALIGNMENT_MODEL_DIR, exist_ok=True)
    model_path = os.path.join(Config.ALIGNMENT_MODEL_DIR, f"alignment_{layer_type}_{student_name}_to_{teacher_name}.pt")
    
    try:
        torch.save(model.state_dict(), model_path)
        print(f"   ✓ Alignment network saved: {model_path}")
    except Exception as e:
        print(f"   Warning: Could not save alignment model: {e}")
        model_path = "not_saved"
    
    return model, {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'best_val_loss': best_val_loss,
        'epochs_trained': epochs_trained,
        'model_path': model_path,
        'config': config
    }


def run_nonlinear_experiment_pipeline_dynamic(data, teacher_name, student_name, layer_type, 
                                      alignment_config, prober_config, model_a_name, model_b_name):
    
    print(f"\n=== {layer_type.upper()} LAYERS ({teacher_name} -> {student_name}) ===")
    
    if teacher_name == model_a_name:
        teacher_key = 'model_a'
        student_key = 'model_b'
        align_teacher_train = data['alignment']['X_a_train']
        align_student_train = data['alignment']['X_b_train']
        align_teacher_val = data['alignment']['X_a_val']
        align_student_val = data['alignment']['X_b_val']
        student_scaler = data['alignment']['scaler_b']
    elif teacher_name == model_b_name:
        teacher_key = 'model_b'
        student_key = 'model_a'
        align_teacher_train = data['alignment']['X_b_train']
        align_student_train = data['alignment']['X_a_train']
        align_teacher_val = data['alignment']['X_b_val']
        align_student_val = data['alignment']['X_a_val']
        student_scaler = data['alignment']['scaler_a']
    else:
        raise ValueError(f"Teacher name {teacher_name} does not match model_a ({model_a_name}) or model_b ({model_b_name})")
        
    teacher_data = data[teacher_key]
    student_data = data[student_key]
    
    print("  Training Teacher MLP Prober...")
    
    num_train = len(teacher_data['X_train'])
    indices = np.arange(num_train)
    np.random.seed(Config.SEED)
    np.random.shuffle(indices)
    prober_val_size = int(num_train * 0.15)
    prober_train_idx = indices[prober_val_size:]
    prober_val_idx = indices[:prober_val_size]
    
    X_prober_train = torch.from_numpy(teacher_data['X_train'][prober_train_idx]).float().to(Config.DEVICE)
    y_prober_train = torch.from_numpy(teacher_data['y_train'][prober_train_idx].astype(np.int64)).long().to(Config.DEVICE)
    X_prober_val = torch.from_numpy(teacher_data['X_train'][prober_val_idx]).float().to(Config.DEVICE)
    y_prober_val = torch.from_numpy(teacher_data['y_train'][prober_val_idx].astype(np.int64)).long().to(Config.DEVICE)
    
    probe_teacher, best_prober_acc, prober_epochs = train_mlp_prober(
        X_prober_train, y_prober_train,
        X_prober_val, y_prober_val,
        input_dim=teacher_data['X_train'].shape[1],
        device=Config.DEVICE,
        prober_config=prober_config
    )
    
    probe_teacher.eval()
    X_teacher_test_t = torch.from_numpy(teacher_data['X_test']).float().to(Config.DEVICE)
    y_pred_t = probe_teacher.predict(X_teacher_test_t).cpu().numpy()
    y_proba_t = probe_teacher.predict_proba(X_teacher_test_t).cpu().numpy()
    
    cm_t = confusion_matrix(teacher_data['y_test'], y_pred_t)
    metrics_teacher = {
        "accuracy": accuracy_score(teacher_data['y_test'], y_pred_t),
        "precision": precision_score(teacher_data['y_test'], y_pred_t),
        "recall": recall_score(teacher_data['y_test'], y_pred_t),
        "f1": f1_score(teacher_data['y_test'], y_pred_t),
        "auroc": roc_auc_score(teacher_data['y_test'], y_proba_t),
        "confusion_matrix": cm_t.tolist()
    }
    print(f"  Teacher: Acc={metrics_teacher['accuracy']:.4f}, F1={metrics_teacher['f1']:.4f}, AUROC={metrics_teacher['auroc']:.4f}")

    print("  Training Alignment Network...")
    alignment_model, alignment_info = train_alignment_network(
        align_student_train, align_teacher_train,
        align_student_val, align_teacher_val,
        alignment_config, layer_type, teacher_name, student_name
    )
    print(f"  Alignment: val_loss={alignment_info['best_val_loss']:.6f}, epochs={alignment_info['epochs_trained']}")
    
    os.makedirs(Config.PROBER_MODEL_DIR, exist_ok=True)
    prober_path = os.path.join(Config.PROBER_MODEL_DIR, f"mlp_prober_{layer_type}_{teacher_name}.pt")
    try:
        torch.save(probe_teacher.state_dict(), prober_path)
        print(f"  ✓ MLP prober saved: {prober_path}")
    except Exception as e:
        print(f"  Warning: Could not save prober: {e}")
        prober_path = "not_saved"
    
    print("  Testing Cross-Model...")
    alignment_model.eval()
    X_student_scaled = student_scaler.transform(student_data['X_test_raw'])
    X_student_tensor = torch.tensor(X_student_scaled, dtype=torch.float32).to(Config.DEVICE)
    
    with torch.no_grad():
        X_student_projected = alignment_model(X_student_tensor)
    
    y_pred_c = probe_teacher.predict(X_student_projected).cpu().numpy()
    y_proba_c = probe_teacher.predict_proba(X_student_projected).cpu().numpy()
    
    cm_c = confusion_matrix(student_data['y_test'], y_pred_c)
    metrics_cross = {
        "accuracy": accuracy_score(student_data['y_test'], y_pred_c),
        "precision": precision_score(student_data['y_test'], y_pred_c),
        "recall": recall_score(student_data['y_test'], y_pred_c),
        "f1": f1_score(student_data['y_test'], y_pred_c),
        "auroc": roc_auc_score(student_data['y_test'], y_proba_c),
        "confusion_matrix": cm_c.tolist()
    }
    print(f"  Cross:   Acc={metrics_cross['accuracy']:.4f}, F1={metrics_cross['f1']:.4f}, AUROC={metrics_cross['auroc']:.4f}")
    
    return {
        "type": layer_type,
        "teacher_name": teacher_name,
        "student_name": student_name,
        "teacher": metrics_teacher,
        "student_on_teacher": metrics_cross,
        "alignment_model": alignment_info,
        "prober_model": {
            "input_dim": int(teacher_data['X_train'].shape[1]),
            "config": prober_config,
            "best_val_acc": float(best_prober_acc),
            "epochs_trained": prober_epochs,
            "model_path": prober_path
        }
    }


def plot_confusion_matrix(cm, layer_type, model_name="", save_dir=Config.CONFUSION_DIR):
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
                xticklabels=['Non-Hallucinated', 'Hallucinated'],
                yticklabels=['Non-Hallucinated', 'Hallucinated'])
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    title = f'Confusion Matrix - {layer_type.upper()} Layers'
    if model_name:
        title += f' ({model_name})'
    ax.set_title(title)
    
    plt.tight_layout()
    filename = os.path.join(save_dir, f'confusion_matrix_{layer_type}_{model_name}.png' if model_name else f'confusion_matrix_{layer_type}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {filename}")


def build_data_splits(dataset_name, model_a_name, model_b_name, config):
    print("Step 1: Loading model statistics...")
    model_a_stats = DataManager.get_stats(model_a_name, dataset_name)
    model_b_stats = DataManager.get_stats(model_b_name, dataset_name)
    print(f"   {model_a_name}: {model_a_stats['total']} total, {model_a_stats['hallucinations']} hallucinations")
    print(f"   {model_b_name}: {model_b_stats['total']} total, {model_b_stats['hallucinations']} hallucinations")

    print("\nStep 2: Concordance analysis and undersampling for ALIGNMENT...")
    alignment_indices, alignment_labels = get_concordant_indices_and_undersample(model_a_stats, model_b_stats, seed=config.SEED)

    n_alignment = len(alignment_indices)
    rng = np.random.RandomState(config.SEED)
    shuffled_alignment_idx = rng.permutation(n_alignment)
    split_idx_align = int(0.7 * n_alignment)
    alignment_train_local_idx = shuffled_alignment_idx[:split_idx_align]
    alignment_val_local_idx = shuffled_alignment_idx[split_idx_align:]

    print(f"   Alignment samples: train={len(alignment_train_local_idx)}, val={len(alignment_val_local_idx)}")

    print("\nStep 3: Preparing full datasets for each LLM...")
    model_a_balanced_idx, model_a_balanced_labels = get_undersampled_indices_per_model(model_a_stats, config.SEED)
    model_b_balanced_idx, model_b_balanced_labels = get_undersampled_indices_per_model(model_b_stats, config.SEED)

    print(f"   {model_a_name} balanced: {len(model_a_balanced_idx)} samples")
    print(f"   {model_b_name} balanced: {len(model_b_balanced_idx)} samples")

    rng_a = np.random.RandomState(config.SEED)
    rng_b = np.random.RandomState(config.SEED + 1)

    shuffled_a = rng_a.permutation(len(model_a_balanced_idx))
    shuffled_b = rng_b.permutation(len(model_b_balanced_idx))

    split_a = int(0.7 * len(model_a_balanced_idx))
    split_b = int(0.7 * len(model_b_balanced_idx))

    model_a_train_local = shuffled_a[:split_a]
    model_a_test_local = shuffled_a[split_a:]
    model_b_train_local = shuffled_b[:split_b]
    model_b_test_local = shuffled_b[split_b:]

    print(f"\n   Split {model_a_name}: train={len(model_a_train_local)}, test={len(model_a_test_local)}")
    print(f"   Split {model_b_name}: train={len(model_b_train_local)}, test={len(model_b_test_local)}")

    print("\n" + "="*80)
    print("PHASE 2: LOADING AND PREPARING DATA PER LAYER TYPE")
    print("="*80 + "\n")

    data_splits = {}
    
    # Get layer config for the specific dataset
    dataset_layer_config = config.LAYER_CONFIG[dataset_name]
    
    for layer_type in config.LAYER_TYPES:
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"--- Processing {layer_type.upper()} ---")
        
        X_model_a_full, _ = DataManager.load_concatenated_layers(model_a_name, dataset_name, dataset_layer_config[model_a_name][layer_type], layer_type)
        X_model_b_full, _ = DataManager.load_concatenated_layers(model_b_name, dataset_name, dataset_layer_config[model_b_name][layer_type], layer_type)
        
        X_align_a_train = X_model_a_full[alignment_indices][alignment_train_local_idx]
        X_align_b_train = X_model_b_full[alignment_indices][alignment_train_local_idx]
        X_align_a_val = X_model_a_full[alignment_indices][alignment_val_local_idx]
        X_align_b_val = X_model_b_full[alignment_indices][alignment_val_local_idx]
        
        X_a_balanced = X_model_a_full[model_a_balanced_idx]
        X_a_train = X_a_balanced[model_a_train_local]
        X_a_test = X_a_balanced[model_a_test_local]
        y_a_train = model_a_balanced_labels[model_a_train_local]
        y_a_test = model_a_balanced_labels[model_a_test_local]
        
        X_b_balanced = X_model_b_full[model_b_balanced_idx]
        X_b_train = X_b_balanced[model_b_train_local]
        X_b_test = X_b_balanced[model_b_test_local]
        y_b_train = model_b_balanced_labels[model_b_train_local]
        y_b_test = model_b_balanced_labels[model_b_test_local]
        
        del X_model_a_full, X_model_b_full, X_a_balanced, X_b_balanced
        gc.collect()
        
        print(f"   [{layer_type.upper()}] Align: train={X_align_a_train.shape[0]}, val={X_align_a_val.shape[0]}")
        
        scaler_align_a, scaler_align_b = StandardScaler(), StandardScaler()
        scaler_a, scaler_b = StandardScaler(), StandardScaler()
        
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
                "scaler_b": scaler_align_b
            },
            "model_a": {
                "X_train": X_a_train_norm, "X_test": X_a_test_norm,
                "y_train": y_a_train, "y_test": y_a_test,
                "X_test_raw": X_a_test
            },
            "model_b": {
                "X_train": X_b_train_norm, "X_test": X_b_test_norm,
                "y_train": y_b_train, "y_test": y_b_test,
                "X_test_raw": X_b_test
            }
        }
        gc.collect()
    
    return data_splits, {
        "alignment_train": len(alignment_train_local_idx),
        "alignment_val": len(alignment_val_local_idx),
        "model_a_train": len(model_a_train_local),
        "model_a_test": len(model_a_test_local),
        "model_b_train": len(model_b_train_local),
        "model_b_test": len(model_b_test_local)
    }


def main():
    print("="*80)
    print("STARTING EXPERIMENTS")
    print("="*80 + "\n")
    
    all_results_json = []
    
    for dataset_name, dataset_config in Config.LAYER_CONFIG.items():
        print(f"\n{'#'*80}")
        print(f"PROCESSING DATASET: {dataset_name}")
        print(f"{'#'*80}\n")
        
        # Extract model names (keys that are not "save_dir")
        model_names = [k for k in dataset_config.keys() if k != "save_dir"]
        if len(model_names) != 2:
            print(f"Skipping {dataset_name}: Expected 2 models, found {len(model_names)}: {model_names}")
            continue
            
        model_a_name = model_names[0]
        model_b_name = model_names[1]
        
        print(f"Models: {model_a_name} vs {model_b_name}")
        

        
        # Create save directory for this dataset
        save_dir = os.path.join(Config.BASE_DIR, dataset_config["save_dir"])
        os.makedirs(save_dir, exist_ok=True)
        print(f"Save directory created: {save_dir}")
        Config.RESULTS_DIR = os.path.join(save_dir, "results_metrics")
        Config.PLOTS_DIR = os.path.join(save_dir, "alignment_plots")
        Config.CONFUSION_DIR = os.path.join(save_dir, "confusion_matrices")
        
        print("\n" + "-"*60)
        print(f"PHASE 1: DATA PREPARATION ({dataset_name})")
        print("-"*60)
        
        try:
            data_splits, metadata = build_data_splits(dataset_name, model_a_name, model_b_name, Config)
        except Exception as e:
            print(f"Error preparing data for {dataset_name}: {e}")
            traceback.print_exc()
            continue
        
        print("\n" + "-"*60)
        print(f"PHASE 3: RUNNING EXPERIMENTS ({dataset_name})")
        print("-"*60)
        
        scenarios = [
            {"teacher": model_a_name, "student": model_b_name},
            {"teacher": model_b_name, "student": model_a_name}
        ]
        
        dataset_results = []
        for scenario in scenarios:
            print(f"\n{'='*60}")
            print(f"SCENARIO: {scenario['teacher']} -> {scenario['student']}")
            print(f"{'='*60}")
            
            results = []
            for layer_type in Config.LAYER_TYPES:
                try:
                    res = run_nonlinear_experiment_pipeline_dynamic(
                        data_splits[layer_type], 
                        scenario['teacher'], 
                        scenario['student'], 
                        layer_type,
                        Config.ALIGNMENT_CONFIG,
                        Config.PROBER_CONFIG,
                        model_a_name,
                        model_b_name
                    )
                    results.append(res)
                    
                    # Add dataset name to plot filename
                    plot_confusion_matrix(
                        np.array(res['teacher']['confusion_matrix']), 
                        layer_type, 
                        f"Teacher_{scenario['teacher']}_{dataset_name}",
                        save_dir=Config.CONFUSION_DIR
                    )
                    plot_confusion_matrix(
                        np.array(res['student_on_teacher']['confusion_matrix']), 
                        layer_type, 
                        f"{scenario['student']}_on_{scenario['teacher']}_{dataset_name}",
                        save_dir=Config.CONFUSION_DIR
                    )
                except Exception as e:
                    print(f"Error in {layer_type}: {e}")
                    traceback.print_exc()
            
            dataset_results.append({"scenario": f"{scenario['teacher']} -> {scenario['student']}", "results": results})
        
        # Format results for this dataset
        for scenario_data in dataset_results:
            scenario_results = []
            for r in scenario_data['results']:
                align_config = r['alignment_model']['config']
                prober_cfg = r['prober_model']['config']
                
                scenario_results.append({
                    "layer_type": r['type'],
                    "teacher_model": r['teacher_name'],
                    "student_model": r['student_name'],
                    "dataset": dataset_name,
                    "data_info": {
                        "alignment_samples_train": metadata["alignment_train"],
                        "alignment_samples_val": metadata["alignment_val"],
                        "model_a_train": metadata["model_a_train"],
                        "model_a_test": metadata["model_a_test"],
                        "model_b_train": metadata["model_b_train"],
                        "model_b_test": metadata["model_b_test"],
                        "concordant_undersampling_for_alignment": True,
                        "separate_undersampling_per_model": True
                    },
                    "alignment_model_info": {
                        "architecture": "AlignmentNetwork",
                        "input_dim": r['alignment_model']['input_dim'],
                        "output_dim": r['alignment_model']['output_dim'],
                        "hidden_dim": align_config['hidden_dim'],
                        "dropout": align_config['dropout'],
                        "activation": "GELU",
                        "normalization": "LayerNorm",
                        "residual_connection": True
                    },
                    "alignment_training_hyperparameters": {
                        "optimizer": align_config['optimizer'],
                        "learning_rate": align_config['learning_rate'],
                        "weight_decay": align_config['weight_decay'],
                        "batch_size": align_config['batch_size'],
                        "max_epochs": align_config['max_epochs'],
                        "scheduler": align_config['scheduler'],
                        "gradient_clip_max_norm": align_config['gradient_clip_max_norm'],
                        "early_stopping_patience": align_config['early_stopping_patience'],
                        "early_stopping_min_delta": align_config['early_stopping_min_delta']
                    },
                    "alignment_loss_function": {
                        "type": "MixedLoss",
                        "mse_weight": align_config['loss_alpha'],
                        "cosine_weight": align_config['loss_beta']
                    },
                    "prober_model_info": {
                        "architecture": "MLPProber",
                        "input_dim": r['prober_model']['input_dim'],
                        "hidden_dim": prober_cfg['hidden_dim'],
                        "dropout": prober_cfg['dropout']
                    },
                    "prober_training_hyperparameters": {
                        "optimizer": prober_cfg['optimizer'],
                        "learning_rate": prober_cfg['learning_rate'],
                        "weight_decay": prober_cfg['weight_decay'],
                        "batch_size": prober_cfg['batch_size'],
                        "max_epochs": prober_cfg['max_epochs'],
                        "scheduler": prober_cfg['scheduler'],
                        "gradient_clip_max_norm": prober_cfg['gradient_clip_max_norm'],
                        "early_stopping_patience": prober_cfg['early_stopping_patience'],
                        "early_stopping_min_delta": prober_cfg['early_stopping_min_delta'],
                        "loss_function": prober_cfg['loss_function'],
                        "use_class_weights": prober_cfg['use_class_weights']
                    },
                    "training_results": {
                        "alignment_network": {
                            "best_val_loss": round(r['alignment_model']['best_val_loss'], 6),
                            "epochs_trained": r['alignment_model']['epochs_trained'],
                            "model_saved_path": r['alignment_model']['model_path']
                        },
                        "mlp_prober": {
                            "best_val_acc": round(r['prober_model']['best_val_acc'], 4),
                            "epochs_trained": r['prober_model']['epochs_trained'],
                            "model_saved_path": r['prober_model']['model_path']
                        }
                    },
                    "metrics": {
                        "teacher": {
                            "accuracy": round(r['teacher']['accuracy'], 4),
                            "precision": round(r['teacher']['precision'], 4),
                            "recall": round(r['teacher']['recall'], 4),
                            "f1_score": round(r['teacher']['f1'], 4),
                            "auroc": round(r['teacher']['auroc'], 4),
                            "confusion_matrix": {
                                "TN": int(r['teacher']['confusion_matrix'][0][0]),
                                "FP": int(r['teacher']['confusion_matrix'][0][1]),
                                "FN": int(r['teacher']['confusion_matrix'][1][0]),
                                "TP": int(r['teacher']['confusion_matrix'][1][1])
                            }
                        },
                        "student_on_teacher": {
                            "accuracy": round(r['student_on_teacher']['accuracy'], 4),
                            "precision": round(r['student_on_teacher']['precision'], 4),
                            "recall": round(r['student_on_teacher']['recall'], 4),
                            "f1_score": round(r['student_on_teacher']['f1'], 4),
                            "auroc": round(r['student_on_teacher']['auroc'], 4),
                            "confusion_matrix": {
                                "TN": int(r['student_on_teacher']['confusion_matrix'][0][0]),
                                "FP": int(r['student_on_teacher']['confusion_matrix'][0][1]),
                                "FN": int(r['student_on_teacher']['confusion_matrix'][1][0]),
                                "TP": int(r['student_on_teacher']['confusion_matrix'][1][1])
                            }
                        }
                    }
                })
            
            all_results_json.append({
                "dataset": dataset_name,
                "scenario": scenario_data['scenario'],
                "results": scenario_results
            })

    print("\n" + "="*80)
    print("PHASE 4: SAVING RESULTS")
    print("="*80 + "\n")
    
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    metrics_file = os.path.join(Config.RESULTS_DIR, "approach1_mlp_prober_results.json")
    
    with open(metrics_file, 'w') as f:
        json.dump(all_results_json, f, indent=2)
    
    print(f"✓ Results saved in: {metrics_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


