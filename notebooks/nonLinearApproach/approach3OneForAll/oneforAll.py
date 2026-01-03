

import gc
import json
import os
import random
import traceback
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ==================================================================
# Configuration
# ==================================================================

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Assuming the script is in a subdir, go up 3 levels to find project root (adjust as needed)
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    CACHE_DIR_NAME = "activation_cache"
    
    # These will be updated dynamically in main()
    RESULTS_DIR = os.path.join(BASE_DIR, "results_metrics")
    MODELS_DIR = os.path.join(BASE_DIR, "models_frozen_head")
    CONFUSION_DIR = os.path.join(BASE_DIR, "confusion_matrices_frozen_head")

    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    LAYER_TYPES = ["attn", "mlp", "hidden"]

    # Provided LAYER_CONFIG
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




    # Approach 3 Specific Configs
    ENCODER_CONFIG = {
        "latent_dim": 256,
        "hidden_dim": 512,
        "dropout": 0.3,
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
        "batch_size": 64,
        "max_epochs": 100,
        "early_stopping_patience": 15,
        "early_stopping_min_delta": 1e-4,
        "gradient_clip_max_norm": 1.0,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "loss_function": "BCEWithLogitsLoss",
        "use_class_weights": True
    }

    HEAD_CONFIG = {
        "latent_dim": 256,
        "hidden_dim": 128,
        "dropout": 0.3,
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
        "batch_size": 64,
        "max_epochs": 100,
        "early_stopping_patience": 15,
        "early_stopping_min_delta": 1e-4,
        "gradient_clip_max_norm": 1.0,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "loss_function": "BCEWithLogitsLoss",
        "use_class_weights": True
    }

# ==================================================================
# Utilities & Reproducibility
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

def plot_confusion_matrix(y_true, y_pred, title, filename, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

# ==================================================================
# Data Selection (Identical to app3.ipynb)
# ==================================================================

def get_balanced_indices(y, seed=Config.SEED):
    """
    Calcola gli indici per bilanciare il dataset tramite undersampling.
    Questa funzione è DETERMINISTICA dato lo stesso seed e le stesse label.
    """
    rng = np.random.RandomState(seed)
    unique_classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    selected_indices = []
    
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        if len(cls_indices) > min_count:
            # Undersampling: seleziona casualmente min_count campioni
            sampled = rng.choice(cls_indices, size=min_count, replace=False)
            selected_indices.extend(sampled)
        else:
            # Classe già al minimo, prendi tutti
            selected_indices.extend(cls_indices)
    
    return np.sort(np.array(selected_indices))

def get_undersampled_indices_per_model(stats, seed=Config.SEED):
    """
    Applica undersampling al dataset di un singolo modello.
    """
    total = stats['total']
    # 'hallucinated_ids' matches 'hallucinated_items' from notebook logic
    hall_set = set(stats['hallucinated_ids']) 
    
    y = np.array([1 if i in hall_set else 0 for i in range(total)])
    balanced_idx = get_balanced_indices(y, seed)
    balanced_labels = y[balanced_idx]
    
    return balanced_idx, balanced_labels

# ==================================================================
# Data Manager
# ==================================================================

class DataManager:
    _cache_root = os.path.join(Config.ROOT_DIR, Config.CACHE_DIR_NAME)

    @classmethod
    def detect_structure_type(cls, model_name, dataset_name):
        base_path = os.path.join(cls._cache_root, model_name, dataset_name, "activation_attn")
        hallucinated_path = os.path.join(base_path, "hallucinated")
        if os.path.isdir(hallucinated_path):
            return 'new'
        return 'old'

    @classmethod
    def get_stats(cls, model_name, dataset_name):
        structure = cls.detect_structure_type(model_name, dataset_name)
        if structure == 'new':
            return cls._stats_from_new_structure(model_name, dataset_name)
        else:
            return cls._stats_per_json(model_name, dataset_name)

    @classmethod
    def _stats_per_json(cls, model_name, dataset_name):
        file_path = os.path.join(cls._cache_root, model_name, dataset_name, "generations", "hallucination_labels.json")
        with open(file_path, 'r') as file:
            data = json.load(file)
        hallucinated_ids = [item['instance_id'] for item in data if item['is_hallucination']]
        not_hallucinated_ids = [item['instance_id'] for item in data if not item['is_hallucination']]
        total = len(data)
        return {
            'total': total,
            'hallucinations': len(hallucinated_ids),
            'hallucinated_ids': hallucinated_ids,
            'not_hallucinated_ids': not_hallucinated_ids,
            'model_name': model_name
        }

    @classmethod
    def _stats_from_new_structure(cls, model_name, dataset_name):
        base_path = os.path.join(cls._cache_root, model_name, dataset_name, "activation_attn")
        with open(os.path.join(base_path, "hallucinated", "layer0_instance_ids.json"), 'r') as f:
            hallucinated_ids = json.load(f)
        with open(os.path.join(base_path, "not_hallucinated", "layer0_instance_ids.json"), 'r') as f:
            not_hallucinated_ids = json.load(f)
        total = len(hallucinated_ids) + len(not_hallucinated_ids)
        return {
            'total': total,
            'hallucinations': len(hallucinated_ids),
            'hallucinated_ids': hallucinated_ids,
            'not_hallucinated_ids': not_hallucinated_ids,
            'model_name': model_name
        }

    @classmethod
    def load_and_split_layers(cls, model_name, dataset_name, layer_indices, layer_type,
                              balanced_indices, balanced_labels, train_indices, test_indices):
        """
        Loads activation layers, selects ONLY balanced indices per layer, concatenates, and splits.
        Strictly follows app3.ipynb logic.
        """
        print(f" Loading {model_name} [{layer_type}]: layers {layer_indices}...")

        structure_type = cls.detect_structure_type(model_name, dataset_name)
        print(f"  Struttura rilevata: {structure_type}")

        all_features = []
        for layer_idx in layer_indices:
            base_path = os.path.join(cls._cache_root, model_name, dataset_name, "activation_" + layer_type)
            
            if structure_type == 'new':
                # Nuova struttura
                hall_path = os.path.join(base_path, "hallucinated", f"layer{layer_idx}_activations.pt")
                not_hall_path = os.path.join(base_path, "not_hallucinated", f"layer{layer_idx}_activations.pt")
                hall_ids_path = os.path.join(base_path, "hallucinated", f"layer{layer_idx}_instance_ids.json")
                not_hall_ids_path = os.path.join(base_path, "not_hallucinated", f"layer{layer_idx}_instance_ids.json")
                
                if not os.path.exists(hall_path) or not os.path.exists(not_hall_path):
                    print(f" Warning: Layer {layer_idx} non trovato. Salto.")
                    continue
                
                print(f"  Loading layer {layer_idx} (new structure)...", end=" ")
                
                acts_hall = torch.load(hall_path, map_location='cpu')
                acts_not_hall = torch.load(not_hall_path, map_location='cpu')
                
                with open(hall_ids_path, 'r') as f: hall_ids = json.load(f)
                with open(not_hall_ids_path, 'r') as f: not_hall_ids = json.load(f)
                
                if isinstance(acts_hall, torch.Tensor): X_hall = acts_hall.float().numpy()
                else: X_hall = acts_hall.astype(np.float32)
                    
                if isinstance(acts_not_hall, torch.Tensor): X_not_hall = acts_not_hall.float().numpy()
                else: X_not_hall = acts_not_hall.astype(np.float32)
                
                if X_hall.ndim > 2: X_hall = X_hall.reshape(X_hall.shape[0], -1)
                if X_not_hall.ndim > 2: X_not_hall = X_not_hall.reshape(X_not_hall.shape[0], -1)
                
                # Reconstruct full array in original ID order
                total_samples = len(hall_ids) + len(not_hall_ids)
                feature_dim = X_hall.shape[1]
                X_layer = np.zeros((total_samples, feature_dim), dtype=np.float32)
                
                for i, inst_id in enumerate(hall_ids):
                    X_layer[inst_id] = X_hall[i]
                for i, inst_id in enumerate(not_hall_ids):
                    X_layer[inst_id] = X_not_hall[i]
                
                del acts_hall, acts_not_hall, X_hall, X_not_hall
                
            else:
                # Vecchia struttura
                file_path = os.path.join(base_path, f"layer{layer_idx}_activations.pt")
                if not os.path.exists(file_path):
                    print(f" Warning: Layer {layer_idx} non trovato. Salto.")
                    continue

                print(f"  Loading layer {layer_idx} (old structure)...", end=" ")
                acts = torch.load(file_path, map_location='cpu')

                X_layer = acts.float().numpy() if isinstance(acts, torch.Tensor) else acts.astype(np.float32)
                if X_layer.ndim > 2:
                    X_layer = X_layer.reshape(X_layer.shape[0], -1)
                
                del acts
                
            # Select ONLY balanced samples
            X_layer = X_layer[balanced_indices]
            all_features.append(X_layer)
            print(f"done ({X_layer.shape})")
            
            gc.collect()

        if not all_features:
            raise ValueError(f"No layers found for {model_name}")

        X_balanced = np.concatenate(all_features, axis=1)
        
        X_train = X_balanced[train_indices]
        X_test = X_balanced[test_indices]
        y_train = balanced_labels[train_indices]
        y_test = balanced_labels[test_indices]
        
        print(f" Completed! Train: {X_train.shape}, Test: {X_test.shape}")

        return X_train, X_test, y_train, y_test

# ==================================================================
# Models
# ==================================================================

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 1024, dropout: float = 0.3):
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
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class ClassificationHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.net(x).squeeze(-1)

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return (torch.sigmoid(logits) > 0.5).long()

# ==================================================================
# Training Logic
# ==================================================================

def train_teacher_pipeline(X_train, y_train, X_val, y_val, input_dim, device, 
                          model_name, encoder_config, head_config):
    """Train encoder + head jointly for teacher model"""
    print(f"   [Teacher] Training full pipeline for {model_name}...")
    set_seed(Config.SEED)
    
    encoder = Encoder(input_dim, encoder_config['latent_dim'], encoder_config['hidden_dim'], encoder_config['dropout']).to(device)
    head = ClassificationHead(encoder_config['latent_dim'], head_config['hidden_dim'], head_config['dropout']).to(device)
    
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = optim.AdamW(params, lr=encoder_config['learning_rate'], weight_decay=encoder_config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=encoder_config['max_epochs'])
    
    # Class weights
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos if n_pos > 0 else 1.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    train_loader = DataLoader(SimpleDataset(X_train, y_train), batch_size=encoder_config['batch_size'], shuffle=True, generator=get_generator(Config.SEED))
    val_loader = DataLoader(SimpleDataset(X_val, y_val), batch_size=encoder_config['batch_size'], shuffle=False)
    
    best_acc, patience, best_states, epochs_trained = 0.0, 0, None, 0
    
    for epoch in range(encoder_config['max_epochs']):
        encoder.train()
        head.train()
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            latents = encoder(X_batch)
            logits = head(latents)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=encoder_config['gradient_clip_max_norm'])
            optimizer.step()
        
        # Validation
        encoder.eval()
        head.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                latents = encoder(X_batch)
                preds = head.predict(latents)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        scheduler.step()
        
        if acc > best_acc:
            best_acc = acc
            patience = 0
            best_states = {'encoder': encoder.state_dict().copy(), 'head': head.state_dict().copy()}
            epochs_trained = epoch + 1
        else:
            patience += 1
            if patience >= encoder_config['early_stopping_patience']:
                break
    
    if best_states:
        encoder.load_state_dict(best_states['encoder'])
        head.load_state_dict(best_states['head'])
    
    return encoder, head, best_acc, epochs_trained

def train_student_adapter(X_train, y_train, X_val, y_val, input_dim, frozen_head, device, 
                         student_name, encoder_config):
    """Train new encoder with frozen head from teacher"""
    print(f"   [Student] Training Adapter Encoder for {student_name} (Head Frozen)...")
    
    # Freeze Head
    frozen_head.eval()
    for param in frozen_head.parameters(): param.requires_grad = False
    
    set_seed(Config.SEED)
    encoder = Encoder(input_dim, encoder_config['latent_dim'], encoder_config['hidden_dim'], encoder_config['dropout']).to(device)
    
    optimizer = optim.AdamW(encoder.parameters(), lr=encoder_config['learning_rate'], weight_decay=encoder_config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=encoder_config['max_epochs'])
    
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos if n_pos > 0 else 1.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    train_loader = DataLoader(SimpleDataset(X_train, y_train), batch_size=encoder_config['batch_size'], shuffle=True, generator=get_generator(Config.SEED))
    val_loader = DataLoader(SimpleDataset(X_val, y_val), batch_size=encoder_config['batch_size'], shuffle=False)
    
    best_acc, patience, best_state, epochs_trained = 0.0, 0, None, 0
    
    for epoch in range(encoder_config['max_epochs']):
        encoder.train()
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            latents = encoder(X_batch)
            logits = frozen_head(latents)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=encoder_config['gradient_clip_max_norm'])
            optimizer.step()
        
        # Validation
        encoder.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                latents = encoder(X_batch)
                preds = frozen_head.predict(latents)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        scheduler.step()
        
        if acc > best_acc:
            best_acc = acc
            patience = 0
            best_state = encoder.state_dict().copy()
            epochs_trained = epoch + 1
        else:
            patience += 1
            if patience >= encoder_config['early_stopping_patience']:
                break
    
    if best_state:
        encoder.load_state_dict(best_state)
    
    return encoder, best_acc, epochs_trained

# ==================================================================
# Core Experiment Logic
# ==================================================================

def run_frozen_head_experiment(teacher_data, teacher_name, student_data, student_name, 
                               layer_type, device, save_dirs):
    """
    Runs the OneForAll logic strictly matching app3.ipynb Phase 0, 1, 2.
    """
    
    # --- PHASE 0: Create SEPARATE train/val splits for Teacher and Student ---
    # Teacher split (Exactly as in app3.ipynb)
    n_tr_t = len(teacher_data["X_train"])
    idx_t = np.arange(n_tr_t)
    np.random.seed(Config.SEED)
    np.random.shuffle(idx_t)
    v_size_t = int(0.15 * n_tr_t)
    tr_idx_t, val_idx_t = idx_t[v_size_t:], idx_t[:v_size_t]
    
    # Student Splits (Exactly as in app3.ipynb)
    n_tr_s = len(student_data["X_train"])
    idx_s = np.arange(n_tr_s)
    np.random.seed(Config.SEED+100) # Different seed for student
    np.random.shuffle(idx_s)
    v_size_s = int(0.15 * n_tr_s)
    tr_idx_s, val_idx_s = idx_s[v_size_s:], idx_s[:v_size_s]
    
    # --- PHASE 1: Train Teacher ---
    enc_teacher, head_shared, best_acc_t, epochs_t = train_teacher_pipeline(
        teacher_data["X_train"][tr_idx_t], teacher_data["y_train"][tr_idx_t],
        teacher_data["X_train"][val_idx_t], teacher_data["y_train"][val_idx_t],
        input_dim=teacher_data["X_train"].shape[1],
        device=device, model_name=teacher_name,
        encoder_config=Config.ENCODER_CONFIG, head_config=Config.HEAD_CONFIG
    )
    
    # Eval Teacher
    enc_teacher.eval(); head_shared.eval()
    with torch.no_grad():
        z_t = enc_teacher(torch.from_numpy(teacher_data["X_test"]).float().to(device))
        preds_t = head_shared.predict(z_t).cpu().numpy()
        probs_t = torch.sigmoid(head_shared(z_t)).cpu().numpy()
        
    metrics_t = {
        "acc": accuracy_score(teacher_data["y_test"], preds_t),
        "f1": f1_score(teacher_data["y_test"], preds_t),
        "prec": precision_score(teacher_data["y_test"], preds_t),
        "rec": recall_score(teacher_data["y_test"], preds_t),
        "auroc": roc_auc_score(teacher_data["y_test"], probs_t),
        "cm": confusion_matrix(teacher_data["y_test"], preds_t).tolist()
    }
    
    plot_confusion_matrix(teacher_data["y_test"], preds_t, 
                          f"Teacher {teacher_name} ({layer_type})", 
                          f"cm_{layer_type}_teacher_{teacher_name}.png", save_dirs['confusion'])

    # --- PHASE 2: Train Student (Adapter) ---
    enc_student, best_acc_s, epochs_s = train_student_adapter(
        student_data["X_train"][tr_idx_s], student_data["y_train"][tr_idx_s],
        student_data["X_train"][val_idx_s], student_data["y_train"][val_idx_s],
        input_dim=student_data["X_train"].shape[1],
        frozen_head=head_shared,
        device=device, student_name=student_name,
        encoder_config=Config.ENCODER_CONFIG
    )
    
    # Eval Student
    enc_student.eval()
    with torch.no_grad():
        z_s = enc_student(torch.from_numpy(student_data["X_test"]).float().to(device))
        preds_s = head_shared.predict(z_s).cpu().numpy()
        probs_s = torch.sigmoid(head_shared(z_s)).cpu().numpy()
        
    metrics_s = {
        "acc": accuracy_score(student_data["y_test"], preds_s),
        "f1": f1_score(student_data["y_test"], preds_s),
        "prec": precision_score(student_data["y_test"], preds_s),
        "rec": recall_score(student_data["y_test"], preds_s),
        "auroc": roc_auc_score(student_data["y_test"], probs_s),
        "cm": confusion_matrix(student_data["y_test"], preds_s).tolist()
    }

    plot_confusion_matrix(student_data["y_test"], preds_s, 
                          f"Student {student_name} ({layer_type})", 
                          f"cm_{layer_type}_{student_name}_adapter.png", save_dirs['confusion'])

    # --- Save Models ---
    os.makedirs(save_dirs['models'], exist_ok=True)
    
    path_t_enc = os.path.join(save_dirs['models'], f"frozen_head_encoder_{teacher_name}.pt")
    torch.save({'model_state_dict': enc_teacher.state_dict(), 'config': Config.ENCODER_CONFIG}, path_t_enc)
    
    path_head = os.path.join(save_dirs['models'], f"frozen_head_shared_head_{teacher_name}.pt")
    torch.save({'model_state_dict': head_shared.state_dict(), 'config': Config.HEAD_CONFIG}, path_head)
    
    path_s_enc = os.path.join(save_dirs['models'], f"frozen_head_encoder_{student_name}_adapter.pt")
    torch.save({'model_state_dict': enc_student.state_dict(), 'config': Config.ENCODER_CONFIG}, path_s_enc)

    return {
        "layer_type": layer_type,
        "teacher_model": teacher_name,
        "student_model": student_name,
        "data_info": {
            "teacher_train_samples": len(teacher_data["X_train"]),
            "teacher_test_samples": len(teacher_data["X_test"]),
            "student_train_samples": len(student_data["X_train"]),
            "student_test_samples": len(student_data["X_test"]),
            "independent_undersampling_per_model": True
        },
        "training_results": {
            "teacher_encoder": {"input_dim": int(teacher_data["X_train"].shape[1]), "epochs_trained": epochs_t, "model_saved_path": path_t_enc},
            "shared_head": {"epochs_trained": epochs_t, "model_saved_path": path_head},
            "student_encoder": {"input_dim": int(student_data["X_train"].shape[1]), "epochs_trained": epochs_s, "model_saved_path": path_s_enc}
        },
        "metrics": {
            "teacher": {
                "accuracy": round(metrics_t['acc'], 4),
                "precision": round(metrics_t['prec'], 4),
                "recall": round(metrics_t['rec'], 4),
                "f1_score": round(metrics_t['f1'], 4),
                "auroc": round(metrics_t['auroc'], 4),
                "confusion_matrix": {
                    "TN": int(metrics_t['cm'][0][0]), "FP": int(metrics_t['cm'][0][1]),
                    "FN": int(metrics_t['cm'][1][0]), "TP": int(metrics_t['cm'][1][1])
                }
            },
            "student_adapter": {
                "accuracy": round(metrics_s['acc'], 4),
                "precision": round(metrics_s['prec'], 4),
                "recall": round(metrics_s['rec'], 4),
                "f1_score": round(metrics_s['f1'], 4),
                "auroc": round(metrics_s['auroc'], 4),
                "confusion_matrix": {
                    "TN": int(metrics_s['cm'][0][0]), "FP": int(metrics_s['cm'][0][1]),
                    "FN": int(metrics_s['cm'][1][0]), "TP": int(metrics_s['cm'][1][1])
                }
            },
            "transfer_gap": {
                "accuracy_gap": round(metrics_t['acc'] - metrics_s['acc'], 4)
            }
        }
    }

# ==================================================================
# Main Execution
# ==================================================================

def main():
    print("="*80)
    print("STARTING APPROACH 3: FROZEN HEAD ADAPTATION (REFACTORED)")
    print("="*80 + "\n")
    
    for dataset_name, dataset_config in Config.LAYER_CONFIG.items():
        print(f"PROCESSING DATASET: {dataset_name}")
        
        # Identify models (keys that are not 'save_dir')
        model_names = [k for k in dataset_config.keys() if k != "save_dir"]
        model_a, model_b = model_names[0], model_names[1]
        
        # Dynamic Directories
        save_base = os.path.join(Config.BASE_DIR, dataset_config["save_dir"])
        Config.RESULTS_DIR = os.path.join(save_base, "results_metrics")
        Config.MODELS_DIR = os.path.join(save_base, "models_frozen_head")
        Config.CONFUSION_DIR = os.path.join(save_base, "confusion_matrices_frozen_head")
        
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        os.makedirs(Config.CONFUSION_DIR, exist_ok=True)

        results_log = []

        # 1. Load Stats
        set_seed(Config.SEED)
        stats_a = DataManager.get_stats(model_a, dataset_name)
        stats_b = DataManager.get_stats(model_b, dataset_name)

        # 2. Get Balanced Indices (Independent for each model)
        # Note: We need labels to calculate this. In new structure, they come from folder names.
        # But get_undersampled_indices_per_model handles constructing the label array using ids.
        idx_a_bal, lbl_a_bal = get_undersampled_indices_per_model(stats_a, Config.SEED)
        idx_b_bal, lbl_b_bal = get_undersampled_indices_per_model(stats_b, Config.SEED)
        
        # 3. Create Local Train/Test Splits (Exactly like notebook)
        print("\nPreparing train/test split for each model...")
        rng_a = np.random.RandomState(Config.SEED)
        shuffled_a = rng_a.permutation(len(idx_a_bal))
        split_a = int(0.7 * len(idx_a_bal))
        train_idx_a, test_idx_a = shuffled_a[:split_a], shuffled_a[split_a:]
        
        rng_b = np.random.RandomState(Config.SEED+1)
        shuffled_b = rng_b.permutation(len(idx_b_bal))
        split_b = int(0.7 * len(idx_b_bal))
        train_idx_b, test_idx_b = shuffled_b[:split_b], shuffled_b[split_b:]

        for layer_type in Config.LAYER_TYPES:
            print(f"\n{'='*40}\nPROCESSING LAYER TYPE: {layer_type.upper()}\n{'='*40}")
            gc.collect()
            torch.cuda.empty_cache()
            
            try:
                # Load MODEL A
                X_a_tr, X_a_te, y_a_tr, y_a_te = DataManager.load_and_split_layers(
                    model_a, dataset_name, dataset_config[model_a][layer_type], layer_type,
                    idx_a_bal, lbl_a_bal, train_idx_a, test_idx_a
                )
                
                # Load MODEL B
                X_b_tr, X_b_te, y_b_tr, y_b_te = DataManager.load_and_split_layers(
                    model_b, dataset_name, dataset_config[model_b][layer_type], layer_type,
                    idx_b_bal, lbl_b_bal, train_idx_b, test_idx_b
                )
                
                # Independent Scaling
                s_a = StandardScaler()
                X_a_tr = s_a.fit_transform(X_a_tr).astype(np.float32)
                X_a_te = s_a.transform(X_a_te).astype(np.float32)
                
                s_b = StandardScaler()
                X_b_tr = s_b.fit_transform(X_b_tr).astype(np.float32)
                X_b_te = s_b.transform(X_b_te).astype(np.float32)
                
                data_map = {
                    model_a: {"X_train": X_a_tr, "y_train": y_a_tr, "X_test": X_a_te, "y_test": y_a_te},
                    model_b: {"X_train": X_b_tr, "y_train": y_b_tr, "X_test": X_b_te, "y_test": y_b_te}
                }
                
                scenarios = [
                    {"teacher": model_a, "student": model_b},
                    {"teacher": model_b, "student": model_a}
                ]
                
                save_dirs = {
                    'models': os.path.join(Config.MODELS_DIR, layer_type),
                    'confusion': Config.CONFUSION_DIR
                }
                
                for sc in scenarios:
                    print(f"   --- Scenario: {sc['teacher']} -> {sc['student']} ---")
                    res = run_frozen_head_experiment(
                        data_map[sc['teacher']], sc['teacher'],
                        data_map[sc['student']], sc['student'],
                        layer_type, Config.DEVICE, save_dirs
                    )
                    results_log.append(res)
                    print(f"     [Result] Gap: {res['metrics']['transfer_gap']['accuracy_gap']:.4f}")

            except Exception as e:
                print(f"CRITICAL ERROR in {layer_type}: {e}")
                traceback.print_exc()
        
        # Save Final Results
        json_path = os.path.join(Config.RESULTS_DIR, "approach3_frozen_head_results.json")
        with open(json_path, 'w') as f:
            json.dump(results_log, f, indent=2)
        print(f"\n✓ Results saved to: {json_path}")

if __name__ == "__main__":
    main()