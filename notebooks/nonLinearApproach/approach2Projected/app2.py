import gc
import json
import os
import random
import traceback
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import copy


class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    CACHE_DIR_NAME = "activation_cache"
    # These will be updated dynamically based on dataset config
    RESULTS_DIR = os.path.join(BASE_DIR, "results_metrics")
    PLOTS_DIR = os.path.join(BASE_DIR, "alignment_plots")
    CONFUSION_DIR = os.path.join(BASE_DIR, "confusion_matrices")
    MODELS_DIR_NAME = "models"

    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LAYER_TYPES = ["attn", "mlp", "hidden"]



    LAYER_CONFIG = {
    "belief_bank_constraints": {
        "Llama-3.1-8B-InstructI3nstruct": {
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
    

    
    
    AUTOENCODER_CONFIG = {
        "latent_dim": 128,
        "hidden_dim": 256,
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "batch_size": 64,
        "max_epochs": 300,
        "early_stopping_patience": 30,
        "early_stopping_min_delta": 1e-4,
        "gradient_clip_max_norm": 1.0,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "loss_function": "MSELoss"
    }

    ALIGNMENT_CONFIG = {
        "hidden_dim": 256,
        "dropout": 0.3,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "batch_size": 32,
        "max_epochs": 500,
        "early_stopping_patience": 50,
        "early_stopping_min_delta": 1e-4,
        "gradient_clip_max_norm": 1.0,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "loss_alpha": 0.5,
        "loss_beta": 0.5
    }

    PROBER_CONFIG = {
        "type": "MLPProber",
        "hidden_dim": 64,
        "dropout": 0.3,
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

# ==================================================================
# Data Loading & Statistics (Exact Notebook Logic)
# ==================================================================

class DataManager:
    _cache_root = os.path.join(Config.ROOT_DIR, Config.CACHE_DIR_NAME)

    @classmethod
    def detect_structure_type(cls, model_name, dataset_name):
        base_path = os.path.join(cls._cache_root, model_name, dataset_name, "activation_attn")
        hallucinated_path = os.path.join(base_path, "hallucinated")
        return 'new' if os.path.isdir(hallucinated_path) else 'old'

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
            'percent_hallucinations': (len(hallucinated_ids) / total) * 100 if total > 0 else 0,
            'hallucinated_ids': hallucinated_ids,
            'not_hallucinated_ids': not_hallucinated_ids,
            'model_name': model_name,
            'dataset_name': dataset_name
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
            'percent_hallucinations': (len(hallucinated_ids) / total) * 100 if total > 0 else 0,
            'hallucinated_ids': hallucinated_ids,
            'not_hallucinated_ids': not_hallucinated_ids,
            'model_name': model_name,
            'dataset_name': dataset_name
        }

    @classmethod
    def load_activations_and_labels(cls, model_name, dataset_name, layer, layer_type):
        structure = cls.detect_structure_type(model_name, dataset_name)
        base_path = os.path.join(cls._cache_root, model_name, dataset_name, f"activation_{layer_type}")
        
        if structure == 'new':
            h_act = torch.load(os.path.join(base_path, "hallucinated", f"layer{layer}_activations.pt"), map_location=Config.DEVICE)
            nh_act = torch.load(os.path.join(base_path, "not_hallucinated", f"layer{layer}_activations.pt"), map_location=Config.DEVICE)
            with open(os.path.join(base_path, "hallucinated", f"layer{layer}_instance_ids.json"), 'r') as f: h_ids = json.load(f)
            with open(os.path.join(base_path, "not_hallucinated", f"layer{layer}_instance_ids.json"), 'r') as f: nh_ids = json.load(f)
            
            h_act = h_act.cpu().numpy().astype(np.float32)
            nh_act = nh_act.cpu().numpy().astype(np.float32)
            
            X = np.vstack([h_act, nh_act])
            y = np.concatenate([np.ones(len(h_act)), np.zeros(len(nh_act))]).astype(int)
            ids = np.array(h_ids + nh_ids)
            
            sort_indices = np.argsort(ids)
            return X[sort_indices], y[sort_indices], ids[sort_indices]
        else:
            activations = torch.load(os.path.join(base_path, f"layer{layer}_activations.pt"), map_location=Config.DEVICE)
            X = activations.cpu().numpy().astype(np.float32)
            with open(os.path.join(cls._cache_root, model_name, dataset_name, "generations", "hallucination_labels.json"), 'r') as f:
                labels_data = json.load(f)
            y = np.array([item['is_hallucination'] for item in labels_data], dtype=int)
            return X, y, np.arange(len(y))

    @classmethod
    def load_concatenated_layers(cls, model_name, dataset_name, layer_indices, layer_type):
        combined, y = [], None
        for layer_idx in layer_indices:
            try:
                X_l, y_l, _ = cls.load_activations_and_labels(model_name, dataset_name, layer_idx, layer_type)
                combined.append(X_l)
                if y is None: y = y_l
            except FileNotFoundError:
                print(f"Warning: Layer {layer_idx} not found for {model_name}. Skipping.")
        return np.concatenate(combined, axis=1), y

def get_balanced_indices(y, seed=Config.SEED):
    rng = np.random.RandomState(seed)
    unique_classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    selected_indices = []
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        if len(cls_indices) > min_count:
            selected_indices.extend(rng.choice(cls_indices, size=min_count, replace=False))
        else:
            selected_indices.extend(cls_indices)
    return np.sort(np.array(selected_indices))

def get_concordant_indices_and_undersample(stats1, stats2, seed=Config.SEED):
    hall_1 = set(stats1['hallucinated_ids'])
    hall_2 = set(stats2['hallucinated_ids'])
    all_1 = set(stats1['hallucinated_ids'] + stats1.get('not_hallucinated_ids', []))
    all_2 = set(stats2['hallucinated_ids'] + stats2.get('not_hallucinated_ids', []))
    
    common = sorted(list(all_1.intersection(all_2)))
    y1 = np.array([1 if x in hall_1 else 0 for x in common])
    y2 = np.array([1 if x in hall_2 else 0 for x in common])
    
    concordant_mask = (y1 == y2)
    concordant_indices = np.array(common)[concordant_mask]
    concordant_labels = y1[concordant_mask]
    
    min_count = min(np.sum(concordant_labels == 1), np.sum(concordant_labels == 0))
    rng = np.random.RandomState(seed)
    
    hall_sampled = rng.choice(concordant_indices[concordant_labels == 1], size=min_count, replace=False)
    non_hall_sampled = rng.choice(concordant_indices[concordant_labels == 0], size=min_count, replace=False)
    
    balanced = np.concatenate([hall_sampled, non_hall_sampled])
    rng.shuffle(balanced)
    return balanced, None # Labels handled later via indexing

def get_undersampled_indices_per_model(stats, seed=Config.SEED):
    total = stats['total']
    hall_set = set(stats['hallucinated_ids'])
    y = np.array([1 if i in hall_set else 0 for i in range(total)])
    idx = get_balanced_indices(y, seed)
    return idx, y[idx]

# ==================================================================
# Models
# ==================================================================

class AutoencoderDataset(Dataset):
    def __init__(self, X): self.X = X
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx]

class AlignmentDataset(Dataset):
    def __init__(self, s, t): self.s = s; self.t = t
    def __len__(self): return self.s.shape[0]
    def __getitem__(self, idx): return self.s[idx], self.t[idx]

class ClassificationDataset(Dataset):
    def __init__(self, X, y): self.X = X; self.y = y
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2), nn.LayerNorm(hidden_dim//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, latent_dim), nn.LayerNorm(latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2), nn.LayerNorm(hidden_dim//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
    
    def encode(self, x): return self.encoder(x)

class AlignmentNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, output_dim, bias=False) if input_dim != output_dim else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(output_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim), nn.Dropout(dropout)
        )
        nn.init.zeros_(self.net[-2].weight)
        if self.net[-2].bias is not None: nn.init.zeros_(self.net[-2].bias)

    def forward(self, x):
        x = self.input_proj(x)
        return x + self.net(x)

class MLPProber(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2), nn.LayerNorm(hidden_dim//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x): return self.net(x).squeeze(-1)
    
    def predict(self, x):
        with torch.no_grad(): return (torch.sigmoid(self.forward(x)) > 0.5).long()
        
    def predict_proba(self, x):
        with torch.no_grad(): return torch.sigmoid(self.forward(x))

class MixedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha, self.beta, self.mse = alpha, beta, nn.MSELoss()
    def forward(self, p, t):
        return self.alpha * self.mse(p, t) + self.beta * (1 - F.cosine_similarity(p, t).mean())

# ==================================================================
# Training Wrappers
# ==================================================================

def train_autoencoder(X_train, X_val, input_dim, device, model_name, config):
    set_seed(Config.SEED)
    model = Autoencoder(input_dim, config['latent_dim'], config['hidden_dim'], config['dropout']).to(device)
    opt = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['max_epochs'])
    crit = nn.MSELoss()
    
    train_loader = DataLoader(AutoencoderDataset(X_train), batch_size=config['batch_size'], shuffle=True, generator=get_generator(Config.SEED))
    val_loader = DataLoader(AutoencoderDataset(X_val), batch_size=config['batch_size'], shuffle=False)
    
    best_loss, patience, best_state, ep_trained = float('inf'), 0, None, 0
    
    for epoch in range(config['max_epochs']):
        model.train()
        for x in train_loader:
            opt.zero_grad()
            recon, _ = model(x)
            loss = crit(recon, x)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_max_norm'])
            opt.step()
            
        model.eval()
        v_loss = sum(crit(model(x)[0], x).item() for x in val_loader) / len(val_loader)
        sched.step()
        
        if (epoch+1)%50==0: print(f"    [AE {model_name}] Ep {epoch+1} Val: {v_loss:.4f}")
        
        if v_loss < best_loss - config['early_stopping_min_delta']:
            best_loss, patience, best_state, ep_trained = v_loss, 0, model.state_dict(), epoch+1
        else:
            patience += 1
            if patience >= config['early_stopping_patience']: break
            
    if best_state: model.load_state_dict(best_state)
    return model, best_loss, ep_trained

def train_mlp_prober(X_train, y_train, X_val, y_val, input_dim, device, config):
    set_seed(Config.SEED)
    model = MLPProber(input_dim, config['hidden_dim'], config['dropout']).to(device)
    
    if config['use_class_weights']:
        n_pos = y_train.sum().item()
        pos_weight = torch.tensor([(len(y_train)-n_pos)/n_pos if n_pos>0 else 1.0]).to(device)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        crit = nn.BCEWithLogitsLoss()
        
    opt = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['max_epochs'])
    
    train_loader = DataLoader(ClassificationDataset(X_train, y_train), batch_size=config['batch_size'], shuffle=True, generator=get_generator(Config.SEED))
    val_loader = DataLoader(ClassificationDataset(X_val, y_val), batch_size=config['batch_size'], shuffle=False)
    
    best_acc, patience, best_state, ep_trained = 0.0, 0, None, 0
    
    for epoch in range(config['max_epochs']):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            loss = crit(model(x), y.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_max_norm'])
            opt.step()
            
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                preds.extend(model.predict(x).cpu().numpy())
                labels.extend(y.cpu().numpy())
        acc = accuracy_score(labels, preds)
        sched.step()
        
        if acc > best_acc + config['early_stopping_min_delta']:
            best_acc, patience, best_state, ep_trained = acc, 0, model.state_dict(), epoch+1
        else:
            patience += 1
            if patience >= config['early_stopping_patience']: break
            
    if best_state: model.load_state_dict(best_state)
    return model, best_acc, ep_trained

def train_alignment_network(X_s, X_t, X_s_v, X_t_v, latent_dim, device, config):
    set_seed(Config.SEED)
    model = AlignmentNetwork(latent_dim, latent_dim, config['hidden_dim'], config['dropout']).to(device)
    opt = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['max_epochs'])
    crit = MixedLoss(config['loss_alpha'], config['loss_beta'])
    
    train_loader = DataLoader(AlignmentDataset(X_s, X_t), batch_size=config['batch_size'], shuffle=True, generator=get_generator(Config.SEED))
    val_loader = DataLoader(AlignmentDataset(X_s_v, X_t_v), batch_size=config['batch_size'], shuffle=False)
    
    best_loss, patience, best_state, ep_trained = float('inf'), 0, None, 0
    
    for epoch in range(config['max_epochs']):
        model.train()
        for s, t in train_loader:
            opt.zero_grad()
            loss = crit(model(s), t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_max_norm'])
            opt.step()
            
        model.eval()
        v_loss = sum(crit(model(s), t).item() for s, t in val_loader) / len(val_loader)
        sched.step()
        
        if v_loss < best_loss - config['early_stopping_min_delta']:
            best_loss, patience, best_state, ep_trained = v_loss, 0, model.state_dict(), epoch+1
        else:
            patience += 1
            if patience >= config['early_stopping_patience']: break
            
    if best_state: model.load_state_dict(best_state)
    return model, best_loss, ep_trained

# ==================================================================
# Core Experiment Function (JSON Construction)
# ==================================================================

def run_experiment(X_teacher, y_teacher, teacher_name, 
                   X_student, y_student, student_name,
                   alignment_data, layer_type, config_name, save_dir):
    
    # 1. Teacher AE (Split on own data)
    num_train_A = len(X_teacher['X_train'])
    idx_A = np.arange(num_train_A)
    np.random.seed(Config.SEED)
    np.random.shuffle(idx_A)
    split_A = int(num_train_A * 0.15)
    
    # NOTE: Notebook logic splits the *Training* set into AE Train and AE Val
    X_A_ae_train = torch.from_numpy(X_teacher['X_train'][idx_A[split_A:]]).float().to(Config.DEVICE)
    X_A_ae_val = torch.from_numpy(X_teacher['X_train'][idx_A[:split_A]]).float().to(Config.DEVICE)
    
    print(f"  Training AE for {teacher_name}...")
    ae_teacher, ae_t_loss, ae_t_ep = train_autoencoder(X_A_ae_train, X_A_ae_val, X_teacher['X_train'].shape[1], Config.DEVICE, teacher_name, Config.AUTOENCODER_CONFIG)

    # 2. Student AE
    num_train_B = len(X_student['X_train'])
    idx_B = np.arange(num_train_B)
    np.random.seed(Config.SEED)
    np.random.shuffle(idx_B)
    split_B = int(num_train_B * 0.15)
    
    X_B_ae_train = torch.from_numpy(X_student['X_train'][idx_B[split_B:]]).float().to(Config.DEVICE)
    X_B_ae_val = torch.from_numpy(X_student['X_train'][idx_B[:split_B]]).float().to(Config.DEVICE)
    
    print(f"  Training AE for {student_name}...")
    ae_student, ae_s_loss, ae_s_ep = train_autoencoder(X_B_ae_train, X_B_ae_val, X_student['X_train'].shape[1], Config.DEVICE, student_name, Config.AUTOENCODER_CONFIG)

    # 3. Encoding
    ae_teacher.eval()
    ae_student.eval()
    with torch.no_grad():
        Z_A_train = ae_teacher.encode(torch.from_numpy(X_teacher['X_train']).float().to(Config.DEVICE))
        Z_A_test = ae_teacher.encode(torch.from_numpy(X_teacher['X_test']).float().to(Config.DEVICE))
        Z_B_test = ae_student.encode(torch.from_numpy(X_student['X_test']).float().to(Config.DEVICE))
        
        Z_align_t_train = ae_teacher.encode(torch.from_numpy(alignment_data['X_teacher_train']).float().to(Config.DEVICE))
        Z_align_t_val = ae_teacher.encode(torch.from_numpy(alignment_data['X_teacher_val']).float().to(Config.DEVICE))
        Z_align_s_train = ae_student.encode(torch.from_numpy(alignment_data['X_student_train']).float().to(Config.DEVICE))
        Z_align_s_val = ae_student.encode(torch.from_numpy(alignment_data['X_student_val']).float().to(Config.DEVICE))

    # 4. Train Prober (Teacher Latent) - Using the same subset logic as notebook
    Z_prob_train = Z_A_train[idx_A[split_A:]]
    y_prob_train = torch.from_numpy(X_teacher['y_train'][idx_A[split_A:]]).float().to(Config.DEVICE)
    Z_prob_val = Z_A_train[idx_A[:split_A]]
    y_prob_val = torch.from_numpy(X_teacher['y_train'][idx_A[:split_A]]).float().to(Config.DEVICE)

    print(f"  Training Prober on {teacher_name} latent...")
    prober, prob_acc, prob_ep = train_mlp_prober(Z_prob_train, y_prob_train, Z_prob_val, y_prob_val, Config.AUTOENCODER_CONFIG['latent_dim'], Config.DEVICE, Config.PROBER_CONFIG)

    # 5. Alignment
    print(f"  Training Alignment {student_name} -> {teacher_name}...")
    aligner, align_loss, align_ep = train_alignment_network(Z_align_s_train, Z_align_t_train, Z_align_s_val, Z_align_t_val, Config.AUTOENCODER_CONFIG['latent_dim'], Config.DEVICE, Config.ALIGNMENT_CONFIG)

    # 6. Evaluation
    def get_metrics(model, X_latent, y_true):
        model.eval()
        y_pred = model.predict(X_latent).cpu().numpy()
        y_prob = model.predict_proba(X_latent).cpu().numpy()
        cm = confusion_matrix(y_true, y_pred)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "auroc": roc_auc_score(y_true, y_prob),
            "confusion_matrix": cm.tolist()
        }

    # Teacher Performance
    metrics_teacher = get_metrics(prober, Z_A_test, X_teacher['y_test'])
    
    # Cross Performance
    aligner.eval()
    with torch.no_grad():
        Z_B_aligned = aligner(Z_B_test)
    metrics_cross = get_metrics(prober, Z_B_aligned, X_student['y_test'])

    # 7. Saving
    models_dir = os.path.join(save_dir, "models", layer_type)
    results_dir = os.path.join(save_dir, "results_metrics")
    plots_dir = os.path.join(save_dir, "alignment_plots")
    confusion_dir = os.path.join(save_dir, "confusion_matrices")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(confusion_dir, exist_ok=True)
    
    def save(model, name, meta):
        path = os.path.join(models_dir, f"{config_name}_{name}.pt")
        torch.save({**meta, "model_state_dict": model.state_dict()}, path)
        return path

    ae_t_path = save(ae_teacher, f"autoencoder_{teacher_name}", {"config": Config.AUTOENCODER_CONFIG})
    ae_s_path = save(ae_student, f"autoencoder_{student_name}", {"config": Config.AUTOENCODER_CONFIG})
    prob_path = save(prober, f"mlp_prober_{teacher_name}", {"config": Config.PROBER_CONFIG})
    align_path = save(aligner, f"aligner_{student_name}_to_{teacher_name}", {"config": Config.ALIGNMENT_CONFIG})

    # 8. Construct JSON Output (Exactly matching notebook schema)
    return {
        "type": layer_type,
        "teacher_name": teacher_name,
        "student_name": student_name,
        "autoencoder_teacher": {
            "input_dim": int(X_teacher['X_train'].shape[1]),
            "config": Config.AUTOENCODER_CONFIG,
            "best_val_loss": float(ae_t_loss),
            "epochs_trained": ae_t_ep,
            "model_path": ae_t_path
        },
        "autoencoder_student": {
            "input_dim": int(X_student['X_train'].shape[1]),
            "config": Config.AUTOENCODER_CONFIG,
            "best_val_loss": float(ae_s_loss),
            "epochs_trained": ae_s_ep,
            "model_path": ae_s_path
        },
        "prober_model": {
            "input_dim": Config.AUTOENCODER_CONFIG['latent_dim'],
            "config": Config.PROBER_CONFIG,
            "best_val_acc": float(prob_acc),
            "epochs_trained": prob_ep,
            "model_path": prob_path
        },
        "alignment_model": {
            "input_dim": Config.AUTOENCODER_CONFIG['latent_dim'],
            "output_dim": Config.AUTOENCODER_CONFIG['latent_dim'],
            "config": Config.ALIGNMENT_CONFIG,
            "best_val_loss": float(align_loss),
            "epochs_trained": align_ep,
            "model_path": align_path
        },
        "metrics": {
            "teacher": {
                "accuracy": round(metrics_teacher['accuracy'], 4),
                "precision": round(metrics_teacher['precision'], 4),
                "recall": round(metrics_teacher['recall'], 4),
                "f1_score": round(metrics_teacher['f1'], 4),
                "auroc": round(metrics_teacher['auroc'], 4),
                "confusion_matrix": {
                    "TN": int(metrics_teacher['confusion_matrix'][0][0]),
                    "FP": int(metrics_teacher['confusion_matrix'][0][1]),
                    "FN": int(metrics_teacher['confusion_matrix'][1][0]),
                    "TP": int(metrics_teacher['confusion_matrix'][1][1])
                }
            },
            "student_on_teacher": {
                "accuracy": round(metrics_cross['accuracy'], 4),
                "precision": round(metrics_cross['precision'], 4),
                "recall": round(metrics_cross['recall'], 4),
                "f1_score": round(metrics_cross['f1'], 4),
                "auroc": round(metrics_cross['auroc'], 4),
                "confusion_matrix": {
                    "TN": int(metrics_cross['confusion_matrix'][0][0]),
                    "FP": int(metrics_cross['confusion_matrix'][0][1]),
                    "FN": int(metrics_cross['confusion_matrix'][1][0]),
                    "TP": int(metrics_cross['confusion_matrix'][1][1])
                }
            }
        }
    }


# ==================================================================
# Main Execution
# ==================================================================

def plot_confusion_matrix(cm, layer_type, model_name="", save_dir=Config.CONFUSION_DIR):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-Hall', 'Hall'], yticklabels=['Non-Hall', 'Hall'])
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(f'Confusion Matrix - {layer_type.upper()} ({model_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_{layer_type}_{model_name}.png'))
    plt.close()

def main():
    print("="*80)
    print("STARTING APPROACH 3: AUTOENCODER ADAPTATION (STRICT NOTEBOOK REPLICATION)")
    print("="*80 + "\n")
    
    for dataset_name, dataset_config in Config.LAYER_CONFIG.items():
        print(f"PROCESSING DATASET: {dataset_name}")
        model_names = [k for k in dataset_config.keys() if k != "save_dir"]
        model_a, model_b = model_names[0], model_names[1]
        save_dir = os.path.join(Config.BASE_DIR, dataset_config["save_dir"])
        Config.RESULTS_DIR = os.path.join(save_dir, "results_metrics")
        Config.PLOTS_DIR = os.path.join(save_dir, "alignment_plots")
        Config.CONFUSION_DIR = os.path.join(save_dir, "confusion_matrices")
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        os.makedirs(Config.PLOTS_DIR, exist_ok=True)
        os.makedirs(Config.CONFUSION_DIR, exist_ok=True)

        # 1. Load Stats (Global)
        set_seed(Config.SEED)
        stats_a = DataManager.get_stats(model_a, dataset_name)
        stats_b = DataManager.get_stats(model_b, dataset_name)
        
        # 2. Indices (Global)
        # 2a. Alignment (Concordant)
        align_indices, _ = get_concordant_indices_and_undersample(stats_a, stats_b, Config.SEED)
        rng = np.random.RandomState(Config.SEED)
        shuffled_align = rng.permutation(len(align_indices))
        split = int(0.7 * len(align_indices))
        align_train_local = shuffled_align[:split]
        align_val_local = shuffled_align[split:]
        
        # 2b. Models (Independent Undersampling)
        idx_a_bal, y_a_bal = get_undersampled_indices_per_model(stats_a, Config.SEED)
        idx_b_bal, y_b_bal = get_undersampled_indices_per_model(stats_b, Config.SEED)
        
        rng_a = np.random.RandomState(Config.SEED)
        rng_b = np.random.RandomState(Config.SEED + 1)  # Different seed for model B as in notebook
        
        shuffled_a = rng_a.permutation(len(idx_a_bal))
        shuffled_b = rng_b.permutation(len(idx_b_bal))
        
        split_a = int(0.7 * len(idx_a_bal))
        split_b = int(0.7 * len(idx_b_bal))
        
        idx_a_train_local = shuffled_a[:split_a]
        idx_a_test_local = shuffled_a[split_a:]
        idx_b_train_local = shuffled_b[:split_b]
        idx_b_test_local = shuffled_b[split_b:]

        all_results_output = []

        scenarios = [
            {"teacher": model_a, "student": model_b},
            {"teacher": model_b, "student": model_a}
        ]
        
        # Result containers mapping to scenarios by index
        scenario_results_map = {0: [], 1: []}

        for layer_type in Config.LAYER_TYPES:
            print(f"\n{'='*40}\nPROCESSING LAYER TYPE: {layer_type.upper()}\n{'='*40}")
            gc.collect()
            torch.cuda.empty_cache()

            # 3. Load & Normalize Data (Per Layer, as in Notebook)
            try:
                X_a_full, _ = DataManager.load_concatenated_layers(model_a, dataset_name, dataset_config[model_a][layer_type], layer_type)
                X_b_full, _ = DataManager.load_concatenated_layers(model_b, dataset_name, dataset_config[model_b][layer_type], layer_type)

                # Alignment Data
                X_align_a_train = X_a_full[align_indices][align_train_local]
                X_align_b_train = X_b_full[align_indices][align_train_local]
                X_align_a_val = X_a_full[align_indices][align_val_local]
                X_align_b_val = X_b_full[align_indices][align_val_local]

                # Model Data (Balanced)
                X_a_bal = X_a_full[idx_a_bal]
                X_b_bal = X_b_full[idx_b_bal]

                X_a_train = X_a_bal[idx_a_train_local]
                X_a_test = X_a_bal[idx_a_test_local]
                y_a_train = y_a_bal[idx_a_train_local]
                y_a_test = y_a_bal[idx_a_test_local]

                X_b_train = X_b_bal[idx_b_train_local]
                X_b_test = X_b_bal[idx_b_test_local]
                y_b_train = y_b_bal[idx_b_train_local]
                y_b_test = y_b_bal[idx_b_test_local]

                # Normalization
                scaler_a = StandardScaler()
                scaler_b = StandardScaler()
                scaler_align_a = StandardScaler()
                scaler_align_b = StandardScaler()

                # Notebook fits alignment scalers on alignment train data
                X_align_a_train_norm = scaler_align_a.fit_transform(X_align_a_train).astype(np.float32)
                X_align_b_train_norm = scaler_align_b.fit_transform(X_align_b_train).astype(np.float32)
                X_align_a_val_norm = scaler_align_a.transform(X_align_a_val).astype(np.float32)
                X_align_b_val_norm = scaler_align_b.transform(X_align_b_val).astype(np.float32)

                # Notebook fits model scalers on model train data
                X_a_train_norm = scaler_a.fit_transform(X_a_train).astype(np.float32)
                X_a_test_norm = scaler_a.transform(X_a_test).astype(np.float32)
                X_b_train_norm = scaler_b.fit_transform(X_b_train).astype(np.float32)
                X_b_test_norm = scaler_b.transform(X_b_test).astype(np.float32)

                # Pack Data
                data_splits = {
                    "alignment": {
                        "X_a_train": X_align_a_train_norm, "X_b_train": X_align_b_train_norm,
                        "X_a_val": X_align_a_val_norm, "X_b_val": X_align_b_val_norm
                    },
                    "model_a": { "X_train": X_a_train_norm, "X_test": X_a_test_norm, "y_train": y_a_train, "y_test": y_a_test },
                    "model_b": { "X_train": X_b_train_norm, "X_test": X_b_test_norm, "y_train": y_b_train, "y_test": y_b_test }
                }

                # 4. Run Scenarios
                for i, scenario in enumerate(scenarios):
                    t_name, s_name = scenario['teacher'], scenario['student']
                    print(f"\n   --- Scenario: {t_name} (Teacher) <- {s_name} (Student) ---")
                    set_seed(Config.SEED)

                    if t_name == model_a:
                        teacher_data = data_splits['model_a']
                        student_data = data_splits['model_b']
                        align_data = {
                            'X_teacher_train': data_splits['alignment']['X_a_train'],
                            'X_teacher_val': data_splits['alignment']['X_a_val'],
                            'X_student_train': data_splits['alignment']['X_b_train'],
                            'X_student_val': data_splits['alignment']['X_b_val']
                        }
                    else:
                        teacher_data = data_splits['model_b']
                        student_data = data_splits['model_a']
                        align_data = {
                            'X_teacher_train': data_splits['alignment']['X_b_train'],
                            'X_teacher_val': data_splits['alignment']['X_b_val'],
                            'X_student_train': data_splits['alignment']['X_a_train'],
                            'X_student_val': data_splits['alignment']['X_a_val']
                        }

                    res = run_experiment(
                        teacher_data, teacher_data['y_train'], t_name,
                        student_data, student_data['y_train'], s_name,
                        align_data, layer_type, "CONFIG1", save_dir
                    )
                    
                    scenario_results_map[i].append(res)
                    
                    # Plot CMs
                    #plot_confusion_matrix(np.array(res['metrics']['teacher']['confusion_matrix'].values()).reshape(2,2), 
                                         # layer_type, f"Teacher_{t_name.replace('.','_').replace('-','_')}")
                    #plot_confusion_matrix(np.array(res['metrics']['student_on_teacher']['confusion_matrix'].values()).reshape(2,2), 
                                          #layer_type, f"{s_name.replace('.','_').replace('-','_')}_on_{t_name.replace('.','_').replace('-','_')}")

            except Exception as e:
                print(f"CRITICAL ERROR in {layer_type}: {e}")
                traceback.print_exc()

        # 5. Final JSON Structure
        for i, scenario in enumerate(scenarios):
            all_results_output.append({
                "scenario": f"{scenario['teacher']} (teacher) → {scenario['student']} (student)",
                "results": scenario_results_map[i]
            })

        json_path = os.path.join(Config.RESULTS_DIR, "approach2_autoencoder_results.json")
        with open(json_path, 'w') as f:
            json.dump(all_results_output, f, indent=2)
        print(f"\n✓ Results saved to: {json_path}")

if __name__ == "__main__":
    main()
