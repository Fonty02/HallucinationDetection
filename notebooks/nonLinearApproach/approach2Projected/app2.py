# %% [markdown]
# # Approach 3: Autoencoder-based Alignment with MLP Prober
# 
# 
# ## Pipeline Overview:
# 1. **Autoencoder for Teacher**: Learn to compress Teacher activations to latent dimension ($X_T \to Z_T$)
# 2. **Autoencoder for Student**: Learn to compress Student activations to the same latent dimension ($X_S \to Z_S$)
# 3. **MLP Prober on Teacher**: Train an MLP classifier on the reduced Teacher space
# 4. **Alignment Network**: Learn to align Student's latent space to Teacher's latent space ($Z_S \to Z_T$)
# 5. **Evaluation**: Test the aligned Student representations on the Teacher's MLP prober
# 
# 

# %%
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import gc
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score
import traceback
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random

# ==================================================================
# DEVICE CONFIGURATION
# ==================================================================
DEVICE = torch.device("cuda")

# ==================================================================
# REPRODUCIBILITY SETTINGS
# ==================================================================
SEED = 42

def set_seed(seed=SEED):
    """Set all seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Set seeds at import time
set_seed(SEED)

# %%

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
CACHE_DIR_NAME = "activation_cache"
HF_DEFAULT_HOME = os.environ.get("HF_HOME", "~\\.cache\\huggingface\\hub")

# Nomi dei modelli (usati come costanti in tutto il notebook)
MODEL_A = "gemma-2-9b-it"
MODEL_B = "Llama-3.1-8B-Instruct"

LAYER_CONFIG = {
    MODEL_A: 
    {
        "attn": [23,27,33],
        "mlp":[24,25,26],
        "hidden": [23,24,27]
    },    
    MODEL_B: 
    {
        "attn": [5,8,12],
        "mlp":[13,14,15],
        "hidden": [13,14,15]
    }  
}
DATASET_NAME = "belief_bank_constraints"

# ==================================================================
# AUTOENCODER CONFIGURATION
# ==================================================================
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

# ==================================================================
# ALIGNMENT CONFIGURATION
# ==================================================================
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
    "loss_alpha": 0.5,  # MSE weight
    "loss_beta": 0.5    # Cosine weight
}

# ==================================================================
# MLP PROBER CONFIGURATION
# ==================================================================
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

# %% [markdown]
# ### Dataset preparation

# %%
def stats_per_json(model_name, dataset_name):
    """
    Versione originale per la vecchia struttura con hallucination_labels.json
    """
    file_path = os.path.join(PROJECT_ROOT, CACHE_DIR_NAME, model_name, dataset_name, "generations", "hallucination_labels.json")
    with open(file_path, 'r') as file:
        data = json.load(file)
    total = len(data)
    hallucinations = sum(1 for item in data if item['is_hallucination'])
    percent_hallucinations = (hallucinations / total) * 100 if total > 0 else 0
    hallucinated_items = [item['instance_id'] for item in data if item['is_hallucination']]
    return {
        'total': total,
        'hallucinations': hallucinations,
        'percent_hallucinations': percent_hallucinations,
        'hallucinated_items': hallucinated_items,
        'model_name': model_name,
        'dataset_name': dataset_name
    }

def stats_from_new_structure(model_name, dataset_name):
    """
    Nuova funzione per la struttura con cartelle hallucinated/ e not_hallucinated/
    """
    base_path = os.path.join(PROJECT_ROOT, CACHE_DIR_NAME, model_name, dataset_name, "activation_attn")
    hallucinated_path = os.path.join(base_path, "hallucinated")
    not_hallucinated_path = os.path.join(base_path, "not_hallucinated")
    
    hall_ids_path = os.path.join(hallucinated_path, "layer0_instance_ids.json")
    not_hall_ids_path = os.path.join(not_hallucinated_path, "layer0_instance_ids.json")
    
    with open(hall_ids_path, 'r') as f:
        hallucinated_ids = json.load(f)
    with open(not_hall_ids_path, 'r') as f:
        not_hallucinated_ids = json.load(f)
    
    total = len(hallucinated_ids) + len(not_hallucinated_ids)
    hallucinations = len(hallucinated_ids)
    percent_hallucinations = (hallucinations / total) * 100 if total > 0 else 0
    
    return {
        'total': total,
        'hallucinations': hallucinations,
        'percent_hallucinations': percent_hallucinations,
        'not_hallucinations': len(not_hallucinated_ids),
        'hallucinated_ids': hallucinated_ids,
        'not_hallucinated_ids': not_hallucinated_ids,
        'hallucinated_items': hallucinated_ids,  # Alias per compatibilità
        'model_name': model_name,
        'dataset_name': dataset_name
    }

def detect_structure_type(model_name, dataset_name):
    """
    Rileva automaticamente se la struttura è vecchia o nuova.
    """
    base_path = os.path.join(PROJECT_ROOT, CACHE_DIR_NAME, model_name, dataset_name, "activation_attn")
    hallucinated_path = os.path.join(base_path, "hallucinated")
    if os.path.isdir(hallucinated_path):
        return 'new'
    return 'old'

def get_stats(model_name, dataset_name):
    """
    Funzione wrapper che rileva automaticamente la struttura e chiama la funzione appropriata.
    """
    structure = detect_structure_type(model_name, dataset_name)
    if structure == 'new':
        return stats_from_new_structure(model_name, dataset_name)
    else:
        return stats_per_json(model_name, dataset_name)


def get_balanced_indices(y, seed=SEED):
    """
    Calcola gli indici per bilanciare il dataset tramite undersampling.
    Questa funzione è DETERMINISTICA dato lo stesso seed e le stesse label.
    
    Args:
        y: numpy array delle label
        seed: seed per la riproducibilità
    
    Returns:
        balanced_indices: numpy array degli indici selezionati (ordinati)
    """
    rng = np.random.RandomState(seed)
    
    # Trova le classi e i loro conteggi
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
    
    # Ordina gli indici per mantenere consistenza
    return np.sort(np.array(selected_indices))


def get_concordant_indices_and_undersample(stats_model1, stats_model2, seed=SEED):
    """
    Trova gli indici dove ENTRAMBI i modelli concordano sull'etichetta,
    poi applica undersampling per bilanciare le classi.
    
    Returns:
        concordant_indices: array di indici concordanti e bilanciati
        labels: array di label corrispondenti (0=non-hallucinated, 1=hallucinated)
    """
    hall_set_1 = set(stats_model1['hallucinated_ids'])
    hall_set_2 = set(stats_model2['hallucinated_ids'])
    
    # Trova tutti gli instance_id per ogni modello
    all_ids_1 = set(stats_model1['hallucinated_ids'] + stats_model1.get('not_hallucinated_ids', []))
    all_ids_2 = set(stats_model2['hallucinated_ids'] + stats_model2.get('not_hallucinated_ids', []))
    
    # Trova instance_id comuni
    common_ids = all_ids_1.intersection(all_ids_2)
    common_ids_sorted = sorted(common_ids)
    
    if not common_ids:
        raise ValueError("Nessun instance_id comune trovato tra i due modelli.")
    
    # Ottieni etichette per gli id comuni
    y1_common = np.array([1 if id in hall_set_1 else 0 for id in common_ids_sorted])
    y2_common = np.array([1 if id in hall_set_2 else 0 for id in common_ids_sorted])
    
    # Trova campioni CONCORDANTI (stessa label in entrambi i modelli)
    concordant_mask = (y1_common == y2_common)
    concordant_indices = np.array(common_ids_sorted)[concordant_mask]
    concordant_labels = y1_common[concordant_mask]
    
    n_hall = np.sum(concordant_labels == 1)
    n_non_hall = np.sum(concordant_labels == 0)
    
    print(f"    - Hallucinated (concordanti): {n_hall}")
    print(f"    - Non-hallucinated (concordanti): {n_non_hall}")
    
    # Undersampling sulla classe maggioritaria
    min_count = min(n_hall, n_non_hall)
    rng = np.random.RandomState(seed)
    
    hall_concordant = concordant_indices[concordant_labels == 1]
    non_hall_concordant = concordant_indices[concordant_labels == 0]
    
    hall_sampled = rng.choice(hall_concordant, size=min_count, replace=False)
    non_hall_sampled = rng.choice(non_hall_concordant, size=min_count, replace=False)
    
    balanced_indices = np.concatenate([hall_sampled, non_hall_sampled])
    balanced_labels = np.concatenate([np.ones(min_count, dtype=np.int8), np.zeros(min_count, dtype=np.int8)])
    
    shuffle_idx = rng.permutation(len(balanced_indices))
    balanced_indices = balanced_indices[shuffle_idx]
    balanced_labels = balanced_labels[shuffle_idx]
    
    print(f"  Dopo undersampling: {len(balanced_indices)} campioni bilanciati ({min_count} per classe)")
    
    return balanced_indices, balanced_labels


def get_undersampled_indices_per_model(model_stats, seed=SEED):
    """
    Applica undersampling al dataset di un singolo modello.
    Usato per addestrare i prober su dati specifici del modello.
    
    Args:
        model_stats: dizionario con statistiche del modello (da get_stats)
        seed: seed per riproducibilità
    
    Returns:
        balanced_idx: array di indici bilanciati
        balanced_labels: array di label corrispondenti
    """
    total = model_stats['total']
    hall_set = set(model_stats['hallucinated_items'])
    
    y = np.array([1 if i in hall_set else 0 for i in range(total)])
    balanced_idx = get_balanced_indices(y, seed)
    balanced_labels = y[balanced_idx]
    
    return balanced_idx, balanced_labels




# %% [markdown]
# ### Model Definitions

# %%
# ------------------------------------------------------------------
# 1. Dataset class for Autoencoder Training
# ------------------------------------------------------------------
class AutoencoderDataset(Dataset):
    def __init__(self, X: torch.Tensor):
        self.X = X
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx]

# ------------------------------------------------------------------
# 2. Dataset class for Alignment
# ------------------------------------------------------------------
class AlignmentDataset(Dataset):
    def __init__(self, x_source: torch.Tensor, x_target: torch.Tensor):
        self.x_source = x_source
        self.x_target = x_target
    
    def __len__(self):
        return self.x_source.shape[0]
    
    def __getitem__(self, idx):
        return self.x_source[idx], self.x_target[idx]

# ------------------------------------------------------------------
# 3. Dataset class for Classification
# ------------------------------------------------------------------
class ClassificationDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ------------------------------------------------------------------
# 4. Autoencoder for Dimensionality Reduction
# ------------------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    

# ------------------------------------------------------------------
# 5. AlignmentNetwork
# ------------------------------------------------------------------
class AlignmentNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
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

    def forward(self, x):
        x_base = self.input_proj(x)
        return x_base + self.net(x_base)
    

# ------------------------------------------------------------------
# 6. MLP Prober
# ------------------------------------------------------------------
class MLPProber(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.3):
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


# ------------------------------------------------------------------
# 7. MixedLoss for Alignment
# ------------------------------------------------------------------
class MixedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        loss_mse = self.mse(pred, target)
        cosine_sim = F.cosine_similarity(pred, target, dim=1).mean()
        loss_cosine = 1 - cosine_sim
        return self.alpha * loss_mse + self.beta * loss_cosine

# %% [markdown]
# ### Training Functions

# %%
def load_activations_and_labels(model_name, dataset_name, layer, layer_type):
    """
    Carica le attivazioni e le label per un dato layer e tipo.
    Supporta sia la vecchia che la nuova struttura dati.
    
    Returns:
        X: numpy array delle attivazioni (n_samples, hidden_dim) - ordinate per instance_id
        y: numpy array delle label (n_samples,) - 1=hallucination, 0=correct
        instance_ids: numpy array degli instance_ids (n_samples,) - ordinati
    """
    structure = detect_structure_type(model_name, dataset_name)
    base_path = os.path.join(PROJECT_ROOT, CACHE_DIR_NAME, model_name, dataset_name, f"activation_{layer_type}")
    
    if structure == 'new':
        hall_act_path = os.path.join(base_path, "hallucinated", f"layer{layer}_activations.pt")
        hall_ids_path = os.path.join(base_path, "hallucinated", f"layer{layer}_instance_ids.json")
        not_hall_act_path = os.path.join(base_path, "not_hallucinated", f"layer{layer}_activations.pt")
        not_hall_ids_path = os.path.join(base_path, "not_hallucinated", f"layer{layer}_instance_ids.json")
        
        hall_activations = torch.load(hall_act_path, map_location=DEVICE)
        not_hall_activations = torch.load(not_hall_act_path, map_location=DEVICE)
        
        with open(hall_ids_path, 'r') as f:
            hall_ids = json.load(f)
        with open(not_hall_ids_path, 'r') as f:
            not_hall_ids = json.load(f)
        
        if isinstance(hall_activations, torch.Tensor):
            hall_activations = hall_activations.cpu().numpy().astype(np.float32)
        if isinstance(not_hall_activations, torch.Tensor):
            not_hall_activations = not_hall_activations.cpu().numpy().astype(np.float32)
        
        X_concat = np.vstack([hall_activations, not_hall_activations])
        y_concat = np.concatenate([
            np.ones(hall_activations.shape[0], dtype=int),
            np.zeros(not_hall_activations.shape[0], dtype=int)
        ])
        ids_concat = np.array(hall_ids + not_hall_ids)
        
        sort_indices = np.argsort(ids_concat)
        X = X_concat[sort_indices]
        y = y_concat[sort_indices]
        instance_ids = ids_concat[sort_indices]
        
        return X, y, instance_ids
    
    else:
        file_path = os.path.join(base_path, f"layer{layer}_activations.pt")
        activations = torch.load(file_path, map_location=DEVICE)
        
        if isinstance(activations, torch.Tensor):
            X = activations.cpu().numpy().astype(np.float32)
        else:
            X = activations.astype(np.float32)
        
        labels_path = os.path.join(PROJECT_ROOT, CACHE_DIR_NAME, model_name, dataset_name, 
                                   "generations", "hallucination_labels.json")
        with open(labels_path, 'r') as f:
            labels_data = json.load(f)
        
        y = np.array([item['is_hallucination'] for item in labels_data], dtype=int)
        instance_ids = np.arange(len(y))
        
        return X, y, instance_ids


def load_concatenated_layers(model_name, dataset_name, layer_indices, type_layer):
    """Carica multipli layer e li concatena."""
    print(f"   Caricamento {model_name} [{type_layer}]: layers {layer_indices}...")
    combined_features = []
    y = None
    
    for layer_idx in layer_indices:
        try:
            X_layer, y_layer, _ = load_activations_and_labels(model_name, dataset_name, layer_idx, type_layer)
            combined_features.append(X_layer)
            if y is None:
                y = y_layer
        except FileNotFoundError as e:
            print(f"Warning: Layer {layer_idx} non trovato: {e}. Salto.")
            continue

    if not combined_features:
        raise ValueError(f"Nessun layer caricato per {model_name}")

    X_final = np.concatenate(combined_features, axis=1)
    return X_final, y


def get_generator(seed=SEED):
    """Create a reproducible generator for DataLoader"""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def train_autoencoder(X_train, X_val, input_dim, device, model_name, autoencoder_config=AUTOENCODER_CONFIG):
    """Train autoencoder for dimensionality reduction with early stopping."""
    
    latent_dim = autoencoder_config['latent_dim']
    hidden_dim = autoencoder_config['hidden_dim']
    
    print(f"   Training Autoencoder for {model_name} ({input_dim} -> {latent_dim})...")
    
    set_seed(SEED)
    autoencoder = Autoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        dropout=autoencoder_config['dropout']
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        autoencoder.parameters(), 
        lr=autoencoder_config['learning_rate'], 
        weight_decay=autoencoder_config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=autoencoder_config['max_epochs'])
    
    # Create dataloaders
    train_dataset = AutoencoderDataset(X_train)
    val_dataset = AutoencoderDataset(X_val)
    train_loader = DataLoader(train_dataset, batch_size=autoencoder_config['batch_size'], 
                             shuffle=True, num_workers=0, generator=get_generator(SEED))
    val_loader = DataLoader(val_dataset, batch_size=autoencoder_config['batch_size'], 
                           shuffle=False, num_workers=0)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    epochs_trained = 0
    
    for epoch in range(autoencoder_config['max_epochs']):
        # Training
        autoencoder.train()
        epoch_loss = 0.0
        for X_batch in train_loader:
            optimizer.zero_grad()
            X_recon, _ = autoencoder(X_batch)
            loss = criterion(X_recon, X_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                autoencoder.parameters(), 
                max_norm=autoencoder_config['gradient_clip_max_norm']
            )
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation
        autoencoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch in val_loader:
                X_recon, _ = autoencoder(X_batch)
                loss = criterion(X_recon, X_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()
        
        if (epoch + 1) % 30 == 0:
            print(f"     Epoch {epoch+1:3d}/{autoencoder_config['max_epochs']} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Early Stopping
        if avg_val_loss < best_val_loss - autoencoder_config['early_stopping_min_delta']:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = autoencoder.state_dict().copy()
            epochs_trained = epoch + 1
        else:
            patience_counter += 1
        
        if patience_counter >= autoencoder_config['early_stopping_patience']:
            print(f"     Early stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.6f}")
            break
    
    if epochs_trained == 0:
        epochs_trained = autoencoder_config['max_epochs']
    
    # Load best model
    if best_model_state is not None:
        autoencoder.load_state_dict(best_model_state)
    
    print(f"   ✓ Autoencoder trained. Final Val Loss: {best_val_loss:.6f}")
    return autoencoder, best_val_loss, epochs_trained


def train_mlp_prober(X_train, y_train, X_val, y_val, input_dim, device, prober_config=PROBER_CONFIG):
    """Train MLP prober with early stopping based on validation accuracy."""
    
    set_seed(SEED)
    prober = MLPProber(
        input_dim=input_dim, 
        hidden_dim=prober_config['hidden_dim'], 
        dropout=prober_config['dropout']
    ).to(device)
    
    # Compute class weights for imbalanced data
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
    
    # Create dataloaders
    train_dataset = ClassificationDataset(X_train, y_train)
    val_dataset = ClassificationDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=prober_config['batch_size'], 
                             shuffle=True, num_workers=0, generator=get_generator(SEED))
    val_loader = DataLoader(val_dataset, batch_size=prober_config['batch_size'], 
                           shuffle=False, num_workers=0)
    
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    epochs_trained = 0
    
    for epoch in range(prober_config['max_epochs']):
        # Training
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
        
        # Validation
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
            print(f"     Epoch {epoch+1:3d}/{prober_config['max_epochs']} | Train Loss: {avg_train_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early Stopping based on accuracy
        if val_acc > best_val_acc + prober_config['early_stopping_min_delta']:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = prober.state_dict().copy()
            epochs_trained = epoch + 1
        else:
            patience_counter += 1
        
        if patience_counter >= prober_config['early_stopping_patience']:
            print(f"     Early stopping at epoch {epoch+1}. Best Val Acc: {best_val_acc:.4f}")
            break
    
    if epochs_trained == 0:
        epochs_trained = prober_config['max_epochs']
    
    # Load best model
    if best_model_state is not None:
        prober.load_state_dict(best_model_state)
    
    return prober, best_val_acc, epochs_trained


def train_alignment_network(X_source_train, X_target_train, X_source_val, X_target_val, 
                            latent_dim, device, alignment_config=ALIGNMENT_CONFIG):
    """Train alignment network to map student latent space to teacher latent space."""
    
    print("   Training Alignment Network...")
    
    set_seed(SEED)
    aligner = AlignmentNetwork(
        input_dim=latent_dim,
        output_dim=latent_dim,
        hidden_dim=alignment_config['hidden_dim'],
        dropout=alignment_config['dropout']
    ).to(device)
    
    criterion = MixedLoss(alpha=alignment_config['loss_alpha'], beta=alignment_config['loss_beta'])
    optimizer = optim.AdamW(
        aligner.parameters(), 
        lr=alignment_config['learning_rate'], 
        weight_decay=alignment_config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=alignment_config['max_epochs'])
    
    # Create dataloaders
    train_dataset = AlignmentDataset(X_source_train, X_target_train)
    val_dataset = AlignmentDataset(X_source_val, X_target_val)
    train_loader = DataLoader(train_dataset, batch_size=alignment_config['batch_size'], 
                             shuffle=True, num_workers=0, generator=get_generator(SEED))
    val_loader = DataLoader(val_dataset, batch_size=alignment_config['batch_size'], 
                           shuffle=False, num_workers=0)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    epochs_trained = 0
    
    for epoch in range(alignment_config['max_epochs']):
        # Training
        aligner.train()
        epoch_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            projected = aligner(data)
            loss = criterion(projected, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                aligner.parameters(), 
                max_norm=alignment_config['gradient_clip_max_norm']
            )
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation
        aligner.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                projected = aligner(data)
                loss = criterion(projected, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"     Epoch {epoch+1:3d}/{alignment_config['max_epochs']} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Early Stopping
        if avg_val_loss < best_val_loss - alignment_config['early_stopping_min_delta']:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = aligner.state_dict().copy()
            epochs_trained = epoch + 1
        else:
            patience_counter += 1
        
        if patience_counter >= alignment_config['early_stopping_patience']:
            print(f"     Early stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.6f}")
            break
    
    if epochs_trained == 0:
        epochs_trained = alignment_config['max_epochs']
    
    # Load best model
    if best_model_state is not None:
        aligner.load_state_dict(best_model_state)
    
    print(f"   ✓ Alignment Network trained. Final Val Loss: {best_val_loss:.6f}")
    return aligner, best_val_loss, epochs_trained

# %% [markdown]
# ### Main Experiment Pipeline (with End-to-End Fine-Tuning)

# %%
def run_experiment_pipeline_with_autoencoder(X_teacher, y_teacher, teacher_name,
                                              X_student, y_student, student_name,
                                              alignment_data,
                                              layer_type, config_name,
                                              autoencoder_config=AUTOENCODER_CONFIG,
                                              alignment_config=ALIGNMENT_CONFIG,
                                              prober_config=PROBER_CONFIG):
    """
    Pipeline con Autoencoder:
    - Autoencoder: addestrato sul dataset PROPRIO di ogni modello (undersampling separato)
    - Alignment: addestrato sul dataset CONCORDANTE (comuni a entrambi i modelli)
    - Prober: addestrato sul dataset PROPRIO del teacher
    
    Args:
        X_teacher, y_teacher: dati del teacher (undersampling proprio)
        X_student, y_student: dati dello student (undersampling proprio)
        alignment_data: dict con dati concordanti per alignment
            - X_teacher_train, X_teacher_val: attivazioni teacher (concordanti)
            - X_student_train, X_student_val: attivazioni student (concordanti)
    """
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {layer_type.upper()} → {teacher_name} ← {student_name}")
    print(f"Using Autoencoder with latent_dim={autoencoder_config['latent_dim']}, hidden_dim={autoencoder_config['hidden_dim']}")
    print(f"{'='*70}")

    # Dati per Autoencoder e Prober (dataset proprio di ogni modello)
    X_A_train_full, X_A_test = X_teacher['X_train'], X_teacher['X_test']
    y_A_train_full, y_A_test = y_teacher['y_train'], y_teacher['y_test']
    X_B_train_full, X_B_test = X_student['X_train'], X_student['X_test']
    y_B_train_full, y_B_test = y_student['y_train'], y_student['y_test']
    
    # Dati per Alignment (dataset concordante)
    X_align_teacher_train = alignment_data['X_teacher_train']
    X_align_teacher_val = alignment_data['X_teacher_val']
    X_align_student_train = alignment_data['X_student_train']
    X_align_student_val = alignment_data['X_student_val']

    device = DEVICE
    print(f"Using device: {device}")
    print(f"   Teacher dataset: {len(X_A_train_full)} train, {len(X_A_test)} test")
    print(f"   Student dataset: {len(X_B_train_full)} train, {len(X_B_test)} test")
    print(f"   Alignment dataset: {len(X_align_teacher_train)} train, {len(X_align_teacher_val)} val")

    # --------------------------------------------------
    # 1. Train Autoencoder for Teacher (sul PROPRIO dataset)
    # --------------------------------------------------
    print("\n1. Training Autoencoder for TEACHER (on teacher's own dataset)...")
    
    num_train_A = len(X_A_train_full)
    indices_A = np.arange(num_train_A)
    np.random.seed(SEED)
    np.random.shuffle(indices_A)
    ae_val_size_A = int(num_train_A * 0.15)
    ae_train_idx_A = indices_A[ae_val_size_A:]
    ae_val_idx_A = indices_A[:ae_val_size_A]
    X_A_ae_train = torch.from_numpy(X_A_train_full[ae_train_idx_A]).float().to(device)
    X_A_ae_val = torch.from_numpy(X_A_train_full[ae_val_idx_A]).float().to(device)

    ae_teacher, ae_teacher_loss, ae_teacher_epochs = train_autoencoder(
        X_A_ae_train, X_A_ae_val,
        input_dim=X_A_train_full.shape[1],
        device=device,
        model_name=teacher_name,
        autoencoder_config=autoencoder_config
    )

    # --------------------------------------------------
    # 2. Train Autoencoder for Student (sul PROPRIO dataset)
    # --------------------------------------------------
    print("\n2. Training Autoencoder for STUDENT (on student's own dataset)...")

    num_train_B = len(X_B_train_full)
    indices_B = np.arange(num_train_B)
    np.random.seed(SEED)
    np.random.shuffle(indices_B)
    ae_val_size_B = int(num_train_B * 0.15)
    ae_train_idx_B = indices_B[ae_val_size_B:]
    ae_val_idx_B = indices_B[:ae_val_size_B]
    X_B_ae_train = torch.from_numpy(X_B_train_full[ae_train_idx_B]).float().to(device)
    X_B_ae_val = torch.from_numpy(X_B_train_full[ae_val_idx_B]).float().to(device)

    ae_student, ae_student_loss, ae_student_epochs = train_autoencoder(
        X_B_ae_train, X_B_ae_val,
        input_dim=X_B_train_full.shape[1],
        device=device,
        model_name=student_name,
        autoencoder_config=autoencoder_config
    )

    # --------------------------------------------------
    # 3. Encode all data to latent space
    # --------------------------------------------------
    print("\n3. Encoding data to latent space...")

    ae_teacher.eval()
    ae_student.eval()

    with torch.no_grad():
        # Teacher encodings (dataset proprio)
        X_A_train_full_t = torch.from_numpy(X_A_train_full).float().to(device)
        X_A_test_t = torch.from_numpy(X_A_test).float().to(device)
        Z_A_train = ae_teacher.encode(X_A_train_full_t)
        Z_A_test = ae_teacher.encode(X_A_test_t)
        
        # Student encodings (dataset proprio)
        X_B_train_full_t = torch.from_numpy(X_B_train_full).float().to(device)
        X_B_test_t = torch.from_numpy(X_B_test).float().to(device)
        Z_B_train = ae_student.encode(X_B_train_full_t)
        Z_B_test = ae_student.encode(X_B_test_t)
        
        # Alignment data encodings (dataset concordante)
        X_align_teacher_train_t = torch.from_numpy(X_align_teacher_train).float().to(device)
        X_align_teacher_val_t = torch.from_numpy(X_align_teacher_val).float().to(device)
        X_align_student_train_t = torch.from_numpy(X_align_student_train).float().to(device)
        X_align_student_val_t = torch.from_numpy(X_align_student_val).float().to(device)
        
        Z_align_teacher_train = ae_teacher.encode(X_align_teacher_train_t)
        Z_align_teacher_val = ae_teacher.encode(X_align_teacher_val_t)
        Z_align_student_train = ae_student.encode(X_align_student_train_t)
        Z_align_student_val = ae_student.encode(X_align_student_val_t)

    print(f"   Teacher latent shape: {Z_A_train.shape}")
    print(f"   Student latent shape: {Z_B_train.shape}")
    print(f"   Alignment latent shape: {Z_align_teacher_train.shape}")

    # --------------------------------------------------
    # 4. Train MLP Prober on Teacher's Latent Space (dataset PROPRIO)
    # --------------------------------------------------
    print("\n4. Training MLP Prober on Teacher's latent space (teacher's own dataset)...")

    # Usa gli stessi indici creati per l'autoencoder del teacher
    Z_A_prober_train = Z_A_train[ae_train_idx_A]
    y_A_prober_train = torch.from_numpy(y_A_train_full[ae_train_idx_A].astype(np.int64)).long().to(device)
    Z_A_prober_val = Z_A_train[ae_val_idx_A]
    y_A_prober_val = torch.from_numpy(y_A_train_full[ae_val_idx_A].astype(np.int64)).long().to(device)

    probe_teacher, best_prober_acc, prober_epochs = train_mlp_prober(
        Z_A_prober_train, y_A_prober_train,
        Z_A_prober_val, y_A_prober_val,
        input_dim=autoencoder_config['latent_dim'],
        device=device,
        prober_config=prober_config
    )
    print(f"   Best prober validation Acc: {best_prober_acc:.4f}")

    # --- Teacher Metrics ---
    probe_teacher.eval()
    y_pred_teacher = probe_teacher.predict(Z_A_test).cpu().numpy()
    y_proba_teacher = probe_teacher.predict_proba(Z_A_test).cpu().numpy()

    cm_teacher = confusion_matrix(y_A_test, y_pred_teacher)
    acc_teacher = accuracy_score(y_A_test, y_pred_teacher)
    prec_teacher = precision_score(y_A_test, y_pred_teacher)
    rec_teacher = recall_score(y_A_test, y_pred_teacher)
    f1_teacher = f1_score(y_A_test, y_pred_teacher)
    auroc_teacher = roc_auc_score(y_A_test, y_proba_teacher)
    print(f"   Teacher Test Acc: {acc_teacher:.4f}, F1: {f1_teacher:.4f}, AUROC: {auroc_teacher:.4f}")

    # --------------------------------------------------
    # 5. Train Alignment Network (dataset CONCORDANTE)
    # --------------------------------------------------
    print("\n5. Training Alignment Network (Student → Teacher latent space, concordant dataset)...")

    aligner, align_loss, align_epochs = train_alignment_network(
        Z_align_student_train, Z_align_teacher_train,
        Z_align_student_val, Z_align_teacher_val,
        latent_dim=autoencoder_config['latent_dim'],
        device=device,
        alignment_config=alignment_config
    )

    # --------------------------------------------------
    # 6. Save Models
    # --------------------------------------------------
    print("\n6. Saving models...")

    model_save_dir = os.path.join("models", layer_type)
    os.makedirs(model_save_dir, exist_ok=True)

    # Save Teacher Autoencoder
    ae_teacher_filename = os.path.join(model_save_dir, f"{config_name}_autoencoder_{teacher_name}.pt")
    torch.save({
        'model_state_dict': ae_teacher.state_dict(),
        'autoencoder_config': autoencoder_config,
        'input_dim': int(X_A_train_full.shape[1]),
        'latent_dim': autoencoder_config['latent_dim'],
        'best_val_loss': ae_teacher_loss,
        'epochs_trained': ae_teacher_epochs,
        'model_name': teacher_name,
    }, ae_teacher_filename)
    print(f"   ✓ Teacher Autoencoder saved: {ae_teacher_filename}")

    # Save Student Autoencoder
    ae_student_filename = os.path.join(model_save_dir, f"{config_name}_autoencoder_{student_name}.pt")
    torch.save({
        'model_state_dict': ae_student.state_dict(),
        'autoencoder_config': autoencoder_config,
        'input_dim': int(X_B_train_full.shape[1]),
        'latent_dim': autoencoder_config['latent_dim'],
        'best_val_loss': ae_student_loss,
        'epochs_trained': ae_student_epochs,
        'model_name': student_name,
    }, ae_student_filename)
    print(f"   ✓ Student Autoencoder saved: {ae_student_filename}")

    # Save MLP Prober
    prober_filename = os.path.join(model_save_dir, f"{config_name}_mlp_prober_{teacher_name}.pt")
    torch.save({
        'model_state_dict': probe_teacher.state_dict(),
        'prober_config': prober_config,
        'input_dim': autoencoder_config['latent_dim'],
        'best_val_acc': best_prober_acc,
        'epochs_trained': prober_epochs,
        'teacher_model': teacher_name,
    }, prober_filename)
    print(f"   ✓ MLP Prober saved: {prober_filename}")

    # Save Alignment Network
    aligner_filename = os.path.join(model_save_dir, f"{config_name}_aligner_{student_name}_to_{teacher_name}.pt")
    torch.save({
        'model_state_dict': aligner.state_dict(),
        'alignment_config': alignment_config,
        'input_dim': autoencoder_config['latent_dim'],
        'output_dim': autoencoder_config['latent_dim'],
        'best_val_loss': align_loss,
        'epochs_trained': align_epochs,
        'student_model': student_name,
        'teacher_model': teacher_name,
    }, aligner_filename)
    print(f"   ✓ Alignment Network saved: {aligner_filename}")

    # --------------------------------------------------
    # 7. Evaluation
    # --------------------------------------------------
    print("\n7. Projecting student test set & evaluating...")

    aligner.eval()
    with torch.no_grad():
        Z_B_aligned = aligner(Z_B_test)
    
    y_pred_cross = probe_teacher.predict(Z_B_aligned).cpu().numpy()
    y_proba_cross = probe_teacher.predict_proba(Z_B_aligned).cpu().numpy()

    # --- Cross-Model Metrics ---
    cm_cross = confusion_matrix(y_B_test, y_pred_cross)
    acc_cross = accuracy_score(y_B_test, y_pred_cross)
    prec_cross = precision_score(y_B_test, y_pred_cross)
    rec_cross = recall_score(y_B_test, y_pred_cross)
    f1_cross = f1_score(y_B_test, y_pred_cross)
    auroc_cross = roc_auc_score(y_B_test, y_proba_cross)

    print(f"\n{'='*50}")
    print(f"FINAL RESULT:")
    print(f"{'='*50}")
    print(f"   Teacher Acc          : {acc_teacher:.4f}, F1: {f1_teacher:.4f}, AUROC: {auroc_teacher:.4f}")
    print(f"   Student → Teacher Acc: {acc_cross:.4f}, F1: {f1_cross:.4f}, AUROC: {auroc_cross:.4f}")
    print(f"   Transfer gap (Acc)   : {acc_teacher - acc_cross:.4f}")
    print(f"   Transfer gap (F1)    : {f1_teacher - f1_cross:.4f}")

    return {
        "type": layer_type,
        "teacher_name": teacher_name,
        "student_name": student_name,
        "autoencoder_teacher": {
            "input_dim": int(X_A_train_full.shape[1]),
            "config": autoencoder_config,
            "best_val_loss": float(ae_teacher_loss),
            "epochs_trained": ae_teacher_epochs,
            "model_path": ae_teacher_filename
        },
        "autoencoder_student": {
            "input_dim": int(X_B_train_full.shape[1]),
            "config": autoencoder_config,
            "best_val_loss": float(ae_student_loss),
            "epochs_trained": ae_student_epochs,
            "model_path": ae_student_filename
        },
        "prober_model": {
            "input_dim": autoencoder_config['latent_dim'],
            "config": prober_config,
            "best_val_acc": float(best_prober_acc),
            "epochs_trained": prober_epochs,
            "model_path": prober_filename
        },
        "alignment_model": {
            "input_dim": autoencoder_config['latent_dim'],
            "output_dim": autoencoder_config['latent_dim'],
            "config": alignment_config,
            "best_val_loss": float(align_loss),
            "epochs_trained": align_epochs,
            "model_path": aligner_filename
        },
        "teacher": {
            "accuracy": acc_teacher,
            "precision": prec_teacher,
            "recall": rec_teacher,
            "f1": f1_teacher,
            "auroc": auroc_teacher,
            "confusion_matrix": cm_teacher.tolist()
        },
        "student_on_teacher": {
            "accuracy": acc_cross,
            "precision": prec_cross,
            "recall": rec_cross,
            "f1": f1_cross,
            "auroc": auroc_cross,
            "confusion_matrix": cm_cross.tolist()
        }
    }


def plot_confusion_matrix(cm, layer_type, model_name="", save_dir="confusion_matrices"):
    """Plot and save confusion matrix as image."""
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

# %% [markdown]
# ### Run Experiments

# %%
print("="*80)
print("PHASE 1: PREPARAZIONE DATI")
print("="*80 + "\n")

set_seed(SEED)

# ============================================
# STEP 1: Ottieni statistiche per entrambi i modelli
# ============================================
print("Step 1: Caricamento statistiche modelli...")
model_a_stats = get_stats(MODEL_A, DATASET_NAME)
model_b_stats = get_stats(MODEL_B, DATASET_NAME)
print(f"   {MODEL_A}: {model_a_stats['total']} totali, {model_a_stats['hallucinations']} allucinazioni")
print(f"   {MODEL_B}: {model_b_stats['total']} totali, {model_b_stats['hallucinations']} allucinazioni")

# ============================================
# STEP 2: Trova campioni concordanti con undersampling (per ALLINEAMENTO)
# ============================================
print("\nStep 2: Analisi concordanza e undersampling per ALLINEAMENTO...")
alignment_indices, alignment_labels = get_concordant_indices_and_undersample(model_a_stats, model_b_stats, seed=SEED)

# ============================================
# STEP 3: Prepara dataset bilanciati per ogni LLM (con undersampling separato)
# ============================================
print("\nStep 3: Preparazione dataset bilanciati per ogni LLM...")
model_a_balanced_idx, model_a_balanced_labels = get_undersampled_indices_per_model(model_a_stats, SEED)
model_b_balanced_idx, model_b_balanced_labels = get_undersampled_indices_per_model(model_b_stats, SEED)

print(f"   {MODEL_A} bilanciato: {len(model_a_balanced_idx)} campioni ({np.sum(model_a_balanced_labels==1)} hall, {np.sum(model_a_balanced_labels==0)} non-hall)")
print(f"   {MODEL_B} bilanciato: {len(model_b_balanced_idx)} campioni ({np.sum(model_b_balanced_labels==1)} hall, {np.sum(model_b_balanced_labels==0)} non-hall)")

# ============================================
# SPLIT per ALIGNMENT (campioni concordanti)
# ============================================
n_alignment = len(alignment_indices)
rng = np.random.RandomState(SEED)
shuffled_alignment_idx = rng.permutation(n_alignment)
split_idx_align = int(0.7 * n_alignment)
alignment_train_local_idx = shuffled_alignment_idx[:split_idx_align]
alignment_val_local_idx = shuffled_alignment_idx[split_idx_align:]

print(f"\nCampioni CONCORDANTI per Alignment/Autoencoder: {n_alignment}")
print(f"  Train: {len(alignment_train_local_idx)}, Val: {len(alignment_val_local_idx)}")

# ============================================
# SPLIT per PROBER (undersampling separato per modello)
# ============================================
rng_a = np.random.RandomState(SEED)
rng_b = np.random.RandomState(SEED + 1)

shuffled_a = rng_a.permutation(len(model_a_balanced_idx))
shuffled_b = rng_b.permutation(len(model_b_balanced_idx))

split_a = int(0.7 * len(model_a_balanced_idx))
split_b = int(0.7 * len(model_b_balanced_idx))

model_a_train_local = shuffled_a[:split_a]
model_a_test_local = shuffled_a[split_a:]
model_b_train_local = shuffled_b[:split_b]
model_b_test_local = shuffled_b[split_b:]

print(f"\nCampioni per PROBER (per modello):")
print(f"  {MODEL_A}: train={len(model_a_train_local)}, test={len(model_a_test_local)}")
print(f"  {MODEL_B}: train={len(model_b_train_local)}, test={len(model_b_test_local)}")
print(f"\nUsing LATENT_DIM={AUTOENCODER_CONFIG['latent_dim']}")

scenarios = [
    {"teacher_model": MODEL_A, "student_model": MODEL_B},
    {"teacher_model": MODEL_B, "student_model": MODEL_A}
]

scenario_results_map = {0: [], 1: []}

# ============================================
# PHASE 2: CARICAMENTO E PREPARAZIONE DATI
# ============================================
print("\n" + "="*80)
print("PHASE 2: CARICAMENTO E PREPARAZIONE DATI PER LAYER TYPE")
print("="*80 + "\n")

for layer_type in ['attn', 'mlp', 'hidden']:
    print(f"\n{'='*40}")
    print(f"PROCESSING LAYER TYPE: {layer_type.upper()}")
    print(f"{'='*40}")
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        # ============================================
        # Carica dati COMPLETI per entrambi i modelli
        # ============================================
        X_model_a_full, _ = load_concatenated_layers(MODEL_A, DATASET_NAME, LAYER_CONFIG[MODEL_A][layer_type], layer_type)
        X_model_b_full, _ = load_concatenated_layers(MODEL_B, DATASET_NAME, LAYER_CONFIG[MODEL_B][layer_type], layer_type)
        
        # ============================================
        # DATI PER ALIGNMENT/AUTOENCODER (concordanti)
        # ============================================
        X_align_a_train = X_model_a_full[alignment_indices][alignment_train_local_idx]
        X_align_b_train = X_model_b_full[alignment_indices][alignment_train_local_idx]
        X_align_a_val = X_model_a_full[alignment_indices][alignment_val_local_idx]
        X_align_b_val = X_model_b_full[alignment_indices][alignment_val_local_idx]
        
        # ============================================
        # DATI PER PROBER MODEL A (undersampling separato)
        # ============================================
        X_a_balanced = X_model_a_full[model_a_balanced_idx]
        X_a_train = X_a_balanced[model_a_train_local]
        X_a_test = X_a_balanced[model_a_test_local]
        y_a_train = model_a_balanced_labels[model_a_train_local]
        y_a_test = model_a_balanced_labels[model_a_test_local]
        
        # ============================================
        # DATI PER PROBER MODEL B (undersampling separato)
        # ============================================
        X_b_balanced = X_model_b_full[model_b_balanced_idx]
        X_b_train = X_b_balanced[model_b_train_local]
        X_b_test = X_b_balanced[model_b_test_local]
        y_b_train = model_b_balanced_labels[model_b_train_local]
        y_b_test = model_b_balanced_labels[model_b_test_local]
        
        del X_model_a_full, X_model_b_full, X_a_balanced, X_b_balanced
        gc.collect()
        
        print(f"   [{layer_type.upper()}] Align: train={X_align_a_train.shape[0]}, val={X_align_a_val.shape[0]}")
        print(f"   [{layer_type.upper()}] {MODEL_A}: train={len(y_a_train)} ({np.bincount(y_a_train)}), test={len(y_a_test)}")
        print(f"   [{layer_type.upper()}] {MODEL_B}: train={len(y_b_train)} ({np.bincount(y_b_train)}), test={len(y_b_test)}")
        
        # ============================================
        # Normalizzazione
        # ============================================
        print("   Normalizing data...")
        scaler_align_a, scaler_align_b = StandardScaler(), StandardScaler()
        scaler_a, scaler_b = StandardScaler(), StandardScaler()
        
        X_align_a_train_norm = scaler_align_a.fit_transform(X_align_a_train).astype(np.float32)
        X_align_b_train_norm = scaler_align_b.fit_transform(X_align_b_train).astype(np.float32)
        X_align_a_val_norm = scaler_align_a.transform(X_align_a_val).astype(np.float32)
        X_align_b_val_norm = scaler_align_b.transform(X_align_b_val).astype(np.float32)
        
        X_a_train_norm = scaler_a.fit_transform(X_a_train).astype(np.float32)
        X_a_test_norm = scaler_a.transform(X_a_test).astype(np.float32)
        
        X_b_train_norm = scaler_b.fit_transform(X_b_train).astype(np.float32)
        X_b_test_norm = scaler_b.transform(X_b_test).astype(np.float32)
        
        # ============================================
        # Prepara struttura dati
        # ============================================
        data_splits = {
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

        # ============================================
        # Esegui esperimenti per entrambi gli scenari
        # ============================================
        for i, scenario in enumerate(scenarios):
            print(f"\n   --- Scenario: {scenario['teacher_model']} (Teacher) <- {scenario['student_model']} (Student) ---")
            
            set_seed(SEED)
            
            if scenario['teacher_model'] == MODEL_A:
                teacher_data = data_splits['model_a']
                student_data = data_splits['model_b']
                align_teacher_train = data_splits['alignment']['X_a_train']
                align_student_train = data_splits['alignment']['X_b_train']
                align_teacher_val = data_splits['alignment']['X_a_val']
                align_student_val = data_splits['alignment']['X_b_val']
                student_scaler = data_splits['alignment']['scaler_b']
            else:
                teacher_data = data_splits['model_b']
                student_data = data_splits['model_a']
                align_teacher_train = data_splits['alignment']['X_b_train']
                align_student_train = data_splits['alignment']['X_a_train']
                align_teacher_val = data_splits['alignment']['X_b_val']
                align_student_val = data_splits['alignment']['X_a_val']
                student_scaler = data_splits['alignment']['scaler_a']
            # Prepara alignment_data per la funzione
            alignment_data = {
                'X_teacher_train': align_teacher_train,
                'X_teacher_val': align_teacher_val,
                'X_student_train': align_student_train,
                'X_student_val': align_student_val
            }
            
            res = run_experiment_pipeline_with_autoencoder(
                {"X_train": teacher_data['X_train'], "X_test": teacher_data['X_test']},
                {"y_train": teacher_data['y_train'], "y_test": teacher_data['y_test']},
                scenario['teacher_model'],
                {"X_train": student_data['X_train'], "X_test": student_data['X_test']},
                {"y_train": student_data['y_train'], "y_test": student_data['y_test']},
                scenario['student_model'],
                alignment_data,
                layer_type, "CONFIG1"
            )
            scenario_results_map[i].append(res)
            
            plot_confusion_matrix(
                np.array(res['teacher']['confusion_matrix']), 
                layer_type, 
                f"Teacher_{scenario['teacher_model'].replace('.', '_').replace('-', '_')}"
            )
            plot_confusion_matrix(
                np.array(res['student_on_teacher']['confusion_matrix']), 
                layer_type, 
                f"{scenario['student_model'].replace('.', '_').replace('-', '_')}_on_{scenario['teacher_model'].replace('.', '_').replace('-', '_')}"
            )

        del data_splits
        gc.collect()
        torch.cuda.empty_cache()
        print(f"   Memory freed for {layer_type}.")

    except Exception as e:
        print(f"Critical error in layer {layer_type}: {e}")
        traceback.print_exc()
        raise

# ============================================
# PHASE 3: SALVATAGGIO RISULTATI
# ============================================
print("\n" + "="*80)
print("PHASE 3: SALVATAGGIO RISULTATI")
print("="*80 + "\n")

# Reconstruct all_results
all_results = []
for i, scenario in enumerate(scenarios):
    all_results.append({
        "scenario": f"{scenario['teacher_model']} (teacher) → {scenario['student_model']} (student)",
        "results": scenario_results_map[i]
    })

# Save JSON
os.makedirs("results_metrics", exist_ok=True)
metrics_file = "results_metrics/approach2_autoencoder_results.json"

all_results_json = []

for scenario_data in all_results:
    scenario_results = []
    
    for r in scenario_data['results']:
        ae_t_config = r['autoencoder_teacher']['config']
        ae_s_config = r['autoencoder_student']['config']
        prober_cfg = r['prober_model']['config']
        align_config = r['alignment_model']['config']
        
        result_entry = {
            "layer_type": r['type'],
            "teacher_model": r['teacher_name'],
            "student_model": r['student_name'],
            "data_info": {
                "alignment_samples_train": int(len(alignment_train_local_idx)),
                "alignment_samples_val": int(len(alignment_val_local_idx)),
                "model_a_train": int(len(model_a_train_local)),
                "model_a_test": int(len(model_a_test_local)),
                "model_b_train": int(len(model_b_train_local)),
                "model_b_test": int(len(model_b_test_local)),
                "concordant_undersampling_for_alignment": True,
                "separate_undersampling_per_model": True
            },
            
            # ==================== TEACHER AUTOENCODER ====================
            "teacher_autoencoder": {
                "architecture": {
                    "input_dim": r['autoencoder_teacher']['input_dim'],
                    "latent_dim": ae_t_config['latent_dim'],
                    "hidden_dim": ae_t_config['hidden_dim'],
                    "dropout": ae_t_config['dropout']
                },
                "training_hyperparameters": {
                    "optimizer": ae_t_config['optimizer'],
                    "learning_rate": ae_t_config['learning_rate'],
                    "weight_decay": ae_t_config['weight_decay'],
                    "batch_size": ae_t_config['batch_size'],
                    "max_epochs": ae_t_config['max_epochs'],
                    "scheduler": ae_t_config['scheduler'],
                    "gradient_clip_max_norm": ae_t_config['gradient_clip_max_norm'],
                    "early_stopping_patience": ae_t_config['early_stopping_patience'],
                    "early_stopping_min_delta": ae_t_config['early_stopping_min_delta'],
                    "loss_function": ae_t_config['loss_function']
                },
                "training_results": {
                    "best_val_loss": round(r['autoencoder_teacher']['best_val_loss'], 6),
                    "epochs_trained": r['autoencoder_teacher']['epochs_trained'],
                    "model_saved_path": r['autoencoder_teacher']['model_path']
                }
            },
            
            # ==================== STUDENT AUTOENCODER ====================
            "student_autoencoder": {
                "architecture": {
                    "input_dim": r['autoencoder_student']['input_dim'],
                    "latent_dim": ae_s_config['latent_dim'],
                    "hidden_dim": ae_s_config['hidden_dim'],
                    "dropout": ae_s_config['dropout']
                },
                "training_hyperparameters": {
                    "optimizer": ae_s_config['optimizer'],
                    "learning_rate": ae_s_config['learning_rate'],
                    "weight_decay": ae_s_config['weight_decay'],
                    "batch_size": ae_s_config['batch_size'],
                    "max_epochs": ae_s_config['max_epochs'],
                    "scheduler": ae_s_config['scheduler'],
                    "gradient_clip_max_norm": ae_s_config['gradient_clip_max_norm'],
                    "early_stopping_patience": ae_s_config['early_stopping_patience'],
                    "early_stopping_min_delta": ae_s_config['early_stopping_min_delta'],
                    "loss_function": ae_s_config['loss_function']
                },
                "training_results": {
                    "best_val_loss": round(r['autoencoder_student']['best_val_loss'], 6),
                    "epochs_trained": r['autoencoder_student']['epochs_trained'],
                    "model_saved_path": r['autoencoder_student']['model_path']
                }
            },
            
            # ==================== PROBER MODEL ====================
            "prober_model": {
                "architecture": {
                    "type": "MLPProber",
                    "input_dim": r['prober_model']['input_dim'],
                    "hidden_dim": prober_cfg['hidden_dim'],
                    "dropout": prober_cfg['dropout']
                },
                "training_hyperparameters": {
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
                    "best_val_acc": round(r['prober_model']['best_val_acc'], 4),
                    "epochs_trained": r['prober_model']['epochs_trained'],
                    "model_saved_path": r['prober_model']['model_path']
                }
            },
            
            # ==================== ALIGNMENT MODEL ====================
            "alignment_model": {
                "architecture": {
                    "type": "AlignmentNetwork",
                    "input_dim": r['alignment_model']['input_dim'],
                    "output_dim": r['alignment_model']['output_dim'],
                    "hidden_dim": align_config['hidden_dim'],
                    "dropout": align_config['dropout']
                },
                "training_hyperparameters": {
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
                "loss_function": {
                    "type": "MixedLoss",
                    "mse_weight": align_config['loss_alpha'],
                    "cosine_weight": align_config['loss_beta']
                },
                "training_results": {
                    "best_val_loss": round(r['alignment_model']['best_val_loss'], 6),
                    "epochs_trained": r['alignment_model']['epochs_trained'],
                    "model_saved_path": r['alignment_model']['model_path']
                }
            },
            
            # ==================== PERFORMANCE METRICS ====================
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
        }
        
        scenario_results.append(result_entry)
    
    all_results_json.append({
        "scenario": scenario_data['scenario'],
        "results": scenario_results
    })

with open(metrics_file, 'w') as f:
    json.dump(all_results_json, f, indent=2)

print(f"✓ Results saved to: {metrics_file}")
print(f"{'='*60}")


