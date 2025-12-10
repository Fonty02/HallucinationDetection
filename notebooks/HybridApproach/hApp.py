# %% [markdown]
# # Hybrid: Non-linear adaptation with AdapterMLP
# 
#  In this notebook a first non-linear approach is tested. We take all the activations of both LLMs, we train 3 classifiers (1 per type of layer) for the Teacher model, with an AdapterMLP we try to adapt the Student latent space to the Teacher one. Finally we test the adapted Student activations with the Teacher classifiers.

# %%
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import gc
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score
import traceback
from sklearn.linear_model import LogisticRegression
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
DEVICE = torch.device("cuda:2")

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

def get_generator(seed=SEED):
    """Create a reproducible generator for DataLoader"""
    g = torch.Generator()
    g.manual_seed(seed)
    return g

# Set seeds at import time
set_seed(SEED)

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

# %%

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.getcwd()))
CACHE_DIR_NAME = "activation_cache"
HF_DEFAULT_HOME = os.environ.get("HF_HOME", "~\\.cache\\huggingface\\hub")

# Nomi dei modelli (usati come costanti in tutto il notebook)
MODEL_A = "gemma-2-9b-it"
MODEL_B = "Llama-3.1-8B-Instruct"

LAYER_CONFIG = {
    MODEL_A: 
    {
        "attn": [21,26,27],
        "mlp":[23,24,28],
        "hidden": [19,24,28]
    },    
    MODEL_B: 
    {
        "attn": [14,15,16],
        "mlp":[13,14,15],
        "hidden": [14,15,16]
    }  
}
DATASET_NAME = "halu_eval"


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


PROBE_CONFIG = {
    "type": "LogisticRegression",
    "max_iter": 1000,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "n_jobs": -1
}

# %% [markdown]
# ### Dataset preparation

# %%
def stats_per_json(model_name, dataset_name):
    """Versione per la vecchia struttura con hallucination_labels.json"""
    file_path = os.path.join(PROJECT_ROOT, CACHE_DIR_NAME, model_name, dataset_name,"generations","hallucination_labels.json")
    with open(file_path, 'r') as file:
        data = json.load(file)
    total = len(data)
    hallucinations = sum(1 for item in data if item['is_hallucination'])
    percent_hallucinations = (hallucinations / total) * 100 if total > 0 else 0
    hallucinated_ids = [item['instance_id'] for item in data if item['is_hallucination']]
    return {
        'total': total,
        'hallucinations': hallucinations,
        'percent_hallucinations': percent_hallucinations,
        'hallucinated_ids': hallucinated_ids,
        'model_name': model_name,
        'dataset_name': dataset_name
    }

def stats_from_new_structure(model_name, dataset_name):
    """Versione per la struttura con cartelle hallucinated/ e not_hallucinated/"""
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
        'not_hallucinations': len(not_hallucinated_ids),
        'percent_hallucinations': percent_hallucinations,
        'hallucinated_ids': hallucinated_ids,
        'not_hallucinated_ids': not_hallucinated_ids,
        'model_name': model_name,
        'dataset_name': dataset_name
    }

def detect_structure_type(model_name, dataset_name):
    """Rileva automaticamente se la struttura è vecchia o nuova."""
    base_path = os.path.join(PROJECT_ROOT, CACHE_DIR_NAME, model_name, dataset_name, "activation_attn")
    hallucinated_path = os.path.join(base_path, "hallucinated")
    if os.path.isdir(hallucinated_path):
        return 'new'
    return 'old'

def get_stats(model_name, dataset_name):
    """Funzione wrapper che rileva automaticamente la struttura e chiama la funzione appropriata."""
    structure = detect_structure_type(model_name, dataset_name)
    if structure == 'new':
        return stats_from_new_structure(model_name, dataset_name)
    else:
        return stats_per_json(model_name, dataset_name)


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

# %%
# ------------------------------------------------------------------
# 1. Dataset class
# ------------------------------------------------------------------
class AlignmentDataset(Dataset):
    def __init__(self, x_source: torch.Tensor, x_target: torch.Tensor):
        # Ora assumiamo che i dati siano già torch.Tensor
        self.x_source = x_source
        self.x_target = x_target
    
    def __len__(self):
        return self.x_source.shape[0]
    
    def __getitem__(self, idx):
        return self.x_source[idx], self.x_target[idx]

# ------------------------------------------------------------------
# 2. AlignmentNetwork
# ------------------------------------------------------------------
class AlignmentNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        
        if input_dim != output_dim:
            self.input_proj = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.input_proj = nn.Identity()

        # Ramo Non-Lineare (Bottleneck Estremo)
        self.net = nn.Sequential(
            nn.Linear(output_dim, hidden_dim), # Compressione forte (es. 10000 -> 128)
            nn.LayerNorm(hidden_dim),          # Normalizzazione
            nn.GELU(),
            nn.Dropout(dropout),               # Dropout aggressivo (0.5)
            nn.Linear(hidden_dim, output_dim), # Decompressione
            nn.Dropout(dropout)                # Dropout finale
        )
        
        # Zero-Init per partire come una funzione lineare pura
        self._init_zero()

    def _init_zero(self):
        nn.init.zeros_(self.net[-2].weight)
        if self.net[-2].bias is not None:
            nn.init.zeros_(self.net[-2].bias)

    def forward(self, x):
        x_base = self.input_proj(x)
        return x_base + self.net(x_base)


class MixedLoss(nn.Module):
    def __init__(self, alpha=0.01, beta=1.0):
        super().__init__()
        self.alpha = alpha  # Peso per MSE
        self.beta = beta    # Peso per Cosine
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        loss_mse = self.mse(pred, target)
        cosine_sim = F.cosine_similarity(pred, target, dim=1).mean()
        loss_cosine = 1 - cosine_sim
        
        # Loss combinata
        return self.alpha * loss_mse + self.beta * loss_cosine

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
    """
    Carica multipli layer e li concatena.
    """
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


def plot_confusion_matrix(cm, layer_type, model_name="", save_dir="confusion_matrices"):
    """
    Plotta e salva la confusion matrix come immagine.
    """
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
    print(f"   ✓ Salvato: {filename}")


def get_undersampled_indices_per_model(model_stats, seed=SEED):
    """Applica undersampling al dataset di un singolo modello."""
    total = model_stats['total']
    hall_set = set(model_stats['hallucinated_ids'])
    
    y = np.array([1 if i in hall_set else 0 for i in range(total)])
    balanced_idx = get_balanced_indices(y, seed)
    balanced_labels = y[balanced_idx]
    
    return balanced_idx, balanced_labels

# %%
print("="*80)
print("FASE 1: PREPARAZIONE DATI")
print("="*80 + "\n")

# ============================================
# STEP 1: Ottieni statistiche per entrambi i modelli
# ============================================
print("Step 1: Caricamento statistiche modelli...")
model_a_stats = get_stats(MODEL_A, DATASET_NAME)
model_b_stats = get_stats(MODEL_B, DATASET_NAME)
print(f"   {MODEL_A}: {model_a_stats['total']} totali, {model_a_stats['hallucinations']} allucinazioni")
print(f"   {MODEL_B}: {model_b_stats['total']} totali, {model_b_stats['hallucinations']} allucinazioni")

# ============================================
# STEP 2: Trova campioni concordanti con undersampling (SOLO per allineamento)
# ============================================
print("\nStep 2: Analisi concordanza e undersampling per ALLINEAMENTO...")
alignment_indices, alignment_labels = get_concordant_indices_and_undersample(model_a_stats, model_b_stats, seed=SEED)

# Split train/val per l'allineamento (70/30) - NOTA: usiamo solo train per allineamento
n_alignment = len(alignment_indices)
rng = np.random.RandomState(SEED)
shuffled_alignment_idx = rng.permutation(n_alignment)
split_idx_align = int(0.7 * n_alignment)
alignment_train_local_idx = shuffled_alignment_idx[:split_idx_align]
alignment_val_local_idx = shuffled_alignment_idx[split_idx_align:]

print(f"   Campioni per allineamento: train={len(alignment_train_local_idx)}, val={len(alignment_val_local_idx)}")

# ============================================
# STEP 3: Prepara dataset completi per ogni LLM (con undersampling separato)
# ============================================
print("\nStep 3: Preparazione dataset completi per ogni LLM...")

# Undersampling separato per ogni modello
model_a_balanced_idx, model_a_balanced_labels = get_undersampled_indices_per_model(model_a_stats, SEED)
model_b_balanced_idx, model_b_balanced_labels = get_undersampled_indices_per_model(model_b_stats, SEED)

print(f"   {MODEL_A} bilanciato: {len(model_a_balanced_idx)} campioni ({np.sum(model_a_balanced_labels==1)} hall, {np.sum(model_a_balanced_labels==0)} non-hall)")
print(f"   {MODEL_B} bilanciato: {len(model_b_balanced_idx)} campioni ({np.sum(model_b_balanced_labels==1)} hall, {np.sum(model_b_balanced_labels==0)} non-hall)")

# Split train/test per ogni modello (70/30)
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

print(f"\n   Split {MODEL_A}: train={len(model_a_train_local)}, test={len(model_a_test_local)}")
print(f"   Split {MODEL_B}: train={len(model_b_train_local)}, test={len(model_b_test_local)}")

# ============================================
# FUNZIONE: Train Alignment Network
# ============================================
def train_alignment_network(X_source_train, X_target_train, X_source_val, X_target_val, config, layer_type, teacher_name, student_name):
    """
    Addestra l'AlignmentNetwork per mappare lo spazio student -> teacher.
    """
    set_seed(SEED)
    
    input_dim = X_source_train.shape[1]
    output_dim = X_target_train.shape[1]
    
    # Converti in tensori
    X_src_train_t = torch.tensor(X_source_train, dtype=torch.float32)
    X_tgt_train_t = torch.tensor(X_target_train, dtype=torch.float32)
    X_src_val_t = torch.tensor(X_source_val, dtype=torch.float32)
    X_tgt_val_t = torch.tensor(X_target_val, dtype=torch.float32)
    
    # Dataset e DataLoader
    train_dataset = AlignmentDataset(X_src_train_t, X_tgt_train_t)
    val_dataset = AlignmentDataset(X_src_val_t, X_tgt_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, generator=get_generator(SEED))
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Modello
    model = AlignmentNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    ).to(DEVICE)
    
    # Loss e Optimizer
    criterion = MixedLoss(alpha=config['loss_alpha'], beta=config['loss_beta'])
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['max_epochs'])
    
    # Training loop con early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(config['max_epochs']):
        # Training
        model.train()
        train_loss = 0.0
        for x_src, x_tgt in train_loader:
            x_src, x_tgt = x_src.to(DEVICE), x_tgt.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x_src)
            loss = criterion(pred, x_tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_max_norm'])
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_src, x_tgt in val_loader:
                x_src, x_tgt = x_src.to(DEVICE), x_tgt.to(DEVICE)
                pred = model(x_src)
                val_loss += criterion(pred, x_tgt).item()
        
        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss - config['early_stopping_min_delta']:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                break
    
    # Ripristina il miglior modello
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Salva il modello
    os.makedirs("alignment_models", exist_ok=True)
    model_path = f"alignment_models/alignment_{layer_type}_{student_name}_to_{teacher_name}.pt"
    torch.save(model.state_dict(), model_path)
    
    return model, {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1,
        'model_path': model_path,
        'config': config
    }


def run_hybrid_experiment_pipeline(data, teacher_name, student_name, layer_type, config):
    """
    Pipeline Hybrid:
    - Prober: addestrato sul dataset bilanciato del teacher
    - Allineamento: AlignmentNetwork addestrata sui dati concordanti
    - Test cross-model: dati student → proiettati via NN → valutati con prober teacher
    """
    print(f"\n=== {layer_type.upper()} LAYERS ({teacher_name} → {student_name}) ===")
    
    if teacher_name == MODEL_A:
        teacher_data, student_data = data['model_a'], data['model_b']
        align_teacher_train = data['alignment']['X_a_train']
        align_student_train = data['alignment']['X_b_train']
        align_teacher_val = data['alignment']['X_a_val']
        align_student_val = data['alignment']['X_b_val']
        student_scaler = data['alignment']['scaler_b']
    else:
        teacher_data, student_data = data['model_b'], data['model_a']
        align_teacher_train = data['alignment']['X_b_train']
        align_student_train = data['alignment']['X_a_train']
        align_teacher_val = data['alignment']['X_b_val']
        align_student_val = data['alignment']['X_a_val']
        student_scaler = data['alignment']['scaler_a']
    
    # STEP 1: Teacher Probing
    print("  Training Teacher Prober...")
    probe = LogisticRegression(
        max_iter=PROBE_CONFIG['max_iter'], 
        class_weight=PROBE_CONFIG['class_weight'], 
        solver=PROBE_CONFIG['solver'], 
        n_jobs=PROBE_CONFIG['n_jobs']
    )
    probe.fit(teacher_data['X_train'], teacher_data['y_train'])
    
    y_pred_t = probe.predict(teacher_data['X_test'])
    y_proba_t = probe.predict_proba(teacher_data['X_test'])[:, 1]
    
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

    # STEP 2: Alignment Network Training
    print("  Training Alignment Network...")
    alignment_model, alignment_info = train_alignment_network(
        align_student_train, align_teacher_train,
        align_student_val, align_teacher_val,
        config, layer_type, teacher_name, student_name
    )
    print(f"  Alignment: val_loss={alignment_info['best_val_loss']:.6f}, epochs={alignment_info['epochs_trained']}")
    
    # STEP 3: Cross-Model Test
    print("  Testing Cross-Model...")
    alignment_model.eval()
    X_student_scaled = student_scaler.transform(student_data['X_test_raw'])
    X_student_tensor = torch.tensor(X_student_scaled, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        X_student_projected = alignment_model(X_student_tensor).cpu().numpy()
    
    y_pred_c = probe.predict(X_student_projected)
    y_proba_c = probe.predict_proba(X_student_projected)[:, 1]
    
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
        "probe_config": PROBE_CONFIG
    }


# ============================================
# STEP 4: Carica e prepara i dati per ogni layer type
# ============================================
print("\n" + "="*80)
print("FASE 2: CARICAMENTO E PREPARAZIONE DATI PER LAYER TYPE")
print("="*80 + "\n")

data_splits = {}
for layer_type in ['attn', 'mlp', 'hidden']:
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"--- Processing {layer_type.upper()} ---")
    
    # Carica i dati COMPLETI per entrambi i modelli
    X_model_a_full, _ = load_concatenated_layers(MODEL_A, DATASET_NAME, LAYER_CONFIG[MODEL_A][layer_type], layer_type)
    X_model_b_full, _ = load_concatenated_layers(MODEL_B, DATASET_NAME, LAYER_CONFIG[MODEL_B][layer_type], layer_type)
    
    # === DATI PER ALLINEAMENTO (concordanti + undersampling) ===
    X_align_a_train = X_model_a_full[alignment_indices][alignment_train_local_idx]
    X_align_b_train = X_model_b_full[alignment_indices][alignment_train_local_idx]
    X_align_a_val = X_model_a_full[alignment_indices][alignment_val_local_idx]
    X_align_b_val = X_model_b_full[alignment_indices][alignment_val_local_idx]
    
    # === DATI PER PROBER MODEL A ===
    X_a_balanced = X_model_a_full[model_a_balanced_idx]
    X_a_train = X_a_balanced[model_a_train_local]
    X_a_test = X_a_balanced[model_a_test_local]
    y_a_train = model_a_balanced_labels[model_a_train_local]
    y_a_test = model_a_balanced_labels[model_a_test_local]
    
    # === DATI PER PROBER MODEL B ===
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
    
    # Normalizzazione
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

gc.collect()
torch.cuda.empty_cache()

# ============================================
# FASE 3: ESECUZIONE ESPERIMENTI
# ============================================
print("\n" + "="*80)
print("FASE 3: ESECUZIONE ESPERIMENTI")
print("="*80 + "\n")

scenarios = [
    {"teacher": MODEL_A, "student": MODEL_B},
    {"teacher": MODEL_B, "student": MODEL_A}
]

all_results = []
for scenario in scenarios:
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario['teacher']} → {scenario['student']}")
    print(f"{'='*60}")
    
    results = []
    for layer_type in ['attn', 'mlp', 'hidden']:
        try:
            res = run_hybrid_experiment_pipeline(
                data_splits[layer_type], 
                scenario['teacher'], 
                scenario['student'], 
                layer_type,
                ALIGNMENT_CONFIG
            )
            results.append(res)
            
            plot_confusion_matrix(np.array(res['teacher']['confusion_matrix']), layer_type, f"Teacher_{scenario['teacher']}")
            plot_confusion_matrix(np.array(res['student_on_teacher']['confusion_matrix']), layer_type, f"{scenario['student']}_on_{scenario['teacher']}")
        except Exception as e:
            print(f"Errore in {layer_type}: {e}")
            traceback.print_exc()
    
    all_results.append({"scenario": f"{scenario['teacher']} → {scenario['student']}", "results": results})

# ============================================
# FASE 4: SALVATAGGIO RISULTATI
# ============================================
print("\n" + "="*80)
print("FASE 4: SALVATAGGIO RISULTATI")
print("="*80 + "\n")

os.makedirs("results_metrics", exist_ok=True)
metrics_file = "results_metrics/hybrid_adapter_logreg_results.json"

all_results_json = []
for scenario_data in all_results:
    scenario_results = []
    for r in scenario_data['results']:
        config = r['alignment_model']['config']
        probe_cfg = r['probe_config']
        
        scenario_results.append({
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
            "alignment_model_info": {
                "architecture": "AlignmentNetwork",
                "input_dim": r['alignment_model']['input_dim'],
                "output_dim": r['alignment_model']['output_dim'],
                "hidden_dim": config['hidden_dim'],
                "dropout": config['dropout'],
                "activation": "GELU",
                "normalization": "LayerNorm",
                "residual_connection": True,
                "initialization": "zero_init"
            },
            "training_hyperparameters": {
                "optimizer": config['optimizer'],
                "learning_rate": config['learning_rate'],
                "weight_decay": config['weight_decay'],
                "batch_size": config['batch_size'],
                "max_epochs": config['max_epochs'],
                "scheduler": config['scheduler'],
                "gradient_clip_max_norm": config['gradient_clip_max_norm'],
                "early_stopping_patience": config['early_stopping_patience'],
                "early_stopping_min_delta": config['early_stopping_min_delta']
            },
            "loss_function": {
                "type": "MixedLoss",
                "mse_weight": config['loss_alpha'],
                "cosine_weight": config['loss_beta']
            },
            "training_results": {
                "alignment_network": {
                    "best_val_loss": round(r['alignment_model']['best_val_loss'], 6),
                    "epochs_trained": r['alignment_model']['epochs_trained'],
                    "model_saved_path": r['alignment_model']['model_path']
                }
            },
            "teacher_probe": {
                "type": probe_cfg['type'],
                "max_iter": probe_cfg['max_iter'],
                "class_weight": probe_cfg['class_weight'],
                "solver": probe_cfg['solver']
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
        "scenario": scenario_data['scenario'],
        "results": scenario_results
    })

with open(metrics_file, 'w') as f:
    json.dump(all_results_json, f, indent=2)

print(f"✓ Risultati salvati in: {metrics_file}")
print(f"{'='*60}")


