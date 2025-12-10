# %% [markdown]
# # This notebook contains a preliminary analysis of the Universal Prober for LLM

# %% [markdown]
# ### Libraries import and defintion of constants

# %%
import json
import os
from sklearn.decomposition import PCA
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import gc
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score
import traceback
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

# %% [markdown]
# ## Dataset stats

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
# Ottieni statistiche per entrambi i modelli
model_a_stats = get_stats(MODEL_A, DATASET_NAME)
model_b_stats = get_stats(MODEL_B, DATASET_NAME)
print(f"{MODEL_A} Hallucination Stats:", model_a_stats)
print(f"{MODEL_B} Hallucination Stats:", model_b_stats)

common_hallucinated = set(model_a_stats['hallucinated_ids']).intersection(set(model_b_stats['hallucinated_ids']))
print(f"Number of common hallucinated instances between {MODEL_A} and {MODEL_B}:", len(common_hallucinated))

# %% [markdown]
# ## Model and activations stats

# %%
def layers_in_model(model, dataset=None):
    """
    Conta il numero di layer nel modello.
    Supporta sia la vecchia che la nuova struttura.
    """
    file_path = os.path.join(PROJECT_ROOT, CACHE_DIR_NAME, model)
    
    # Se non viene specificato il dataset, prendi il primo disponibile
    if dataset is None:
        subdirs = [d for d in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, d))]
        if not subdirs:
            raise ValueError(f"No subdirectories found in {file_path}")
        dataset = subdirs[0]
    
    layer_dir = os.path.join(file_path, dataset, "activation_attn")
    
    # Controlla se è la nuova struttura (con cartelle hallucinated/not_hallucinated)
    hallucinated_path = os.path.join(layer_dir, "hallucinated")
    if os.path.isdir(hallucinated_path):
        # Nuova struttura: conta i file layer*_activations.pt nella cartella hallucinated
        layer_files = [f for f in os.listdir(hallucinated_path) if f.endswith('_activations.pt')]
        return len(layer_files)
    else:
        # Vecchia struttura: conta i file layer*_activations.pt direttamente
        layer_files = [f for f in os.listdir(layer_dir) if f.endswith('_activations.pt')]
        return len(layer_files)

model_a_layers = layers_in_model(MODEL_A, DATASET_NAME)
model_b_layers = layers_in_model(MODEL_B, DATASET_NAME)
print(f"Number of layers in {MODEL_A}:", model_a_layers)
print(f"Number of layers in {MODEL_B}:", model_b_layers)

# %%
def load_activations_for_layer(model, dataset, layer, layer_type):
    """
    Carica le attivazioni per un singolo layer.
    Supporta sia la vecchia che la nuova struttura dati.
    
    Returns:
        activations: numpy array (n_samples, hidden_dim) ordinato per instance_id
        hallucinated_indices: set degli indici (nella posizione ordinata) che sono allucinazioni
    """
    structure = detect_structure_type(model, dataset)
    base_path = os.path.join(PROJECT_ROOT, CACHE_DIR_NAME, model, dataset, f"activation_{layer_type}")
    
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
            hall_activations = hall_activations.cpu().numpy()
        if isinstance(not_hall_activations, torch.Tensor):
            not_hall_activations = not_hall_activations.cpu().numpy()
        
        activations_concat = np.vstack([hall_activations, not_hall_activations])
        ids_concat = np.array(hall_ids + not_hall_ids)
        labels_concat = np.concatenate([
            np.ones(hall_activations.shape[0], dtype=int),
            np.zeros(not_hall_activations.shape[0], dtype=int)
        ])
        
        sort_indices = np.argsort(ids_concat)
        activations = activations_concat[sort_indices]
        labels = labels_concat[sort_indices]
        
        hallucinated_indices = set(np.where(labels == 1)[0])
        return activations, hallucinated_indices
    
    else:
        file_path = os.path.join(base_path, f"layer{layer}_activations.pt")
        activations = torch.load(file_path, map_location=DEVICE)
        if isinstance(activations, torch.Tensor):
            activations = activations.cpu().numpy()
        
        stats = get_stats(model, dataset)
        hallucinated_indices = set(stats['hallucinated_ids'])
        return activations, hallucinated_indices


# %% [markdown]
# ## Classifier

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


def run_experiment_pipeline(data, teacher_name, student_name, layer_type):
    """
    Pipeline:
    - Prober: addestrato sul dataset bilanciato del teacher
    - Allineamento: addestrato sui dati concordanti
    - Test cross-model: dati student → proiettati → valutati con prober teacher
    """
    print(f"\n=== {layer_type.upper()} LAYERS ({teacher_name} → {student_name}) ===")
    
    if teacher_name == MODEL_A:
        teacher_data, student_data = data['model_a'], data['model_b']
        align_teacher, align_student = data['alignment']['X_a_train'], data['alignment']['X_b_train']
        student_scaler = data['alignment']['scaler_b']
    else:
        teacher_data, student_data = data['model_b'], data['model_a']
        align_teacher, align_student = data['alignment']['X_b_train'], data['alignment']['X_a_train']
        student_scaler = data['alignment']['scaler_a']
    
    # STEP 1: Teacher Probing
    probe = LogisticRegression(max_iter=10000, class_weight='balanced', solver='lbfgs', n_jobs=-1)
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

    # STEP 2: Alignment
    aligner = Ridge(alpha=1000.0, fit_intercept=False)
    aligner.fit(align_student, align_teacher)
    
    # STEP 3: Cross-Model Test
    X_student_projected = aligner.predict(student_scaler.transform(student_data['X_test_raw']))
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
        "student_on_teacher": metrics_cross
    }


# %%
# ==============================
# FASE 1: PREPARAZIONE DATI
# ==============================

print("="*80)
print("FASE 1: PREPARAZIONE DATI")
print("="*80 + "\n")

# ============================================
# STEP 1: Trova campioni concordanti con undersampling (SOLO per allineamento)
# ============================================
print("Step 1: Analisi concordanza e undersampling per ALLINEAMENTO...")
alignment_indices, alignment_labels = get_concordant_indices_and_undersample(model_a_stats, model_b_stats, seed=SEED)

# Split train/test per l'allineamento (70/30) - NOTA: usiamo solo train per allineamento
n_alignment = len(alignment_indices)
rng = np.random.RandomState(SEED)
shuffled_alignment_idx = rng.permutation(n_alignment)
split_idx_align = int(0.7 * n_alignment)
alignment_train_local_idx = shuffled_alignment_idx[:split_idx_align]

print(f"   Campioni per allineamento (train): {len(alignment_train_local_idx)}")

# ============================================
# STEP 2: Prepara dataset completi per ogni LLM (con undersampling separato)
# ============================================
print("\nStep 2: Preparazione dataset completi per ogni LLM...")



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
# STEP 3: Carica e prepara i dati per ogni layer type
# ============================================
print("\nStep 3: Caricamento e preparazione dati per ogni layer type...")

data_splits = {}
for layer_type in ['attn', 'mlp', 'hidden']:
    gc.collect()
    
    # Carica i dati COMPLETI per entrambi i modelli
    X_model_a_full, _ = load_concatenated_layers(MODEL_A, DATASET_NAME, LAYER_CONFIG[MODEL_A][layer_type], layer_type)
    X_model_b_full, _ = load_concatenated_layers(MODEL_B, DATASET_NAME, LAYER_CONFIG[MODEL_B][layer_type], layer_type)
    
    # === DATI PER ALLINEAMENTO (concordanti + undersampling) ===
    X_align_a_train = X_model_a_full[alignment_indices][alignment_train_local_idx]
    X_align_b_train = X_model_b_full[alignment_indices][alignment_train_local_idx]
    
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
    
    print(f"   [{layer_type.upper()}] Align: {X_align_a_train.shape[0]} | {MODEL_A}: {np.bincount(y_a_train)} | {MODEL_B}: {np.bincount(y_b_train)}")
    
    # Normalizzazione
    scaler_align_a, scaler_align_b = StandardScaler(), StandardScaler()
    scaler_a, scaler_b = StandardScaler(), StandardScaler()
    
    X_align_a_train_norm = scaler_align_a.fit_transform(X_align_a_train)
    X_align_b_train_norm = scaler_align_b.fit_transform(X_align_b_train)
    
    X_a_train_norm = scaler_a.fit_transform(X_a_train)
    X_a_test_norm = scaler_a.transform(X_a_test)
    
    X_b_train_norm = scaler_b.fit_transform(X_b_train)
    X_b_test_norm = scaler_b.transform(X_b_test)
    
    data_splits[layer_type] = {
        "alignment": {
            "X_a_train": X_align_a_train_norm,
            "X_b_train": X_align_b_train_norm,
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

# ============================================
# FASE 2: ESECUZIONE ESPERIMENTI
# ============================================
print("\n" + "="*80)
print("FASE 2: ESECUZIONE ESPERIMENTI")
print("="*80 + "\n")


# Esegui esperimenti
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
            res = run_experiment_pipeline(data_splits[layer_type], scenario['teacher'], scenario['student'], layer_type)
            results.append(res)
            
            plot_confusion_matrix(np.array(res['teacher']['confusion_matrix']), layer_type, f"Teacher_{scenario['teacher']}")
            plot_confusion_matrix(np.array(res['student_on_teacher']['confusion_matrix']), layer_type, f"{scenario['student']}_on_{scenario['teacher']}")
        except Exception as e:
            print(f"Errore in {layer_type}: {e}")
            traceback.print_exc()
    
    all_results.append({"scenario": f"{scenario['teacher']} → {scenario['student']}", "results": results})

# Salva risultati
os.makedirs("results_metrics", exist_ok=True)
with open("results_metrics/linear_probe_results.json", 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n✓ Risultati salvati in: results_metrics/linear_probe_results.json")

# %%
import os
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from matplotlib.patches import Patch

def createSubplots_Clean(model, dataset, num_layers, type, model_stats, dim_type, directory_to_save):
    """
    Versione modificata: Senza legenda, titoli più grandi, font più leggibili.
    Compatibile con la nuova struttura dati.
    """
    cols = 4
    rows = (num_layers + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, max(5, rows * 5)))
    axs = axs.flatten()

    for layer in range(num_layers):
        try:
            activations, hallucinated_indices = load_activations_for_layer(model, dataset, layer, type)
        except FileNotFoundError as e:
            print(f"File non trovato per layer {layer}: {e}")
            continue

        activations_2d = None
        var_text = ""

        if dim_type == "PCA":
            pca = PCA(n_components=2)
            activations_2d = pca.fit_transform(activations)
            var_text = f'\nVar: {pca.explained_variance_ratio_[0]:.1%} - {pca.explained_variance_ratio_[1]:.1%}'

        if activations_2d is not None:
            colors = ['red' if i in hallucinated_indices else 'blue' for i in range(activations_2d.shape[0])]
            axs[layer].scatter(activations_2d[:, 0], activations_2d[:, 1], c=colors, alpha=0.6, s=15)
            axs[layer].set_xlabel(f'{dim_type} 1', fontsize=16)
            axs[layer].set_ylabel(f'{dim_type} 2', fontsize=16)
            axs[layer].tick_params(axis='both', which='major', labelsize=14)
            axs[layer].grid(True, alpha=0.3)
            axs[layer].set_aspect('equal', adjustable='datalim')
    
    for i in range(num_layers, len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    os.makedirs(directory_to_save, exist_ok=True)
    filename = os.path.join(directory_to_save, f'{model}_{dataset}_{type}_activations_{dim_type}_CLEAN.pdf')
    plt.savefig(filename, dpi=150, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"Salvato plot GRIGLIA pulito: {filename}")


def createSinglePlot(model, dataset, layer_idx, type, model_stats, dim_type, directory_to_save):
    """
    Crea e salva un singolo grafico per una specifica combinazione Modello/Layer/Tipo.
    Compatibile con la nuova struttura dati.
    """
    try:
        activations, hallucinated_indices = load_activations_for_layer(model, dataset, layer_idx, type)
    except FileNotFoundError as e:
        print(f"Errore: File non trovato per Layer {layer_idx} -> {e}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    activations_2d = None
    title_suffix = ""

    if dim_type == "PCA":
        pca = PCA(n_components=2)
        activations_2d = pca.fit_transform(activations)
        var_1 = pca.explained_variance_ratio_[0]
        var_2 = pca.explained_variance_ratio_[1]
        title_suffix = f" (Explained Var: {var_1:.1%}, {var_2:.1%})"

    if activations_2d is not None:
        colors = ['red' if i in hallucinated_indices else 'blue' for i in range(activations_2d.shape[0])]
        scatter = ax.scatter(activations_2d[:, 0], activations_2d[:, 1], c=colors, alpha=0.5, s=25)
        ax.set_xlabel(f'{dim_type} 1', fontsize=16)
        ax.set_ylabel(f'{dim_type} 2', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal', adjustable='datalim')

        n_hall = len(hallucinated_indices)
        n_total = activations_2d.shape[0]
        legend_elements = [
            Patch(facecolor='blue', label=f'True ({n_total - n_hall})'),
            Patch(facecolor='red', label=f'Hallucination ({n_hall})')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=14, title="Labels", title_fontsize=16)

    os.makedirs(directory_to_save, exist_ok=True)
    filename = os.path.join(directory_to_save, f'SINGLE_{model}_{type}_L{layer_idx}_{dim_type}.pdf')
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"Salvato plot SINGOLO: {filename}")

# --- ESEMPIO DI UTILIZZO ---

# 1. Rigenera le griglie più leggibili
for dim_type in ["PCA"]:
    directory = f"activation_plots_{dim_type}_v2"
    os.makedirs(directory, exist_ok=True)
    
    # Griglie per entrambi i modelli
    for layer_type in ["attn", "mlp", "hidden"]:
        createSubplots_Clean(MODEL_A, DATASET_NAME, model_a_layers, layer_type, model_a_stats, dim_type, directory)
        createSubplots_Clean(MODEL_B, DATASET_NAME, model_b_layers, layer_type, model_b_stats, dim_type, directory)

# 2. Genera grafici singoli specifici (Esempio: layer specifici dalla config)
single_plot_dir = "activation_plots_single"
# Usa il primo layer dalla config per attn
#createSinglePlot(MODEL_A, DATASET_NAME, LAYER_CONFIG[MODEL_A]["attn"][0], "attn", model_a_stats, "PCA", single_plot_dir)
#createSinglePlot(MODEL_B, DATASET_NAME, LAYER_CONFIG[MODEL_B]["attn"][0], "attn", model_b_stats, "PCA", single_plot_dir)



