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
import gc
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
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
        "attn": [22,23,24],
        "mlp":[21,23,24],
        "hidden": [21,22,23]
    },    
    MODEL_B: 
    {
        "attn": [8,13,14],
        "mlp":[14,5,16],
        "hidden": [16,17,19]
    }  
}
DATASET_NAME = "belief_bank_facts"

# %% [markdown]
# ## Dataset stats

# %%
def stats_per_json(model_name, dataset_name):
    """
    Versione originale per la vecchia struttura con hallucination_labels.json
    """
    file_path = os.path.join(PROJECT_ROOT, CACHE_DIR_NAME, model_name, dataset_name,"generations","hallucination_labels.json")
    with open(file_path, 'r') as file:
        data = json.load(file)
    total = len(data)
    hallucinations = sum(1 for item in data if item['is_hallucination'])
    percent_hallucinations = (hallucinations / total) * 100 if total > 0 else 0
    allucinated_items = [item['instance_id'] for item in data if item['is_hallucination']]
    return {
        'total': total,
        'hallucinations': hallucinations,
        'percent_hallucinations': percent_hallucinations,
        'hallucinated_items': allucinated_items,
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
    
    # Carica gli instance_ids da un layer (layer0) per contare i campioni
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
    """
    Rileva automaticamente se la struttura è vecchia o nuova.
    Ritorna 'new' se esistono le cartelle hallucinated/not_hallucinated,
    altrimenti 'old'.
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

# %%
# Ottieni statistiche per entrambi i modelli usando le costanti
model_a_stats = get_stats(MODEL_A, DATASET_NAME)
model_b_stats = get_stats(MODEL_B, DATASET_NAME)
print(f"{MODEL_A} Hallucination Stats:", model_a_stats)
print(f"{MODEL_B} Hallucination Stats:", model_b_stats)

# Ottieni gli hallucinated_items in base alla struttura
if 'hallucinated_ids' in model_a_stats:
    model_a_hall_items = model_a_stats['hallucinated_ids']
else:
    model_a_hall_items = model_a_stats['hallucinated_items']
    
if 'hallucinated_ids' in model_b_stats:
    model_b_hall_items = model_b_stats['hallucinated_ids']
else:
    model_b_hall_items = model_b_stats['hallucinated_items']

common_hallucinated = set(model_a_hall_items).intersection(set(model_b_hall_items))
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
    Ordina per instance_id per garantire consistenza.
    
    Returns:
        activations: numpy array (n_samples, hidden_dim) ordinato per instance_id
        hallucinated_indices: set degli indici (nella posizione ordinata) che sono allucinazioni
    """
    structure = detect_structure_type(model, dataset)
    base_path = os.path.join(PROJECT_ROOT, CACHE_DIR_NAME, model, dataset, f"activation_{layer_type}")
    
    if structure == 'new':
        # Nuova struttura: carica da hallucinated/ e not_hallucinated/
        hall_act_path = os.path.join(base_path, "hallucinated", f"layer{layer}_activations.pt")
        hall_ids_path = os.path.join(base_path, "hallucinated", f"layer{layer}_instance_ids.json")
        not_hall_act_path = os.path.join(base_path, "not_hallucinated", f"layer{layer}_activations.pt")
        not_hall_ids_path = os.path.join(base_path, "not_hallucinated", f"layer{layer}_instance_ids.json")
        
        # Carica attivazioni
        hall_activations = torch.load(hall_act_path, map_location=DEVICE)
        not_hall_activations = torch.load(not_hall_act_path, map_location=DEVICE)
        
        # Carica instance_ids
        with open(hall_ids_path, 'r') as f:
            hall_ids = json.load(f)
        with open(not_hall_ids_path, 'r') as f:
            not_hall_ids = json.load(f)
        
        # Converti in numpy
        if isinstance(hall_activations, torch.Tensor):
            hall_activations = hall_activations.cpu().numpy()
        if isinstance(not_hall_activations, torch.Tensor):
            not_hall_activations = not_hall_activations.cpu().numpy()
        
        # Concatena attivazioni e ids
        activations_concat = np.vstack([hall_activations, not_hall_activations])
        ids_concat = np.array(hall_ids + not_hall_ids)
        labels_concat = np.concatenate([
            np.ones(hall_activations.shape[0], dtype=int),
            np.zeros(not_hall_activations.shape[0], dtype=int)
        ])
        
        # Ordina tutto in base agli instance_ids
        sort_indices = np.argsort(ids_concat)
        activations = activations_concat[sort_indices]
        labels = labels_concat[sort_indices]
        
        # Crea set degli indici allucinati (posizione nell'array ordinato)
        hallucinated_indices = set(np.where(labels == 1)[0])
        
        return activations, hallucinated_indices
    
    else:
        # Vecchia struttura
        file_path = os.path.join(base_path, f"layer{layer}_activations.pt")
        activations = torch.load(file_path, map_location=DEVICE)
        if isinstance(activations, torch.Tensor):
            activations = activations.cpu().numpy()
        
        # Usa model_stats per ottenere gli indici allucinati
        stats = get_stats(model, dataset)
        if 'hallucinated_ids' in stats:
            hallucinated_indices = set(stats['hallucinated_ids'])
        else:
            hallucinated_indices = set(stats['hallucinated_items'])
        
        return activations, hallucinated_indices


def createSubplots(model, dataset, num_layers, type, model_stats, dim_type, directory_to_save):
    # Calculate grid dimensions
    cols = 4
    rows = (num_layers + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(32, 8*rows))
    axs = axs.flatten()  # Flatten to handle single row/col cases

    for layer in range(num_layers):
        try:
            activations, hallucinated_indices = load_activations_for_layer(model, dataset, layer, type)
        except FileNotFoundError as e:
            print(f"File non trovato per layer {layer}: {e}")
            continue

        activations_2d = None
        var_text = ""

        # --- IMPLEMENTAZIONE DIMENSION REDUCTION ---
        if dim_type == "PCA":
            pca = PCA(n_components=2)
            activations_2d = pca.fit_transform(activations)
            var_text = f'(Var: {pca.explained_variance_ratio_[0]:.2%}, {pca.explained_variance_ratio_[1]:.2%})'

        # --- PLOTTING ---
        if activations_2d is not None:
            colors = ['red' if i in hallucinated_indices else 'blue' for i in range(activations_2d.shape[0])]
            
            # Scatter plot
            axs[layer].scatter(activations_2d[:, 0], activations_2d[:, 1], c=colors, alpha=0.5, s=10)
            
            axs[layer].set_title(f'Layer {layer} {var_text}', fontsize=12, fontweight='bold')
            axs[layer].set_xlabel(f'{dim_type} 1', fontsize=10)
            axs[layer].set_ylabel(f'{dim_type} 2', fontsize=10)
            axs[layer].grid(True, alpha=0.3)
    
    # Leave unused subplots empty
    for i in range(num_layers, len(axs)):
        axs[i].axis('off')
    
    # Add legend to figure (top right corner of the entire figure)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Hallucinated'),
                       Patch(facecolor='blue', label='Non-hallucinated')]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=12, bbox_to_anchor=(0.98, 0.98))
    
    fig.suptitle(f'Activations {dim_type} for {model} - {type} layers\n(Red: Hallucinated, Blue: Non-hallucinated)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(directory_to_save, f'{model}_{dataset}_{type}_activations_{dim_type}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Salvato plot per {model} - {type} - {dim_type}")

# %%
for dim_type in ["PCA"]:
    directory_to_save = f"activation_plots_{dim_type}"
    os.makedirs(directory_to_save, exist_ok=True)
    for type in ['attn', 'mlp', "hidden"]:
        createSubplots(MODEL_A, DATASET_NAME, model_a_layers, type, model_a_stats, dim_type, directory_to_save)
        createSubplots(MODEL_B, DATASET_NAME, model_b_layers, type, model_b_stats, dim_type, directory_to_save)

# %% [markdown]
# ## Classifier

# %%
def load_activations_and_labels(model_name, dataset_name, layer, layer_type):
    """
    Carica le attivazioni e le label per un dato layer e tipo.
    Supporta sia la vecchia che la nuova struttura dati.
    
    IMPORTANTE: Per la nuova struttura, le attivazioni vengono ordinate
    in base agli instance_ids per garantire la corretta corrispondenza
    tra campioni di diversi layer/tipi.
    
    Returns:
        X: numpy array delle attivazioni (n_samples, hidden_dim) - ordinate per instance_id
        y: numpy array delle label (n_samples,) - 1=hallucination, 0=correct
        instance_ids: numpy array degli instance_ids (n_samples,) - ordinati
    """
    structure = detect_structure_type(model_name, dataset_name)
    base_path = os.path.join(PROJECT_ROOT, CACHE_DIR_NAME, model_name, dataset_name, f"activation_{layer_type}")
    
    if structure == 'new':
        # Nuova struttura: carica da hallucinated/ e not_hallucinated/
        hall_act_path = os.path.join(base_path, "hallucinated", f"layer{layer}_activations.pt")
        hall_ids_path = os.path.join(base_path, "hallucinated", f"layer{layer}_instance_ids.json")
        not_hall_act_path = os.path.join(base_path, "not_hallucinated", f"layer{layer}_activations.pt")
        not_hall_ids_path = os.path.join(base_path, "not_hallucinated", f"layer{layer}_instance_ids.json")
        
        # Carica attivazioni
        hall_activations = torch.load(hall_act_path, map_location=DEVICE)
        not_hall_activations = torch.load(not_hall_act_path, map_location=DEVICE)
        
        # Carica instance_ids
        with open(hall_ids_path, 'r') as f:
            hall_ids = json.load(f)
        with open(not_hall_ids_path, 'r') as f:
            not_hall_ids = json.load(f)
        
        # Converti in numpy
        if isinstance(hall_activations, torch.Tensor):
            hall_activations = hall_activations.cpu().numpy().astype(np.float32)
        if isinstance(not_hall_activations, torch.Tensor):
            not_hall_activations = not_hall_activations.cpu().numpy().astype(np.float32)
        
        # Concatena attivazioni, label e ids
        X_concat = np.vstack([hall_activations, not_hall_activations])
        y_concat = np.concatenate([
            np.ones(hall_activations.shape[0], dtype=int),
            np.zeros(not_hall_activations.shape[0], dtype=int)
        ])
        ids_concat = np.array(hall_ids + not_hall_ids)
        
        # Ordina tutto in base agli instance_ids
        sort_indices = np.argsort(ids_concat)
        X = X_concat[sort_indices]
        y = y_concat[sort_indices]
        instance_ids = ids_concat[sort_indices]
        
        return X, y, instance_ids
    
    else:
        # Vecchia struttura: carica tutto insieme e usa hallucination_labels.json
        file_path = os.path.join(base_path, f"layer{layer}_activations.pt")
        activations = torch.load(file_path, map_location=DEVICE)
        
        if isinstance(activations, torch.Tensor):
            X = activations.cpu().numpy().astype(np.float32)
        else:
            X = activations.astype(np.float32)
        
        # Carica le label dal JSON
        labels_path = os.path.join(PROJECT_ROOT, CACHE_DIR_NAME, model_name, dataset_name, 
                                   "generations", "hallucination_labels.json")
        with open(labels_path, 'r') as f:
            labels_data = json.load(f)
        
        y = np.array([item['is_hallucination'] for item in labels_data], dtype=int)
        instance_ids = np.arange(len(y))  # IDs sequenziali per la vecchia struttura
        
        return X, y, instance_ids


def load_concatenated_layers(model_name, dataset_name, layer_indices, type_layer, stats):
    """
    Carica multipli layer e li concatena.
    Supporta sia la vecchia che la nuova struttura dati.
    """
    print(f"   Caricamento {model_name} [{type_layer}]: layers {layer_indices}...")
    combined_features = []
    y = None
    instance_ids = None
    
    for layer_idx in layer_indices:
        try:
            X_layer, y_layer, ids_layer = load_activations_and_labels(model_name, dataset_name, layer_idx, type_layer)
            combined_features.append(X_layer)
            
            if y is None:
                y = y_layer
                instance_ids = ids_layer
        except FileNotFoundError as e:
            print(f"Warning: Layer {layer_idx} non trovato: {e}. Salto.")
            continue

    if not combined_features:
        raise ValueError(f"Nessun layer caricato per {model_name}")

    X_final = np.concatenate(combined_features, axis=1)
    return X_final, y

def run_experiment_pipeline_cached(X_teacher, y_teacher, teacher_name, 
                                   X_student, y_student, student_name, layer_type):
    """
    Esegue l'esperimento con dati già splittati e normalizzati.
    (X_teacher, y_teacher, X_student, y_student sono già train/test split e normalizzati)
    """
    print(f"\n=== EXPERIMENT: {layer_type.upper()} LAYERS ({teacher_name} → {student_name}) ===")
    print(f"Teacher Input Shape ({teacher_name}): Train={X_teacher['X_train'].shape}, Test={X_teacher['X_test'].shape}")
    print(f"Student Input Shape ({student_name}): Train={X_student['X_train'].shape}, Test={X_student['X_test'].shape}")
    
    X_A_train = X_teacher['X_train']
    X_A_test = X_teacher['X_test']
    y_A_train = y_teacher['y_train']
    y_A_test = y_teacher['y_test']
    
    X_B_train = X_student['X_train']
    X_B_test = X_student['X_test']
    y_B_train = y_student['y_train']
    y_B_test = y_student['y_test']

    # --- STEP 1: Teacher Probing ---
    print(f"1. Training Teacher Probe ({teacher_name})...")
    probe_teacher = LogisticRegression(max_iter=10000, class_weight='balanced', solver='saga', n_jobs=-1)
    probe_teacher.fit(X_A_train, y_A_train)
    
    # --- METRICHE TEACHER ---
    y_pred_teacher = probe_teacher.predict(X_A_test)
    cm_teacher = confusion_matrix(y_A_test, y_pred_teacher)
    acc_teacher = accuracy_score(y_A_test, y_pred_teacher)
    prec_teacher = precision_score(y_A_test, y_pred_teacher)
    rec_teacher = recall_score(y_A_test, y_pred_teacher)
    f1_teacher = f1_score(y_A_test, y_pred_teacher)
    

    # --- STEP 2: Alignment ---
    print(f"2. Learning Linear Projection ({student_name} → {teacher_name})...")
    aligner = Ridge(alpha=1000.0, fit_intercept=False) 
    aligner.fit(X_B_train, X_A_train) 
    
    # --- STEP 3: StudentOnTeacher (Cross-Model) ---
    print(f"3. Projecting {student_name} & Testing with {teacher_name} Probe...")
    X_B_test_projected = aligner.predict(X_B_test)
    y_pred_cross = probe_teacher.predict(X_B_test_projected)
    
    # --- METRICHE CROSS-MODEL ---
    cm_cross = confusion_matrix(y_B_test, y_pred_cross)
    acc_cross = accuracy_score(y_B_test, y_pred_cross)
    prec_cross = precision_score(y_B_test, y_pred_cross)
    rec_cross = recall_score(y_B_test, y_pred_cross)
    f1_cross = f1_score(y_B_test, y_pred_cross)
    
    print(f"   -> {student_name} on {teacher_name} Accuracy: {acc_cross:.4f}")
    
    return {
        "type": layer_type,
        "teacher_name": teacher_name,
        "student_name": student_name,
        "teacher": {
            "accuracy": acc_teacher,
            "precision": prec_teacher,
            "recall": rec_teacher,
            "f1": f1_teacher,
            "confusion_matrix": cm_teacher.tolist()
        },
        "student_on_teacher": {
            "accuracy": acc_cross,
            "precision": prec_cross,
            "recall": rec_cross,
            "f1": f1_cross,
            "confusion_matrix": cm_cross.tolist()
        }
    }

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

# %%
print("="*80)
print("FASE 1: PRE-CARICAMENTO E SPLITTING DEI DATI (stessi indici shuffled per TUTTI i layer type)")
print("="*80 + "\n")


n_samples = model_a_stats['total'] 
rng = np.random.RandomState(42)
shuffled_indices = rng.permutation(n_samples)
split_idx = int(0.7 * n_samples)

train_indices = shuffled_indices[:split_idx]
test_indices = shuffled_indices[split_idx:]


data_splits = {}
for layer_type in ['attn', 'mlp', 'hidden']:
    gc.collect()
    
    # Carica Model A
    X_model_a, y_model_a = load_concatenated_layers(
        MODEL_A, DATASET_NAME, 
        LAYER_CONFIG[MODEL_A][layer_type], 
        layer_type, model_a_stats
    )
    
    # Carica Model B
    X_model_b, y_model_b = load_concatenated_layers(
        MODEL_B, DATASET_NAME, 
        LAYER_CONFIG[MODEL_B][layer_type], 
        layer_type, model_b_stats
    )
    
    # Applica gli STESSI indici a entrambi i modelli
    X_model_a_train, X_model_a_test = X_model_a[train_indices], X_model_a[test_indices]
    y_model_a_train, y_model_a_test = y_model_a[train_indices], y_model_a[test_indices]
    
    X_model_b_train, X_model_b_test = X_model_b[train_indices], X_model_b[test_indices]
    y_model_b_train, y_model_b_test = y_model_b[train_indices], y_model_b[test_indices]
    
    # Normalizza una sola volta
    scaler_model_a = StandardScaler()
    X_model_a_train = scaler_model_a.fit_transform(X_model_a_train)
    X_model_a_test = scaler_model_a.transform(X_model_a_test)
    
    scaler_model_b = StandardScaler()
    X_model_b_train = scaler_model_b.fit_transform(X_model_b_train)
    X_model_b_test = scaler_model_b.transform(X_model_b_test)
    
    # Salva in un dizionario le informazioni
    data_splits[layer_type] = {
        "model_a": {
            "X_train": X_model_a_train,
            "X_test": X_model_a_test,
            "y_train": y_model_a_train,
            "y_test": y_model_a_test
        },
        "model_b": {
            "X_train": X_model_b_train,
            "X_test": X_model_b_test,
            "y_train": y_model_b_train,
            "y_test": y_model_b_test
        }
    }
    
    # li cancello poichè ho tutto quello che mi serve nel dizionario
    del X_model_a, y_model_a, X_model_b, y_model_b

print("\n" + "="*80)
print("FASE 2: ESECUZIONE ESPERIMENTI SU ENTRAMBI GLI SCENARI")
print("="*80 + "\n")



# Definisci gli scenari di esperimento usando le costanti
scenarios = [
    {
        "teacher_model": MODEL_A,
        "student_model": MODEL_B,
    },
    {
        "teacher_model": MODEL_B,
        "student_model": MODEL_A,
    }
]

all_results = []

# Loop su entrambi gli scenari
for scenario_idx, scenario in enumerate(scenarios, 1):
    print(f"\n{'='*80}")
    print(f"SCENARIO {scenario_idx}: {scenario['teacher_model']} → {scenario['student_model']}")
    print(f"{'='*80}\n")
    
    results = []
    
    # Loop sui 3 tipi di layer richiesti
    for layer_type in ['attn', 'mlp', 'hidden']:
        
        try:
            # Recupera i dati pre-splittati e normalizzati
            if scenario['teacher_model'] == MODEL_A:
                X_teacher_data = data_splits[layer_type]['model_a']
                X_student_data = data_splits[layer_type]['model_b']
            else:
                X_teacher_data = data_splits[layer_type]['model_b']
                X_student_data = data_splits[layer_type]['model_a']
            
            # Esegui pipeline con dati della cache
            res = run_experiment_pipeline_cached(
                X_teacher_data, X_teacher_data, scenario['teacher_model'],
                X_student_data, X_student_data, scenario['student_model'],
                layer_type
            )
            results.append(res)
            
            # 4. Plotta confusion matrices
            print(f"\n   Creazione visualizzazioni confusion matrices...")
            plot_confusion_matrix(
                np.array(res['teacher']['confusion_matrix']), 
                layer_type, 
                f"Teacher_{scenario['teacher_model'].split('.')[0]}"
            )
            plot_confusion_matrix(
                np.array(res['student_on_teacher']['confusion_matrix']), 
                layer_type, 
                f"{scenario['student_model'].split('.')[0]}_on_{scenario['teacher_model'].split('.')[0]}"
            )
            
        except Exception as e:
            print(f"Errore critico nel layer {layer_type}: {e}")
            traceback.print_exc()
    
    all_results.append({
        "scenario": f"{scenario['teacher_model']} (teacher) → {scenario['student_model']} (student)",
        "results": results
    })

# Salva tutti i risultati in JSON
os.makedirs("results_metrics", exist_ok=True)
metrics_file = "results_metrics/experiment_results_all_scenarios.json"

all_results_json = []
for scenario_data in all_results:
    scenario_results = []
    for r in scenario_data['results']:
        scenario_results.append({
            "layer_type": r['type'],
            "teacher_model": r['teacher_name'],
            "student_model": r['student_name'],
            "teacher": {
                "accuracy": round(r['teacher']['accuracy'], 4),
                "precision": round(r['teacher']['precision'], 4),
                "recall": round(r['teacher']['recall'], 4),
                "f1_score": round(r['teacher']['f1'], 4),
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
                "confusion_matrix": {
                    "TN": int(r['student_on_teacher']['confusion_matrix'][0][0]),
                    "FP": int(r['student_on_teacher']['confusion_matrix'][0][1]),
                    "FN": int(r['student_on_teacher']['confusion_matrix'][1][0]),
                    "TP": int(r['student_on_teacher']['confusion_matrix'][1][1])
                }
            }
        })
    
    all_results_json.append({
        "scenario": scenario_data['scenario'],
        "results": scenario_results
    })

with open(metrics_file, 'w') as f:
    json.dump(all_results_json, f, indent=2)

print(f"\n✓ Risultati salvati in: {metrics_file}")

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
    # Calcolo dimensioni griglia
    cols = 4
    rows = (num_layers + cols - 1) // cols
    
    # Aumento la dimensione verticale per dare spazio ai titoli
    fig, axs = plt.subplots(rows, cols, figsize=(32, 10 * rows))
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
            # Format della varianza più pulito
            var_text = f'\nVar: {pca.explained_variance_ratio_[0]:.1%} - {pca.explained_variance_ratio_[1]:.1%}'

        if activations_2d is not None:
            colors = ['red' if i in hallucinated_indices else 'blue' for i in range(activations_2d.shape[0])]
            
            # Scatter plot
            axs[layer].scatter(activations_2d[:, 0], activations_2d[:, 1], c=colors, alpha=0.6, s=15)
            
            # --- MODIFICHE RICHIESTE ---
            # Titolo molto più grande
            axs[layer].set_title(f'Layer {layer}{var_text}', fontsize=24, fontweight='bold')
            
            # Etichette assi rimosse o ingrandite (qui le tengo ma grandi)
            axs[layer].set_xlabel(f'{dim_type} 1', fontsize=16)
            axs[layer].set_ylabel(f'{dim_type} 2', fontsize=16)
            axs[layer].tick_params(axis='both', which='major', labelsize=14)
            axs[layer].grid(True, alpha=0.3)
    
    # Rimuovi assi vuoti
    for i in range(num_layers, len(axs)):
        axs[i].axis('off')
    
    # --- LEGENDA RIMOSSA COME RICHIESTO ---
    
    # Titolo generale ancora più grande
    fig.suptitle(f'{dim_type} Analysis: {model} - {type} layers', fontsize=40, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    filename = os.path.join(directory_to_save, f'{model}_{dataset}_{type}_activations_{dim_type}_CLEAN.jpg') # Salvo in JPG per compatibilità
    plt.savefig(filename, dpi=150, bbox_inches='tight')
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
    
    # Crea figura singola grande
    fig, ax = plt.subplots(figsize=(12, 10))
    
    activations_2d = None
    title_suffix = ""

    # PCA
    if dim_type == "PCA":
        pca = PCA(n_components=2)
        activations_2d = pca.fit_transform(activations)
        var_1 = pca.explained_variance_ratio_[0]
        var_2 = pca.explained_variance_ratio_[1]
        title_suffix = f" (Explained Var: {var_1:.1%}, {var_2:.1%})"

    if activations_2d is not None:
        colors = ['red' if i in hallucinated_indices else 'blue' for i in range(activations_2d.shape[0])]
        
        # Plot
        scatter = ax.scatter(activations_2d[:, 0], activations_2d[:, 1], c=colors, alpha=0.5, s=25)
        
        # Styling
        ax.set_title(f'{model} - {type.upper()} Layer {layer_idx}\n{dim_type}{title_suffix}', fontsize=20, fontweight='bold')
        ax.set_xlabel(f'Principal Component 1', fontsize=16)
        ax.set_ylabel(f'Principal Component 2', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Legenda (utile nel singolo plot)
        n_hall = len(hallucinated_indices)
        n_total = activations_2d.shape[0]
        legend_elements = [
            Patch(facecolor='blue', label=f'True ({n_total - n_hall})'),
            Patch(facecolor='red', label=f'Hallucination ({n_hall})')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=14, title="Labels", title_fontsize=16)

    # Save
    os.makedirs(directory_to_save, exist_ok=True)
    filename = os.path.join(directory_to_save, f'SINGLE_{model}_{type}_L{layer_idx}_{dim_type}.png')
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
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
createSinglePlot(MODEL_A, DATASET_NAME, LAYER_CONFIG[MODEL_A]["attn"][0], "attn", model_a_stats, "PCA", single_plot_dir)
createSinglePlot(MODEL_B, DATASET_NAME, LAYER_CONFIG[MODEL_B]["attn"][0], "attn", model_b_stats, "PCA", single_plot_dir)


