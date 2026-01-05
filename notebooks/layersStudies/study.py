

import json
import os
import numpy as np
import torch
import random
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import gc
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)

def get_balanced_indices(y, seed=SEED):
    
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

# %%

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.getcwd()))
CACHE_DIR_NAME = "activation_cache"
HF_DEFAULT_HOME = os.environ.get("HF_HOME", "~\\.cache\\huggingface\\hub")




def stats_per_json(model_name, dataset_name):
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
    
    base_path = os.path.join(PROJECT_ROOT, CACHE_DIR_NAME, model_name, dataset_name, "activation_attn")
    hallucinated_path = os.path.join(base_path, "hallucinated")
    if os.path.isdir(hallucinated_path):
        return 'new'
    return 'old'

def get_stats(model_name, dataset_name):
    
    structure = detect_structure_type(model_name, dataset_name)
    if structure == 'new':
        return stats_from_new_structure(model_name, dataset_name)
    else:
        return stats_per_json(model_name, dataset_name)


available_models = os.listdir(os.path.join(PROJECT_ROOT, CACHE_DIR_NAME))
print("Available models:", available_models)

# Choose model and dataset
MODEL_NAME = "gemma-2-9b-it"  # Change as needed
DATASET_NAME = "belief_bank_constraints"      # Change as needed
# Check the structure
structure_type = detect_structure_type(MODEL_NAME, DATASET_NAME)
print(f"Data structure detected for {MODEL_NAME}/{DATASET_NAME}: {structure_type}")

# Get statistics
stats = get_stats(MODEL_NAME, DATASET_NAME)
print(f"\nStatistics for {MODEL_NAME}:")
print(f"  Total samples: {stats['total']}")
print(f"  Hallucinations: {stats['hallucinations']} ({stats['percent_hallucinations']:.2f}%)")

# If you want to compare multiple models
if "Llama-3.1-8B-Instruct" in available_models:
    gemma_stats = get_stats("Llama-3.1-8B-Instruct", DATASET_NAME)
    print(f"\nStatistics for Llama-3.1-8B-Instruct")
    print(f"  Total samples: {gemma_stats['total']}")
    print(f"  Hallucinations: {gemma_stats['hallucinations']} ({gemma_stats['percent_hallucinations']:.2f}%)")

# %%
def layers_in_model(model, dataset=None):
    
    file_path = os.path.join(PROJECT_ROOT, CACHE_DIR_NAME, model)
    
    # If dataset is not specified, take the first available one
    if dataset is None:
        subdirs = [d for d in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, d))]
        if not subdirs:
            raise ValueError(f"No subdirectories found in {file_path}")
        dataset = subdirs[0]
    
    layer_dir = os.path.join(file_path, dataset, "activation_attn")
    
    # Check if it's the new structure (with hallucinated/not_hallucinated folders)
    hallucinated_path = os.path.join(layer_dir, "hallucinated")
    if os.path.isdir(hallucinated_path):
        # New structure: count layer*_activations.pt files in hallucinated folder
        layer_files = [f for f in os.listdir(hallucinated_path) if f.endswith('_activations.pt')]
        return len(layer_files)
    else:
        # Old structure: count layer*_activations.pt files directly
        layer_files = [f for f in os.listdir(layer_dir) if f.endswith('_activations.pt')]
        return len(layer_files)


def load_activations_and_labels(model_name, dataset_name, layer, layer_type):
    
    structure = detect_structure_type(model_name, dataset_name)
    base_path = os.path.join(PROJECT_ROOT, CACHE_DIR_NAME, model_name, dataset_name, f"activation_{layer_type}")
    
    if structure == 'new':
        # New structure: load from hallucinated/ and not_hallucinated/
        hall_act_path = os.path.join(base_path, "hallucinated", f"layer{layer}_activations.pt")
        hall_ids_path = os.path.join(base_path, "hallucinated", f"layer{layer}_instance_ids.json")
        not_hall_act_path = os.path.join(base_path, "not_hallucinated", f"layer{layer}_activations.pt")
        not_hall_ids_path = os.path.join(base_path, "not_hallucinated", f"layer{layer}_instance_ids.json")
        
        # Load activations
        hall_activations = torch.load(hall_act_path)
        not_hall_activations = torch.load(not_hall_act_path)
        
        # Load instance_ids
        with open(hall_ids_path, 'r') as f:
            hall_ids = json.load(f)
        with open(not_hall_ids_path, 'r') as f:
            not_hall_ids = json.load(f)
        
        # Convert to numpy
        if isinstance(hall_activations, torch.Tensor):
            hall_activations = hall_activations.cpu().numpy().astype(np.float32)
        if isinstance(not_hall_activations, torch.Tensor):
            not_hall_activations = not_hall_activations.cpu().numpy().astype(np.float32)
        
        # Concatenate activations, labels and ids
        X_concat = np.vstack([hall_activations, not_hall_activations])
        y_concat = np.concatenate([
            np.ones(hall_activations.shape[0], dtype=int),
            np.zeros(not_hall_activations.shape[0], dtype=int)
        ])
        ids_concat = np.array(hall_ids + not_hall_ids)
        
        # Sort everything by instance_ids
        sort_indices = np.argsort(ids_concat)
        X = X_concat[sort_indices]
        y = y_concat[sort_indices]
        instance_ids = ids_concat[sort_indices]
        
        return X, y, instance_ids
    
    else:
        # Old structure: load everything together and use hallucination_labels.json
        file_path = os.path.join(base_path, f"layer{layer}_activations.pt")
        activations = torch.load(file_path)
        
        if isinstance(activations, torch.Tensor):
            X = activations.cpu().numpy().astype(np.float32)
        else:
            X = activations.astype(np.float32)
        
        # Load labels from JSON
        labels_path = os.path.join(PROJECT_ROOT, CACHE_DIR_NAME, model_name, dataset_name, 
                                   "generations", "hallucination_labels.json")
        with open(labels_path, 'r') as f:
            labels_data = json.load(f)
        
        y = np.array([item['is_hallucination'] for item in labels_data], dtype=int)
        instance_ids = np.arange(len(y))  # Sequential IDs for old structure
        
        return X, y, instance_ids


def verify_ordering(model_name, dataset_name, layer=0, layer_type="attn"):
   
    X, y, instance_ids = load_activations_and_labels(model_name, dataset_name, layer, layer_type)
    
    print(f"=== Verify ordering for {model_name}/{dataset_name} ===")
    print(f"Layer: {layer}, Type: {layer_type}")
    print(f"Number of samples: {len(instance_ids)}")
    print(f"\nFirst 20 instance_ids: {instance_ids[:20].tolist()}")
    print(f"Last 20 instance_ids: {instance_ids[-20:].tolist()}")
    
    # Check if sorted
    is_sorted = np.all(instance_ids[:-1] <= instance_ids[1:])
    print(f"\nInstance_ids are sorted in ascending order: {is_sorted}")
    
    # Check label correspondence
    print(f"\nFirst 20 labels (y): {y[:20].tolist()}")
    print(f"Last 20 labels (y): {y[-20:].tolist()}")
    
    # Label statistics
    print(f"\nLabel distribution:")
    print(f"  Hallucination (y=1): {np.sum(y == 1)}")
    print(f"  Not hallucination (y=0): {np.sum(y == 0)}")
    
    return X, y, instance_ids

# Run verification
X_test, y_test, ids_test = verify_ordering(MODEL_NAME, DATASET_NAME, layer=0, layer_type="attn")

# Also verify that different layers/types have the same ordering
print("\n" + "="*60)
print("Checking consistency across different layers/types...")
print("="*60)

_, y_attn, ids_attn = load_activations_and_labels(MODEL_NAME, DATASET_NAME, 0, "attn")
_, y_mlp, ids_mlp = load_activations_and_labels(MODEL_NAME, DATASET_NAME, 0, "mlp")
_, y_hidden, ids_hidden = load_activations_and_labels(MODEL_NAME, DATASET_NAME, 0, "hidden")

print(f"IDs attn == IDs mlp: {np.array_equal(ids_attn, ids_mlp)}")
print(f"IDs attn == IDs hidden: {np.array_equal(ids_attn, ids_hidden)}")
print(f"Labels attn == Labels mlp: {np.array_equal(y_attn, y_mlp)}")
print(f"Labels attn == Labels hidden: {np.array_equal(y_attn, y_hidden)}")


MODELS_TO_ANALYZE = [MODEL_NAME]
if "Llama-3.1-8B-Instruct" in available_models:
    MODELS_TO_ANALYZE.append("Llama-3.1-8B-Instruct")

DATASET = DATASET_NAME

# Initialize results
results = {model: {"attn": {}, "mlp": {}, "hidden": {}} for model in MODELS_TO_ANALYZE}

# For each model
for model in MODELS_TO_ANALYZE:
    print(f"\n{'='*60}")
    print(f"Processing model: {model}")
    print(f"{'='*60}")
    
    num_layers = layers_in_model(model, DATASET)
    print(f"Number of detected layers: {num_layers}")
    
    # ============================================
    # CALCULATE INDICES ONLY ONCE
    # ============================================
    X_sample, y_sample, _ = load_activations_and_labels(model, DATASET, 0, "attn")
    n_samples = X_sample.shape[0]
    print(f"Number of original samples: {n_samples}")
    print(f"Original distribution: {np.bincount(y_sample)}")
    
    del X_sample  # Free memory immediately
    
    balanced_indices = get_balanced_indices(y_sample, seed=SEED)
    y_balanced = y_sample[balanced_indices]
    print(f"After undersampling: {len(balanced_indices)} samples")
    print(f"Balanced distribution: {np.bincount(y_balanced)}")
    
    # Stratified split on balanced data
    train_rel_idx, test_rel_idx = train_test_split(
        np.arange(len(balanced_indices)),
        test_size=0.3,
        random_state=SEED,
        stratify=y_balanced
    )
    
    # Converti in indici assoluti
    train_indices = balanced_indices[train_rel_idx]
    test_indices = balanced_indices[test_rel_idx]
    
    print(f"Train: {len(train_indices)}, Test: {len(test_indices)}")
    print(f"Train dist: {np.bincount(y_sample[train_indices])}, Test dist: {np.bincount(y_sample[test_indices])}")
    
    del y_sample, y_balanced
    gc.collect()
    
    # ============================================
    # LOOP SUI LAYER
    # ============================================
    for layer in range(num_layers):
        for layer_type in ["attn", "mlp", "hidden"]:
            X_layer, y, _ = load_activations_and_labels(model, DATASET, layer, layer_type)
            
            X_train = X_layer[train_indices]
            y_train = y[train_indices]
            X_test = X_layer[test_indices]
            y_test = y[test_indices]
            
            del X_layer, y  # Free memory immediately
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Training (lbfgs is faster for small datasets)
            clf = LogisticRegression(max_iter=10000, class_weight='balanced', solver='lbfgs', n_jobs=-1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]
            
            # Metriche
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auroc = roc_auc_score(y_test, y_proba)
            
            # Save the results
            results[model][layer_type][layer] = (accuracy, f1, auroc)
            
            print(f"  Layer {layer} {layer_type}: Acc={accuracy:.4f}, F1={f1:.4f}, AUROC={auroc:.4f}")
            
            del X_train, X_test, y_train, y_test, scaler, clf
            gc.collect()

print("\n" + "="*60)
print("Training completato!")
print("="*60)

# %%
# Function to sort all layers by accuracy and save to JSON
def sort_and_save_all_results(results, output_file="sorted_results.json"):
    
    sorted_results = {}
    
    for model_name, layer_types in results.items():
        sorted_results[model_name] = {}
        
        for layer_type, layer_data in layer_types.items():
            # Sort layers by descending accuracy
            sorted_layers = sorted(
                [(layer, acc, f1, auroc) for layer, (acc, f1, auroc) in layer_data.items()],
                key=lambda x: x[1],  # sort by accuracy
                reverse=True  # descending order
            )
            
            # Save in sorted list format
            sorted_results[model_name][layer_type] = [
                {
                    "layer": layer,
                    "accuracy": round(acc, 4),
                    "f1_score": round(f1, 4),
                    "auroc": round(auroc, 4)
                }
                for layer, acc, f1, auroc in sorted_layers
            ]
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(sorted_results, f, indent=4)
    print(f"All sorted results saved to {output_file}")
    
    return sorted_results

# Save all sorted results
sorted_all = sort_and_save_all_results(results, "all_layers_sorted.json")

# %%
import matplotlib.pyplot as plt

def plot_accuracy_from_json(json_data, model_name=None, dataset="Dataset"):
    
    
    # 1. Model selection
    if model_name is None:
        model_name = list(json_data.keys())[0]
    
    if model_name not in json_data:
        print(f"Error: Model '{model_name}' not found in JSON.")
        return

    data = json_data[model_name]

    # 2. Style Configuration
    plt.rcParams.update({
        "font.family": "serif",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.labelsize": 24,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 12,
        "legend.title_fontsize": 14,
        "lines.linewidth": 2
    })

    fig, ax = plt.subplots(figsize=(12, 8))

    # Color mapping
    styles = {
        "hidden": {"color": "red", "label": "hidden"},
        "mlp":    {"color": "blue", "label": "mlp"},
        "attn":   {"color": "green", "label": "attn"}
    }

    for key in ["hidden", "mlp", "attn"]:
        if key in data:
            points = data[key]
            sorted_points = sorted(points, key=lambda x: x['layer'])
            layers = [item['layer'] for item in sorted_points]
            accuracies = [item['accuracy'] for item in sorted_points]
            ax.plot(layers, accuracies, 
                    color=styles[key]["color"], 
                    label=styles[key]["label"])

    # 4. Graphic Refinement
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.grid(True, linestyle='-', alpha=1.0)
    
    legend = ax.legend(title="activation", loc="upper left", frameon=True)
    plt.setp(legend.get_title(), fontweight='bold')

    plt.tight_layout()
    
    os.makedirs("img", exist_ok=True)
    plt.savefig(f"img/{model_name}_{dataset}_activations.pdf")
    #plt.show()

# Load and display results
content = json.load(open('all_layers_sorted_GEMMA_LLAMA_BBF.json', 'r'))
DATASET = DATASET_NAME
for model_name in content.keys():
    plot_accuracy_from_json(content, model_name, DATASET)

# %%
import matplotlib.pyplot as plt

def plot_single_model_multi_dataset(json_files, dataset_names, model_name, output_filename=None):
    
    
    all_data = {}
    for json_file, dataset_name in zip(json_files, dataset_names):
        with open(json_file, 'r') as f:
            all_data[dataset_name] = json.load(f)
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "lines.linewidth": 2
    })
    
    styles = {
        "hidden": {"color": "red", "label": "hidden"},
        "mlp":    {"color": "blue", "label": "mlp"},
        "attn":   {"color": "green", "label": "attn"}
    }
    
    fig, axes = plt.subplots(1, len(dataset_names), figsize=(15, 4))
    
    full_model_name = None
    for key in all_data[dataset_names[0]].keys():
        if model_name.lower() in key.lower():
            full_model_name = key
            break
    
    if full_model_name is None:
        print(f"Modello '{model_name}' non trovato nei dati.")
        return
    
    for col_idx, dataset_name in enumerate(dataset_names):
        ax = axes[col_idx]
        
        data = all_data[dataset_name]
        
        matching_model = None
        for key in data.keys():
            if model_name.lower() in key.lower():
                matching_model = key
                break
        
        if matching_model is None:
            ax.set_title(dataset_name)
            ax.text(0.5, 0.5, "Model not found", ha='center', va='center', transform=ax.transAxes)
            continue
        
        model_data = data[matching_model]
        
        for key in ["hidden", "mlp", "attn"]:
            if key in model_data:
                points = model_data[key]
                sorted_points = sorted(points, key=lambda x: x['layer'])
                layers = [item['layer'] for item in sorted_points]
                accuracies = [item['accuracy'] for item in sorted_points]
                ax.plot(layers, accuracies, 
                        color=styles[key]["color"], 
                        label=styles[key]["label"])
        
        ax.set_title(dataset_name)
        
        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy")
        
        ax.grid(True, linestyle='-', alpha=1.0)
        
        legend = ax.legend(title="activation", loc="upper left", frameon=True)
        plt.setp(legend.get_title(), fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs("img", exist_ok=True)
    
    if output_filename is None:
        output_filename = f"{model_name}_3datasets_comparison.pdf"
    
    plt.savefig(f"img/{output_filename}")
    plt.show()
    print(f"Figure saved to img/{output_filename}")

# Available JSON files
json_files = [
    "all_layers_sorted_GEMMA_LLAMA_BBC.json",
    "all_layers_sorted_GEMMA_LLAMA_BBF.json",
    "all_layers_sorted_GEMMA_LLAMA_HE.json"
]

# Dataset names (for column titles)
dataset_names = [
    "Belief Bank Constraints",
    "Belief Bank Facts", 
    "HaluEval"
]

# Generate a plot for Gemma
plot_single_model_multi_dataset(json_files, dataset_names, "gemma", "gemma_3datasets_comparison.pdf")

# Generate a plot for Llama
plot_single_model_multi_dataset(json_files, dataset_names, "llama", "llama_3datasets_comparison.pdf")


