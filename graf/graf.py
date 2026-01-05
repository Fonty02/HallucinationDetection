"""
Script to generate PCA plots of activations for each LLM-LayerType combination.
For each combination, generates a figure with 3 subplots (one per dataset).
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ==================================================================
# CONFIGURAZIONE
# ==================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42

# Data structure: {dataset_folder: {model: {layer_type: layer_num}}}
# The layers are those available in the .pt files
DATASET_CONFIG = {
    "beliefbank_constr": {
        "dataset_name": "belief_bank_constraints",
        "display_name": "Belief Bank Constraints",
        "models": {
            "gemma-2-9b-it": {
                "attn": 23,
                "hidden": 23,
                "mlp": 24
            },
            "Llama-3.1-8B-Instruct": {
                "attn": 5,
                "hidden": 13,
                "mlp": 12
            }
        }
    },
    "beliefbank_fact": {
        "dataset_name": "belief_bank_facts",
        "display_name": "Belief Bank Facts",
        "models": {
            "gemma-2-9b-it": {
                "attn": 21,
                "hidden": 23,
                "mlp": 22
            },
            "Llama-3.1-8B-Instruct": {
                "attn": 8,
                "hidden": 14,
                "mlp": 14
            }
        }
    },
    "halu_eval": {
        "dataset_name": "halu_eval",
        "display_name": "HaluEval",
        "models": {
            "gemma-2-9b-it": {
                "attn": 21,
                "hidden": 19,
                "mlp": 23
            },
            "Llama-3.1-8B-Instruct": {
                "attn": 14,
                "hidden": 14,
                "mlp": 13
            }
        }
    }
}

MODELS = ["gemma-2-9b-it", "Llama-3.1-8B-Instruct"]
LAYER_TYPES = ["attn", "mlp", "hidden"]
DATASET_FOLDERS = ["beliefbank_constr", "beliefbank_fact", "halu_eval"]


def load_activations(dataset_folder, model_name, layer_type, layer_num):
    """
    Load activations from .pt files for hallucinated and not_hallucinated.
    
    Returns:
        X: numpy array delle attivazioni concatenate
        y: numpy array delle label (1=hallucination, 0=not hallucination)
    """
    dataset_name = DATASET_CONFIG[dataset_folder]["dataset_name"]
    base_path = os.path.join(
        SCRIPT_DIR, 
        dataset_folder, 
        "activation_cache", 
        model_name, 
        dataset_name, 
        f"activation_{layer_type}"
    )
    
    hall_path = os.path.join(base_path, "hallucinated", f"layer{layer_num}_activations.pt")
    not_hall_path = os.path.join(base_path, "not_hallucinated", f"layer{layer_num}_activations.pt")
    
    # Verify that files exist
    if not os.path.exists(hall_path) or not os.path.exists(not_hall_path):
        print(f"  [WARN] Files not found for {model_name}/{dataset_folder}/{layer_type}/layer{layer_num}")
        return None, None
    
    # Load activations
    hall_activations = torch.load(hall_path, map_location='cpu')
    not_hall_activations = torch.load(not_hall_path, map_location='cpu')
    
    # Limit to 1000 samples per label
    max_samples = 1000
    if isinstance(hall_activations, torch.Tensor):
        hall_activations = hall_activations[:max_samples] if hall_activations.shape[0] > max_samples else hall_activations
        hall_activations = hall_activations.cpu().numpy().astype(np.float32)
    if isinstance(not_hall_activations, torch.Tensor):
        not_hall_activations = not_hall_activations[:max_samples] if not_hall_activations.shape[0] > max_samples else not_hall_activations
        not_hall_activations = not_hall_activations.cpu().numpy().astype(np.float32)
    
    # Try to load instance ids and reorder as in FullLinear.ipynb
    hall_ids_path = os.path.join(base_path, "hallucinated", f"layer{layer_num}_instance_ids.json")
    not_hall_ids_path = os.path.join(base_path, "not_hallucinated", f"layer{layer_num}_instance_ids.json")
    try:
        with open(hall_ids_path, 'r') as f:
            hall_ids = json.load(f)
        with open(not_hall_ids_path, 'r') as f:
            not_hall_ids = json.load(f)

        X_concat = np.vstack([hall_activations, not_hall_activations])
        y_concat = np.concatenate([
            np.ones(hall_activations.shape[0], dtype=int),
            np.zeros(not_hall_activations.shape[0], dtype=int)
        ])
        ids_concat = np.array(hall_ids + not_hall_ids)
        sort_idx = np.argsort(ids_concat)
        X = X_concat[sort_idx]
        y = y_concat[sort_idx]
    except Exception:
        # Fallback: keep simple concatenation if ids files are not present
        X = np.vstack([hall_activations, not_hall_activations])
        y = np.concatenate([
            np.ones(hall_activations.shape[0], dtype=int),
            np.zeros(not_hall_activations.shape[0], dtype=int)
        ])

    return X, y


def create_pca_subplot(ax, X, y, title, show_ylabel=True):
    """
    Create a single PCA subplot.
    
    Args:
        ax: matplotlib axes
        X: activations (n_samples, hidden_dim)
        y: labels (n_samples,)
        title: subplot title
        show_ylabel: whether to show Y-axis label
    """
    if X is None or y is None:
        ax.text(0.5, 0.5, 'Data not available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        return
    
    # Apply PCA directly without StandardScaler (as in original notebooks)
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X)
    
    # Separate by class
    mask_hall = y == 1
    mask_not_hall = y == 0
    
    # Plot - first not_hallucinated (blue), then hallucinated (red) for visibility
    ax.scatter(X_pca[mask_not_hall, 0], X_pca[mask_not_hall, 1], 
               c='blue', alpha=0.5, s=25, label='Not Hallucinated', zorder=1)
    ax.scatter(X_pca[mask_hall, 0], X_pca[mask_hall, 1], 
               c='red', alpha=0.6, s=25, label='Hallucinated', zorder=2)
    
    ax.set_xlabel('PCA 1', fontsize=12)
    if show_ylabel:
        ax.set_ylabel('PCA 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)


def plot_model_layer_combination(model_name, layer_type, output_dir="output_plots"):
    """
    Generate a figure with 3 subplots (one per dataset) for a model-layer_type combination.
    
    Args:
        model_name: model name
        layer_type: layer type (attn, mlp, hidden)
        output_dir: output directory for plots
    """
    # Create output directory if it doesn't exist
    output_path = os.path.join(SCRIPT_DIR, output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # Style configuration
    plt.rcParams.update({
        "font.family": "serif",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Model name for title (abbreviated)
    model_short = "Gemma-2-9B" if "gemma" in model_name.lower() else "Llama-3.1-8B"
    
    
    for idx, dataset_folder in enumerate(DATASET_FOLDERS):
        config = DATASET_CONFIG[dataset_folder]
        display_name = config["display_name"]
        layer_num = config["models"][model_name][layer_type]
        
        print(f"  Loading {dataset_folder} - layer {layer_num}...")
        X, y = load_activations(dataset_folder, model_name, layer_type, layer_num)
        
        if X is not None:
            print(f"    Samples: {len(y)} (Hall: {np.sum(y==1)}, Not Hall: {np.sum(y==0)})")
        
        create_pca_subplot(
            axes[idx], 
            X, 
            y, 
            f"{display_name}\n(Layer {layer_num})",
            show_ylabel=(idx == 0)
        )
    
    # Add shared legend: position it at center above title to avoid overlaps
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=True)
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"PCA_{model_short}_{layer_type}.pdf"
    filepath = os.path.join(output_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    
    # Also save as PNG for quick preview
    filename_png = f"PCA_{model_short}_{layer_type}.png"
    filepath_png = os.path.join(output_path, filename_png)
    plt.savefig(filepath_png, dpi=150, bbox_inches='tight')
    
    plt.close(fig)


def generate_all_plots():
    """
    Generate all plots for each LLM-LayerType combination.
    """
    print("="*60)
    print("Generating PCA plots for all combinations")
    print("="*60)
    
    for model in MODELS:
        for layer_type in LAYER_TYPES:
            print(f"\n[{model}] - [{layer_type}]")
            plot_model_layer_combination(model, layer_type)
    
    print("\n" + "="*60)
    print("Generation completed!")
    print("="*60)


if __name__ == "__main__":
    generate_all_plots()
