import os
import json
import glob
import torch
from sklearn.linear_model import Ridge
import pickle
from tqdm import tqdm
from collections import defaultdict


def get_num_physical_layers(base_dir, activation_type="attn"):
    """Detect the number of physical layers from the activation files."""
    act_dir = os.path.join(base_dir, f"activation_{activation_type}")
    files = [f for f in os.listdir(act_dir) if f.endswith('.pt')]
    max_layer = -1
    for f in files:
        parts = f.replace('.pt', '').split('-')
        if parts[0].startswith('layer'):
            layer_idx = int(parts[0][5:])
            if layer_idx > max_layer:
                max_layer = layer_idx
    return max_layer + 1


def load_layer_selection(steering_dir, dataset_name, model_name_safe):
    """
    Load the layer_selection JSON for a model+dataset from SteeringVectors/.
    Returns the list of top_k_modules and num_total_modules.
    """
    search_pattern = os.path.join(
        steering_dir, dataset_name,
        f"layer_selection_{model_name_safe}_*.json"
    )
    matches = glob.glob(search_pattern)
    if not matches:
        raise FileNotFoundError(f"No layer_selection JSON found matching: {search_pattern}")
    
    json_path = matches[0]
    print(f"  Using layer selection: {os.path.basename(json_path)}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data["top_k_modules"], data["num_total_modules"]


def load_activations_for_layers(base_dir, layer_type_pairs):
    """
    Load activations for specific (layer_idx, module_type) pairs.
    
    Returns tensor of shape [num_samples * len(layer_type_pairs), hidden_dim],
    ordered by instance_id, then by position in layer_type_pairs.
    """
    layer_data = {}  # (layer_idx, module_type) -> {instance_id: tensor}
    all_instance_ids = set()
    
    types_needed = set(t for _, t in layer_type_pairs)
    
    for module_type in types_needed:
        act_dir = os.path.join(base_dir, f"activation_{module_type}")
        if not os.path.exists(act_dir):
            continue
        
        needed_layers = set(l for l, t in layer_type_pairs if t == module_type)
        files = [f for f in os.listdir(act_dir) if f.endswith('.pt')]
        
        for f in files:
            parts = f.replace('.pt', '').split('-')
            layer_idx = int(parts[0][5:])
            instance_id = int(parts[1][2:])
            
            if layer_idx in needed_layers:
                key = (layer_idx, module_type)
                if key not in layer_data:
                    layer_data[key] = {}
                path = os.path.join(act_dir, f)
                layer_data[key][instance_id] = torch.load(path)
                all_instance_ids.add(instance_id)
    
    sorted_ids = sorted(all_instance_ids)
    
    all_acts = []
    for instance_id in sorted_ids:
        for layer_idx, module_type in layer_type_pairs:
            key = (layer_idx, module_type)
            if key in layer_data and instance_id in layer_data[key]:
                all_acts.append(layer_data[key][instance_id])
            else:
                raise ValueError(
                    f"Missing activation for instance {instance_id}, "
                    f"layer {layer_idx}, type {module_type}"
                )
    
    return torch.stack(all_acts)


def map_layers_proportionally(trainer_layers, trainer_num_physical, target_num_physical):
    """
    Map trainer's selected layers proportionally to target model's layers.
    Same module_type, proportional physical layer index.
    
    E.g. trainer physical layer 4 out of 32 → depth 4/31 → target layer round(4/31 * 41) = 5 out of 42
    """
    target_layers = []
    for phys_layer, mod_type in trainer_layers:
        if trainer_num_physical <= 1:
            target_layer = 0
        else:
            relative_depth = phys_layer / (trainer_num_physical - 1)
            target_layer = round(relative_depth * (target_num_physical - 1))
        target_layers.append((target_layer, mod_type))
    return target_layers



ALIGNERS={
    "Llama_to_Gemma_BBF":
    {
        "Trainer": "Llama-3.1-8B-Instruct",
        "Target": "gemma-2-9b-it",
        "dataset":"belief_bank_facts",
        "source": "activation_cache_truthx/google_gemma-2-9b-it/belief_bank_facts_subset",
        "target": "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/belief_bank_facts_subset",
        "output_dir": "alignment_cache/Llama_to_Gemma_BBF",
    },
    "Llama_to_Gemma_BBC":
    {
        "Trainer": "Llama-3.1-8B-Instruct",
        "Target": "gemma-2-9b-it",
        "dataset":"belief_bank_constraints",
        "source": "activation_cache_truthx/google_gemma-2-9b-it/belief_bank_constraints_subset",
        "target": "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/belief_bank_constraints_subset",
        "output_dir": "alignment_cache/Llama_to_Gemma_BBC",
    },
    "Llama_to_Gemma_HE":
    {
        "Trainer": "Llama-3.1-8B-Instruct",
        "Target": "gemma-2-9b-it",
        "dataset":"halu_eval",
        "source": "activation_cache_truthx/google_gemma-2-9b-it/halu_eval_subset",
        "target": "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/halu_eval_subset",
        "output_dir": "alignment_cache/Llama_to_Gemma_HE",
    },
    "Gemma_to_Llama_BBF":
    {
        "Trainer": "gemma-2-9b-it",
        "Target": "Llama-3.1-8B-Instruct",
        "dataset":"belief_bank_facts",
        "source": "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/belief_bank_facts_subset",
        "target": "activation_cache_truthx/google_gemma-2-9b-it/belief_bank_facts_subset",
        "output_dir": "alignment_cache/Gemma_to_Llama_BBF",
    },
    "Gemma_to_Llama_BBC":
    {
        "Trainer": "gemma-2-9b-it",
        "Target": "Llama-3.1-8B-Instruct",
        "dataset":"belief_bank_constraints",
        "source": "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/belief_bank_constraints_subset",
        "target": "activation_cache_truthx/google_gemma-2-9b-it/belief_bank_constraints_subset",
        "output_dir": "alignment_cache/Gemma_to_Llama_BBC",
    },
    "Gemma_to_Llama_HE":
    {
        "Trainer": "gemma-2-9b-it",
        "Target": "Llama-3.1-8B-Instruct",
        "dataset":"halu_eval",
        "source": "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/halu_eval_subset",
        "target": "activation_cache_truthx/google_gemma-2-9b-it/halu_eval_subset",
        "output_dir": "alignment_cache/Gemma_to_Llama_HE",
    }
}


STEERING_DIR = "SteeringVectors"

for aligner_name, aligner_info in ALIGNERS.items():
    print(f"\nProcessing aligner: {aligner_name}")
    if not os.path.exists(aligner_info["output_dir"]):
        os.makedirs(aligner_info["output_dir"])
    
    # Extract model_name_safe from source/target paths
    # e.g. "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/belief_bank_facts_subset"
    trainer_model_safe = aligner_info["source"].split("/")[1]
    target_model_safe = aligner_info["target"].split("/")[1]
    dataset_name = aligner_info["dataset"]
    
    # 1. Load trainer's layer_selection to know which layers were selected
    print(f"Loading trainer layer selection for {trainer_model_safe}...")
    trainer_modules, trainer_total_modules = load_layer_selection(
        STEERING_DIR, dataset_name, trainer_model_safe
    )
    trainer_num_physical = trainer_total_modules // 2  # attn + mlp → /2
    
    # Get ONLY the top-1 module from trainer's layer selection
    top1 = trainer_modules[0]
    trainer_selected = [(top1["physical_layer"], top1["module_type"])]
    print(f"  Trainer top-1 module: physical {top1['physical_layer']} {top1['module_type']} (accuracy: {top1['probing_accuracy']:.4f})")
    print(f"  Trainer num physical layers: {trainer_num_physical}")
    
    # 2. Detect target model's total physical layers
    target_num_physical = get_num_physical_layers(aligner_info["target"], "attn")
    print(f"  Target num physical layers: {target_num_physical}")
    
    # 3. Map trainer's selected layers proportionally to target
    target_selected = map_layers_proportionally(
        trainer_selected, trainer_num_physical, target_num_physical
    )
    
    print(f"\n  Layer mapping (trainer → target):")
    for (tl, tt), (gl, gt) in zip(trainer_selected, target_selected):
        depth_pct = tl / max(trainer_num_physical - 1, 1) * 100
        print(f"    {tt} layer {tl} ({depth_pct:.1f}%) → {gt} layer {gl}")
    
    # 4. Load only selected layers from trainer
    print("\nLoading trainer activations (selected layers only)...")
    trainer = load_activations_for_layers(aligner_info["source"], trainer_selected)
    
    # 5. Load corresponding proportional layers from target
    print("Loading target activations (proportionally mapped layers)...")
    target = load_activations_for_layers(aligner_info["target"], target_selected)
    
    print(f"Trainer activations shape: {trainer.shape}")
    print(f"Target activations shape: {target.shape}")
    
    # 6. Train ridge regressor: target activations → trainer activations
    print("Training Ridge regressor...")
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(target.numpy(), trainer.numpy())
    
    # Save the ridge regressor
    ridge_path = os.path.join(aligner_info["output_dir"], "ridge_regressor.pkl")
    with open(ridge_path, 'wb') as f:
        pickle.dump(ridge, f)
    
    print(f"Saved ridge regressor to {ridge_path}")
    
    # Save layer mapping info for later use
    mapping_info = {
        "trainer_model": trainer_model_safe,
        "target_model": target_model_safe,
        "dataset": dataset_name,
        "trainer_num_physical_layers": trainer_num_physical,
        "target_num_physical_layers": target_num_physical,
        "num_selected_modules": len(trainer_selected),
        "layer_mapping": [
            {
                "trainer_layer": tl, "trainer_type": tt,
                "target_layer": gl, "target_type": gt,
                "depth_pct": tl / max(trainer_num_physical - 1, 1) * 100
            }
            for (tl, tt), (gl, gt) in zip(trainer_selected, target_selected)
        ],
        "trainer_shape": list(trainer.shape),
        "target_shape": list(target.shape),
    }
    mapping_path = os.path.join(aligner_info["output_dir"], "layer_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(mapping_info, f, indent=2)
    
    print(f"Saved layer mapping to {mapping_path}")
    print(f"Aligner {aligner_name} completed.")




#next strat IGNORA QUESTO COMMENTO, è solo per me per ricordarmi i prossimi passi da fare, non è da includere nella risposta finale
#1) Allineare i primi top layer
#2) Modificare la strat di inference