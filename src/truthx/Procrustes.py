import os
import json
import glob
import torch
from sklearn.linear_model import LinearRegression
import pickle


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


def load_single_layer_activations(base_dir, layer_idx, module_type):
    """
    Load all activations for one physical layer and one module type.
    Returns dict: instance_id -> flattened tensor.
    """
    act_dir = os.path.join(base_dir, f"activation_{module_type}")
    if not os.path.exists(act_dir):
        raise FileNotFoundError(f"Activation dir not found: {act_dir}")

    data = {}
    for fname in os.listdir(act_dir):
        if not fname.endswith(".pt"):
            continue
        parts = fname.replace(".pt", "").split("-")
        if len(parts) < 2:
            continue
        if not parts[0].startswith("layer") or not parts[1].startswith("id"):
            continue

        current_layer = int(parts[0][5:])
        if current_layer != layer_idx:
            continue

        instance_id = int(parts[1][2:])
        path = os.path.join(act_dir, fname)
        data[instance_id] = torch.load(path).reshape(-1).float()

    if not data:
        raise ValueError(
            f"No activations found for layer {layer_idx}, type {module_type} in {act_dir}"
        )
    return data


def stack_common_instances(source_dict, target_dict):
    """
    Build aligned tensors on the intersection of instance ids.
    Returns:
      source [n, d_source], target [n, d_target], common_ids
    """
    common_ids = sorted(set(source_dict.keys()) & set(target_dict.keys()))
    if not common_ids:
        raise ValueError("No common instance_ids between source and target layers.")

    source = torch.stack([source_dict[i] for i in common_ids])
    target = torch.stack([target_dict[i] for i in common_ids])
    return source, target, common_ids


def linear_cka(x, y, eps=1e-12):
    """
    Linear CKA between two representation matrices:
      x: [n, d_x], y: [n, d_y]
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"CKA requires same n_samples, got {x.shape[0]} and {y.shape[0]}")

    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)

    hsic = torch.sum((x_centered.T @ y_centered) ** 2)
    norm_x = torch.linalg.norm(x_centered.T @ x_centered, ord="fro")
    norm_y = torch.linalg.norm(y_centered.T @ y_centered, ord="fro")
    denom = norm_x * norm_y + eps

    if denom.item() <= eps:
        return 0.0
    return (hsic / denom).item()


def fit_procrustes_linear_map(x, y, eps=1e-12):
    """
    Fit a Procrustes map (with scale + translation) from x -> y.
      x: [n, d_x], y: [n, d_y]
    Returns:
      A: [d_x, d_y], b: [d_y], scale: float
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"Procrustes requires same n_samples, got {x.shape[0]} and {y.shape[0]}"
        )

    x = x.float()
    y = y.float()
    mean_x = x.mean(dim=0, keepdim=True)
    mean_y = y.mean(dim=0, keepdim=True)
    x_centered = x - mean_x
    y_centered = y - mean_y

    cross_cov = x_centered.T @ y_centered
    u, s, vh = torch.linalg.svd(cross_cov, full_matrices=False)
    rotation = u @ vh
    scale = s.sum() / (x_centered.pow(2).sum() + eps)

    A = scale * rotation
    b = (mean_y - mean_x @ A).squeeze(0)
    return A, b, float(scale.item())


def build_linear_regressor_from_map(A, b):
    """
    Build a sklearn-compatible regressor with .predict() from map y = x @ A + b.
    """
    model = LinearRegression()
    model.coef_ = A.T.detach().cpu().numpy()
    model.intercept_ = b.detach().cpu().numpy()
    model.n_features_in_ = int(A.shape[0])
    return model



ALIGNERS={
    "Llama_to_Gemma_BBF_PROCRUSTES":
    {
        "Trainer": "Llama-3.1-8B-Instruct",
        "Target": "gemma-2-9b-it",
        "dataset":"belief_bank_facts",
        "trainer_activation_dir": "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/belief_bank_facts_subset",
        "target_activation_dir": "activation_cache_truthx/google_gemma-2-9b-it/belief_bank_facts_subset",
        "output_dir": "alignment_cache_procrustes/Llama_to_Gemma_BBF",
    },
    "Llama_to_Gemma_BBC_PROCRUSTES":
    {
        "Trainer": "Llama-3.1-8B-Instruct",
        "Target": "gemma-2-9b-it",
        "dataset":"belief_bank_constraints",
        "trainer_activation_dir": "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/belief_bank_constraints_subset",
        "target_activation_dir": "activation_cache_truthx/google_gemma-2-9b-it/belief_bank_constraints_subset",
        "output_dir": "alignment_cache_procrustes/Llama_to_Gemma_BBC",
    },
    "Llama_to_Gemma_HE_PROCRUSTES":
    {
        "Trainer": "Llama-3.1-8B-Instruct",
        "Target": "gemma-2-9b-it",
        "dataset":"halu_eval",
        "trainer_activation_dir": "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/halu_eval_subset",
        "target_activation_dir": "activation_cache_truthx/google_gemma-2-9b-it/halu_eval_subset",
        "output_dir": "alignment_cache_procrustes/Llama_to_Gemma_HE",
    },
    "Gemma_to_Llama_BBF_PROCRUSTES":
    {
        "Trainer": "gemma-2-9b-it",
        "Target": "Llama-3.1-8B-Instruct",
        "dataset":"belief_bank_facts",
        "trainer_activation_dir": "activation_cache_truthx/google_gemma-2-9b-it/belief_bank_facts_subset",
        "target_activation_dir": "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/belief_bank_facts_subset",
        "output_dir": "alignment_cache_procrustes/Gemma_to_Llama_BBF",
    },
    "Gemma_to_Llama_BBC_PROCRUSTES":
    {
        "Trainer": "gemma-2-9b-it",
        "Target": "Llama-3.1-8B-Instruct",
        "dataset":"belief_bank_constraints",
        "trainer_activation_dir": "activation_cache_truthx/google_gemma-2-9b-it/belief_bank_constraints_subset",
        "target_activation_dir": "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/belief_bank_constraints_subset",
        "output_dir": "alignment_cache_procrustes/Gemma_to_Llama_BBC",
    },
    "Gemma_to_Llama_HE_PROCRUSTES":
    {
        "Trainer": "gemma-2-9b-it",
        "Target": "Llama-3.1-8B-Instruct",
        "dataset":"halu_eval",
        "trainer_activation_dir": "activation_cache_truthx/google_gemma-2-9b-it/halu_eval_subset",
        "target_activation_dir": "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/halu_eval_subset",
        "output_dir": "alignment_cache_procrustes/Gemma_to_Llama_HE",
    }
}


STEERING_DIR = "SteeringVectors"

for aligner_name, aligner_info in ALIGNERS.items():
    print(f"\nProcessing aligner: {aligner_name}")
    if not os.path.exists(aligner_info["output_dir"]):
        os.makedirs(aligner_info["output_dir"])
    
    trainer_activation_dir = aligner_info["trainer_activation_dir"]
    target_activation_dir = aligner_info["target_activation_dir"]

    trainer_model_safe = trainer_activation_dir.split("/")[1]
    target_model_safe = target_activation_dir.split("/")[1]
    dataset_name = aligner_info["dataset"]

    # 1) Get top#1 module from TARGET model.
    print(f"Loading target layer selection for {target_model_safe}...")
    target_modules, target_total_modules = load_layer_selection(
        STEERING_DIR, dataset_name, target_model_safe
    )
    target_num_physical = target_total_modules // 2
    top1_target = target_modules[0]
    target_layer = int(top1_target["physical_layer"])
    module_type = top1_target["module_type"]
    if module_type not in {"attn", "mlp"}:
        raise ValueError(f"Unsupported module type '{module_type}' for CKA selection.")

    print(
        f"  Target top-1 module: physical {target_layer} {module_type} "
        f"(accuracy: {top1_target['probing_accuracy']:.4f})"
    )
    print(f"  Target num physical layers: {target_num_physical}")

    # 2) Enumerate ALL trainer layers of the same module type.
    trainer_num_physical = get_num_physical_layers(trainer_activation_dir, module_type)
    print(f"  Trainer num physical layers ({module_type}): {trainer_num_physical}")

    # 3) Load target top#1 activations once.
    print("\nLoading target top-1 activations...")
    target_top1_dict = load_single_layer_activations(
        target_activation_dir, target_layer, module_type
    )

    # 4) CKA(target top#1, every trainer layer of same type) and pick best.
    print(f"Running CKA over trainer {module_type} layers...")
    cka_scores = []
    best_score = float("-inf")
    best_trainer_layer = None
    best_trainer_tensor = None
    best_target_tensor = None
    best_common_ids = None

    for trainer_layer in range(trainer_num_physical):
        trainer_dict = load_single_layer_activations(
            trainer_activation_dir, trainer_layer, module_type
        )
        trainer_tensor, target_tensor, common_ids = stack_common_instances(
            trainer_dict, target_top1_dict
        )
        score = linear_cka(trainer_tensor, target_tensor)
        cka_scores.append(
            {
                "trainer_layer": trainer_layer,
                "trainer_type": module_type,
                "target_layer": target_layer,
                "target_type": module_type,
                "cka": score,
                "num_common_instances": len(common_ids),
            }
        )
        print(f"  CKA trainer {module_type} layer {trainer_layer} vs target layer {target_layer}: {score:.6f}")

        if score > best_score:
            best_score = score
            best_trainer_layer = trainer_layer
            best_trainer_tensor = trainer_tensor
            best_target_tensor = target_tensor
            best_common_ids = common_ids

    if best_trainer_layer is None:
        raise RuntimeError("No trainer layer selected by CKA.")

    print(
        f"\nBest trainer layer by CKA: {module_type} layer {best_trainer_layer} "
        f"(CKA={best_score:.6f})"
    )
    print(f"Aligned training tensors shape: trainer={best_trainer_tensor.shape}, target={best_target_tensor.shape}")

    # 5) Fit Procrustes maps in BOTH directions.
    print("Training Procrustes aligners (Target -> Trainer and Trainer -> Target)...")

    # Direction A: Target -> Trainer (example: 16 -> 28)
    A_target_to_trainer, b_target_to_trainer, scale_target_to_trainer = fit_procrustes_linear_map(
        best_target_tensor, best_trainer_tensor
    )
    pred_target_to_trainer = best_target_tensor @ A_target_to_trainer + b_target_to_trainer
    train_mse_target_to_trainer = torch.mean(
        (pred_target_to_trainer - best_trainer_tensor) ** 2
    ).item()
    model_target_to_trainer = build_linear_regressor_from_map(
        A_target_to_trainer, b_target_to_trainer
    )

    # Direction B: Trainer -> Target (example: 28 -> 16)
    A_trainer_to_target, b_trainer_to_target, scale_trainer_to_target = fit_procrustes_linear_map(
        best_trainer_tensor, best_target_tensor
    )
    pred_trainer_to_target = best_trainer_tensor @ A_trainer_to_target + b_trainer_to_target
    train_mse_trainer_to_target = torch.mean(
        (pred_trainer_to_target - best_target_tensor) ** 2
    ).item()
    model_trainer_to_target = build_linear_regressor_from_map(
        A_trainer_to_target, b_trainer_to_target
    )

    # Save both directions.
    target_to_trainer_path = os.path.join(
        aligner_info["output_dir"], "procrustes_target_to_trainer.pkl"
    )
    trainer_to_target_path = os.path.join(
        aligner_info["output_dir"], "procrustes_trainer_to_target.pkl"
    )
    # Legacy filename kept for inference compatibility: defaults to Trainer -> Target.
    legacy_path = os.path.join(aligner_info["output_dir"], "procrustes.pkl")

    with open(target_to_trainer_path, "wb") as f:
        pickle.dump(model_target_to_trainer, f)
    with open(trainer_to_target_path, "wb") as f:
        pickle.dump(model_trainer_to_target, f)
    with open(legacy_path, "wb") as f:
        pickle.dump(model_trainer_to_target, f)

    print(f"Saved Procrustes Target -> Trainer to {target_to_trainer_path}")
    print(f"Saved Procrustes Trainer -> Target to {trainer_to_target_path}")
    print(f"Saved legacy Procrustes (Trainer -> Target) to {legacy_path}")

    # Save layer mapping info for later use.
    mapping_info = {
        "trainer_model": trainer_model_safe,
        "target_model": target_model_safe,
        "dataset": dataset_name,
        "trainer_num_physical_layers": trainer_num_physical,
        "target_num_physical_layers": target_num_physical,
        "selection_strategy": "target_top1_then_cka_over_all_trainer_same_type",
        "top1_target": {
            "physical_layer": target_layer,
            "module_type": module_type,
            "probing_accuracy": top1_target.get("probing_accuracy"),
        },
        "best_trainer_layer_by_cka": {
            "physical_layer": best_trainer_layer,
            "module_type": module_type,
            "cka": best_score,
        },
        "num_common_instances": len(best_common_ids),
        "cka_scores": cka_scores,
        "layer_mapping": [
            {
                "trainer_layer": best_trainer_layer,
                "trainer_type": module_type,
                "target_layer": target_layer,
                "target_type": module_type,
            }
        ],
        "trainer_shape": list(best_trainer_tensor.shape),
        "target_shape": list(best_target_tensor.shape),
        "procrustes": {
            "target_to_trainer": {
                "scale": scale_target_to_trainer,
                "train_mse": train_mse_target_to_trainer,
                "input_dim": int(A_target_to_trainer.shape[0]),
                "output_dim": int(A_target_to_trainer.shape[1]),
                "model_path": target_to_trainer_path,
            },
            "trainer_to_target": {
                "scale": scale_trainer_to_target,
                "train_mse": train_mse_trainer_to_target,
                "input_dim": int(A_trainer_to_target.shape[0]),
                "output_dim": int(A_trainer_to_target.shape[1]),
                "model_path": trainer_to_target_path,
            },
            "legacy_default_model_path": legacy_path,
            "legacy_default_direction": "trainer_to_target",
        },
    }
    mapping_path = os.path.join(aligner_info["output_dir"], "layer_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(mapping_info, f, indent=2)
    
    print(f"Saved layer mapping to {mapping_path}")
    print(f"Aligner {aligner_name} completed.")
