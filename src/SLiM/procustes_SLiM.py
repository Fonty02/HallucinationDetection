import glob
import json
import os
import pickle
import re

import torch
from sklearn.linear_model import LinearRegression


def parse_activation_filename(fname):
    """
    Parse expected activation filename format: layer{idx}-id{instance}.pt
    Returns (layer_idx, instance_id) or None if format is invalid.
    """
    if not fname.endswith(".pt"):
        return None

    parts = fname[:-3].split("-")
    if len(parts) < 2:
        return None
    if not parts[0].startswith("layer") or not parts[1].startswith("id"):
        return None

    try:
        layer_idx = int(parts[0][5:])
        instance_id = int(parts[1][2:])
    except ValueError:
        return None

    return layer_idx, instance_id


def get_available_physical_layers(base_dir, activation_type="hidden"):
    """
    Return sorted list of physical layer indices available in activation files.
    """
    act_dir = os.path.join(base_dir, f"activation_{activation_type}")
    if not os.path.exists(act_dir):
        raise FileNotFoundError(f"Activation dir not found: {act_dir}")

    layers = set()
    for fname in os.listdir(act_dir):
        parsed = parse_activation_filename(fname)
        if parsed is None:
            continue
        layer_idx, _ = parsed
        layers.add(layer_idx)

    if not layers:
        raise ValueError(f"No valid activation files found in: {act_dir}")
    return sorted(layers)


def get_num_physical_layers(base_dir, activation_type="hidden"):
    """
    Estimate number of physical layers as max_layer + 1.
    """
    layers = get_available_physical_layers(base_dir, activation_type)
    return max(layers) + 1


def load_latest_slim_config(steering_slim_dir, dataset_name, model_name_safe):
    """
    Load latest SLiM training config for model+dataset and return selected layer.
    """
    search_pattern = os.path.join(
        steering_slim_dir,
        model_name_safe,
        dataset_name,
        "slim_*_config.json",
    )
    matches = glob.glob(search_pattern)
    if not matches:
        raise FileNotFoundError(f"No SLiM config found matching: {search_pattern}")

    config_path = max(matches, key=os.path.getmtime)
    print(f"  Using SLiM config: {os.path.basename(config_path)}")

    with open(config_path, "r") as f:
        config = json.load(f)

    target_layer = config.get("target_layer")
    if target_layer is None:
        # Fallback: parse from filename token "..._layer{n}_..."
        name_match = re.search(r"_layer(\d+)_", os.path.basename(config_path))
        if name_match is None:
            raise ValueError(
                f"target_layer missing in config and not inferable from filename: {config_path}"
            )
        target_layer = int(name_match.group(1))

    module_type = config.get("module_type", "hidden")
    if module_type == "ffn":
        module_type = "mlp"
    if module_type not in {"hidden", "attn", "mlp"}:
        print(f"  Warning: unsupported module_type '{module_type}', falling back to 'hidden'")
        module_type = "hidden"

    return int(target_layer), module_type, config_path, config


def load_single_layer_activations(base_dir, layer_idx, module_type):
    """
    Load all activations for one physical layer and one module type.
    Returns dict: instance_id -> flattened float tensor.
    """
    act_dir = os.path.join(base_dir, f"activation_{module_type}")
    if not os.path.exists(act_dir):
        raise FileNotFoundError(f"Activation dir not found: {act_dir}")

    data = {}
    for fname in os.listdir(act_dir):
        parsed = parse_activation_filename(fname)
        if parsed is None:
            continue
        current_layer, instance_id = parsed
        if current_layer != layer_idx:
            continue

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


ALIGNERS = {
    "Llama_to_Gemma_BBF_PROCRUSTES": {
        "Trainer": "Llama-3.1-8B-Instruct",
        "Target": "gemma-2-9b-it",
        "dataset": "belief_bank_facts",
        "trainer_activation_dir": "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/belief_bank_facts_subset",
        "target_activation_dir": "activation_cache_truthx/google_gemma-2-9b-it/belief_bank_facts_subset",
        "output_dir": "alignment_procustes_SLiM/Llama_to_Gemma_BBF_PROCRUSTES",
    },
    "Llama_to_Gemma_BBC_PROCRUSTES": {
        "Trainer": "Llama-3.1-8B-Instruct",
        "Target": "gemma-2-9b-it",
        "dataset": "belief_bank_constraints",
        "trainer_activation_dir": "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/belief_bank_constraints_subset",
        "target_activation_dir": "activation_cache_truthx/google_gemma-2-9b-it/belief_bank_constraints_subset",
        "output_dir": "alignment_procustes_SLiM/Llama_to_Gemma_BBC_PROCRUSTES",
    },
    "Llama_to_Gemma_HE_PROCRUSTES": {
        "Trainer": "Llama-3.1-8B-Instruct",
        "Target": "gemma-2-9b-it",
        "dataset": "halu_eval",
        "trainer_activation_dir": "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/halu_eval_subset",
        "target_activation_dir": "activation_cache_truthx/google_gemma-2-9b-it/halu_eval_subset",
        "output_dir": "alignment_procustes_SLiM/Llama_to_Gemma_HE_PROCRUSTES",
    },
    "Gemma_to_Llama_BBF_PROCRUSTES": {
        "Trainer": "gemma-2-9b-it",
        "Target": "Llama-3.1-8B-Instruct",
        "dataset": "belief_bank_facts",
        "trainer_activation_dir": "activation_cache_truthx/google_gemma-2-9b-it/belief_bank_facts_subset",
        "target_activation_dir": "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/belief_bank_facts_subset",
        "output_dir": "alignment_procustes_SLiM/Gemma_to_Llama_BBF_PROCRUSTES",
    },
    "Gemma_to_Llama_BBC_PROCRUSTES": {
        "Trainer": "gemma-2-9b-it",
        "Target": "Llama-3.1-8B-Instruct",
        "dataset": "belief_bank_constraints",
        "trainer_activation_dir": "activation_cache_truthx/google_gemma-2-9b-it/belief_bank_constraints_subset",
        "target_activation_dir": "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/belief_bank_constraints_subset",
        "output_dir": "alignment_procustes_SLiM/Gemma_to_Llama_BBC_PROCRUSTES",
    },
    "Gemma_to_Llama_HE_PROCRUSTES": {
        "Trainer": "gemma-2-9b-it",
        "Target": "Llama-3.1-8B-Instruct",
        "dataset": "halu_eval",
        "trainer_activation_dir": "activation_cache_truthx/google_gemma-2-9b-it/halu_eval_subset",
        "target_activation_dir": "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/halu_eval_subset",
        "output_dir": "alignment_procustes_SLiM/Gemma_to_Llama_HE_PROCRUSTES",
    },
}


STEERING_SLIM_DIR = "SteeringVectors/SLiM"


def run_single_aligner(aligner_name, aligner_info):
    print(f"\nProcessing aligner: {aligner_name}")
    os.makedirs(aligner_info["output_dir"], exist_ok=True)

    trainer_activation_dir = aligner_info["trainer_activation_dir"]
    target_activation_dir = aligner_info["target_activation_dir"]

    trainer_model_safe = trainer_activation_dir.split("/")[1]
    target_model_safe = target_activation_dir.split("/")[1]
    dataset_name = aligner_info["dataset"]

    print(f"Loading SLiM target layer for {target_model_safe}...")
    target_layer, module_type, slim_config_path, slim_config = load_latest_slim_config(
        STEERING_SLIM_DIR, dataset_name, target_model_safe
    )
    print(
        f"  Target manual layer: physical {target_layer} {module_type} "
        f"(from SLiM config)"
    )

    trainer_layers_available = get_available_physical_layers(
        trainer_activation_dir, module_type
    )
    trainer_num_physical = max(trainer_layers_available) + 1

    target_layers_available = get_available_physical_layers(
        target_activation_dir, module_type
    )
    target_num_physical = max(target_layers_available) + 1

    print(
        f"  Trainer available physical layers ({module_type}): "
        f"{len(trainer_layers_available)} / inferred {trainer_num_physical}"
    )
    print(
        f"  Target available physical layers ({module_type}): "
        f"{len(target_layers_available)} / inferred {target_num_physical}"
    )

    print("\nLoading target selected-layer activations...")
    target_selected_dict = load_single_layer_activations(
        target_activation_dir, target_layer, module_type
    )

    print(f"Running CKA over trainer {module_type} layers...")
    cka_scores = []
    best_score = float("-inf")
    best_trainer_layer = None
    best_trainer_tensor = None
    best_target_tensor = None
    best_common_ids = None

    for trainer_layer in trainer_layers_available:
        trainer_dict = load_single_layer_activations(
            trainer_activation_dir, trainer_layer, module_type
        )
        trainer_tensor, target_tensor, common_ids = stack_common_instances(
            trainer_dict, target_selected_dict
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
        print(
            f"  CKA trainer {module_type} layer {trainer_layer} "
            f"vs target layer {target_layer}: {score:.6f}"
        )

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
    print(
        "Aligned training tensors shape: "
        f"trainer={best_trainer_tensor.shape}, target={best_target_tensor.shape}"
    )

    print("Training Procrustes aligners (Target -> Trainer and Trainer -> Target)...")

    A_target_to_trainer, b_target_to_trainer, scale_target_to_trainer = (
        fit_procrustes_linear_map(best_target_tensor, best_trainer_tensor)
    )
    pred_target_to_trainer = best_target_tensor @ A_target_to_trainer + b_target_to_trainer
    train_mse_target_to_trainer = torch.mean(
        (pred_target_to_trainer - best_trainer_tensor) ** 2
    ).item()
    model_target_to_trainer = build_linear_regressor_from_map(
        A_target_to_trainer, b_target_to_trainer
    )

    A_trainer_to_target, b_trainer_to_target, scale_trainer_to_target = (
        fit_procrustes_linear_map(best_trainer_tensor, best_target_tensor)
    )
    pred_trainer_to_target = best_trainer_tensor @ A_trainer_to_target + b_trainer_to_target
    train_mse_trainer_to_target = torch.mean(
        (pred_trainer_to_target - best_target_tensor) ** 2
    ).item()
    model_trainer_to_target = build_linear_regressor_from_map(
        A_trainer_to_target, b_trainer_to_target
    )

    target_to_trainer_path = os.path.join(
        aligner_info["output_dir"], "procrustes_target_to_trainer.pkl"
    )
    trainer_to_target_path = os.path.join(
        aligner_info["output_dir"], "procrustes_trainer_to_target.pkl"
    )
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

    mapping_info = {
        "trainer_model": trainer_model_safe,
        "target_model": target_model_safe,
        "dataset": dataset_name,
        "trainer_num_physical_layers": trainer_num_physical,
        "target_num_physical_layers": target_num_physical,
        "selection_strategy": "manual_target_layer_from_slim_config_then_cka_over_all_trainer_same_type",
        "target_from_slim_config": {
            "physical_layer": target_layer,
            "module_type": module_type,
            "slim_config_path": slim_config_path,
        },
        "slim_config": {
            "num_pairs": slim_config.get("num_pairs"),
            "epochs": slim_config.get("epochs"),
            "batch_size": slim_config.get("batch_size"),
            "lr": slim_config.get("lr"),
            "target_layer": slim_config.get("target_layer"),
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
    with open(mapping_path, "w") as f:
        json.dump(mapping_info, f, indent=2)

    print(f"Saved layer mapping to {mapping_path}")
    print(f"Aligner {aligner_name} completed.")


def main():
    for aligner_name, aligner_info in ALIGNERS.items():
        run_single_aligner(aligner_name, aligner_info)


if __name__ == "__main__":
    main()
