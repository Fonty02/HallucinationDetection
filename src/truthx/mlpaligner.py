import os
import json
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
from tqdm import tqdm
import pickle

from MLP import MLP


def get_num_physical_layers(base_dir, activation_type="attn"):
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
    layer_data = {}
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
    target_layers = []
    for phys_layer, mod_type in trainer_layers:
        if trainer_num_physical <= 1:
            target_layer = 0
        else:
            relative_depth = phys_layer / (trainer_num_physical - 1)
            target_layer = round(relative_depth * (target_num_physical - 1))
        target_layers.append((target_layer, mod_type))
    return target_layers


ALIGNERS = {
    "Llama_to_Gemma_BBF":
    {
        "Trainer": "Llama-3.1-8B-Instruct",
        "Target": "gemma-2-9b-it",
        "dataset":"belief_bank_facts",
        "source": "activation_cache_truthx/google_gemma-2-9b-it/belief_bank_facts_subset",
        "target": "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/belief_bank_facts_subset",
    },
    "Llama_to_Gemma_BBC":
    {
        "Trainer": "Llama-3.1-8B-Instruct",
        "Target": "gemma-2-9b-it",
        "dataset":"belief_bank_constraints",
        "source": "activation_cache_truthx/google_gemma-2-9b-it/belief_bank_constraints_subset",
        "target": "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/belief_bank_constraints_subset",
    },
    "Llama_to_Gemma_HE":
    {
        "Trainer": "Llama-3.1-8B-Instruct",
        "Target": "gemma-2-9b-it",
        "dataset":"halu_eval",
        "source": "activation_cache_truthx/google_gemma-2-9b-it/halu_eval_subset",
        "target": "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/halu_eval_subset",
    },
    "Gemma_to_Llama_BBF":
    {
        "Trainer": "gemma-2-9b-it",
        "Target": "Llama-3.1-8B-Instruct",
        "dataset":"belief_bank_facts",
        "source": "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/belief_bank_facts_subset",
        "target": "activation_cache_truthx/google_gemma-2-9b-it/belief_bank_facts_subset",
    },
    "Gemma_to_Llama_BBC":
    {
        "Trainer": "gemma-2-9b-it",
        "Target": "Llama-3.1-8B-Instruct",
        "dataset":"belief_bank_constraints",
        "source": "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/belief_bank_constraints_subset",
        "target": "activation_cache_truthx/google_gemma-2-9b-it/belief_bank_constraints_subset",
    },
    "Gemma_to_Llama_HE":
    {
        "Trainer": "gemma-2-9b-it",
        "Target": "Llama-3.1-8B-Instruct",
        "dataset":"halu_eval",
        "source": "activation_cache_truthx/meta-llama_Llama-3.1-8B-Instruct/halu_eval_subset",
        "target": "activation_cache_truthx/google_gemma-2-9b-it/halu_eval_subset",
    }
}

STEERING_DIR = "SteeringVectors"


def train_mlp(X_train, y_train, X_val, y_val, input_size, output_size, device,
              epochs=1000, patience=50, lr=1e-3, batch_size=512):
    hidden_size = 1024
    model = MLP(input_size, hidden_size, output_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val = float('inf')
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            count += xb.size(0)
        train_loss = running / max(1, count)

        model.eval()
        with torch.no_grad():
            running_val = 0.0
            count_val = 0
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                running_val += loss.item() * xb.size(0)
                count_val += xb.size(0)
            val_loss = running_val / max(1, count_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-12:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 50 == 0 or epoch <= 5:
            print(f"  Epoch {epoch:4d} | train_loss={train_loss:.6e} val_loss={val_loss:.6e} best_val={best_val:.6e} wait={wait}")

        if wait >= patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_val, epoch


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for aligner_name, aligner_info in ALIGNERS.items():
        print(f"\nProcessing MLP aligner: {aligner_name}")
        out_base = os.path.join("alignment_cache_mlp", aligner_name)
        os.makedirs(out_base, exist_ok=True)

        trainer_model_safe = aligner_info["source"].split("/")[1]
        target_model_safe = aligner_info["target"].split("/")[1]
        dataset_name = aligner_info["dataset"]

        print(f"Loading trainer layer selection for {trainer_model_safe}...")
        trainer_modules, trainer_total_modules = load_layer_selection(
            STEERING_DIR, dataset_name, trainer_model_safe
        )
        trainer_num_physical = trainer_total_modules // 2

        top1 = trainer_modules[0]
        trainer_selected = [(top1["physical_layer"], top1["module_type"])]
        print(f"  Trainer top-1 module: physical {top1['physical_layer']} {top1['module_type']} (accuracy: {top1['probing_accuracy']:.4f})")
        print(f"  Trainer num physical layers: {trainer_num_physical}")

        target_num_physical = get_num_physical_layers(aligner_info["target"], "attn")
        print(f"  Target num physical layers: {target_num_physical}")

        target_selected = map_layers_proportionally(
            trainer_selected, trainer_num_physical, target_num_physical
        )

        print("\nLoading trainer activations (selected layers only)...")
        trainer = load_activations_for_layers(aligner_info["source"], trainer_selected)

        print("Loading target activations (proportionally mapped layers)...")
        target = load_activations_for_layers(aligner_info["target"], target_selected)

        print(f"Trainer activations shape: {trainer.shape}")
        print(f"Target activations shape: {target.shape}")

        # Ensure float tensors
        X = target.float()
        Y = trainer.float()

        n = X.shape[0]
        if n < 2:
            print(f"  Not enough samples ({n}) to train MLP. Skipping.")
            continue

        perm = torch.randperm(n)
        train_n = int(0.7 * n)
        train_idx = perm[:train_n]
        val_idx = perm[train_n:]

        X_train = X[train_idx]
        y_train = Y[train_idx]
        X_val = X[val_idx]
        y_val = Y[val_idx]

        print(f"  Split: train={len(train_idx)} val={len(val_idx)} (70/30)")

        model, history, best_val, epochs_trained = train_mlp(
            X_train, y_train, X_val, y_val,
            input_size=X.shape[1],
            output_size=Y.shape[1],
            device=device,
            epochs=1000,
            patience=50,
            lr=1e-3,
            batch_size=512,
        )

        # Save model checkpoint
        ckpt_path = os.path.join(out_base, "mlp_regressor.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_size': X.shape[1],
            'output_size': Y.shape[1],
            'hidden_size': max(X.shape[1], Y.shape[1])
        }, ckpt_path)

        # Save training history
        history_path = os.path.join(out_base, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)

        # Save layer mapping and metadata
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
            "mlp_training": {
                "optimizer": "AdamW",
                "loss": "MSE",
                "epochs_run": int(epochs_trained),
                "best_val_loss": float(best_val),
                "patience": 50,
                "split": "70-30",
                "seed": 42,
            }
        }
        mapping_path = os.path.join(out_base, "layer_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(mapping_info, f, indent=2)

        print(f"Saved MLP model to {ckpt_path}")
        print(f"Saved mapping + metadata to {mapping_path}")

    print("\nAll MLP aligners completed.")
