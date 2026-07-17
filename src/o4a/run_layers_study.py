"""Layer-wise hallucination study with multi-seed train/test splits.

For a given (model, dataset, layer_type), this script:
1) loads activations layer-by-layer from activation_cache
2) builds 10 deterministic stratified train/test splits (via split seeds),
   then undersamples only the training portion for balanced classifier training
3) trains Logistic Regression per layer and per split seed
4) stores per-seed metrics and aggregated stats in a JSON file
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR_NAME = "activation_cache"
DEFAULT_SPLIT_SEEDS = [2, 4, 24, 42, 64, 67, 104, 123, 420, 511]
LAYER_TYPES = ["attn", "mlp", "hidden"]
MODEL_ALIASES = {
    "LLama3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
}
METRIC_NAMES = ["precision", "recall", "accuracy", "f1", "auroc"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_model_name(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name, model_name)


def parse_split_seeds(raw_value: str) -> list[int]:
    if not raw_value.strip():
        return []
    tokens = [token.strip() for token in raw_value.replace(";", ",").split(",")]
    seeds: list[int] = []
    for token in tokens:
        if not token:
            continue
        seed = int(token)
        if seed not in seeds:
            seeds.append(seed)
    return seeds


def get_activation_dir(model_name: str, dataset_name: str, layer_type: str) -> Path:
    resolved_model = resolve_model_name(model_name)
    activation_dir = PROJECT_ROOT / CACHE_DIR_NAME / resolved_model / dataset_name / f"activation_{layer_type}"
    if not activation_dir.exists():
        raise FileNotFoundError(f"Activation directory not found: {activation_dir}")
    return activation_dir


def detect_structure_type(model_name: str, dataset_name: str, layer_type: str) -> str:
    activation_dir = get_activation_dir(model_name, dataset_name, layer_type)
    if (activation_dir / "hallucinated").is_dir():
        return "new"
    return "old"


def extract_layer_index(file_name: str) -> int | None:
    match = re.match(r"layer(\d+)_activations\.pt$", file_name)
    if match is None:
        return None
    return int(match.group(1))


def list_available_layers(model_name: str, dataset_name: str, layer_type: str) -> list[int]:
    structure = detect_structure_type(model_name, dataset_name, layer_type)
    activation_dir = get_activation_dir(model_name, dataset_name, layer_type)
    source_dir = activation_dir / "hallucinated" if structure == "new" else activation_dir

    layers: list[int] = []
    for file_path in source_dir.glob("layer*_activations.pt"):
        layer_idx = extract_layer_index(file_path.name)
        if layer_idx is not None:
            layers.append(layer_idx)

    unique_layers = sorted(set(layers))
    if not unique_layers:
        raise RuntimeError(
            f"No layer activation files found for model={model_name}, "
            f"dataset={dataset_name}, layer_type={layer_type} in {source_dir}"
        )
    return unique_layers


def _to_numpy_float32(tensor_or_array: object) -> np.ndarray:
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.cpu().numpy().astype(np.float32)
    return np.asarray(tensor_or_array, dtype=np.float32)


def load_activations_and_labels(
    model_name: str,
    dataset_name: str,
    layer_type: str,
    layer_idx: int,
    map_location: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    structure = detect_structure_type(model_name, dataset_name, layer_type)
    activation_dir = get_activation_dir(model_name, dataset_name, layer_type)

    if structure == "new":
        hall_act_path = activation_dir / "hallucinated" / f"layer{layer_idx}_activations.pt"
        hall_ids_path = activation_dir / "hallucinated" / f"layer{layer_idx}_instance_ids.json"
        non_hall_act_path = activation_dir / "not_hallucinated" / f"layer{layer_idx}_activations.pt"
        non_hall_ids_path = activation_dir / "not_hallucinated" / f"layer{layer_idx}_instance_ids.json"

        hall_activations = torch.load(hall_act_path, map_location=map_location)
        non_hall_activations = torch.load(non_hall_act_path, map_location=map_location)

        with open(hall_ids_path, "r", encoding="utf-8") as file:
            hall_ids = json.load(file)
        with open(non_hall_ids_path, "r", encoding="utf-8") as file:
            non_hall_ids = json.load(file)

        hall_np = _to_numpy_float32(hall_activations)
        non_hall_np = _to_numpy_float32(non_hall_activations)

        x_concat = np.vstack([hall_np, non_hall_np])
        y_concat = np.concatenate(
            [
                np.ones(hall_np.shape[0], dtype=np.int64),
                np.zeros(non_hall_np.shape[0], dtype=np.int64),
            ]
        )
        ids_concat = np.asarray(hall_ids + non_hall_ids)
        sort_idx = np.argsort(ids_concat)
        return x_concat[sort_idx], y_concat[sort_idx], ids_concat[sort_idx]

    act_path = activation_dir / f"layer{layer_idx}_activations.pt"
    activations = torch.load(act_path, map_location=map_location)
    x_np = _to_numpy_float32(activations)

    resolved_model = resolve_model_name(model_name)
    labels_path = (
        PROJECT_ROOT
        / CACHE_DIR_NAME
        / resolved_model
        / dataset_name
        / "generations"
        / "hallucination_labels.json"
    )
    with open(labels_path, "r", encoding="utf-8") as file:
        labels_data = json.load(file)

    y_np = np.asarray([int(item["is_hallucination"]) for item in labels_data], dtype=np.int64)
    ids_np = np.arange(len(y_np))
    return x_np, y_np, ids_np


def get_balanced_indices(y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    unique_classes, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min())

    selected_indices: list[int] = []
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        if len(cls_indices) > min_count:
            sampled = rng.choice(cls_indices, size=min_count, replace=False)
            selected_indices.extend(sampled.tolist())
        else:
            selected_indices.extend(cls_indices.tolist())

    return np.sort(np.asarray(selected_indices, dtype=np.int64))


def build_split_indices(
    y_reference: np.ndarray,
    split_seeds: list[int],
    test_size: float,
) -> dict[int, dict[str, object]]:
    split_cache: dict[int, dict[str, object]] = {}

    for split_seed in split_seeds:
        # Step 1: stratified train/test split on ALL data (preserves original proportions)
        all_idx = np.arange(len(y_reference))
        train_rel_idx, test_rel_idx = train_test_split(
            all_idx,
            test_size=test_size,
            random_state=split_seed,
            stratify=y_reference,
        )
        # Step 2: undersample only the training portion (balanced 50/50 for classifier)
        y_train_part = y_reference[train_rel_idx]
        bal_train = get_balanced_indices(y_train_part, split_seed)
        train_indices = train_rel_idx[bal_train]
        test_indices = test_rel_idx

        split_cache[split_seed] = {
            "train_indices": train_indices,
            "test_indices": test_indices,
            "train_class_distribution": {
                "hallucination_1": int(np.sum(y_train_part[bal_train] == 1)),
                "not_hallucination_0": int(np.sum(y_train_part[bal_train] == 0)),
            },
            "test_class_distribution": {
                "hallucination_1": int(np.sum(y_reference[test_indices] == 1)),
                "not_hallucination_0": int(np.sum(y_reference[test_indices] == 0)),
            },
            "train_size": int(len(train_indices)),
            "test_size": int(len(test_indices)),
        }
    return split_cache


def sanitize_metric_value(value: float) -> float | None:
    if np.isnan(value):
        return None
    return float(value)


def compute_layer_metrics(
    x_layer: np.ndarray,
    y_layer: np.ndarray,
    split_cache: dict[int, dict[str, object]],
    max_iter: int,
    logreg_n_jobs: int,
) -> tuple[list[dict[str, object]], dict[str, float | None], dict[str, float | None]]:
    per_seed_results: list[dict[str, object]] = []
    per_metric_values: dict[str, list[float]] = {metric: [] for metric in METRIC_NAMES}

    for split_seed, split_data in split_cache.items():
        train_indices = np.asarray(split_data["train_indices"], dtype=np.int64)
        test_indices = np.asarray(split_data["test_indices"], dtype=np.int64)

        x_train = x_layer[train_indices]
        y_train = y_layer[train_indices]
        x_test = x_layer[test_indices]
        y_test = y_layer[test_indices]

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        clf = LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=logreg_n_jobs,
        )
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)
        y_proba = clf.predict_proba(x_test)[:, 1]

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            auroc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auroc = float("nan")

        metric_payload = {
            "precision": sanitize_metric_value(float(precision)),
            "recall": sanitize_metric_value(float(recall)),
            "accuracy": sanitize_metric_value(float(accuracy)),
            "f1": sanitize_metric_value(float(f1)),
            "auroc": sanitize_metric_value(float(auroc)),
        }
        per_seed_results.append(
            {
                "split_seed": int(split_seed),
                "train_size": int(split_data["train_size"]),
                "test_size": int(split_data["test_size"]),
                "train_class_distribution": split_data["train_class_distribution"],
                "test_class_distribution": split_data["test_class_distribution"],
                "metrics": metric_payload,
            }
        )

        for metric_name, metric_value in metric_payload.items():
            if metric_value is not None:
                per_metric_values[metric_name].append(metric_value)

        del x_train, y_train, x_test, y_test, scaler, clf
        gc.collect()

    mean_metrics: dict[str, float | None] = {}
    std_metrics: dict[str, float | None] = {}
    for metric_name in METRIC_NAMES:
        metric_values = per_metric_values[metric_name]
        if not metric_values:
            mean_metrics[metric_name] = None
            std_metrics[metric_name] = None
            continue
        metric_np = np.asarray(metric_values, dtype=np.float64)
        mean_metrics[metric_name] = float(np.mean(metric_np))
        std_metrics[metric_name] = float(np.std(metric_np))

    return per_seed_results, mean_metrics, std_metrics


def build_rankings(layer_results: list[dict[str, object]], metric_name: str) -> list[dict[str, object]]:
    ranking_rows: list[dict[str, object]] = []
    for item in layer_results:
        mean_metrics = item["mean_metrics"]
        std_metrics = item["std_metrics"]
        ranking_rows.append(
            {
                "layer": int(item["layer"]),
                metric_name: mean_metrics.get(metric_name),
                f"{metric_name}_std": std_metrics.get(metric_name),
            }
        )

    ranking_rows.sort(
        key=lambda row: float("-inf") if row[metric_name] is None else float(row[metric_name]),
        reverse=True,
    )
    return ranking_rows


def run_study(
    model_name: str,
    dataset_name: str,
    layer_type: str,
    split_seeds: list[int],
    test_size: float,
    device: str,
    max_iter: int,
    logreg_n_jobs: int,
) -> dict[str, object]:
    set_seed(42)
    available_layers = list_available_layers(model_name, dataset_name, layer_type)
    structure_type = detect_structure_type(model_name, dataset_name, layer_type)

    print("=" * 80)
    print(f"Layer study start: model={model_name} dataset={dataset_name} layer_type={layer_type}")
    print(f"Structure type: {structure_type}")
    print(f"Available layers: {len(available_layers)} -> {available_layers[0]}..{available_layers[-1]}")
    print(f"Split seeds ({len(split_seeds)}): {split_seeds}")
    print("=" * 80)

    first_layer = available_layers[0]
    _, y_reference, ids_reference = load_activations_and_labels(
        model_name=model_name,
        dataset_name=dataset_name,
        layer_type=layer_type,
        layer_idx=first_layer,
        map_location=device,
    )
    split_cache = build_split_indices(y_reference=y_reference, split_seeds=split_seeds, test_size=test_size)

    layer_results: list[dict[str, object]] = []

    for pos, layer_idx in enumerate(available_layers, start=1):
        print(f"[{pos}/{len(available_layers)}] Processing layer {layer_idx} ...")
        x_layer, y_layer, ids_layer = load_activations_and_labels(
            model_name=model_name,
            dataset_name=dataset_name,
            layer_type=layer_type,
            layer_idx=layer_idx,
            map_location=device,
        )

        if not np.array_equal(y_reference, y_layer):
            raise RuntimeError(
                f"Label mismatch between reference layer {first_layer} and layer {layer_idx} "
                f"for model={model_name}, dataset={dataset_name}, layer_type={layer_type}."
            )
        if not np.array_equal(ids_reference, ids_layer):
            raise RuntimeError(
                f"Instance ID ordering mismatch between reference layer {first_layer} and layer {layer_idx} "
                f"for model={model_name}, dataset={dataset_name}, layer_type={layer_type}."
            )

        per_seed_results, mean_metrics, std_metrics = compute_layer_metrics(
            x_layer=x_layer,
            y_layer=y_layer,
            split_cache=split_cache,
            max_iter=max_iter,
            logreg_n_jobs=logreg_n_jobs,
        )

        layer_results.append(
            {
                "layer": int(layer_idx),
                "num_samples": int(x_layer.shape[0]),
                "per_seed": per_seed_results,
                "mean_metrics": mean_metrics,
                "std_metrics": std_metrics,
            }
        )

        del x_layer, y_layer, ids_layer
        gc.collect()

    rankings = {
        "by_mean_accuracy_desc": build_rankings(layer_results, "accuracy"),
        "by_mean_f1_desc": build_rankings(layer_results, "f1"),
        "by_mean_auroc_desc": build_rankings(layer_results, "auroc"),
    }

    split_summary = {
        str(split_seed): {
            "train_size": int(split_data["train_size"]),
            "test_size": int(split_data["test_size"]),
            "train_class_distribution": split_data["train_class_distribution"],
            "test_class_distribution": split_data["test_class_distribution"],
        }
        for split_seed, split_data in split_cache.items()
    }

    return {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "project_root": str(PROJECT_ROOT),
            "cache_dir_name": CACHE_DIR_NAME,
            "model_requested": model_name,
            "model_resolved": resolve_model_name(model_name),
            "dataset": dataset_name,
            "layer_type": layer_type,
            "structure_type": structure_type,
            "available_layers": available_layers,
            "num_layers": len(available_layers),
            "split_seeds": split_seeds,
            "test_size": test_size,
            "device": device,
            "logistic_regression": {
                "solver": "lbfgs",
                "class_weight": "balanced",
                "max_iter": max_iter,
                "n_jobs": logreg_n_jobs,
            },
        },
        "split_summary": split_summary,
        "layers": layer_results,
        "rankings": rankings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run layer-wise hallucination study for one model/dataset/layer_type "
            "using multiple split seeds and save JSON metrics."
        )
    )
    parser.add_argument("--model", required=True, help="Model folder name in activation_cache.")
    parser.add_argument("--dataset", required=True, help="Dataset folder name.")
    parser.add_argument("--layer-type", required=True, choices=LAYER_TYPES, help="Activation layer type.")
    parser.add_argument(
        "--split-seeds",
        default=",".join(str(seed) for seed in DEFAULT_SPLIT_SEEDS),
        help="Comma-separated split seeds (default: standard 10 O4A seeds).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Test split ratio on original imbalanced data (default: 0.3).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch.load map_location value (default: cpu).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10000,
        help="Max iterations for LogisticRegression (default: 10000).",
    )
    parser.add_argument(
        "--logreg-n-jobs",
        type=int,
        default=-1,
        help="n_jobs for LogisticRegression (default: -1).",
    )
    parser.add_argument("--output", required=True, help="Output JSON file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_seeds = parse_split_seeds(args.split_seeds)
    if not split_seeds:
        raise ValueError("No split seeds provided. Use --split-seeds with at least one integer seed.")
    if not (0.0 < args.test_size < 1.0):
        raise ValueError(f"test_size must be in (0,1), got {args.test_size}")

    output_path = Path(args.output).resolve()

    # Check if already completed
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if "layers" in existing and len(existing["layers"]) > 0 and "rankings" in existing:
                print(f"[SKIP] Output already exists and appears complete: {output_path}")
                print(f"       ({len(existing['layers'])} layers, {len(existing.get('rankings', {}))} rankings)")
                return
        except Exception:
            print(f"[RESUME] Output exists but is incomplete/corrupt, re-running: {output_path}")

    result_payload = run_study(
        model_name=args.model,
        dataset_name=args.dataset,
        layer_type=args.layer_type,
        split_seeds=split_seeds,
        test_size=args.test_size,
        device=args.device,
        max_iter=args.max_iter,
        logreg_n_jobs=args.logreg_n_jobs,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result_payload, file, indent=2)

    print("=" * 80)
    print(f"Study completed. JSON written to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
