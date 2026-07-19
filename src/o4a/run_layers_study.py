"""Layer-wise hallucination study — standard + automatic cross-domain — CSV output.

For a given (model, dataset, layer_type):
1) Runs the standard layer study (train on dataset, eval on same dataset test split)
2) Automatically discovers other datasets in activation_cache for the same model
   and runs cross-domain evaluation for each (train on --dataset, eval on each other dataset)
3) Writes one flat CSV with per-seed rows + mean/std aggregate rows
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import re
import time
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

CSV_COLUMNS = [
    "model",
    "train_dataset",
    "layer_type",
    "layer",
    "num_samples",
    "eval_dataset",
    "split_seed",
    "train_size",
    "test_size",
    "train_hall",
    "train_not_hall",
    "test_hall",
    "test_not_hall",
    "fit_time_s",
    "model_params",
    "precision",
    "recall",
    "accuracy",
    "f1",
    "auroc",
    "total_runtime_s",
    "num_splits",
    "created_at_utc",
]


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
    activation_dir = (
        PROJECT_ROOT / CACHE_DIR_NAME / resolved_model / dataset_name / f"activation_{layer_type}"
    )
    if not activation_dir.exists():
        raise FileNotFoundError(f"Activation directory not found: {activation_dir}")
    return activation_dir


def detect_structure_type(model_name: str, dataset_name: str, layer_type: str) -> str:
    activation_dir = get_activation_dir(model_name, dataset_name, layer_type)
    if (activation_dir / "hallucinated").is_dir():
        return "new"
    return "old"


def discover_other_datasets(model_name: str) -> list[str]:
    """List all dataset directories for a model in activation_cache."""
    resolved_model = resolve_model_name(model_name)
    model_cache_dir = PROJECT_ROOT / CACHE_DIR_NAME / resolved_model
    if not model_cache_dir.is_dir():
        return []
    return sorted(
        d.name
        for d in model_cache_dir.iterdir()
        if d.is_dir() and d.name != "generations"
    )


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
        all_idx = np.arange(len(y_reference))
        train_rel_idx, test_rel_idx = train_test_split(
            all_idx,
            test_size=test_size,
            random_state=split_seed,
            stratify=y_reference,
        )
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


def evaluate_single_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    max_iter: int,
    logreg_n_jobs: int,
) -> dict[str, object]:
    """Train LogisticRegression on one split and return per-seed metrics."""
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    clf = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=logreg_n_jobs,
    )

    t0_fit = time.time()
    clf.fit(x_train, y_train)
    fit_time_s = round(time.time() - t0_fit, 4)

    model_params = int(np.prod(clf.coef_.shape) + clf.intercept_.shape[0])

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

    return {
        "fit_time_s": fit_time_s,
        "model_params": model_params,
        "precision": sanitize_metric_value(float(precision)),
        "recall": sanitize_metric_value(float(recall)),
        "accuracy": sanitize_metric_value(float(accuracy)),
        "f1": sanitize_metric_value(float(f1)),
        "auroc": sanitize_metric_value(float(auroc)),
    }


def _row_for_seed(
    model_name: str,
    train_dataset: str,
    layer_type: str,
    layer_idx: int,
    num_samples: int,
    eval_dataset: str,
    split_seed: str,
    train_size: int,
    test_size: int,
    train_hall: int,
    train_not_hall: int,
    test_hall: int,
    test_not_hall: int,
    fit_time_s: float,
    model_params: int,
    precision: float | None,
    recall: float | None,
    accuracy: float | None,
    f1: float | None,
    auroc: float | None,
    total_runtime_s: float,
    num_splits: int,
    created_at_utc: str,
) -> dict[str, object]:
    return {
        "model": resolve_model_name(model_name),
        "train_dataset": train_dataset,
        "layer_type": layer_type,
        "layer": layer_idx,
        "num_samples": num_samples,
        "eval_dataset": eval_dataset,
        "split_seed": split_seed,
        "train_size": train_size,
        "test_size": test_size,
        "train_hall": train_hall,
        "train_not_hall": train_not_hall,
        "test_hall": test_hall,
        "test_not_hall": test_not_hall,
        "fit_time_s": fit_time_s,
        "model_params": model_params,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "auroc": auroc,
        "total_runtime_s": total_runtime_s,
        "num_splits": num_splits,
        "created_at_utc": created_at_utc,
    }


def _build_aggregate_rows(
    per_seed_results: list[dict[str, object]],
    base_row: dict[str, object],
    total_runtime_s: float,
    num_splits: int,
) -> list[dict[str, object]]:
    """Given per-seed rows, compute mean and std aggregate rows."""
    if not per_seed_results:
        return []

    metric_keys = ["precision", "recall", "accuracy", "f1", "auroc", "fit_time_s", "model_params"]
    agg_mean: dict[str, float | None] = {}
    agg_std: dict[str, float | None] = {}

    for key in metric_keys:
        vals = []
        for r in per_seed_results:
            v = r.get(key)
            if v is not None:
                vals.append(float(v))
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            agg_mean[key] = float(np.mean(arr))
            agg_std[key] = float(np.std(arr))
        else:
            agg_mean[key] = None
            agg_std[key] = None

    base = dict(base_row)
    rows: list[dict[str, object]] = []

    mean_row = dict(base)
    mean_row["split_seed"] = "mean"
    mean_row["total_runtime_s"] = total_runtime_s
    mean_row["num_splits"] = num_splits
    for key in metric_keys:
        mean_row[key] = agg_mean.get(key)
    rows.append(mean_row)

    std_row = dict(base)
    std_row["split_seed"] = "std"
    std_row["total_runtime_s"] = total_runtime_s
    std_row["num_splits"] = num_splits
    for key in metric_keys:
        std_row[key] = agg_std.get(key)
    rows.append(std_row)

    return rows


def run_study(
    model_name: str,
    dataset_name: str,
    layer_type: str,
    split_seeds: list[int],
    test_size: float,
    device: str,
    max_iter: int,
    logreg_n_jobs: int,
    no_cross_domain: bool = False,
) -> list[dict[str, object]]:
    """
    Run in-domain layer study, then automatically cross-domain on all
    other datasets found in activation_cache. Returns flat CSV rows.
    """
    t0_study = time.time()
    set_seed(42)

    resolved_model = resolve_model_name(model_name)
    created_at_utc = datetime.now(timezone.utc).isoformat()

    all_datasets = discover_other_datasets(model_name)
    cross_domain_datasets = [] if no_cross_domain else [d for d in all_datasets if d != dataset_name]

    available_layers = list_available_layers(model_name, dataset_name, layer_type)
    structure_type = detect_structure_type(model_name, dataset_name, layer_type)

    print("=" * 80)
    print(f"Layer study: model={model_name}  train_dataset={dataset_name}  layer_type={layer_type}")
    print(f"Train structure: {structure_type}")
    print(f"Available layers: {len(available_layers)} ({available_layers[0]}..{available_layers[-1]})")
    print(f"Split seeds ({len(split_seeds)}): {split_seeds}")
    if cross_domain_datasets:
        print(f"Cross-domain eval datasets (auto): {cross_domain_datasets}")
    else:
        print("Cross-domain:  none (no other datasets found)")
    print("=" * 80)

    # Build split indices from training dataset (shared for all)
    first_layer = available_layers[0]
    _, y_reference, ids_reference = load_activations_and_labels(
        model_name=model_name,
        dataset_name=dataset_name,
        layer_type=layer_type,
        layer_idx=first_layer,
        map_location=device,
    )
    split_cache = build_split_indices(y_reference=y_reference, split_seeds=split_seeds, test_size=test_size)
    num_splits = len(split_cache)

    all_rows: list[dict[str, object]] = []

    # Pre-load cross-domain data for each eval dataset (all layers)
    cd_data: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for cd_dataset in cross_domain_datasets:
        try:
            cd_available = list_available_layers(model_name, cd_dataset, layer_type)
            cd_data[cd_dataset] = {}
            for layer_idx in available_layers:
                cd_layer = layer_idx if layer_idx in cd_available else cd_available[0]
                x_cd, y_cd, _ = load_activations_and_labels(
                    model_name=model_name,
                    dataset_name=cd_dataset,
                    layer_type=layer_type,
                    layer_idx=cd_layer,
                    map_location=device,
                )
                cd_data[cd_dataset][layer_idx] = (x_cd, y_cd)
        except FileNotFoundError as e:
            print(f"  [WARN] Cannot load cross-domain dataset {cd_dataset}: {e}")

    # Process layers
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

        num_samples = int(x_layer.shape[0])

        # ---- IN-DOMAIN ----
        in_domain_seed_rows: list[dict[str, object]] = []
        for split_seed, split_data in split_cache.items():
            train_idx = np.asarray(split_data["train_indices"], dtype=np.int64)
            test_idx = np.asarray(split_data["test_indices"], dtype=np.int64)

            x_train = x_layer[train_idx]
            y_train = y_layer[train_idx]
            x_test = x_layer[test_idx]
            y_test = y_layer[test_idx]

            result = evaluate_single_split(x_train, y_train, x_test, y_test, max_iter, logreg_n_jobs)
            cd = split_data["train_class_distribution"]
            td = split_data["test_class_distribution"]

            row = _row_for_seed(
                model_name=model_name,
                train_dataset=dataset_name,
                layer_type=layer_type,
                layer_idx=layer_idx,
                num_samples=num_samples,
                eval_dataset=dataset_name,
                split_seed=str(split_seed),
                train_size=int(split_data["train_size"]),
                test_size=int(split_data["test_size"]),
                train_hall=int(cd["hallucination_1"]),
                train_not_hall=int(cd["not_hallucination_0"]),
                test_hall=int(td["hallucination_1"]),
                test_not_hall=int(td["not_hallucination_0"]),
                fit_time_s=result["fit_time_s"],
                model_params=int(result["model_params"]),
                precision=result["precision"],
                recall=result["recall"],
                accuracy=result["accuracy"],
                f1=result["f1"],
                auroc=result["auroc"],
                total_runtime_s=round(time.time() - t0_study, 3),
                num_splits=num_splits,
                created_at_utc=created_at_utc,
            )
            in_domain_seed_rows.append(row)
            all_rows.append(row)
            del x_train, y_train, x_test, y_test
            gc.collect()

        # In-domain aggregate rows
        base_row = _row_for_seed(
            model_name=model_name,
            train_dataset=dataset_name,
            layer_type=layer_type,
            layer_idx=layer_idx,
            num_samples=num_samples,
            eval_dataset=dataset_name,
            split_seed="",
            train_size=0,
            test_size=0,
            train_hall=0,
            train_not_hall=0,
            test_hall=0,
            test_not_hall=0,
            fit_time_s=0.0,
            model_params=0,
            precision=None,
            recall=None,
            accuracy=None,
            f1=None,
            auroc=None,
            total_runtime_s=0.0,
            num_splits=num_splits,
            created_at_utc=created_at_utc,
        )
        agg_rows = _build_aggregate_rows(in_domain_seed_rows, base_row, round(time.time() - t0_study, 3), num_splits)
        all_rows.extend(agg_rows)

        # ---- CROSS-DOMAIN ----
        for cd_dataset in cross_domain_datasets:
            if layer_idx not in cd_data.get(cd_dataset, {}):
                continue
            x_cd, y_cd = cd_data[cd_dataset][layer_idx]
            cd_seed_rows: list[dict[str, object]] = []

            for split_seed, split_data in split_cache.items():
                train_idx = np.asarray(split_data["train_indices"], dtype=np.int64)
                y_train = y_layer[train_idx]

                x_train = x_layer[train_idx]
                x_test = x_cd
                y_test = y_cd

                result = evaluate_single_split(x_train, y_train, x_test, y_test, max_iter, logreg_n_jobs)
                cd_dist = split_data["train_class_distribution"]

                row = _row_for_seed(
                    model_name=model_name,
                    train_dataset=dataset_name,
                    layer_type=layer_type,
                    layer_idx=layer_idx,
                    num_samples=num_samples,
                    eval_dataset=cd_dataset,
                    split_seed=str(split_seed),
                    train_size=int(split_data["train_size"]),
                    test_size=int(x_cd.shape[0]),
                    train_hall=int(cd_dist["hallucination_1"]),
                    train_not_hall=int(cd_dist["not_hallucination_0"]),
                    test_hall=int(np.sum(y_cd == 1)),
                    test_not_hall=int(np.sum(y_cd == 0)),
                    fit_time_s=result["fit_time_s"],
                    model_params=int(result["model_params"]),
                    precision=result["precision"],
                    recall=result["recall"],
                    accuracy=result["accuracy"],
                    f1=result["f1"],
                    auroc=result["auroc"],
                    total_runtime_s=round(time.time() - t0_study, 3),
                    num_splits=num_splits,
                    created_at_utc=created_at_utc,
                )
                cd_seed_rows.append(row)
                all_rows.append(row)
                del x_train, x_test, y_train, y_test
                gc.collect()

            cd_base_row = _row_for_seed(
                model_name=model_name,
                train_dataset=dataset_name,
                layer_type=layer_type,
                layer_idx=layer_idx,
                num_samples=num_samples,
                eval_dataset=cd_dataset,
                split_seed="",
                train_size=0,
                test_size=0,
                train_hall=0,
                train_not_hall=0,
                test_hall=0,
                test_not_hall=0,
                fit_time_s=0.0,
                model_params=0,
                precision=None,
                recall=None,
                accuracy=None,
                f1=None,
                auroc=None,
                total_runtime_s=0.0,
                num_splits=num_splits,
                created_at_utc=created_at_utc,
            )
            cd_agg = _build_aggregate_rows(cd_seed_rows, cd_base_row, round(time.time() - t0_study, 3), num_splits)
            all_rows.extend(cd_agg)

        del x_layer, y_layer, ids_layer
        gc.collect()

    # Free cross-domain data
    for cd_dataset in cross_domain_datasets:
        if cd_dataset in cd_data:
            del cd_data[cd_dataset]
    gc.collect()

    total_runtime_s = round(time.time() - t0_study, 3)
    for row in all_rows:
        if row["split_seed"] in ("mean", "std") or not row["split_seed"].isdigit():
            row["total_runtime_s"] = total_runtime_s

    print(f"Total runtime: {total_runtime_s:.1f}s  |  Rows: {len(all_rows)}")
    return all_rows


def _write_csv(output_path: str, rows: list[dict[str, object]]):
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_json(rows: list[dict[str, object]], output_csv_path: str):
    """Write a compact summary JSON alongside the CSV for quick inspection."""
    output_dir = os.path.dirname(output_csv_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    first_row = rows[0] if rows else {}
    model = str(first_row.get("model", "unknown"))
    train_ds = str(first_row.get("train_dataset", "unknown"))
    layer_type = str(first_row.get("layer_type", "unknown"))
    created_at = str(first_row.get("created_at_utc", ""))

    summary = {
        "model": model,
        "train_dataset": train_ds,
        "layer_type": layer_type,
        "created_at_utc": created_at,
        "evaluations": {},
    }

    for row in rows:
        seed = str(row.get("split_seed", ""))
        if seed not in ("mean", "std"):
            continue
        eval_ds = str(row.get("eval_dataset", ""))
        if eval_ds not in summary["evaluations"]:
            eval_type = "in_domain" if eval_ds == train_ds else "cross_domain"
            summary["evaluations"][eval_ds] = {"type": eval_type, "metrics": {}}
        summary["evaluations"][eval_ds]["metrics"][seed] = {
            "accuracy": row.get("accuracy"),
            "f1": row.get("f1"),
            "auroc": row.get("auroc"),
            "precision": row.get("precision"),
            "recall": row.get("recall"),
        }

    safe_model = model.replace("/", "__").replace("\\", "__")
    summary_name = f"summary_{safe_model}_{train_ds}_{layer_type}.json"
    summary_path = os.path.join(output_dir, summary_name)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary JSON written to: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run layer-wise hallucination study for one model/dataset/layer_type "
            "with automatic cross-domain evaluation and CSV output."
        )
    )
    parser.add_argument("--model", required=True, help="Model folder name in activation_cache.")
    parser.add_argument("--dataset", required=True, help="Training dataset folder name.")
    parser.add_argument("--layer-type", required=True, choices=LAYER_TYPES, help="Activation layer type.")
    parser.add_argument(
        "--split-seeds",
        default=",".join(str(seed) for seed in DEFAULT_SPLIT_SEEDS),
        help="Comma-separated split seeds (default: 10 standard O4A seeds).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Test split ratio (default: 0.3).",
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
    parser.add_argument("--output", required=True, help="Output CSV file path.")
    parser.add_argument(
        "--no-cross-domain",
        action="store_true",
        help="Disable automatic cross-domain evaluation.",
    )
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
            with open(output_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                existing = list(reader)
            if len(existing) > 0:
                has_mean_std = any(r.get("split_seed") in ("mean", "std") for r in existing)
                if has_mean_std:
                    print(f"[SKIP] Output already exists and appears complete: {output_path}")
                    print(f"       ({len(existing)} rows)")
                    return
        except Exception:
            print(f"[RESUME] Output exists but is incomplete/corrupt, re-running: {output_path}")

    if not args.no_cross_domain:
        all_datasets = discover_other_datasets(args.model)
        cross_datasets = [d for d in all_datasets if d != args.dataset]
        print(f"[INFO] Cross-domain eval datasets (auto-discovered): {cross_datasets or 'none'}")

    rows = run_study(
        model_name=args.model,
        dataset_name=args.dataset,
        layer_type=args.layer_type,
        split_seeds=split_seeds,
        test_size=args.test_size,
        device=args.device,
        max_iter=args.max_iter,
        logreg_n_jobs=args.logreg_n_jobs,
        no_cross_domain=args.no_cross_domain,
    )

    _write_csv(str(output_path), rows)
    _write_summary_json(rows, str(output_path))

    print("=" * 80)
    print(f"Study completed. CSV written to: {output_path}  ({len(rows)} rows)")
    print("=" * 80)


if __name__ == "__main__":
    main()
