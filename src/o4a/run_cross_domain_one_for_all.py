"""Cross-domain multi-method runner: train all methods on one dataset, eval on another."""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from o4a.config import (  # noqa: E402
    DEVICE,
    EXPERIMENTS,
    LAYER_TYPES,
    METRICS,
    METHODS,
    ROOT_DIR,
    SEED,
    RIDGE_REGRESSOR_CONFIG,
    PROCRUSTES_CONFIG,
    CKA_CONFIG,
    CCA_CONFIG,
    HYBRID_CONFIG,
    FULL_NONLINEAR_CONFIG,
    REDUCED_NONLINEAR_CONFIG,
    ONE_FOR_ALL_CONFIG,
)
from o4a.data import (  # noqa: E402
    DataManager,
    prepare_shared_data,
    set_seed,
)
from o4a.methods.ridge_regressor import run_ridge_regressor  # noqa: E402
from o4a.methods.procrustes import run_procrustes  # noqa: E402
from o4a.methods.cka import run_cka  # noqa: E402
from o4a.methods.cca import run_cca  # noqa: E402
from o4a.methods.hybrid import run_hybrid  # noqa: E402
from o4a.methods.full_nonlinear import run_full_nonlinear  # noqa: E402
from o4a.methods.reduced_nonlinear import run_reduced_nonlinear  # noqa: E402
from o4a.methods.one_for_all import run_one_for_all  # noqa: E402


METHOD_REGISTRY = {
    "ridge_regressor": (run_ridge_regressor, RIDGE_REGRESSOR_CONFIG),
    "procrustes": (run_procrustes, PROCRUSTES_CONFIG),
    "cka": (run_cka, CKA_CONFIG),
    "cca": (run_cca, CCA_CONFIG),
    "hybrid": (run_hybrid, HYBRID_CONFIG),
    "full_nonlinear": (run_full_nonlinear, FULL_NONLINEAR_CONFIG),
    "reduced_nonlinear": (run_reduced_nonlinear, REDUCED_NONLINEAR_CONFIG),
    "one_for_all": (run_one_for_all, ONE_FOR_ALL_CONFIG),
}

ROLES = ["trainer", "tester"]
META_SUFFIXES = [
    "detector_params", "detector_time_s", "detector_train_n",
    "aligner_params", "aligner_time_s", "aligner_train_n",
]
EXTRA_SUFFIXES = [
    "ae_trainer_params", "ae_trainer_time_s",
    "ae_tester_params", "ae_tester_time_s",
]


def _load_stratified_test_set(
    model_name: str,
    dataset_name: str,
    layer_indices: list[int],
    layer_type: str,
    test_size: float = 0.15,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a stratified test split from the activation dataset."""
    X_full, _, _ = DataManager.load_concatenated_layers(model_name, dataset_name, layer_indices, layer_type)
    stats = DataManager.get_stats(model_name, dataset_name, layer_type=layer_type)
    hall_set = set(stats["hallucinated_ids"])
    y = np.array([1 if i in hall_set else 0 for i in range(stats["total"])], dtype=np.int8)
    all_idx = np.arange(stats["total"])
    _, test_idx = train_test_split(all_idx, test_size=test_size, random_state=seed, stratify=y)
    return X_full[test_idx], y[test_idx]


def _prepare_cross_domain_shared_data(
    exp_cfg: dict[str, Any],
    layer_type: str,
    activation_dataset: str,
) -> dict[str, Any]:
    """Build shared_data with source-domain train/val and target-domain test splits."""
    source = prepare_shared_data(exp_cfg, layer_type)

    if activation_dataset == exp_cfg["dataset"]:
        return source

    trainer_layers = exp_cfg["trainer_layers"][layer_type]
    tester_layers = exp_cfg["tester_layers"][layer_type]

    X_trainer_test_raw, y_trainer_test = _load_stratified_test_set(
        exp_cfg["trainer"], activation_dataset, trainer_layers, layer_type, seed=SEED,
    )
    X_tester_test_raw, y_tester_test = _load_stratified_test_set(
        exp_cfg["tester"], activation_dataset, tester_layers, layer_type, seed=SEED,
    )

    source["trainer"]["X_test_raw"] = X_trainer_test_raw
    source["trainer"]["y_test"] = y_trainer_test
    source["trainer"]["X_test"] = source["trainer"]["scaler"].transform(X_trainer_test_raw)

    source["tester"]["X_test_raw"] = X_tester_test_raw
    source["tester"]["y_test"] = y_tester_test
    source["tester"]["X_test"] = source["tester"]["scaler"].transform(X_tester_test_raw)

    return source


def _build_csv_header() -> list[str]:
    info_cols = [
        "experiment", "seed", "dataset", "trainer", "tester",
        "layer_type", "trainer_layers", "tester_layers", "activation_dataset",
    ]
    metric_cols = []
    for method in METHODS:
        for role in ROLES:
            for metric in METRICS:
                metric_cols.append(f"{method}_{role}_{metric}")
        for suffix in META_SUFFIXES:
            metric_cols.append(f"{method}_{suffix}")
        for suffix in EXTRA_SUFFIXES:
            metric_cols.append(f"{method}_{suffix}")
    return info_cols + metric_cols + ["runtime_seconds", "status"]


def _load_existing_rows(output_csv: str) -> tuple[list[dict], set[tuple], list[str] | None]:
    if not os.path.exists(output_csv):
        return [], set(), None

    header = _build_csv_header()
    info_cols = header[:9]
    method_cols = header[9:-2]  # between info and runtime/status

    rows = []
    with open(output_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))

    completed = set()
    for row in rows:
        key = (row.get("experiment", ""), row.get("layer_type", ""), int(row.get("seed", 0)))
        row_complete = True
        for col in method_cols:
            val = row.get(col, "")
            if not val or val == "ERROR":
                row_complete = False
                break
        if row_complete:
            completed.add(key)

    print(f"[RESUME] Loaded {len(rows)} existing rows, {len(completed)} fully complete.")
    return rows, completed, header


def run_cross_domain_one_for_all(
    encoder_experiment: str,
    layer_types: list[str],
    activation_dataset: str | None,
    head_experiment: str | None = None,
    output_csv: str | None = None,
    save_dir: str | None = None,
    methods: list[str] | None = None,
) -> list[dict[str, Any]]:

    if encoder_experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {encoder_experiment}")
    if head_experiment is not None and head_experiment != encoder_experiment:
        raise ValueError(
            f"--head-experiment must match --encoder-experiment. "
            f"Got encoder={encoder_experiment}, head={head_experiment}."
        )

    enc_cfg = EXPERIMENTS[encoder_experiment]
    activation_dataset = activation_dataset or enc_cfg["dataset"]
    methods = methods or METHODS

    header = _build_csv_header()
    info_cols = header[:9]
    method_cols = header[9:-2]

    existing_rows, completed_keys, _ = _load_existing_rows(output_csv) if output_csv else ([], set(), None)
    all_rows = list(existing_rows) if existing_rows else []

    if output_csv:
        out_dir = os.path.dirname(output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Experiment:         {encoder_experiment} (dataset={enc_cfg['dataset']})")
    print(f"[INFO] Activation dataset: {activation_dataset}")
    print(f"[INFO] Trainer/Tester:     {enc_cfg['trainer']} -> {enc_cfg['tester']}")
    print(f"[INFO] Methods:            {methods}")

    total = len(layer_types)
    done = 0
    skipped = 0

    for layer_type in layer_types:
        done += 1
        key = (encoder_experiment, layer_type, SEED)

        if key in completed_keys:
            print(f"\n[SKIP {done}/{total}] {layer_type} — already complete")
            skipped += 1
            continue

        print(f"\n{'=' * 72}")
        print(f"[{done}/{total}] {encoder_experiment}  |  layer_type={layer_type}  |  seed={SEED}")
        print(f"{'=' * 72}")

        t0_layer = time.time()

        row = {
            "experiment": encoder_experiment,
            "seed": SEED,
            "dataset": enc_cfg["dataset"],
            "trainer": enc_cfg["trainer"],
            "tester": enc_cfg["tester"],
            "layer_type": layer_type,
            "trainer_layers": str(enc_cfg["trainer_layers"][layer_type]),
            "tester_layers": str(enc_cfg["tester_layers"][layer_type]),
            "activation_dataset": activation_dataset,
        }
        for col in method_cols + ["runtime_seconds", "status"]:
            row[col] = ""

        for existing in (existing_rows or []):
            if (existing.get("experiment", "") == encoder_experiment
                    and existing.get("layer_type", "") == layer_type
                    and int(existing.get("seed", 0)) == SEED):
                for col in method_cols:
                    val = existing.get(col, "")
                    if val and val != "ERROR":
                        row[col] = val
                break

        shared_data = None
        try:
            set_seed(SEED)
            shared_data = _prepare_cross_domain_shared_data(enc_cfg, layer_type, activation_dataset)
        except Exception:
            print(f"  !! DATA LOADING FAILED for {encoder_experiment}/{layer_type}:")
            traceback.print_exc()
            row["status"] = "data_error"
            row["runtime_seconds"] = round(time.time() - t0_layer, 3)
            all_rows.append(row)
            _persist_csv(output_csv, header, all_rows)
            continue

        all_methods_ok = True
        try:
            for method_name in methods:
                if method_name not in METHOD_REGISTRY:
                    print(f"  !! Unknown method: {method_name} — skipping")
                    continue

                fn, cfg = METHOD_REGISTRY[method_name]
                method_save_dir = None
                if save_dir is not None:
                    method_save_dir = os.path.join(save_dir, encoder_experiment, layer_type, method_name, str(SEED))

                print(f"  -> Running {method_name} ... ", end="", flush=True)
                t0_m = time.time()
                try:
                    set_seed(SEED)
                    result = fn(shared_data, cfg, save_dir=method_save_dir)
                    elapsed = time.time() - t0_m
                    print(f"done ({elapsed:.1f}s)")

                    for role in ROLES:
                        for metric in METRICS:
                            col = f"{method_name}_{role}_{metric}"
                            row[col] = result[role].get(metric, "")

                    meta = result.get("_meta", {})
                    for suffix in META_SUFFIXES:
                        row[f"{method_name}_{suffix}"] = meta.get(suffix, "")
                    for suffix in EXTRA_SUFFIXES:
                        row[f"{method_name}_{suffix}"] = meta.get(suffix, "")

                except Exception:
                    elapsed = time.time() - t0_m
                    print(f"FAILED ({elapsed:.1f}s)")
                    traceback.print_exc()
                    all_methods_ok = False
                    for role in ROLES:
                        for metric in METRICS:
                            row[f"{method_name}_{role}_{metric}"] = "ERROR"
        finally:
            if shared_data is not None:
                del shared_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        row["runtime_seconds"] = round(time.time() - t0_layer, 3)
        row["status"] = "ok" if all_methods_ok else "partial_error"
        all_rows.append(row)
        _persist_csv(output_csv, header, all_rows)

    print(f"\nResults written to {output_csv}  ({len(all_rows)} rows, {skipped} skipped)")
    return all_rows


def _persist_csv(output_csv: str | None, header: list[str], rows: list[dict[str, Any]]):
    if output_csv is None:
        return
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-domain evaluation: train on source dataset, evaluate on target dataset — for all methods."
    )
    parser.add_argument("--encoder-experiment", required=True,
                        help="Experiment config for training (must exist in o4a.config.EXPERIMENTS).")
    parser.add_argument("--head-experiment", default=None,
                        help="Deprecated; must match --encoder-experiment.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--activation-dataset", default=None,
                        help="Target dataset for evaluation. Default: encoder experiment dataset.")
    parser.add_argument("--layer-types", nargs="+", choices=LAYER_TYPES, default=None,
                        help="Optional subset of layer types (default: all).")
    parser.add_argument("--methods", nargs="+", choices=list(METHOD_REGISTRY.keys()), default=None,
                        help="Optional subset of methods (default: all 8 methods).")
    args = parser.parse_args()

    selected_layer_types = args.layer_types if args.layer_types else LAYER_TYPES
    selected_methods = list(args.methods) if args.methods else METHODS

    print(f"[DEBUG] Root: {ROOT_DIR}")
    print(f"[DEBUG] Seed: {SEED}")
    print(f"[DEBUG] Device: {DEVICE}")
    print(f"[DEBUG] Layer types: {selected_layer_types}")
    print(f"[DEBUG] Methods: {selected_methods}")

    default_save_dir = os.path.join(ROOT_DIR, "saved_models", "cross_domain")

    run_cross_domain_one_for_all(
        encoder_experiment=args.encoder_experiment,
        layer_types=selected_layer_types,
        activation_dataset=args.activation_dataset,
        head_experiment=args.head_experiment,
        output_csv=args.output,
        save_dir=default_save_dir,
        methods=selected_methods,
    )


if __name__ == "__main__":
    main()
