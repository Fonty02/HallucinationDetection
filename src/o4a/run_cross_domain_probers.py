"""Single-LLM cross-domain runner: train every prober on one dataset, eval on another.

Counterpart of run_cross_domain_one_for_all.py without the cross-LLM part:
for one model, each prober (see o4a.probers) is trained on --train-dataset using
the model's selected layers for that dataset, then evaluated on a stratified
test split of --activation-dataset (same layer indices).
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
import traceback
from typing import Any

import torch

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from o4a.config import DEVICE, LAYER_TYPES, METRICS, ROOT_DIR, SEED  # noqa: E402
from o4a.data import load_stratified_test_set, prepare_single_model_data, set_seed  # noqa: E402
from o4a.experiments import load_selected_layers  # noqa: E402
from o4a.probers import PROBER_REGISTRY, PROBERS  # noqa: E402


INFO_COLS = ["model", "seed", "train_dataset", "activation_dataset", "layer_type", "layers"]
META_SUFFIXES = ["params", "train_time_s", "train_n", "ae_params", "ae_time_s"]
# Probers without an autoencoder stage have zero AE params and zero AE training time.
META_DEFAULT = 0


def _prober_cols(prober: str) -> list[str]:
    return [f"{prober}_{m}" for m in METRICS] + [f"{prober}_{s}" for s in META_SUFFIXES]


def _build_csv_header() -> list[str]:
    cols = list(INFO_COLS)
    for prober in PROBERS:
        cols.extend(_prober_cols(prober))
    return cols + ["runtime_seconds", "status"]


def _is_done(row: dict, prober: str) -> bool:
    return all(row.get(col, "") not in ("", "ERROR") for col in _prober_cols(prober))


def _load_existing_rows(output_csv: str | None) -> dict[tuple[str, int], dict]:
    """Existing rows keyed by (layer_type, seed)."""
    if not output_csv or not os.path.exists(output_csv):
        return {}
    with open(output_csv, "r", newline="") as f:
        rows = {(r["layer_type"], int(r["seed"])): dict(r) for r in csv.DictReader(f)}
    print(f"[RESUME] Loaded {len(rows)} existing rows from {output_csv}")
    return rows


def _persist_csv(output_csv: str | None, rows: dict[tuple[str, int], dict]) -> None:
    if output_csv is None:
        return
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_build_csv_header())
        writer.writeheader()
        writer.writerows(rows.values())


def _prepare_cross_domain_data(
    model: str, train_dataset: str, activation_dataset: str, layers: list[int], layer_type: str,
) -> dict[str, Any]:
    """Source-domain train/val splits, target-domain test split (scaled with the source scaler)."""
    data = prepare_single_model_data(model, train_dataset, layers, layer_type)
    if activation_dataset == train_dataset:
        return data

    X_test_raw, y_test = load_stratified_test_set(model, activation_dataset, layers, layer_type, seed=SEED)
    data["X_test_raw"] = X_test_raw
    data["X_test"] = data["scaler"].transform(X_test_raw)
    data["y_test"] = y_test
    return data


def run_cross_domain_probers(
    model: str,
    train_dataset: str,
    activation_dataset: str | None,
    layer_types: list[str],
    probers: list[str] | None = None,
    output_csv: str | None = None,
    save_dir: str | None = None,
) -> dict[tuple[str, int], dict]:
    selected_layers = load_selected_layers()
    if train_dataset not in selected_layers.get(model, {}):
        raise ValueError(f"No layer selection for model={model}, dataset={train_dataset}")
    model_layers = selected_layers[model][train_dataset]

    activation_dataset = activation_dataset or train_dataset
    probers = probers or PROBERS

    if output_csv:
        out_dir = os.path.dirname(output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    all_rows = _load_existing_rows(output_csv)

    print(f"[INFO] Model:              {model}")
    print(f"[INFO] Train dataset:      {train_dataset}")
    print(f"[INFO] Activation dataset: {activation_dataset}")
    print(f"[INFO] Probers:            {probers}")

    for pos, layer_type in enumerate(layer_types, start=1):
        key = (layer_type, SEED)
        row = all_rows.get(key)
        if row is None:
            row = {col: "" for col in _build_csv_header()}
            row.update({
                "model": model,
                "seed": SEED,
                "train_dataset": train_dataset,
                "activation_dataset": activation_dataset,
                "layer_type": layer_type,
                "layers": str(model_layers[layer_type]),
            })

        pending = [p for p in probers if not _is_done(row, p)]
        if not pending:
            print(f"\n[SKIP {pos}/{len(layer_types)}] {layer_type} — already complete")
            continue

        print(f"\n{'=' * 72}")
        print(f"[{pos}/{len(layer_types)}] {model} {train_dataset} -> {activation_dataset}  |  "
              f"layer_type={layer_type}  |  seed={SEED}  |  probers={pending}")
        print(f"{'=' * 72}")
        t0_layer = time.time()

        data = None
        try:
            set_seed(SEED)
            data = _prepare_cross_domain_data(
                model, train_dataset, activation_dataset, model_layers[layer_type], layer_type,
            )
        except Exception:
            print(f"  !! DATA LOADING FAILED for {model}/{train_dataset}/{layer_type}:")
            traceback.print_exc()
            row["status"] = "data_error"
            row["runtime_seconds"] = round(time.time() - t0_layer, 3)
            all_rows[key] = row
            _persist_csv(output_csv, all_rows)
            continue

        try:
            for prober in pending:
                fn, cfg = PROBER_REGISTRY[prober]
                prober_save_dir = None
                if save_dir is not None:
                    prober_save_dir = os.path.join(
                        save_dir, model, f"{train_dataset}__activation__{activation_dataset}",
                        layer_type, prober, str(SEED),
                    )

                print(f"  -> Running {prober} ... ", end="", flush=True)
                t0_p = time.time()
                try:
                    set_seed(SEED)
                    result = fn(data, cfg, save_dir=prober_save_dir)
                    print(f"done ({time.time() - t0_p:.1f}s)")
                    for metric in METRICS:
                        row[f"{prober}_{metric}"] = result["metrics"][metric]
                    for suffix in META_SUFFIXES:
                        row[f"{prober}_{suffix}"] = result["_meta"].get(suffix, META_DEFAULT)
                except Exception:
                    print(f"FAILED ({time.time() - t0_p:.1f}s)")
                    traceback.print_exc()
                    for col in _prober_cols(prober):
                        row[col] = "ERROR"
        finally:
            del data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        row["runtime_seconds"] = round(time.time() - t0_layer, 3)
        row["status"] = "ok" if all(_is_done(row, p) for p in probers) else "partial_error"
        all_rows[key] = row
        _persist_csv(output_csv, all_rows)

    print(f"\nResults written to {output_csv}  ({len(all_rows)} rows)")
    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-LLM cross-domain evaluation: train probers on one dataset, evaluate on another."
    )
    parser.add_argument("--model", required=True, help="Model folder name in activation_cache.")
    parser.add_argument("--train-dataset", required=True, help="Dataset the probers are trained on.")
    parser.add_argument("--activation-dataset", default=None,
                        help="Target dataset for evaluation. Default: --train-dataset (in-domain).")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--layer-types", nargs="+", choices=LAYER_TYPES, default=None,
                        help="Optional subset of layer types (default: all).")
    parser.add_argument("--probers", nargs="+", choices=PROBERS, default=None,
                        help=f"Optional subset of probers (default: all {len(PROBERS)}).")
    parser.add_argument("--save-models", action="store_true",
                        help="Save trained probers under saved_models/cross_domain_probers (default: off).")
    args = parser.parse_args()

    layer_types = args.layer_types or LAYER_TYPES
    probers = args.probers or PROBERS

    print(f"[DEBUG] Root: {ROOT_DIR}")
    print(f"[DEBUG] Seed: {SEED}")
    print(f"[DEBUG] Device: {DEVICE}")
    print(f"[DEBUG] Layer types: {layer_types}")
    print(f"[DEBUG] Probers: {probers}")

    run_cross_domain_probers(
        model=args.model,
        train_dataset=args.train_dataset,
        activation_dataset=args.activation_dataset,
        layer_types=layer_types,
        probers=probers,
        output_csv=args.output,
        save_dir=os.path.join(ROOT_DIR, "saved_models", "cross_domain_probers") if args.save_models else None,
    )


if __name__ == "__main__":
    main()
