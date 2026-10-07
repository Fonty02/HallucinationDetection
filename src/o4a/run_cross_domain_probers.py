"""Cross-domain prober runner: train on one dataset, eval on another.

Single-LLM mode (default): for one model, each prober (see o4a.probers) is trained
on --train-dataset using the model's selected layers for that dataset, then evaluated
on a stratified test split of --activation-dataset (same layer indices).

Cross-LLM mode (--tester-model): cross-LLM and cross-domain at the same time, with the
same mechanism as run_cross_domain_one_for_all.py. Every method (see METHOD_REGISTRY there)
trains its detector on --model and its cross-LLM stage (aligner / student encoder) towards
--tester-model, both on --train-dataset; both roles are evaluated on --activation-dataset.

Training on --train-dataset is identical to run_experiments, so by default models already saved
by run_experiments are loaded instead of retrained (see o4a.pretrained); --no-pretrained retrains.
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
from o4a.pretrained import evaluate_pretrained_method, evaluate_pretrained_prober  # noqa: E402
from o4a.probers import PROBER_REGISTRY, PROBERS  # noqa: E402
from o4a.run_cross_domain_one_for_all import (  # noqa: E402
    EXTRA_DEFAULT,
    EXTRA_SUFFIXES,
    META_SUFFIXES as CROSS_LLM_META_SUFFIXES,
    METHOD_REGISTRY,
    ROLES as CROSS_LLM_ROLES,
    _prepare_cross_domain_shared_data,
)


INFO_COLS = ["model", "seed", "train_dataset", "activation_dataset", "layer_type", "layers"]
META_SUFFIXES = ["params", "train_time_s", "train_n", "ae_params", "ae_time_s"]
# Probers without an autoencoder stage have zero AE params and zero AE training time.
META_DEFAULT = 0

# Cross-LLM mode: every method, reported for both roles as in run_cross_domain_one_for_all.
CROSS_LLM_INFO_COLS = [
    "model", "tester_model", "seed", "train_dataset", "activation_dataset",
    "layer_type", "layers", "tester_layers",
]


def _prober_cols(prober: str, cross_llm: bool = False) -> list[str]:
    if cross_llm:
        return ([f"{prober}_{r}_{m}" for r in CROSS_LLM_ROLES for m in METRICS]
                + [f"{prober}_{s}" for s in CROSS_LLM_META_SUFFIXES + EXTRA_SUFFIXES])
    return [f"{prober}_{m}" for m in METRICS] + [f"{prober}_{s}" for s in META_SUFFIXES]


def _source_col(prober: str) -> str:
    """Where the prober's models come from: 'trained' here, or the saved_models dir they were loaded from."""
    return f"{prober}_source"


def _build_csv_header(cross_llm: bool = False) -> list[str]:
    cols = list(CROSS_LLM_INFO_COLS if cross_llm else INFO_COLS)
    for prober in PROBERS:
        cols.extend(_prober_cols(prober, cross_llm))
        cols.append(_source_col(prober))
    return cols + ["runtime_seconds", "status"]


def _is_done(row: dict, prober: str, cross_llm: bool = False) -> bool:
    return all(row.get(col, "") not in ("", "ERROR") for col in _prober_cols(prober, cross_llm))


def _load_existing_rows(output_csv: str | None) -> dict[tuple[str, int], dict]:
    """Existing rows keyed by (layer_type, seed)."""
    if not output_csv or not os.path.exists(output_csv):
        return {}
    with open(output_csv, "r", newline="") as f:
        rows = {(r["layer_type"], int(r["seed"])): dict(r) for r in csv.DictReader(f)}
    print(f"[RESUME] Loaded {len(rows)} existing rows from {output_csv}")
    return rows


def _persist_csv(output_csv: str | None, rows: dict[tuple[str, int], dict], cross_llm: bool = False) -> None:
    if output_csv is None:
        return
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_build_csv_header(cross_llm))
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


def _record_result(row: dict, prober: str, result: dict, cross_llm: bool) -> None:
    if cross_llm:
        for role in CROSS_LLM_ROLES:
            for metric in METRICS:
                row[f"{prober}_{role}_{metric}"] = result[role][metric]
        for suffix in CROSS_LLM_META_SUFFIXES:
            row[f"{prober}_{suffix}"] = result["_meta"].get(suffix, "")
        for suffix in EXTRA_SUFFIXES:
            row[f"{prober}_{suffix}"] = result["_meta"].get(suffix, EXTRA_DEFAULT)
        return
    for metric in METRICS:
        row[f"{prober}_{metric}"] = result["metrics"][metric]
    for suffix in META_SUFFIXES:
        row[f"{prober}_{suffix}"] = result["_meta"].get(suffix, META_DEFAULT)


def run_cross_domain_probers(
    model: str,
    train_dataset: str,
    activation_dataset: str | None,
    layer_types: list[str],
    probers: list[str] | None = None,
    output_csv: str | None = None,
    save_dir: str | None = None,
    tester_model: str | None = None,
    use_pretrained: bool = True,
) -> dict[tuple[str, int], dict]:
    cross_llm = tester_model is not None
    selected_layers = load_selected_layers()
    for m in ([model, tester_model] if cross_llm else [model]):
        if train_dataset not in selected_layers.get(m, {}):
            raise ValueError(f"No layer selection for model={m}, dataset={train_dataset}")
    model_layers = selected_layers[model][train_dataset]

    activation_dataset = activation_dataset or train_dataset
    if cross_llm:
        if tester_model == model:
            raise ValueError(f"--tester-model must differ from --model (got {model}).")
        tester_layers = selected_layers[tester_model][train_dataset]
        # Same experiment config as o4a.experiments.build_experiments (trainer -> tester on train_dataset).
        exp_cfg = {
            "dataset": train_dataset,
            "trainer": model,
            "tester": tester_model,
            "trainer_layers": model_layers,
            "tester_layers": tester_layers,
        }
    probers = probers or PROBERS

    if output_csv:
        out_dir = os.path.dirname(output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    all_rows = _load_existing_rows(output_csv)

    print(f"[INFO] Model:              {model}")
    if cross_llm:
        print(f"[INFO] Tester model:       {tester_model}")
    print(f"[INFO] Train dataset:      {train_dataset}")
    print(f"[INFO] Activation dataset: {activation_dataset}")
    print(f"[INFO] Probers:            {probers}")
    print(f"[INFO] Reuse saved models: {use_pretrained}")

    for pos, layer_type in enumerate(layer_types, start=1):
        key = (layer_type, SEED)
        row = all_rows.get(key)
        if row is None:
            row = {col: "" for col in _build_csv_header(cross_llm)}
            row.update({
                "model": model,
                "seed": SEED,
                "train_dataset": train_dataset,
                "activation_dataset": activation_dataset,
                "layer_type": layer_type,
                "layers": str(model_layers[layer_type]),
            })
            if cross_llm:
                row["tester_model"] = tester_model
                row["tester_layers"] = str(tester_layers[layer_type])

        pending = [p for p in probers if not _is_done(row, p, cross_llm)]
        if not pending:
            print(f"\n[SKIP {pos}/{len(layer_types)}] {layer_type} — already complete")
            continue

        print(f"\n{'=' * 72}")
        model_desc = f"{model} -> {tester_model}" if cross_llm else model
        print(f"[{pos}/{len(layer_types)}] {model_desc} {train_dataset} -> {activation_dataset}  |  "
              f"layer_type={layer_type}  |  seed={SEED}  |  probers={pending}")
        print(f"{'=' * 72}")
        t0_layer = time.time()

        data = None
        try:
            set_seed(SEED)
            if cross_llm:
                data = _prepare_cross_domain_shared_data(exp_cfg, layer_type, activation_dataset)
            else:
                data = _prepare_cross_domain_data(
                    model, train_dataset, activation_dataset, model_layers[layer_type], layer_type,
                )
        except Exception:
            print(f"  !! DATA LOADING FAILED for {model}/{train_dataset}/{layer_type}:")
            traceback.print_exc()
            row["status"] = "data_error"
            row["runtime_seconds"] = round(time.time() - t0_layer, 3)
            all_rows[key] = row
            _persist_csv(output_csv, all_rows, cross_llm)
            continue

        try:
            for prober in pending:
                fn, cfg = (METHOD_REGISTRY if cross_llm else PROBER_REGISTRY)[prober]
                prober_save_dir = None
                if save_dir is not None:
                    prober_save_dir = os.path.join(
                        save_dir, f"{model}__to__{tester_model}" if cross_llm else model,
                        f"{train_dataset}__activation__{activation_dataset}",
                        layer_type, prober, str(SEED),
                    )

                print(f"  -> Running {prober} ... ", end="", flush=True)
                t0_p = time.time()
                try:
                    set_seed(SEED)
                    result = None
                    if use_pretrained:
                        if cross_llm:
                            result = evaluate_pretrained_method(
                                prober, data, model, tester_model, train_dataset, layer_type, SEED,
                                model_layers[layer_type], tester_layers[layer_type],
                            )
                        else:
                            result = evaluate_pretrained_prober(
                                prober, data, model, train_dataset, layer_type, SEED, model_layers[layer_type],
                            )
                    if result is None:
                        result = fn(data, cfg, save_dir=prober_save_dir)
                        row[_source_col(prober)] = "trained"
                    else:
                        row[_source_col(prober)] = result["_source"]
                    print(f"done ({time.time() - t0_p:.1f}s, {row[_source_col(prober)]})")
                    _record_result(row, prober, result, cross_llm)
                except Exception:
                    print(f"FAILED ({time.time() - t0_p:.1f}s)")
                    traceback.print_exc()
                    for col in _prober_cols(prober, cross_llm):
                        row[col] = "ERROR"
        finally:
            del data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        row["runtime_seconds"] = round(time.time() - t0_layer, 3)
        row["status"] = "ok" if all(_is_done(row, p, cross_llm) for p in probers) else "partial_error"
        all_rows[key] = row
        _persist_csv(output_csv, all_rows, cross_llm)

    print(f"\nResults written to {output_csv}  ({len(all_rows)} rows)")
    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-domain evaluation: train probers on one dataset, evaluate on another "
                    "(optionally also cross-LLM, via --tester-model)."
    )
    parser.add_argument("--model", required=True, help="Model folder name in activation_cache (trainer).")
    parser.add_argument("--tester-model", default=None,
                        help="Enable cross-LLM + cross-domain mode: each method's detector is trained on "
                             "--model and its cross-LLM stage towards this model.")
    parser.add_argument("--train-dataset", required=True, help="Dataset the probers are trained on.")
    parser.add_argument("--activation-dataset", default=None,
                        help="Target dataset for evaluation. Default: --train-dataset (in-domain).")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--layer-types", nargs="+", choices=LAYER_TYPES, default=None,
                        help="Optional subset of layer types (default: all).")
    parser.add_argument("--probers", nargs="+", choices=PROBERS, default=None,
                        help=f"Optional subset of probers (default: all {len(PROBERS)}).")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Always retrain instead of loading models saved by run_experiments.")
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
    print(f"[DEBUG] Tester model: {args.tester_model}")

    run_cross_domain_probers(
        model=args.model,
        train_dataset=args.train_dataset,
        activation_dataset=args.activation_dataset,
        layer_types=layer_types,
        probers=probers,
        output_csv=args.output,
        save_dir=os.path.join(ROOT_DIR, "saved_models", "cross_domain_probers") if args.save_models else None,
        tester_model=args.tester_model,
        use_pretrained=not args.no_pretrained,
    )


if __name__ == "__main__":
    main()
