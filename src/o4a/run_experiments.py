"""Main experiment runner. Loops over EXPERIMENTS × LAYER_TYPES, runs all methods, writes CSV."""

import argparse
import csv
import os
import sys
import time
import traceback

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from o4a.config import (
    EXPERIMENTS,
    LAYER_TYPES,
    METHODS,
    METRICS,
    SEED,
    ROOT_DIR,
    RIDGE_REGRESSOR_CONFIG,
    PROCRUSTES_CONFIG,
    CKA_CONFIG,
    CCA_CONFIG,
    HYBRID_CONFIG,
    FULL_NONLINEAR_CONFIG,
    REDUCED_NONLINEAR_CONFIG,
    ONE_FOR_ALL_CONFIG,
)
from o4a.data import prepare_shared_data, set_seed
from o4a.methods.ridge_regressor import run_ridge_regressor
from o4a.methods.procrustes import run_procrustes
from o4a.methods.cka import run_cka
from o4a.methods.cca import run_cca
from o4a.methods.hybrid import run_hybrid
from o4a.methods.full_nonlinear import run_full_nonlinear
from o4a.methods.reduced_nonlinear import run_reduced_nonlinear
from o4a.methods.one_for_all import run_one_for_all

# Mapping from method name → (function, config)
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

DEFAULT_SAVE_DIR = os.path.join(ROOT_DIR, "saved_models")


META_SUFFIXES = [
    "detector_params", "detector_time_s", "detector_train_n",
    "aligner_params", "aligner_time_s", "aligner_train_n",
]
EXTRA_SUFFIXES = [
    "ae_trainer_params", "ae_trainer_time_s",
    "ae_tester_params", "ae_tester_time_s",
]


def _build_csv_header() -> list:
    """Build the CSV header row."""
    info_cols = [
        "experiment", "seed", "dataset", "trainer", "tester",
        "layer_type", "trainer_layers", "tester_layers",
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
    return info_cols + metric_cols


def _load_existing_rows(output_csv: str) -> tuple[list, dict, list]:
    """
    Load existing CSV rows. Returns:
      - existing_rows: list[dict] of parsed rows
      - completed_keys: set of (experiment, layer_type, seed) tuples that are fully complete
      - header: header list or None if file doesn't exist
    """
    if not os.path.exists(output_csv):
        return [], set(), None

    header = _build_csv_header()
    info_cols = header[:8]  # first 8 are info cols
    method_cols = header[8:]

    rows = []
    with open(output_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))

    completed = set()
    for row in rows:
        key = (row.get("experiment", ""), row.get("layer_type", ""), int(row.get("seed", 0)))
        # A row is complete if all method columns have non-ERROR, non-empty values
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


def run_all(
    experiments: dict | None = None,
    layer_types: list[str] | None = None,
    methods: list[str] | None = None,
    output_csv: str = "experiments_results.csv",
    save_dir: str | None = DEFAULT_SAVE_DIR,
    seeds: list[int] | None = None,
):
    """
    Main entry point. For each (experiment, layer_type, seed) tuple, prepare shared
    data once and run all requested methods. Results are accumulated and
    written to a CSV file.

    Parameters
    ----------
    seeds : list[int]
        Seeds to run. REQUIRED - must be passed as [SEED] from main().
        SEED comes from O4A_SEED environment variable (set by HTC).
    """
    experiments = experiments or EXPERIMENTS
    layer_types = layer_types or LAYER_TYPES
    methods_override = methods is not None
    methods = methods or METHODS

    # Seeds MUST be provided (from O4A_SEED via HTC)
    if seeds is None:
        raise RuntimeError("seeds parameter is required. Must be passed from O4A_SEED environment variable via HTC.")
    seeds_to_run = seeds

    header = _build_csv_header()
    info_cols = header[:8]
    method_cols = header[8:]

    # Load existing results for resume support
    existing_rows, completed_keys, _ = _load_existing_rows(output_csv)
    all_rows = list(existing_rows)

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total = len(experiments) * len(layer_types) * len(seeds_to_run)
    done = 0
    skipped = 0

    for exp_name, exp_cfg in experiments.items():
        for lt in layer_types:
            for seed in seeds_to_run:
                done += 1
                key = (exp_name, lt, seed)

                if key in completed_keys:
                    print(f"\n[SKIP {done}/{total}] {exp_name}  |  layer_type={lt}  |  seed={seed}  (already complete)")
                    skipped += 1
                    continue

                print(f"\n{'='*70}")
                print(f"[{done}/{total}] {exp_name}  |  layer_type={lt}  |  seed={seed}")
                print(f"{'='*70}")

                # Prepare shared data once for all methods
                set_seed(seed)
                try:
                    shared_data = prepare_shared_data(exp_cfg, lt)
                except Exception:
                    print(f"  !! DATA LOADING FAILED for {exp_name}/{lt}:")
                    traceback.print_exc()
                    continue

                # Info columns
                row = {
                    "experiment": exp_name,
                    "seed": seed,
                    "dataset": exp_cfg["dataset"],
                    "trainer": exp_cfg["trainer"],
                    "tester": exp_cfg["tester"],
                    "layer_type": lt,
                    "trainer_layers": str(exp_cfg["trainer_layers"][lt]),
                    "tester_layers": str(exp_cfg["tester_layers"][lt]),
                }
                # Initialize method columns to empty
                for col in method_cols:
                    row[col] = ""

                # Restore any previously computed method values for this key
                for existing in existing_rows:
                    if (existing.get("experiment", "") == exp_name and
                        existing.get("layer_type", "") == lt and
                        int(existing.get("seed", 0)) == seed):
                        for col in method_cols:
                            val = existing.get(col, "")
                            if val and val != "ERROR":
                                row[col] = val
                        break

                exp_methods = methods if methods_override else exp_cfg.get("methods", methods)
                for method_name in exp_methods:
                    if method_name not in METHOD_REGISTRY:
                        print(f"  !! Unknown method: {method_name} — skipping")
                        continue

                    fn, cfg = METHOD_REGISTRY[method_name]
                    method_save_dir = None
                    if save_dir is not None:
                        method_save_dir = os.path.join(save_dir, exp_name, lt, method_name, str(seed))
                    print(f"  -> Running {method_name} ... ", end="", flush=True)
                    t0 = time.time()
                    try:
                        set_seed(seed)
                        result = fn(shared_data, cfg, save_dir=method_save_dir)
                        elapsed = time.time() - t0
                        print(f"done ({elapsed:.1f}s)")
                        for role in ROLES:
                            for metric in METRICS:
                                key = f"{method_name}_{role}_{metric}"
                                row[key] = result[role].get(metric, "")
                        meta = result.get("_meta", {})
                        for suffix in META_SUFFIXES:
                            row[f"{method_name}_{suffix}"] = meta.get(suffix, "")
                        for suffix in EXTRA_SUFFIXES:
                            row[f"{method_name}_{suffix}"] = meta.get(suffix, "")
                    except Exception:
                        elapsed = time.time() - t0
                        print(f"FAILED ({elapsed:.1f}s)")
                        traceback.print_exc()
                        for role in ROLES:
                            for metric in METRICS:
                                key = f"{method_name}_{role}_{metric}"
                                row[key] = "ERROR"

                all_rows.append(row)
                # Persist incrementally
                with open(output_csv, "w", newline="") as out_f:
                    writer = csv.DictWriter(out_f, fieldnames=header)
                    writer.writeheader()
                    writer.writerows(all_rows)

    print(f"\nResults written to {output_csv}  ({len(all_rows)} rows, {skipped} skipped)")
    return all_rows


def main():
    parser = argparse.ArgumentParser(description="Run hallucination detection experiments via HTC")
    parser.add_argument("--experiments", nargs="+", required=True,
                        help="Experiment name(s) to run (REQUIRED - passed by HTC)")
    parser.add_argument("--output", required=True,
                        help="Output CSV path (REQUIRED - passed by HTC)")
    parser.add_argument(
        "--layer-types",
        nargs="+",
        choices=LAYER_TYPES,
        default=None,
        help="Optional subset of layer types to run (default: all configured layer types).",
    )
    args = parser.parse_args()

    selected_layer_types = args.layer_types if args.layer_types else LAYER_TYPES

    print(f"[DEBUG] SEED (from O4A_SEED): {SEED}")
    print(f"[DEBUG] Experiments: {args.experiments}")
    print(f"[DEBUG] Layer types: {selected_layer_types}")
    print(f"[DEBUG] Output: {args.output}")

    # Filter experiments - MUST be provided
    exps = {k: v for k, v in EXPERIMENTS.items() if k in args.experiments}
    if not exps:
        print(f"ERROR: No matching experiments found.\nRequested: {args.experiments}\nAvailable: {list(EXPERIMENTS.keys())}")
        sys.exit(1)

    # All parameters come from HTC via environment variables set in config.py
    # SEED and DEVICE are mandatory and enforced in config.py
    # LAYER_TYPES and METHODS use defaults from config.py
    run_all(
        experiments=exps,
        layer_types=selected_layer_types,
        methods=METHODS,
        output_csv=args.output,
        save_dir=DEFAULT_SAVE_DIR,
        seeds=[SEED],  # Single seed from O4A_SEED env var (mandatory)
    )


if __name__ == "__main__":
    main()
