#!/usr/bin/env python3
"""
Export O4A sampling/intersection stats and hallucination rates to JSON.

This script mirrors the counting logic used in src/o4a/data.py (prepare_shared_data):
  - per-model undersampling to minority class size
  - train/test split for probing
  - common-id intersection across the 2 models in each experiment
  - concordant-only balancing used for alignment
"""

from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export O4A sampling stats to JSON.")
    parser.add_argument(
        "--config-path",
        type=str,
        default="src/o4a/config.py",
        help="Path to O4A config.py containing EXPERIMENTS and split constants.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="",
        help="Activation cache root (default: <project_root>/<CACHE_DIR_NAME from config>).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/o4a_experiments/sampling_hallucination_stats.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--stats-layer-type",
        type=str,
        default="attn",
        choices=["attn", "mlp", "hidden"],
        help=(
            "Layer type used to read stats for new cache structure. "
            "Default is 'attn' to match current prepare_shared_data behavior."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if an experiment/model-dataset pair is missing data instead of reporting an error entry.",
    )
    return parser.parse_args()


def _literal_from_config(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise KeyError(f"'{name}' not found as literal assignment in config.")


def load_o4a_config_literals(config_path: Path) -> dict:
    tree = ast.parse(config_path.read_text(encoding="utf-8"), filename=str(config_path))
    return {
        "EXPERIMENTS": _literal_from_config(tree, "EXPERIMENTS"),
        "MODEL_ALIASES": _literal_from_config(tree, "MODEL_ALIASES"),
        "CACHE_DIR_NAME": _literal_from_config(tree, "CACHE_DIR_NAME"),
        "LAYER_TYPES": _literal_from_config(tree, "LAYER_TYPES"),
        "TRAIN_SPLIT": _literal_from_config(tree, "TRAIN_SPLIT"),
        "ALIGNMENT_SPLIT": _literal_from_config(tree, "ALIGNMENT_SPLIT"),
        "PROBER_VAL_SPLIT": _literal_from_config(tree, "PROBER_VAL_SPLIT"),
    }


def resolve_model_dir(cache_root: Path, model_name: str, model_aliases: dict[str, str]) -> str:
    preferred = model_aliases.get(model_name, model_name)
    candidates = [preferred, model_name]
    if "/" in preferred:
        candidates.extend([preferred.split("/")[-1], preferred.replace("/", "_"), preferred.replace("/", "-")])
    if "/" in model_name:
        candidates.extend([model_name.split("/")[-1], model_name.replace("/", "_"), model_name.replace("/", "-")])

    for candidate in candidates:
        if (cache_root / candidate).is_dir():
            return candidate

    available = [p.name for p in cache_root.iterdir() if p.is_dir()] if cache_root.exists() else []
    available_lc = {name.lower(): name for name in available}
    for candidate in candidates:
        resolved = available_lc.get(candidate.lower())
        if resolved is not None:
            return resolved

    raise FileNotFoundError(
        f"Model folder for '{model_name}' not found in {cache_root}. "
        f"Available models: {', '.join(sorted(available))}"
    )


def _read_int_list(path: Path) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        values = json.load(f)
    return [int(v) for v in values]


def load_model_dataset_stats(
    cache_root: Path,
    model_name: str,
    dataset: str,
    model_aliases: dict[str, str],
    layer_type_for_new_structure: str,
) -> dict:
    model_dir = resolve_model_dir(cache_root, model_name, model_aliases)
    dataset_dir = cache_root / model_dir / dataset
    base = dataset_dir / f"activation_{layer_type_for_new_structure}"
    hall_dir = base / "hallucinated"
    not_hall_dir = base / "not_hallucinated"

    if hall_dir.is_dir() and not_hall_dir.is_dir():
        hall_ids_path = hall_dir / "layer0_instance_ids.json"
        not_hall_ids_path = not_hall_dir / "layer0_instance_ids.json"
        if not hall_ids_path.exists() or not not_hall_ids_path.exists():
            raise FileNotFoundError(
                f"Missing layer0 id files in new structure for {model_name}/{dataset} at {base}"
            )
        hall_ids = _read_int_list(hall_ids_path)
        not_hall_ids = _read_int_list(not_hall_ids_path)
        structure = "new"
    else:
        labels_path = dataset_dir / "generations" / "hallucination_labels.json"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Neither new structure nor generations labels found for {model_name}/{dataset} in {dataset_dir}"
            )
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        hall_ids = [int(item["instance_id"]) for item in labels if int(item["is_hallucination"]) == 1]
        not_hall_ids = [int(item["instance_id"]) for item in labels if int(item["is_hallucination"]) == 0]
        structure = "old"

    hall_count = len(hall_ids)
    not_hall_count = len(not_hall_ids)
    total = hall_count + not_hall_count
    rate = float(hall_count / total) if total > 0 else None

    return {
        "model_name": model_name,
        "resolved_model_dir": model_dir,
        "dataset": dataset,
        "structure": structure,
        "total_instances": total,
        "hallucinated_count": hall_count,
        "not_hallucinated_count": not_hall_count,
        "hallucination_rate": rate,
        "hallucinated_ids_set": set(hall_ids),
        "not_hallucinated_ids_set": set(not_hall_ids),
    }


def per_model_sampling_counts(stats: dict, train_split: float) -> dict:
    hall = int(stats["hallucinated_count"])
    non = int(stats["not_hallucinated_count"])
    balanced_per_class = min(hall, non)
    balanced_total = 2 * balanced_per_class
    train_count = int(train_split * balanced_total)
    test_count = balanced_total - train_count
    return {
        "raw_total_instances": int(stats["total_instances"]),
        "raw_hallucinated": hall,
        "raw_not_hallucinated": non,
        "balanced_per_class": balanced_per_class,
        "balanced_total_for_probing": balanced_total,
        "train_count_after_split": train_count,
        "test_count_after_split": test_count,
    }


def intersection_counts(stats_a: dict, stats_b: dict, alignment_split: float) -> dict:
    hall_a = stats_a["hallucinated_ids_set"]
    hall_b = stats_b["hallucinated_ids_set"]
    all_a = hall_a | stats_a["not_hallucinated_ids_set"]
    all_b = hall_b | stats_b["not_hallucinated_ids_set"]

    common = all_a & all_b
    common_count = len(common)

    concordant_hall = 0
    concordant_non = 0
    discordant = 0
    for idx in common:
        ya = 1 if idx in hall_a else 0
        yb = 1 if idx in hall_b else 0
        if ya == yb == 1:
            concordant_hall += 1
        elif ya == yb == 0:
            concordant_non += 1
        else:
            discordant += 1

    concordant_total = concordant_hall + concordant_non
    balanced_per_class = min(concordant_hall, concordant_non)
    balanced_total = 2 * balanced_per_class
    align_train = int(alignment_split * balanced_total)
    align_val = balanced_total - align_train

    return {
        "common_instance_ids_count": common_count,
        "concordant_total_count": concordant_total,
        "concordant_hallucinated_count": concordant_hall,
        "concordant_not_hallucinated_count": concordant_non,
        "discordant_count": discordant,
        "balanced_concordant_per_class": balanced_per_class,
        "balanced_concordant_total_for_alignment": balanced_total,
        "alignment_train_count_after_split": align_train,
        "alignment_val_count_after_split": align_val,
    }


def _json_safe_stats(stats: dict) -> dict:
    return {
        "model_name": stats["model_name"],
        "resolved_model_dir": stats["resolved_model_dir"],
        "dataset": stats["dataset"],
        "structure": stats["structure"],
        "total_instances": stats["total_instances"],
        "hallucinated_count": stats["hallucinated_count"],
        "not_hallucinated_count": stats["not_hallucinated_count"],
        "hallucination_rate": stats["hallucination_rate"],
    }


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]
    config_path = (project_root / args.config_path).resolve()
    cfg = load_o4a_config_literals(config_path)

    cache_root = (project_root / args.cache_dir).resolve() if args.cache_dir else (project_root / cfg["CACHE_DIR_NAME"]).resolve()
    output_path = (project_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    experiments: dict = cfg["EXPERIMENTS"]
    model_aliases: dict = cfg["MODEL_ALIASES"]
    train_split: float = float(cfg["TRAIN_SPLIT"])
    alignment_split: float = float(cfg["ALIGNMENT_SPLIT"])
    prober_val_split: float = float(cfg["PROBER_VAL_SPLIT"])
    layer_types: list[str] = list(cfg["LAYER_TYPES"])

    print("=" * 72)
    print("Export O4A Sampling Stats")
    print("=" * 72)
    print(f"Config: {config_path}")
    print(f"Cache root: {cache_root}")
    print(f"Stats layer type for new structure: {args.stats_layer_type}")
    print(f"Experiments: {len(experiments)}")

    # Cache loaded stats per (model,dataset)
    stats_cache: dict[tuple[str, str], dict] = {}
    missing_pairs: dict[tuple[str, str], str] = {}

    def get_stats(model: str, dataset: str) -> dict | None:
        key = (model, dataset)
        if key in stats_cache:
            return stats_cache[key]
        if key in missing_pairs:
            return None
        try:
            stats = load_model_dataset_stats(
                cache_root=cache_root,
                model_name=model,
                dataset=dataset,
                model_aliases=model_aliases,
                layer_type_for_new_structure=args.stats_layer_type,
            )
            stats_cache[key] = stats
            return stats
        except Exception as exc:
            msg = str(exc)
            missing_pairs[key] = msg
            if args.strict:
                raise
            return None

    # Collect all pairs used by experiments
    all_pairs = set()
    for exp_cfg in experiments.values():
        all_pairs.add((exp_cfg["trainer"], exp_cfg["dataset"]))
        all_pairs.add((exp_cfg["tester"], exp_cfg["dataset"]))

    for model, dataset in sorted(all_pairs):
        get_stats(model, dataset)

    # Hallucination rate section (independent from experiment; grouped by model and dataset)
    hallucination_rate_by_model: dict[str, dict] = {}
    for (model, dataset), stats in sorted(stats_cache.items()):
        m = hallucination_rate_by_model.setdefault(
            model, {"datasets": {}, "aggregate": {"total_instances": 0, "hallucinated_count": 0}}
        )
        m["datasets"][dataset] = {
            "total_instances": stats["total_instances"],
            "hallucinated_count": stats["hallucinated_count"],
            "not_hallucinated_count": stats["not_hallucinated_count"],
            "hallucination_rate": stats["hallucination_rate"],
            "structure": stats["structure"],
            "resolved_model_dir": stats["resolved_model_dir"],
        }
        m["aggregate"]["total_instances"] += int(stats["total_instances"])
        m["aggregate"]["hallucinated_count"] += int(stats["hallucinated_count"])

    for model, payload in hallucination_rate_by_model.items():
        total = payload["aggregate"]["total_instances"]
        hall = payload["aggregate"]["hallucinated_count"]
        payload["aggregate"]["hallucination_rate_weighted"] = (float(hall / total) if total > 0 else None)
        payload["aggregate"]["not_hallucinated_count"] = total - hall
        payload["aggregate"]["num_datasets"] = len(payload["datasets"])

    # Experiment section
    experiments_out: dict[str, dict] = {}
    for exp_name, exp_cfg in sorted(experiments.items()):
        trainer = exp_cfg["trainer"]
        tester = exp_cfg["tester"]
        dataset = exp_cfg["dataset"]

        trainer_stats = get_stats(trainer, dataset)
        tester_stats = get_stats(tester, dataset)

        entry = {
            "dataset": dataset,
            "trainer": trainer,
            "tester": tester,
            "layer_types": layer_types,
            "count_logic_reference": "src/o4a/data.py::prepare_shared_data",
            "stats_layer_type_used_for_new_structure": args.stats_layer_type,
            "splits": {
                "train_split": train_split,
                "alignment_split": alignment_split,
                "prober_val_split": prober_val_split,
            },
        }

        if trainer_stats is None or tester_stats is None:
            errors = {}
            if trainer_stats is None:
                errors["trainer"] = missing_pairs.get((trainer, dataset), "missing trainer stats")
            if tester_stats is None:
                errors["tester"] = missing_pairs.get((tester, dataset), "missing tester stats")
            entry["status"] = "missing_data"
            entry["errors"] = errors
            experiments_out[exp_name] = entry
            continue

        trainer_counts = per_model_sampling_counts(trainer_stats, train_split)
        tester_counts = per_model_sampling_counts(tester_stats, train_split)
        inter_counts = intersection_counts(trainer_stats, tester_stats, alignment_split)

        # In prepare_shared_data, prober split is applied on trainer train split only.
        trainer_prober_val = int(prober_val_split * trainer_counts["train_count_after_split"])
        trainer_prober_train = trainer_counts["train_count_after_split"] - trainer_prober_val

        entry["status"] = "ok"
        entry["models"] = {
            "trainer": {
                "stats": _json_safe_stats(trainer_stats),
                "sampling_counts": trainer_counts,
            },
            "tester": {
                "stats": _json_safe_stats(tester_stats),
                "sampling_counts": tester_counts,
            },
        }
        entry["intersection_and_alignment_counts"] = inter_counts
        entry["trainer_prober_split_counts"] = {
            "train_after_prober_split": trainer_prober_train,
            "val_after_prober_split": trainer_prober_val,
            "source_train_count": trainer_counts["train_count_after_split"],
        }
        experiments_out[exp_name] = entry

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "config_path": str(config_path),
        "cache_root": str(cache_root),
        "stats_layer_type": args.stats_layer_type,
        "num_experiments": len(experiments),
        "num_model_dataset_pairs_found": len(stats_cache),
        "num_model_dataset_pairs_missing": len(missing_pairs),
        "missing_model_dataset_pairs": [
            {"model": m, "dataset": d, "error": err}
            for (m, d), err in sorted(missing_pairs.items())
        ],
        "hallucination_rate_by_model": hallucination_rate_by_model,
        "experiments": experiments_out,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Saved JSON: {output_path}")
    print(f"Found model/dataset pairs: {len(stats_cache)}")
    print(f"Missing model/dataset pairs: {len(missing_pairs)}")


if __name__ == "__main__":
    main()
