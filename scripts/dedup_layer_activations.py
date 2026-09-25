#!/usr/bin/env python3
"""Detect and remove duplicated instances from cached layer activations.

Layout assumed:
    activation_cache/<model_name>/<dataset_name>/activation_<type>/layer<N>_activations.pt
    activation_cache/<model_name>/<dataset_name>/activation_<type>/layer<N>_instance_ids.json

Instance ids are 0-based and contiguous, so a clean file satisfies
    max(ids) + 1 == len(ids)
For every (model, dataset, activation type, layer) the instance-id file is
checked against that invariant. When it fails and the file contains repeated
ids, the first occurrence of each id is kept and the duplicated rows are
dropped from both the .pt tensor and the .json file (row i of the tensor
belongs to id i of the json). The .pt file is only loaded for files that fail
the check.

A mismatch without duplicates (i.e. gaps in the ids) cannot be fixed by
deduplication and is only reported.

Usage:
    python dedup_layer_activations.py [--root activation_cache] [--model M ...]
        [--dataset D ...] [--dry-run] [--backup]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import torch

ACTIVATION_DIR_PREFIX = "activation_"
INSTANCE_IDS_RE = re.compile(r"^layer(\d+)_instance_ids\.json$")


def subdirs(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_dir())


def find_layer_files(activation_type_dir: Path) -> list[tuple[int, Path, Path]]:
    """Return (layer_idx, ids_path, activations_path) sorted by layer."""
    found = []
    for p in activation_type_dir.iterdir():
        m = INSTANCE_IDS_RE.match(p.name)
        if m:
            layer_idx = int(m.group(1))
            found.append((layer_idx, p, activation_type_dir / f"layer{layer_idx}_activations.pt"))
    return sorted(found)


def first_occurrence_indices(ids: list) -> list[int]:
    seen: set[int] = set()
    keep = []
    for i, x in enumerate(ids):
        key = int(x)
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return keep


def atomic_write_json(path: Path, data: list) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(data, f, indent=4)
    os.replace(tmp_path, path)


def atomic_write_tensor(path: Path, tensor: torch.Tensor) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor, tmp_path)
    os.replace(tmp_path, path)


def check_and_fix(ids_path: Path, act_path: Path, dry_run: bool, backup: bool) -> str:
    """Return one of: 'ok', 'fixed', 'gaps', 'error'."""
    try:
        with ids_path.open("r") as f:
            ids = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"ERROR     {ids_path}: cannot read ids: {e}", file=sys.stderr)
        return "error"

    if not isinstance(ids, list) or not ids:
        print(f"ERROR     {ids_path}: expected a non-empty list of ids", file=sys.stderr)
        return "error"

    max_id = max(int(x) for x in ids)
    if max_id + 1 == len(ids):
        return "ok"

    keep = first_occurrence_indices(ids)
    n_dups = len(ids) - len(keep)
    if n_dups == 0:
        print(f"GAPS      {ids_path}: max_id={max_id} but {len(ids)} ids and no duplicates; not fixable by dedup")
        return "gaps"

    if not act_path.is_file():
        print(f"ERROR     {ids_path}: {n_dups} duplicates but activations file is missing: {act_path}", file=sys.stderr)
        return "error"

    try:
        activations = torch.load(act_path, map_location="cpu")
    except Exception as e:  # noqa: BLE001 - report any load failure and move on
        print(f"ERROR     {act_path}: cannot load: {e}", file=sys.stderr)
        return "error"

    if not isinstance(activations, torch.Tensor):
        print(f"ERROR     {act_path}: expected a tensor, got {type(activations).__name__}", file=sys.stderr)
        return "error"

    if activations.shape[0] != len(ids):
        print(
            f"ERROR     {act_path}: tensor has {activations.shape[0]} rows but ids file has {len(ids)}; "
            "rows cannot be matched to ids, skipped",
            file=sys.stderr,
        )
        return "error"

    new_ids = [ids[i] for i in keep]
    new_activations = activations[torch.as_tensor(keep, dtype=torch.long)].clone()
    new_max = max(int(x) for x in new_ids)
    residual = "" if new_max + 1 == len(new_ids) else f" (still max_id={new_max} vs {len(new_ids)} ids: gaps remain)"

    action = "WOULD FIX" if dry_run else "FIXED"
    print(
        f"{action:9s} {act_path.parent}/layer*: -{n_dups} duplicates, "
        f"{tuple(activations.shape)} -> {tuple(new_activations.shape)}{residual}"
    )

    if not dry_run:
        if backup:
            shutil.copy2(ids_path, ids_path.with_suffix(ids_path.suffix + ".bak"))
            shutil.copy2(act_path, act_path.with_suffix(act_path.suffix + ".bak"))
        atomic_write_tensor(act_path, new_activations)
        atomic_write_json(ids_path, new_ids)

    return "fixed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="activation_cache", help="root activation_cache directory (default: %(default)s)")
    parser.add_argument("--model", nargs="*", help="restrict to these model names (default: all)")
    parser.add_argument("--dataset", nargs="*", help="restrict to these dataset names (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="report what would change without writing anything")
    parser.add_argument("--backup", action="store_true", help="keep .bak copies of every file that is rewritten")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"error: root directory not found: {root}", file=sys.stderr)
        return 1

    counts = {"ok": 0, "fixed": 0, "gaps": 0, "error": 0}
    for model_dir in subdirs(root):
        if args.model and model_dir.name not in args.model:
            continue
        for dataset_dir in subdirs(model_dir):
            if args.dataset and dataset_dir.name not in args.dataset:
                continue
            for activation_type_dir in subdirs(dataset_dir):
                if not activation_type_dir.name.startswith(ACTIVATION_DIR_PREFIX):
                    continue
                layer_files = find_layer_files(activation_type_dir)
                before = dict(counts)
                for _, ids_path, act_path in layer_files:
                    counts[check_and_fix(ids_path, act_path, args.dry_run, args.backup)] += 1
                n_bad = sum(counts[k] - before[k] for k in ("fixed", "gaps", "error"))
                print(f"checked   {activation_type_dir.relative_to(root)}: {len(layer_files)} layers, {n_bad} flagged")

    verb = "would be fixed" if args.dry_run else "fixed"
    total = sum(counts.values())
    print(
        f"\nDone. {total} layer files checked: {counts['ok']} ok, {counts['fixed']} {verb}, "
        f"{counts['gaps']} with gaps, {counts['error']} errors."
    )
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
