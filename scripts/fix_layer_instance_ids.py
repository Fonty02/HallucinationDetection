#!/usr/bin/env python3
"""Fix layer*_instance_ids.json files that are missing instance ids.

Layout assumed:
    activation_cache/<model_name>/<dataset_name>/activation_<type>/layer*_instance_ids.json
    activation_cache/<model_name>/<dataset_name>/generations/generation_<id>.json

The `generations` folder has exactly one file per instance id in the dataset,
so it is treated as the ground truth. Instance ids do not depend on the
activation type, so the ground truth is read from disk exactly once and then
reused to fix every layer*_instance_ids.json file under every
activation_<type> directory.

Usage:
    python fix_layer_instance_ids.py --model MODEL --dataset DATASET \
        [--root activation_cache] [--dry-run] [--no-backup] [--strict] \
        [--jobs N] [-v]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GENERATION_RE = re.compile(r"^generation_(.+)\.json$")
LAYER_FILE_PREFIX = "layer"
LAYER_FILE_SUFFIX = "_instance_ids.json"
ACTIVATION_DIR_PREFIX = "activation_"


def natural_sort_key(value):
    """Sort numeric-looking ids numerically, everything else lexically."""
    s = str(value)
    return (0, int(s)) if s.isdigit() else (1, s)


def load_ground_truth_ids(generations_dir: Path) -> list[str]:
    """Scan the generations folder once and return the sorted list of ids."""
    ids: list[str] = []
    with os.scandir(generations_dir) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            m = GENERATION_RE.match(entry.name)
            if m:
                ids.append(m.group(1))
    ids.sort(key=natural_sort_key)
    return ids


def find_activation_type_dirs(dataset_dir: Path) -> list[Path]:
    with os.scandir(dataset_dir) as entries:
        return [
            Path(e.path)
            for e in entries
            if e.is_dir() and e.name.startswith(ACTIVATION_DIR_PREFIX)
        ]


def find_layer_instance_files(activation_type_dir: Path) -> list[Path]:
    with os.scandir(activation_type_dir) as entries:
        return [
            Path(e.path)
            for e in entries
            if e.is_file()
            and e.name.startswith(LAYER_FILE_PREFIX)
            and e.name.endswith(LAYER_FILE_SUFFIX)
        ]


def cast_like(id_str: str, sample):
    """Cast a ground-truth id (str) to match the type already used in a file."""
    if isinstance(sample, int):
        try:
            return int(id_str)
        except ValueError:
            return id_str
    return id_str


def build_fixed_content(current: list, ground_truth: list[str], strict: bool):
    """Return (new_list, added, removed) or (None, [], []) if already correct."""
    sample = current[0] if current else None
    current_as_str = {str(x) for x in current}
    truth_set = set(ground_truth)

    missing = [gid for gid in ground_truth if gid not in current_as_str]
    extra = [x for x in current if str(x) not in truth_set]

    if not missing and (not strict or not extra):
        return None, [], []

    new_list = list(current)
    if strict and extra:
        extra_set = {str(x) for x in extra}
        new_list = [x for x in new_list if str(x) not in extra_set]
    if missing:
        new_list.extend(cast_like(gid, sample) for gid in missing)

    return new_list, missing, (extra if strict else [])


def fix_file(path: Path, ground_truth: list[str], strict: bool, dry_run: bool, verbose: bool):
    try:
        with path.open("r") as f:
            current = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return path, None, None, f"error reading file: {e}"

    if not isinstance(current, list):
        return path, None, None, "unexpected JSON shape (expected a list), skipped"

    new_list, added, removed = build_fixed_content(current, ground_truth, strict)
    if new_list is None:
        if verbose:
            print(f"OK      {path} ({len(current)} ids, already complete)")
        return path, [], [], None

    if not dry_run:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w") as f:
            json.dump(new_list, f)
        os.replace(tmp_path, path)

    action = "WOULD FIX" if dry_run else "FIXED"
    print(f"{action:9s} {path}: +{len(added)} missing" + (f", -{len(removed)} stray" if removed else ""))
    return path, added, removed, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model_name under activation_cache/")
    parser.add_argument("--dataset", required=True, help="dataset_name under activation_cache/<model_name>/")
    parser.add_argument("--root", default="activation_cache", help="root activation_cache directory (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true", help="report what would change without writing anything")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also drop ids present in a layer file but absent from the generations ground truth",
    )
    parser.add_argument("--jobs", type=int, default=min(32, (os.cpu_count() or 4) * 4), help="parallel workers for fixing files")
    parser.add_argument("-v", "--verbose", action="store_true", help="print a line for every up-to-date file too")
    args = parser.parse_args()

    dataset_dir = Path(args.root) / args.model / args.dataset
    generations_dir = dataset_dir / "generations"

    if not generations_dir.is_dir():
        print(f"error: generations directory not found: {generations_dir}", file=sys.stderr)
        return 1

    ground_truth = load_ground_truth_ids(generations_dir)
    if not ground_truth:
        print(f"error: no generation_<id>.json files found under {generations_dir}", file=sys.stderr)
        return 1
    print(f"Ground truth: {len(ground_truth)} instance ids (read once from {generations_dir})")

    activation_type_dirs = find_activation_type_dirs(dataset_dir)
    if not activation_type_dirs:
        print(f"error: no activation_<type> directories found under {dataset_dir}", file=sys.stderr)
        return 1

    layer_files: list[Path] = []
    for d in activation_type_dirs:
        layer_files.extend(find_layer_instance_files(d))

    if not layer_files:
        print(f"error: no layer*_instance_ids.json files found under {dataset_dir}", file=sys.stderr)
        return 1

    print(f"Checking {len(layer_files)} layer instance-id files across {len(activation_type_dirs)} activation types...")

    n_fixed = n_errors = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(fix_file, p, ground_truth, args.strict, args.dry_run, args.verbose)
            for p in layer_files
        ]
        for fut in futures:
            path, added, removed, err = fut.result()
            if err:
                n_errors += 1
                print(f"ERROR   {path}: {err}", file=sys.stderr)
            elif added:
                n_fixed += 1

    verb = "would be fixed" if args.dry_run else "fixed"
    print(f"\nDone. {n_fixed}/{len(layer_files)} files {verb}, {n_errors} errors.")
    return 1 if n_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
