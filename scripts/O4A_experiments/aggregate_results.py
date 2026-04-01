"""Aggregate all per-seed O4A experiment CSVs into a single CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge all results.csv files under results/experiments into one CSV."
    )
    parser.add_argument(
        "--input-dir",
        default="results/experiments",
        help="Directory that contains experiment subfolders (default: results/experiments).",
    )
    parser.add_argument(
        "--output",
        default="results/experiments/merged_results.csv",
        help="Output CSV path (default: results/experiments/merged_results.csv).",
    )
    parser.add_argument(
        "--filename",
        default="results.csv",
        help="Result filename to search recursively (default: results.csv).",
    )
    return parser.parse_args()


def discover_csv_files(input_dir: Path, filename: str, output_path: Path) -> list[Path]:
    files = sorted(p for p in input_dir.rglob(filename) if p.is_file())
    output_resolved = output_path.resolve()
    return [p for p in files if p.resolve() != output_resolved]


def aggregate(files: list[Path], repo_root: Path, input_dir: Path) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames: list[str] = ["source_file", "source_experiment", "source_seed"]
    rows: list[dict[str, str]] = []

    for csv_path in files:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue

            for col in reader.fieldnames:
                if col not in fieldnames:
                    fieldnames.append(col)

            try:
                rel_source = str(csv_path.relative_to(repo_root))
            except ValueError:
                rel_source = str(csv_path)

            rel_to_input = csv_path.relative_to(input_dir)
            source_experiment = rel_to_input.parts[0] if len(rel_to_input.parts) >= 3 else ""
            source_seed = rel_to_input.parts[1] if len(rel_to_input.parts) >= 3 else csv_path.parent.name

            for row in reader:
                normalized = {key: value for key, value in row.items() if key is not None}
                normalized["source_file"] = rel_source
                normalized["source_experiment"] = source_experiment
                normalized["source_seed"] = source_seed
                rows.append(normalized)

    return fieldnames, rows


def write_output(output_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            safe_row = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(safe_row)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output).resolve()
    repo_root = Path.cwd().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = discover_csv_files(input_dir=input_dir, filename=args.filename, output_path=output_path)
    if not files:
        raise FileNotFoundError(f"No '{args.filename}' files found under: {input_dir}")

    fieldnames, rows = aggregate(files=files, repo_root=repo_root, input_dir=input_dir)
    write_output(output_path=output_path, fieldnames=fieldnames, rows=rows)

    print(f"Discovered files: {len(files)}")
    print(f"Aggregated rows: {len(rows)}")
    print(f"Output CSV: {output_path}")


if __name__ == "__main__":
    main()
