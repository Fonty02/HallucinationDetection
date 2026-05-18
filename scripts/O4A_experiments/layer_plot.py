#!/usr/bin/env python3
"""
Plot layer-wise metrics from all_layers_sorted_*.json files.

Compared to the notebook plot:
1) metric is configurable via CLI (--metric)
2) one figure per layer type (hidden/mlp/attn)
3) in each subplot (dataset) curves represent different models
4) output is saved as PDF
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASET_CODE_TO_NAME = {
    "BBC": "Belief Bank Constraints",
    "BBF": "Belief Bank Facts",
    "HE": "HaluEval",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create per-layer-type multi-dataset plots with one curve per model."
    )
    parser.add_argument(
        "--json-files",
        nargs="*",
        default=[],
        help=(
            "Explicit JSON paths. If omitted, files are discovered with "
            "--json-dir/--json-pattern."
        ),
    )
    parser.add_argument(
        "--json-dir",
        default="notebooks/layersStudies",
        help="Directory used for JSON discovery (default: notebooks/layersStudies).",
    )
    parser.add_argument(
        "--json-pattern",
        default="all_layers_sorted_*.json",
        help="Glob pattern for JSON discovery (default: all_layers_sorted_*.json).",
    )
    parser.add_argument(
        "--dataset-order",
        nargs="+",
        default=["BBC", "BBF", "HE"],
        help=(
            "Dataset codes order for subplot columns "
            "(default: BBC BBF HE; unknown codes are appended)."
        ),
    )
    parser.add_argument(
        "--layer-types",
        nargs="+",
        default=["hidden", "mlp", "attn"],
        help="Layer types to plot (default: hidden mlp attn).",
    )
    parser.add_argument(
        "--metric",
        default="accuracy",
        help="Metric key to plot (use --list-metrics to inspect available keys).",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="Print available metrics, datasets and layer types, then exit.",
    )
    parser.add_argument(
        "--output-dir",
        default="notebooks/layersStudies/img",
        help="Directory where PDFs are saved (default: notebooks/layersStudies/img).",
    )
    parser.add_argument(
        "--filename-prefix",
        default="",
        help="Optional prefix added to output filenames.",
    )
    return parser.parse_args()


def discover_json_files(json_dir: Path, json_pattern: str) -> list[Path]:
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")
    files = sorted(p for p in json_dir.glob(json_pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(
            f"No JSON files found in {json_dir} with pattern '{json_pattern}'."
        )
    return files


def extract_dataset_code(json_path: Path) -> str:
    stem_parts = json_path.stem.split("_")
    if not stem_parts:
        raise ValueError(f"Cannot extract dataset code from file name: {json_path.name}")
    return stem_parts[-1].upper()


def dataset_title(dataset_code: str) -> str:
    return DATASET_CODE_TO_NAME.get(dataset_code, dataset_code)


def sanitize_filename_part(value: str) -> str:
    safe = []
    for ch in value:
        safe.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    normalized = "".join(safe).strip("_")
    return normalized or "plot"


def load_grouped_data(
    json_files: list[Path],
) -> tuple[
    dict[str, dict[str, dict[str, list[dict[str, float]]]]],
    set[str],
    set[str],
]:
    grouped: dict[str, dict[str, dict[str, list[dict[str, float]]]]] = {}
    metric_keys: set[str] = set()
    layer_types: set[str] = set()

    for json_path in json_files:
        dataset_code = extract_dataset_code(json_path)
        grouped.setdefault(dataset_code, {})

        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, dict):
            raise ValueError(f"Unexpected JSON structure in {json_path}: root must be a dict.")

        for model_name, model_data in data.items():
            if not isinstance(model_data, dict):
                raise ValueError(
                    f"Unexpected model payload for '{model_name}' in {json_path}: expected dict."
                )
            grouped[dataset_code][model_name] = model_data
            for layer_type, points in model_data.items():
                layer_types.add(layer_type)
                for point in points:
                    metric_keys.update(k for k in point.keys() if k != "layer")

    return grouped, metric_keys, layer_types


def sort_dataset_codes(
    grouped_data: dict[str, dict[str, dict[str, list[dict[str, float]]]]],
    requested_order: list[str],
) -> list[str]:
    available = list(grouped_data.keys())
    ordered = []
    for code in requested_order:
        upper_code = code.upper()
        if upper_code in grouped_data and upper_code not in ordered:
            ordered.append(upper_code)
    remaining = sorted(code for code in available if code not in ordered)
    return ordered + remaining


def metric_title(metric: str) -> str:
    upper_aliases = {"auroc": "AUROC", "auc": "AUC", "f1": "F1"}
    if metric.lower() in upper_aliases:
        return upper_aliases[metric.lower()]
    return metric.replace("_", " ").title()


def plot_layer_type(
    grouped_data: dict[str, dict[str, dict[str, list[dict[str, float]]]]],
    dataset_codes: list[str],
    layer_type: str,
    metric: str,
    output_dir: Path,
    filename_prefix: str,
) -> Path:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 9,
            "legend.title_fontsize": 10,
            "lines.linewidth": 2,
        }
    )

    fig, axes = plt.subplots(1, len(dataset_codes), figsize=(5 * len(dataset_codes), 4), squeeze=False)

    for col_idx, dataset_code in enumerate(dataset_codes):
        ax = axes[0][col_idx]
        models = grouped_data.get(dataset_code, {})
        plotted_any = False

        for model_name in sorted(models.keys()):
            model_payload = models[model_name]
            points = model_payload.get(layer_type, [])
            if not points:
                continue

            sorted_points = sorted(points, key=lambda item: item["layer"])
            filtered = [item for item in sorted_points if metric in item]
            if not filtered:
                continue

            layers = [item["layer"] for item in filtered]
            values = [item[metric] for item in filtered]
            line, = ax.plot(layers, values, label=model_name)
            ax.plot(
                layers[-1],
                values[-1],
                marker="x",
                linestyle="None",
                color=line.get_color(),
                markersize=7,
                markeredgewidth=2,
                label="_nolegend_",
            )
            plotted_any = True

        ax.set_title(dataset_title(dataset_code))
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric_title(metric))
        ax.grid(True, linestyle="-", alpha=0.9)

        if plotted_any:
            legend = ax.legend(title="model", loc="upper left", frameon=True)
            plt.setp(legend.get_title(), fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = sanitize_filename_part(filename_prefix) if filename_prefix else ""
    metric_part = sanitize_filename_part(metric)
    layer_part = sanitize_filename_part(layer_type)
    filename = f"{metric_part}_{layer_part}_models_by_dataset.pdf"
    if prefix:
        filename = f"{prefix}_{filename}"

    output_path = output_dir / filename
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()

    json_files = [Path(path).resolve() for path in args.json_files]
    if not json_files:
        json_files = discover_json_files(
            json_dir=Path(args.json_dir).resolve(),
            json_pattern=args.json_pattern,
        )

    grouped_data, metric_keys, available_layer_types = load_grouped_data(json_files)
    dataset_codes = sort_dataset_codes(grouped_data, args.dataset_order)

    if args.list_metrics:
        print("Available metrics:", ", ".join(sorted(metric_keys)))
        print("Available layer types:", ", ".join(sorted(available_layer_types)))
        print("Available dataset codes:", ", ".join(dataset_codes))
        print("Discovered JSON files:")
        for path in json_files:
            print(f"  - {path}")
        return

    if args.metric not in metric_keys:
        raise ValueError(
            f"Metric '{args.metric}' not found. Available metrics: {', '.join(sorted(metric_keys))}"
        )

    requested_layer_types = []
    for layer_type in args.layer_types:
        normalized = layer_type.strip()
        if normalized not in available_layer_types:
            raise ValueError(
                f"Layer type '{normalized}' not found. "
                f"Available layer types: {', '.join(sorted(available_layer_types))}"
            )
        requested_layer_types.append(normalized)

    output_dir = Path(args.output_dir).resolve()
    print("Creating plots...")
    print(f"Metric: {args.metric}")
    print(f"Layer types: {', '.join(requested_layer_types)}")
    print(f"Datasets: {', '.join(dataset_codes)}")
    print(f"Output directory: {output_dir}")

    for layer_type in requested_layer_types:
        output_path = plot_layer_type(
            grouped_data=grouped_data,
            dataset_codes=dataset_codes,
            layer_type=layer_type,
            metric=args.metric,
            output_dir=output_dir,
            filename_prefix=args.filename_prefix,
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
