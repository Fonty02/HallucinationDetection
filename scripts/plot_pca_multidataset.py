#!/usr/bin/env python3
"""
Plot PCA scatter charts for one LLM across multiple datasets/layers.

Example:
    python scripts/plot_pca_multidataset.py \
        --model-name gemma-2-9b-it \
        --dataset-layers belief_bank_constraints:23,belief_bank_facts:21,halu_eval:21 \
        --layer-type attn \
        --max-points-per-class 1500 \
        --balance
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DATASET_TITLES = {
    "belief_bank_constraints": "Belief Bank Constraints",
    "belief_bank_facts": "Belief Bank Facts",
    "halu_eval": "HaluEval",
}

LAYER_TYPE_TITLES = {
    "attn": "Attention",
    "mlp": "MLP",
    "hidden": "Hidden-state",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PCA multi-dataset scatter plots (hallucinated vs not hallucinated)."
    )
    parser.add_argument("--cache-dir", type=str, default="activation_cache")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument(
        "--dataset-layer",
        action="append",
        default=[],
        help="Single pair in format dataset:layer (can be repeated).",
    )
    parser.add_argument(
        "--dataset-layers",
        type=str,
        default="",
        help="Comma-separated pairs: dataset:layer,dataset:layer,...",
    )
    parser.add_argument("--layer-type", type=str, default="attn", choices=["attn", "mlp", "hidden"])
    parser.add_argument("--output-dir", type=str, default=os.path.join("results", "pca_multidataset"))
    parser.add_argument("--filename", type=str, default="")
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--max-points-per-class", type=int, default=1500)
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--standardize", dest="standardize", action="store_true")
    parser.add_argument("--no-standardize", dest="standardize", action="store_false")
    parser.add_argument(
        "--axis-quantile",
        type=float,
        default=0.995,
        help=(
            "Robust axis range quantile in (0.5, 1.0]. "
            "Example: 0.995 keeps the central 99.5%% of projected points per axis."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(standardize=True)
    return parser.parse_args()


def parse_dataset_layers(dataset_layer_args: list[str], dataset_layers_compact: str) -> list[tuple[str, int]]:
    raw_items: list[str] = []
    if dataset_layers_compact.strip():
        raw_items.extend([item.strip() for item in dataset_layers_compact.split(",") if item.strip()])
    raw_items.extend(dataset_layer_args)

    if not raw_items:
        raise ValueError(
            "No dataset/layer pairs provided. Use --dataset-layer dataset:layer or --dataset-layers."
        )

    parsed: list[tuple[str, int]] = []
    seen: set[str] = set()
    for item in raw_items:
        if ":" not in item:
            raise ValueError(f"Invalid dataset/layer pair '{item}'. Expected format dataset:layer.")
        dataset_raw, layer_raw = item.split(":", 1)
        dataset = dataset_raw.strip()
        layer_str = layer_raw.strip()
        if not dataset:
            raise ValueError(f"Invalid dataset name in pair '{item}'.")
        if dataset in seen:
            raise ValueError(f"Dataset '{dataset}' appears multiple times. Use each dataset once.")
        try:
            layer = int(layer_str)
        except ValueError as exc:
            raise ValueError(f"Layer must be integer in pair '{item}'.") from exc
        seen.add(dataset)
        parsed.append((dataset, layer))
    return parsed


def resolve_model_dir(cache_root: Path, model_name: str) -> str:
    if not cache_root.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_root}")

    available = [p.name for p in cache_root.iterdir() if p.is_dir()]
    candidates = [model_name]
    if "/" in model_name:
        last = model_name.split("/")[-1]
        candidates.extend([last, model_name.replace("/", "_"), model_name.replace("/", "-")])

    for candidate in candidates:
        if (cache_root / candidate).is_dir():
            return candidate

    available_lower = {name.lower(): name for name in available}
    for candidate in candidates:
        resolved = available_lower.get(candidate.lower())
        if resolved is not None:
            return resolved

    available_str = ", ".join(sorted(available))
    raise FileNotFoundError(
        f"Model folder for '{model_name}' not found in {cache_root}. Available: {available_str}"
    )


def load_tensor_as_matrix(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().float().numpy()
    elif isinstance(obj, np.ndarray):
        arr = obj.astype(np.float32, copy=False)
    else:
        arr = np.asarray(obj, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def load_labels_old_structure(labels_path: Path, instance_ids: list[int] | None) -> np.ndarray:
    with open(labels_path, "r", encoding="utf-8") as f:
        labels_data = json.load(f)

    if instance_ids is None:
        return np.asarray([int(item["is_hallucination"]) for item in labels_data], dtype=np.int8)

    label_by_id = {int(item["instance_id"]): int(item["is_hallucination"]) for item in labels_data}
    labels = []
    for instance_id in instance_ids:
        idx = int(instance_id)
        if idx not in label_by_id:
            raise KeyError(f"Instance id {idx} not found in {labels_path}")
        labels.append(label_by_id[idx])
    return np.asarray(labels, dtype=np.int8)


def load_dataset_layer(
    cache_root: Path,
    model_dir: str,
    dataset: str,
    layer: int,
    layer_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    base_dir = cache_root / model_dir / dataset / f"activation_{layer_type}"
    if not base_dir.exists():
        raise FileNotFoundError(f"Activation directory not found: {base_dir}")

    hall_dir = base_dir / "hallucinated"
    not_hall_dir = base_dir / "not_hallucinated"

    # New structure: activation_<type>/hallucinated + not_hallucinated
    if hall_dir.is_dir() and not_hall_dir.is_dir():
        hall_path = hall_dir / f"layer{layer}_activations.pt"
        not_hall_path = not_hall_dir / f"layer{layer}_activations.pt"
        if not hall_path.exists():
            raise FileNotFoundError(f"Missing file: {hall_path}")
        if not not_hall_path.exists():
            raise FileNotFoundError(f"Missing file: {not_hall_path}")

        hall = load_tensor_as_matrix(hall_path)
        not_hall = load_tensor_as_matrix(not_hall_path)
        if hall.shape[1] != not_hall.shape[1]:
            raise ValueError(
                f"Feature mismatch in {dataset} layer {layer}: "
                f"{hall.shape[1]} vs {not_hall.shape[1]}"
            )
        x = np.vstack([not_hall, hall])
        y = np.concatenate(
            [
                np.zeros(not_hall.shape[0], dtype=np.int8),
                np.ones(hall.shape[0], dtype=np.int8),
            ]
        )
        return x, y

    # Old structure: activation_<type>/layerX_activations + generations labels
    act_path = base_dir / f"layer{layer}_activations.pt"
    ids_path = base_dir / f"layer{layer}_instance_ids.json"
    labels_path = cache_root / model_dir / dataset / "generations" / "hallucination_labels.json"

    if not act_path.exists():
        raise FileNotFoundError(f"Missing file: {act_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    x = load_tensor_as_matrix(act_path)
    instance_ids = None
    if ids_path.exists():
        with open(ids_path, "r", encoding="utf-8") as f:
            instance_ids = json.load(f)
    y = load_labels_old_structure(labels_path, instance_ids)

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"Samples/labels mismatch for {dataset} layer {layer}: {x.shape[0]} vs {y.shape[0]}"
        )
    return x, y


def sample_points(
    x: np.ndarray,
    y: np.ndarray,
    max_points_per_class: int | None,
    balance: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    idx_not = np.flatnonzero(y == 0)
    idx_hall = np.flatnonzero(y == 1)
    if len(idx_not) == 0 or len(idx_hall) == 0:
        raise ValueError("At least one class is empty. Cannot create comparison plot.")

    rng = np.random.RandomState(seed)
    if balance:
        target = min(len(idx_not), len(idx_hall))
        if max_points_per_class is not None:
            target = min(target, max_points_per_class)
        if len(idx_not) > target:
            idx_not = rng.choice(idx_not, size=target, replace=False)
        if len(idx_hall) > target:
            idx_hall = rng.choice(idx_hall, size=target, replace=False)
    elif max_points_per_class is not None:
        if len(idx_not) > max_points_per_class:
            idx_not = rng.choice(idx_not, size=max_points_per_class, replace=False)
        if len(idx_hall) > max_points_per_class:
            idx_hall = rng.choice(idx_hall, size=max_points_per_class, replace=False)

    selected = np.concatenate([idx_not, idx_hall])
    rng.shuffle(selected)
    return x[selected], y[selected]


def project_to_pca(x: np.ndarray, standardize: bool, seed: int) -> tuple[np.ndarray, np.ndarray]:
    x_proc = StandardScaler().fit_transform(x) if standardize else x
    pca = PCA(n_components=2, random_state=seed)
    projection = pca.fit_transform(x_proc)
    return projection, pca.explained_variance_ratio_


def pretty_dataset_name(dataset: str) -> str:
    return DATASET_TITLES.get(dataset, dataset.replace("_", " ").title())


def sanitize_filename(text: str) -> str:
    return (
        text.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "")
        .replace(",", "_")
        .replace(":", "-")
    )


def build_default_filename(model_dir: str, layer_type: str) -> str:
    return f"pca_{sanitize_filename(model_dir)}_{layer_type}.pdf"


def build_default_title(model_dir: str, layer_type: str) -> str:
    layer_type_title = LAYER_TYPE_TITLES.get(layer_type, layer_type)
    return f"PCA of {layer_type_title} activations of {model_dir} across datasets"


def plot_multidataset(
    results: list[dict],
    model_dir: str,
    layer_type: str,
    output_path: Path,
    title_override: str,
    axis_quantile: float,
) -> None:
    n_plots = len(results)
    n_cols = min(3, n_plots)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 5.2 * n_rows), squeeze=False)

    for i, result in enumerate(results):
        ax = axes[i // n_cols][i % n_cols]
        proj = result["projection"]
        y = result["labels"]
        layer = result["layer"]
        dataset = result["dataset"]
        variance = result["variance"]

        mask_not = y == 0
        mask_hall = y == 1

        ax.scatter(
            proj[mask_not, 0],
            proj[mask_not, 1],
            s=24,
            c="#1F4EFF",
            alpha=0.6,
            linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            proj[mask_hall, 0],
            proj[mask_hall, 1],
            s=24,
            c="#E52323",
            alpha=0.6,
            linewidths=0,
            rasterized=True,
        )

        ax.set_box_aspect(1)

        # Robust axis limits improve readability when a few points are extreme outliers.
        if 0.5 < axis_quantile < 1.0:
            q_lo = 1.0 - axis_quantile
            x_lo, x_hi = np.quantile(proj[:, 0], [q_lo, axis_quantile])
            y_lo, y_hi = np.quantile(proj[:, 1], [q_lo, axis_quantile])
            x_pad = max(1e-6, (x_hi - x_lo) * 0.05)
            y_pad = max(1e-6, (y_hi - y_lo) * 0.05)
            ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
            ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

        ax.set_title(f"{pretty_dataset_name(dataset)}\n(Layer {layer})", fontsize=12, fontweight="bold")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.grid(True, alpha=0.2, linewidth=0.5)

    for j in range(n_plots, n_rows * n_cols):
        fig.delaxes(axes[j // n_cols][j % n_cols])

    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=6, color="#1F4EFF", label="Not Hallucinated"),
        Line2D([], [], marker="o", linestyle="", markersize=6, color="#E52323", label="Hallucinated"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.995))

    fig.tight_layout(rect=[0, 0, 1, 0.95], w_pad=0.05, h_pad=0.0025)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", format="pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_layer_pairs = parse_dataset_layers(args.dataset_layer, args.dataset_layers)
    max_points_per_class = args.max_points_per_class if args.max_points_per_class > 0 else None

    cache_root = Path(args.cache_dir)
    model_dir = resolve_model_dir(cache_root, args.model_name)

    print("=" * 72)
    print("PCA multi-dataset plot")
    print("=" * 72)
    print(f"Model input: {args.model_name}")
    print(f"Resolved model folder: {model_dir}")
    print(f"Layer type: {args.layer_type}")
    print(f"Dataset/layer pairs: {dataset_layer_pairs}")
    print(f"Balance classes: {args.balance}")
    print(f"Standardize: {args.standardize}")
    print(f"Axis quantile: {args.axis_quantile}")
    print(f"Max points per class: {max_points_per_class}")

    results: list[dict] = []
    for dataset, layer in dataset_layer_pairs:
        print(f"\n[Dataset: {dataset}] loading layer {layer} ...")
        x_raw, y_raw = load_dataset_layer(cache_root, model_dir, dataset, layer, args.layer_type)
        print(
            f"  Raw shape: {x_raw.shape}, labels: "
            f"not_hall={(y_raw == 0).sum()}, hall={(y_raw == 1).sum()}"
        )

        x_use, y_use = sample_points(
            x_raw,
            y_raw,
            max_points_per_class=max_points_per_class,
            balance=args.balance,
            seed=args.seed,
        )
        print(
            f"  Used samples: {x_use.shape[0]}, labels: "
            f"not_hall={(y_use == 0).sum()}, hall={(y_use == 1).sum()}"
        )

        projection, variance = project_to_pca(x_use, standardize=args.standardize, seed=args.seed)
        results.append(
            {
                "dataset": dataset,
                "layer": layer,
                "projection": projection,
                "labels": y_use,
                "variance": variance,
            }
        )

    filename = args.filename if args.filename else build_default_filename(model_dir, args.layer_type)
    output_path = Path(args.output_dir) / filename
    if output_path.suffix == ".png":
        output_path = output_path.with_suffix(".pdf")
        
    plot_multidataset(
        results=results,
        model_dir=model_dir,
        layer_type=args.layer_type,
        output_path=output_path,
        title_override=args.title,
        axis_quantile=args.axis_quantile,
    )
    print(f"\nSaved plot in: {output_path}")


if __name__ == "__main__":
    main()
