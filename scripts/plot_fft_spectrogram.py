"""
Plot spectrograms of the per-sample 2D FFT of attention activations.

The attention tensor has shape [N, L, H]:
  - N = number of samples (the "time" / sequence axis in the spectrogram)
  - L = number of layers (frequency axis 1)
  - H = number of attention heads (frequency axis 2)

For each sample, a 2D FFT is applied on the (L, H) plane.
The resulting magnitude matrix is [L, H] per sample.
Stacking all N samples gives a "spectrogram" [N, L, H].

Visualizations produced:
  1. Average magnitude spectrogram across all samples
  2. Average magnitude spectrogram for truthful vs hallucinated
  3. Difference spectrogram (hallucinated - truthful)
  4. Individual sample spectrograms (a few examples)
  5. Spectrogram "timeline" - samples as rows, flattened freq bins as columns

Usage:
    python scripts/plot_fft_spectrogram.py \
        --cache-dir activation_cache \
        --model-name Qwen2.5-7B \
        --dataset-name belief_bank_facts \
        --max-samples 5000 \
        --output-dir results/fft_spectrograms
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.WaveSteer.train_fft_detector import load_stacked_attention_tensor


def compute_fft2_magnitude(tensor: torch.Tensor, batch_size: int = 100) -> torch.Tensor:
    """
    Apply per-sample 2D FFT and return magnitude, processed in batches.

    Args:
        tensor: [N, L, H]
        batch_size: number of samples per batch to avoid OOM

    Returns:
        magnitude: [N, L, H] - magnitude of the 2D FFT for each sample
    """
    N = tensor.shape[0]
    results = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        chunk = tensor[start:end].float()
        fft_complex = torch.fft.fft2(chunk, dim=(-2, -1), norm="ortho")
        results.append(torch.abs(fft_complex).cpu())
        del chunk, fft_complex
    return torch.cat(results, dim=0)


def plot_single_spectrogram(
    matrix: np.ndarray,
    title: str,
    save_path: str,
    xlabel: str = "Frequency bin (heads axis)",
    ylabel: str = "Frequency bin (layers axis)",
    cmap: str = "magma",
    use_log: bool = False,
):
    """Plot a single 2D spectrogram (heatmap)."""
    fig, ax = plt.subplots(figsize=(14, 8))

    vmin = matrix.min()
    vmax = matrix.max()

    if use_log and vmin > 0:
        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap=cmap,
            norm=LogNorm(vmin=max(vmin, 1e-8), vmax=vmax),
            interpolation="nearest",
            origin="lower",
        )
    else:
        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
            origin="lower",
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Magnitude", fontsize=12)

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_timeline_spectrogram(
    fft_magnitudes: np.ndarray,
    title: str,
    save_path: str,
    cmap: str = "inferno",
    max_display: int = 2000,
):
    """
    Plot a spectrogram where:
      - X axis = frequency bin index (flattened L*H)
      - Y axis = sample index ("time")

    Args:
        fft_magnitudes: [N, L, H]
    """
    N, L, H = fft_magnitudes.shape
    n_show = min(N, max_display)
    flat = fft_magnitudes[:n_show].reshape(n_show, L * H)

    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(
        flat,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        origin="lower",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("FFT Magnitude", fontsize=12)

    ax.set_xlabel("Frequency bin (flattened L×H)", fontsize=13)
    ax.set_ylabel("Sample index (time)", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")

    # Add layer boundary markers on x-axis
    for layer_idx in range(1, L):
        ax.axvline(x=layer_idx * H - 0.5, color="white", alpha=0.15, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_class_comparison(
    truth_avg: np.ndarray,
    hall_avg: np.ndarray,
    diff: np.ndarray,
    save_path: str,
):
    """Side-by-side comparison: truthful avg, hallucinated avg, difference."""
    fig = plt.figure(figsize=(22, 7))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.25)

    titles = ["Truthful (avg)", "Hallucinated (avg)", "Difference (hall - truth)"]
    matrices = [truth_avg, hall_avg, diff]
    cmaps = ["magma", "magma", "RdBu_r"]

    axes = []
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)

        if i < 2:
            vmin = min(truth_avg.min(), hall_avg.min())
            vmax = max(truth_avg.max(), hall_avg.max())
            im = ax.imshow(
                matrices[i],
                aspect="auto",
                cmap=cmaps[i],
                interpolation="nearest",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
        else:
            abs_max = max(abs(diff.min()), abs(diff.max()))
            im = ax.imshow(
                matrices[i],
                aspect="auto",
                cmap=cmaps[i],
                interpolation="nearest",
                origin="lower",
                vmin=-abs_max,
                vmax=abs_max,
            )

        ax.set_title(titles[i], fontsize=13, fontweight="bold")
        ax.set_xlabel("Freq bin (heads)", fontsize=11)
        if i == 0:
            ax.set_ylabel("Freq bin (layers)", fontsize=11)

    # Shared colorbar for first two
    cax = fig.add_subplot(gs[0, 3])
    fig.colorbar(
        plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
            cmap="magma",
        ),
        cax=cax,
        label="Magnitude",
    )

    fig.suptitle(
        "FFT2D Spectrogram: Truthful vs Hallucinated",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_individual_samples(
    fft_mag: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    n_examples: int = 6,
):
    """Plot FFT spectrograms for individual samples (3 truthful + 3 hallucinated)."""
    truth_idx = np.where(labels == 0)[0]
    hall_idx = np.where(labels == 1)[0]

    n_each = n_examples // 2
    selected_truth = truth_idx[:n_each] if len(truth_idx) >= n_each else truth_idx
    selected_hall = hall_idx[:n_each] if len(hall_idx) >= n_each else hall_idx

    n_total = len(selected_truth) + len(selected_hall)
    fig, axes = plt.subplots(2, n_each, figsize=(6 * n_each, 12))

    for col, idx in enumerate(selected_truth):
        ax = axes[0, col] if n_each > 1 else axes[0]
        im = ax.imshow(
            fft_mag[idx],
            aspect="auto",
            cmap="magma",
            interpolation="nearest",
            origin="lower",
        )
        ax.set_title(f"Truthful (sample {idx})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Freq bin (heads)", fontsize=10)
        if col == 0:
            ax.set_ylabel("Freq bin (layers)", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for col, idx in enumerate(selected_hall):
        ax = axes[1, col] if n_each > 1 else axes[1]
        im = ax.imshow(
            fft_mag[idx],
            aspect="auto",
            cmap="magma",
            interpolation="nearest",
            origin="lower",
        )
        ax.set_title(f"Hallucinated (sample {idx})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Freq bin (heads)", fontsize=10)
        if col == 0:
            ax.set_ylabel("Freq bin (layers)", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Individual Sample FFT2D Spectrograms",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_timeline_by_class(
    fft_mag: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    max_per_class: int = 1000,
):
    """
    Two stacked spectrograms: truthful samples on top, hallucinated on bottom.
    X = freq bins (L*H), Y = sample index (time within each class).
    """
    truth_idx = np.where(labels == 0)[0][:max_per_class]
    hall_idx = np.where(labels == 1)[0][:max_per_class]

    N, L, H = fft_mag.shape
    truth_flat = fft_mag[truth_idx].reshape(len(truth_idx), L * H)
    hall_flat = fft_mag[hall_idx].reshape(len(hall_idx), L * H)

    vmin = min(truth_flat.min(), hall_flat.min())
    vmax = max(truth_flat.max(), hall_flat.max())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14), sharex=True)

    im1 = ax1.imshow(
        truth_flat,
        aspect="auto",
        cmap="inferno",
        interpolation="nearest",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_ylabel("Sample index (truthful)", fontsize=12)
    ax1.set_title("Truthful samples – FFT2D Spectrogram Timeline", fontsize=13, fontweight="bold")
    fig.colorbar(im1, ax=ax1, fraction=0.02, pad=0.02)

    # Layer boundary markers
    for layer_idx in range(1, L):
        ax1.axvline(x=layer_idx * H - 0.5, color="white", alpha=0.15, linewidth=0.5)
        ax2.axvline(x=layer_idx * H - 0.5, color="white", alpha=0.15, linewidth=0.5)

    im2 = ax2.imshow(
        hall_flat,
        aspect="auto",
        cmap="inferno",
        interpolation="nearest",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_ylabel("Sample index (hallucinated)", fontsize=12)
    ax2.set_xlabel("Frequency bin (flattened L×H)", fontsize=12)
    ax2.set_title("Hallucinated samples – FFT2D Spectrogram Timeline", fontsize=13, fontweight="bold")
    fig.colorbar(im2, ax=ax2, fraction=0.02, pad=0.02)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot FFT spectrograms of attention activations")
    parser.add_argument("--cache-dir", type=str, default="activation_cache")
    parser.add_argument("--model-name", type=str, default="Qwen2.5-7B")
    parser.add_argument("--dataset-name", type=str, default="belief_bank_facts")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default=os.path.join("results", "fft_spectrograms"))
    parser.add_argument("--device", type=str, default="cpu", help="Device for FFT computation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("FFT2D Spectrogram Visualization")
    print("=" * 70)

    # 1. Load data
    print("\n[1/4] Loading attention tensors...")
    features, labels, instance_ids, layer_ids = load_stacked_attention_tensor(
        cache_dir=args.cache_dir,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        max_samples=args.max_samples,
        dtype=torch.float32,
    )

    N, L, H = features.shape
    n_truth_orig = int((labels == 0).sum())
    n_hall_orig = int((labels == 1).sum())
    print(f"  Tensor shape: [{N}, {L}, {H}]  (samples × layers × heads)")
    print(f"  Labels (before balancing): {n_truth_orig} truthful, {n_hall_orig} hallucinated")
    print(f"  Layer IDs: {layer_ids}")

    # 1b. Downsample majority class for balanced comparison
    print("\n  Balancing classes by downsampling majority...")
    truth_idx = np.where(labels.numpy() == 0)[0]
    hall_idx = np.where(labels.numpy() == 1)[0]
    n_minority = min(len(truth_idx), len(hall_idx))

    rng = np.random.RandomState(42)
    if len(truth_idx) > n_minority:
        truth_idx = rng.choice(truth_idx, size=n_minority, replace=False)
    if len(hall_idx) > n_minority:
        hall_idx = rng.choice(hall_idx, size=n_minority, replace=False)

    balanced_idx = np.sort(np.concatenate([truth_idx, hall_idx]))
    features = features[balanced_idx]
    labels = labels[balanced_idx]
    N = features.shape[0]
    print(f"  After balancing: {N} samples ({int((labels == 0).sum())} truthful, {int((labels == 1).sum())} hallucinated)")

    # 2. Compute per-sample 2D FFT magnitude
    print("\n[2/4] Computing per-sample 2D FFT...")
    fft_mag = compute_fft2_magnitude(features, batch_size=50).numpy()
    del features  # free memory
    labels_np = labels.numpy()
    print(f"  FFT magnitude shape: {fft_mag.shape}")

    # 3. Compute statistics
    print("\n[3/4] Computing statistics...")
    global_avg = fft_mag.mean(axis=0)  # [L, H]
    truth_mask = labels_np == 0
    hall_mask = labels_np == 1
    truth_avg = fft_mag[truth_mask].mean(axis=0)  # [L, H]
    hall_avg = fft_mag[hall_mask].mean(axis=0)  # [L, H]
    diff = hall_avg - truth_avg

    print(f"  Global avg range: [{global_avg.min():.4f}, {global_avg.max():.4f}]")
    print(f"  Truthful avg range: [{truth_avg.min():.4f}, {truth_avg.max():.4f}]")
    print(f"  Hallucinated avg range: [{hall_avg.min():.4f}, {hall_avg.max():.4f}]")
    print(f"  Diff range: [{diff.min():.4f}, {diff.max():.4f}]")

    # 4. Generate plots
    print("\n[4/4] Generating plots...")

    # 4a. Global average spectrogram
    plot_single_spectrogram(
        global_avg,
        title=f"Average FFT2D Spectrogram (all {N} samples)\n"
              f"Layers={L}, Heads={H}",
        save_path=os.path.join(args.output_dir, "spectrogram_average.png"),
        cmap="magma",
    )

    # 4b. Log-scale version
    plot_single_spectrogram(
        global_avg,
        title=f"Average FFT2D Spectrogram (log scale)\n"
              f"Layers={L}, Heads={H}",
        save_path=os.path.join(args.output_dir, "spectrogram_average_log.png"),
        cmap="magma",
        use_log=True,
    )

    # 4c. Class comparison
    plot_class_comparison(
        truth_avg, hall_avg, diff,
        save_path=os.path.join(args.output_dir, "spectrogram_class_comparison.png"),
    )

    # 4d. Individual samples
    plot_individual_samples(
        fft_mag, labels_np,
        save_path=os.path.join(args.output_dir, "spectrogram_individual_samples.png"),
        n_examples=6,
    )

    # 4e. Timeline spectrogram (all samples)
    plot_timeline_spectrogram(
        fft_mag,
        title=f"FFT2D Spectrogram Timeline (all {min(N, 2000)} samples)\n"
              f"Frequency bins = {L}×{H} = {L*H}",
        save_path=os.path.join(args.output_dir, "spectrogram_timeline_all.png"),
        max_display=2000,
    )

    # 4f. Timeline by class
    plot_timeline_by_class(
        fft_mag, labels_np,
        save_path=os.path.join(args.output_dir, "spectrogram_timeline_by_class.png"),
        max_per_class=1000,
    )

    print(f"\n{'=' * 70}")
    print(f"All plots saved to: {args.output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
