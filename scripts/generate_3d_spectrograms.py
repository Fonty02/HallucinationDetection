import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.WaveSteer.train_fft_detector import load_stacked_attention_tensor
from scripts.plot_fft_spectrogram import compute_fft2_magnitude

def plot_3d_surface(matrix, title, save_path):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    L, H = matrix.shape
    
    # Create meshgrid
    # To avoid the 3D plot becoming incredibly dense and unreadable due to H=3584,
    # we can downsample the H dimension (e.g., take every 16th point) for the 3D plot
    step_H = max(1, H // 200) # Keep around 200 points in the H dimension
    step_L = 1 # L is only 28, keep all
    
    X, Y = np.meshgrid(np.arange(0, H, step_H), np.arange(0, L, step_L))
    Z = matrix[::step_L, ::step_H]
    
    surf = ax.plot_surface(X, Y, Z, cmap='magma', edgecolor='none', alpha=0.9, antialiased=True)
    
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Head Dimension (downsampled)", fontsize=11, labelpad=10)
    ax.set_ylabel("Layer", fontsize=11, labelpad=10)
    ax.set_zlabel("FFT Magnitude", fontsize=11, labelpad=10)
    
    # Adjust viewing angle for better visualization
    ax.view_init(elev=30, azim=-45)
    
    fig.colorbar(surf, ax=ax, fraction=0.03, pad=0.1)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    out_dir = "results/spectrograms_3d"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading attention tensors...")
    features, labels, instance_ids, layer_ids = load_stacked_attention_tensor(
        cache_dir="activation_cache",
        model_name="Qwen2.5-7B",
        dataset_name="belief_bank_facts",
        max_samples=2000,
        dtype=torch.float32,
    )
    
    print("Computing FFT2D magnitudes...")
    fft_mag = compute_fft2_magnitude(features, batch_size=50).numpy()
    labels_np = labels.numpy()
    
    truth_idx = np.where(labels_np == 0)[0][:10]
    hall_idx = np.where(labels_np == 1)[0][:10]
    
    print("Generating 3D Truthful plots...")
    for i, idx in enumerate(truth_idx):
        plot_3d_surface(
            fft_mag[idx],
            title=f"3D Truthful (sample {idx})",
            save_path=os.path.join(out_dir, f"truthful_3d_{i+1:02d}.png")
        )
        
    print("Generating 3D Hallucinated plots...")
    for i, idx in enumerate(hall_idx):
        plot_3d_surface(
            fft_mag[idx],
            title=f"3D Hallucinated (sample {idx})",
            save_path=os.path.join(out_dir, f"hallucinated_3d_{i+1:02d}.png")
        )
        
    print(f"Saved 20 3D spectrograms to {out_dir}")

if __name__ == "__main__":
    main()
