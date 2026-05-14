import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.WaveSteer.train_fft_detector import load_stacked_attention_tensor
from src.WaveSteer.fft_detector import FFT2DAttentionClassifier, apply_fft2_features

def compute_saliency_map(model, input_tensor, target_class=None):
    """
    Compute input x gradient saliency map.
    """
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    
    # Forward pass
    logits = model(input_tensor)
    
    # Target class
    if target_class is None:
        target_class = logits.argmax(dim=1).item()
        
    score = logits[0, target_class]
    
    # Backward pass
    model.zero_grad()
    score.backward()
    
    # Grad-CAM / Saliency equivalent for flat inputs
    # Use Input * Gradient and take absolute value or ReLU to highlight positive contributions
    saliency = (input_tensor.grad * input_tensor).detach()
    # Apply ReLU to only keep features that push the score higher (like in Grad-CAM)
    saliency = torch.nn.functional.relu(saliency).squeeze().numpy()
    
    # Normalize for visualization
    if saliency.max() > 0:
        saliency = saliency / saliency.max()
        
    return saliency, target_class

def plot_saliency(original_fft, saliency, save_path, title_prefix, L, H):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original Spectrogram
    im1 = axes[0].imshow(
        original_fft.reshape(L, H),
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
        origin="lower",
    )
    axes[0].set_title(f"{title_prefix} - Original FFT", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Saliency Map
    im2 = axes[1].imshow(
        saliency.reshape(L, H),
        aspect="auto",
        cmap="jet", # Jet is typically used for Grad-CAM
        interpolation="nearest", # Can use bilinear for smoother look
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title(f"{title_prefix} - Saliency Map (Grad-CAM eq)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Head")
    axes[1].set_ylabel("Layer")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    out_dir = "results/saliency_maps"
    os.makedirs(out_dir, exist_ok=True)
    
    ckpt_path = "results/fft_detector/Qwen2.5-7B_belief_bank_facts_attn_fft2d.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    model = FFT2DAttentionClassifier(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_classes=ckpt["num_classes"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    features, labels, instance_ids, layer_ids = load_stacked_attention_tensor(
        cache_dir="activation_cache",
        model_name="Qwen2.5-7B",
        dataset_name="belief_bank_facts",
        max_samples=2000,
        dtype=torch.float32,
    )
    
    labels_np = labels.numpy()
    truth_idx = np.where(labels_np == 0)[0][:5]
    hall_idx = np.where(labels_np == 1)[0][:5]
    
    mean = ckpt["mean"]
    std = ckpt["std"]
    
    L = features.shape[1]
    H = features.shape[2]
    
    for i, idx in enumerate(truth_idx):
        tensor = features[idx:idx+1]
        fft_mag = apply_fft2_features(tensor, feature_mode="magnitude")
        x = (fft_mag - mean) / std
        
        saliency, pred_class = compute_saliency_map(model, x, target_class=0)
        
        plot_saliency(
            fft_mag[0].numpy(), 
            saliency, 
            os.path.join(out_dir, f"truthful_{i+1:02d}.png"),
            f"Truthful Sample {idx}",
            L, H
        )
        
    for i, idx in enumerate(hall_idx):
        tensor = features[idx:idx+1]
        fft_mag = apply_fft2_features(tensor, feature_mode="magnitude")
        x = (fft_mag - mean) / std
        
        saliency, pred_class = compute_saliency_map(model, x, target_class=1)
        
        plot_saliency(
            fft_mag[0].numpy(), 
            saliency, 
            os.path.join(out_dir, f"hallucinated_{i+1:02d}.png"),
            f"Hallucinated Sample {idx}",
            L, H
        )
        
    print(f"Saved saliency maps to {out_dir}")

if __name__ == "__main__":
    main()
