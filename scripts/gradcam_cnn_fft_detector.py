import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.WaveSteer.train_fft_detector import load_stacked_attention_tensor
from src.WaveSteer.fft_detector import ConvolutionalFFT2DAttentionClassifier, apply_fft2_features

def compute_gradcam(model, input_tensor, target_class=None):
    # Hook for gradients
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
        
    def forward_hook(module, input, output):
        activations.append(output.detach())
        
    # Register hooks on the conv layer
    handle_forward = model.conv.register_forward_hook(forward_hook)
    handle_backward = model.conv.register_backward_hook(backward_hook)
    
    # Forward pass
    logits = model(input_tensor)
    
    if target_class is None:
        target_class = logits.argmax(dim=1).item()
        
    score = logits[0, target_class]
    
    # Backward pass
    model.zero_grad()
    score.backward()
    
    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()
    
    # Get activations and gradients
    A = activations[0][0] # Shape: [C, L, H]
    grad_A = gradients[0][0] # Shape: [C, L, H]
    
    # Global average pooling of gradients to get weights alpha
    alpha = torch.mean(grad_A, dim=(1, 2)) # Shape: [C]
    
    # Weighted combination of activations
    cam = torch.zeros(A.shape[1:], dtype=torch.float32)
    for i, w in enumerate(alpha):
        cam += w * A[i]
        
    # Apply ReLU
    cam = F.relu(cam).numpy()
    
    # Normalize
    if cam.max() > 0:
        cam = cam / cam.max()
        
    return cam, target_class

def plot_gradcam(original_fft, cam, save_path, title_prefix, L, H):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
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
    
    im2 = axes[1].imshow(
        cam,
        aspect="auto",
        cmap="jet",
        interpolation="nearest",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title(f"{title_prefix} - Grad-CAM (Conv2d)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Head")
    axes[1].set_ylabel("Layer")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    out_dir = "results/gradcam_cnn"
    os.makedirs(out_dir, exist_ok=True)
    
    ckpt_path = "results/fft_detector/Qwen2.5-7B_belief_bank_facts_attn_fft2d_cnn.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    L = len(ckpt["layer_ids"])
    H = ckpt["input_dim"] // L
    
    model = ConvolutionalFFT2DAttentionClassifier(
        L=L,
        H=H,
        in_channels=1,
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
    
    for i, idx in enumerate(truth_idx):
        tensor = features[idx:idx+1]
        fft_mag = apply_fft2_features(tensor, feature_mode="magnitude")
        fft_mag_flat = fft_mag.reshape(fft_mag.shape[0], -1)
        x = (fft_mag_flat - mean) / std
        
        cam, pred_class = compute_gradcam(model, x, target_class=0)
        
        plot_gradcam(
            fft_mag[0].numpy(), 
            cam, 
            os.path.join(out_dir, f"truthful_{i+1:02d}.png"),
            f"Truthful Sample {idx}",
            L, H
        )
        
    for i, idx in enumerate(hall_idx):
        tensor = features[idx:idx+1]
        fft_mag = apply_fft2_features(tensor, feature_mode="magnitude")
        fft_mag_flat = fft_mag.reshape(fft_mag.shape[0], -1)
        x = (fft_mag_flat - mean) / std
        
        cam, pred_class = compute_gradcam(model, x, target_class=1)
        
        plot_gradcam(
            fft_mag[0].numpy(), 
            cam, 
            os.path.join(out_dir, f"hallucinated_{i+1:02d}.png"),
            f"Hallucinated Sample {idx}",
            L, H
        )
        
    print(f"Saved Grad-CAM maps to {out_dir}")

if __name__ == "__main__":
    main()
