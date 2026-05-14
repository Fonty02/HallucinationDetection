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
from scripts.plot_fft_spectrogram import compute_fft2_magnitude

def compute_gradcam(model, input_tensor, target_class=None):
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
        
    def forward_hook(module, input, output):
        activations.append(output.detach())
        
    handle_forward = model.conv.register_forward_hook(forward_hook)
    handle_backward = model.conv.register_full_backward_hook(backward_hook)
    
    logits = model(input_tensor)
    
    if target_class is None:
        target_class = logits.argmax(dim=1).item()
        
    score = logits[0, target_class]
    
    model.zero_grad()
    score.backward()
    
    handle_forward.remove()
    handle_backward.remove()
    
    A = activations[0][0]
    grad_A = gradients[0][0]
    
    alpha = torch.mean(grad_A, dim=(1, 2))
    
    cam = torch.zeros(A.shape[1:], dtype=torch.float32)
    for i, w in enumerate(alpha):
        cam += w * A[i]
        
    cam = F.relu(cam).numpy()
    
    if cam.max() > 0:
        cam = cam / cam.max()
        
    return cam, A[0].cpu().numpy()

def plot_gradcam(conv_out, cam, save_path, title_prefix, L, H):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(
        conv_out,
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
        origin="lower",
        extent=[0, H, 0, L] # Stretch it like the CAM
    )
    axes[0].set_title(f"{title_prefix}\nConv2d Output", fontsize=12, fontweight="bold")
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
        extent=[0, H, 0, L] # Ensures alignment with the original FFT plot
    )
    axes[1].set_title(f"{title_prefix}\nGrad-CAM (Conv2d)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Head")
    axes[1].set_ylabel("Layer")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    out_dir = "results/gradcam_cnn_high_conf"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading CNN model...")
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
    
    mean = ckpt["mean"]
    std = ckpt["std"]
    
    print("Loading attention tensors...")
    features, labels, instance_ids, layer_ids = load_stacked_attention_tensor(
        cache_dir="activation_cache",
        model_name="Qwen2.5-7B",
        dataset_name="belief_bank_facts",
        max_samples=2500,
        dtype=torch.float32,
    )
    
    print("Computing FFT2D magnitudes...")
    fft_mag = compute_fft2_magnitude(features, batch_size=50)
    fft_mag_flat = fft_mag.reshape(fft_mag.shape[0], -1)
    
    print("Running Inference for Confidence Scoring...")
    normalized_features = (fft_mag_flat - mean) / std
    
    with torch.no_grad():
        logits = model(normalized_features)
        probs = F.softmax(logits, dim=1)
    
    probs_np = probs.numpy()
    labels_np = labels.numpy()
    preds = np.argmax(probs_np, axis=1)
    
    conf_truth = probs_np[:, 0]
    conf_hall = probs_np[:, 1]
    
    idx_pred_truth = np.where(preds == 0)[0]
    idx_pred_hall = np.where(preds == 1)[0]
    
    top_truth_idx = idx_pred_truth[np.argsort(conf_truth[idx_pred_truth])[::-1]][:10]
    top_hall_idx = idx_pred_hall[np.argsort(conf_hall[idx_pred_hall])[::-1]][:10]
    
    class_map = {0: "Truth", 1: "Hallucination"}
    
    print("Generating Grad-CAM for Top Truthful predictions...")
    for i, idx in enumerate(top_truth_idx):
        gt_class = class_map[labels_np[idx]]
        pred_class = class_map[preds[idx]]
        confidence = conf_truth[idx]
        
        x = normalized_features[idx:idx+1]
        cam, conv_out = compute_gradcam(model, x, target_class=0)
        
        title = f"GT: {gt_class} | Pred: {pred_class} | Conf: {confidence:.4f}"
        
        plot_gradcam(
            conv_out, 
            cam, 
            os.path.join(out_dir, f"top_truthful_{i+1:02d}.png"),
            title,
            L, H
        )
        
    print("Generating Grad-CAM for Top Hallucinated predictions...")
    for i, idx in enumerate(top_hall_idx):
        gt_class = class_map[labels_np[idx]]
        pred_class = class_map[preds[idx]]
        confidence = conf_hall[idx]
        
        x = normalized_features[idx:idx+1]
        cam, conv_out = compute_gradcam(model, x, target_class=1)
        
        title = f"GT: {gt_class} | Pred: {pred_class} | Conf: {confidence:.4f}"
        
        plot_gradcam(
            conv_out, 
            cam, 
            os.path.join(out_dir, f"top_hallucinated_{i+1:02d}.png"),
            title,
            L, H
        )
        
    print(f"Saved high confidence Grad-CAM maps to {out_dir}")

if __name__ == "__main__":
    main()
