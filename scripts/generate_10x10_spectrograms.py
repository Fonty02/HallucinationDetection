import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.WaveSteer.train_fft_detector import load_stacked_attention_tensor
from scripts.plot_fft_spectrogram import compute_fft2_magnitude, plot_single_spectrogram

def main():
    out_dir = "results/spectrograms_10x10"
    os.makedirs(out_dir, exist_ok=True)
    
    features, labels, instance_ids, layer_ids = load_stacked_attention_tensor(
        cache_dir="activation_cache",
        model_name="Qwen2.5-7B",
        dataset_name="belief_bank_facts",
        max_samples=2000,
        dtype=torch.float32,
    )
    
    fft_mag = compute_fft2_magnitude(features, batch_size=50).numpy()
    labels_np = labels.numpy()
    
    truth_idx = np.where(labels_np == 0)[0][:10]
    hall_idx = np.where(labels_np == 1)[0][:10]
    
    for i, idx in enumerate(truth_idx):
        plot_single_spectrogram(
            fft_mag[idx],
            title=f"Truthful (sample {idx}, inst {instance_ids[idx]})",
            save_path=os.path.join(out_dir, f"truthful_{i+1:02d}.png")
        )
        
    for i, idx in enumerate(hall_idx):
        plot_single_spectrogram(
            fft_mag[idx],
            title=f"Hallucinated (sample {idx}, inst {instance_ids[idx]})",
            save_path=os.path.join(out_dir, f"hallucinated_{i+1:02d}.png")
        )
        
    print(f"Saved 20 spectrograms to {out_dir}")

if __name__ == "__main__":
    main()
