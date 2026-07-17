"""FullNonLinear method: AlignmentNetwork + MLPProber (both non-linear, full dim)."""

import os
import time

import torch

from ..config import DEVICE, FULL_NONLINEAR_CONFIG
from .training import compute_metrics, count_params, train_alignment_network, train_mlp_prober


def run_full_nonlinear(shared_data: dict, config: dict = None, save_dir: str = None) -> dict:
    """AlignmentNetwork + MLPProber, both non-linear, full dimension."""
    cfg = config or FULL_NONLINEAR_CONFIG
    trainer = shared_data["trainer"]
    tester = shared_data["tester"]
    alignment = shared_data["alignment"]

    # Train prober on trainer (balanced train, imbalanced val for early stopping)
    t0_detector = time.time()
    prober, prober_info = train_mlp_prober(
        trainer["X_train"], trainer["y_train"],
        trainer["X_val"], trainer["y_val"],
        input_dim=trainer["X_train"].shape[1], cfg=cfg,
    )
    detector_time = time.time() - t0_detector

    detector_params = count_params(prober)
    detector_train_n = int(len(trainer["y_train"]))

    # Evaluate trainer
    prober.eval()
    with torch.no_grad():
        X_t = torch.tensor(trainer["X_test"], dtype=torch.float32, device=DEVICE)
        pred_t = prober.predict(X_t).cpu().numpy()
        proba_t = torch.sigmoid(prober(X_t)).cpu().numpy()
    metrics_trainer = compute_metrics(trainer["y_test"], pred_t, proba_t)

    # Train alignment (tester → trainer)
    t0_aligner = time.time()
    align_model, align_info = train_alignment_network(
        alignment["X_tester_train"], alignment["X_trainer_train"],
        alignment["X_tester_val"], alignment["X_trainer_val"], cfg,
    )
    aligner_time = time.time() - t0_aligner

    aligner_params = count_params(align_model)
    aligner_train_n = int(alignment["X_tester_train"].shape[0])

    # Project tester & evaluate
    align_model.eval()
    X_tester_scaled = alignment["scaler_tester"].transform(tester["X_test_raw"])
    with torch.no_grad():
        projected = align_model(torch.tensor(X_tester_scaled, dtype=torch.float32, device=DEVICE)).cpu().numpy()
        X_proj_t = torch.tensor(projected, dtype=torch.float32, device=DEVICE)
        pred_s = prober.predict(X_proj_t).cpu().numpy()
        proba_s = torch.sigmoid(prober(X_proj_t)).cpu().numpy()
    metrics_tester = compute_metrics(tester["y_test"], pred_s, proba_s)

    # Save models
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(prober.state_dict(), os.path.join(save_dir, "detector.pt"))
        torch.save(align_model.state_dict(), os.path.join(save_dir, "aligner.pt"))

    return {
        "trainer": metrics_trainer,
        "tester": metrics_tester,
        "_meta": {
            "detector_params": detector_params,
            "detector_time_s": detector_time,
            "detector_train_n": detector_train_n,
            "aligner_params": aligner_params,
            "aligner_time_s": aligner_time,
            "aligner_train_n": aligner_train_n,
        },
    }
