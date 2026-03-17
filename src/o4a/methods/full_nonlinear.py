"""FullNonLinear method: AlignmentNetwork + MLPProber (both non-linear, full dim)."""

import os

import torch

from ..config import DEVICE, FULL_NONLINEAR_CONFIG
from .training import compute_metrics, train_alignment_network, train_mlp_prober


def run_full_nonlinear(shared_data: dict, config: dict = None, save_dir: str = None) -> dict:
    """AlignmentNetwork + MLPProber, both non-linear, full dimension."""
    cfg = config or FULL_NONLINEAR_CONFIG
    trainer = shared_data["trainer"]
    tester = shared_data["tester"]
    alignment = shared_data["alignment"]

    # Shared prober split (fixed across methods)
    prober_split = shared_data["prober_split"]
    tr_idx = prober_split["train_idx"]
    val_idx = prober_split["val_idx"]

    # Train prober on trainer
    prober, prober_info = train_mlp_prober(
        trainer["X_train"][tr_idx], trainer["y_train"][tr_idx],
        trainer["X_train"][val_idx], trainer["y_train"][val_idx],
        input_dim=trainer["X_train"].shape[1], cfg=cfg,
    )

    # Evaluate trainer
    prober.eval()
    with torch.no_grad():
        X_t = torch.tensor(trainer["X_test"], dtype=torch.float32, device=DEVICE)
        pred_t = prober.predict(X_t).cpu().numpy()
        proba_t = torch.sigmoid(prober(X_t)).cpu().numpy()
    metrics_trainer = compute_metrics(trainer["y_test"], pred_t, proba_t)

    # Train alignment (tester → trainer)
    align_model, align_info = train_alignment_network(
        alignment["X_tester_train"], alignment["X_trainer_train"],
        alignment["X_tester_val"], alignment["X_trainer_val"], cfg,
    )

    # Project tester & evaluate
    align_model.eval()
    X_tester_scaled = alignment["scaler_tester"].transform(tester["X_test_raw"])
    with torch.no_grad():
        projected = align_model(torch.tensor(X_tester_scaled, dtype=torch.float32, device=DEVICE)).cpu().numpy()
        X_proj_t = torch.tensor(projected, dtype=torch.float32, device=DEVICE)
        pred_s = prober.predict(X_proj_t).cpu().numpy()
        proba_s = torch.sigmoid(prober(X_proj_t)).cpu().numpy()
    metrics_tester = compute_metrics(tester["y_test"], pred_s, proba_s)

    # Save checkpoint
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            "method": "full_nonlinear",
            "prober": {
                "model_class": "MLPProber",
                "state_dict": prober.state_dict(),
                "architecture": {
                    "input_dim": trainer["X_train"].shape[1],
                    "hidden_dim": cfg["prober_hidden_dim"],
                    "dropout": cfg["prober_dropout"],
                },
                "training_info": prober_info,
            },
            "alignment_network": {
                "model_class": "AlignmentNetwork",
                "state_dict": align_model.state_dict(),
                "architecture": {
                    "input_dim": alignment["X_tester_train"].shape[1],
                    "output_dim": alignment["X_trainer_train"].shape[1],
                    "hidden_dim": cfg["alignment_hidden_dim"],
                    "dropout": cfg["alignment_dropout"],
                },
                "training_info": align_info,
            },
            "config": cfg,
        }
        torch.save(checkpoint, os.path.join(save_dir, "checkpoint.pt"))

    return {"trainer": metrics_trainer, "tester": metrics_tester}
