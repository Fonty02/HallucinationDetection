"""Hybrid method: AlignmentNetwork (neural) + LogisticRegression prober."""

import os

import torch
from sklearn.linear_model import LogisticRegression

from ..config import DEVICE, SEED, HYBRID_CONFIG
from .training import compute_metrics, train_alignment_network


def run_hybrid(shared_data: dict, config: dict = None, save_dir: str = None) -> dict:
    """AlignmentNetwork + LogisticRegression."""
    cfg = config or HYBRID_CONFIG
    trainer = shared_data["trainer"]
    tester = shared_data["tester"]
    alignment = shared_data["alignment"]

    # Trainer prober (LogReg) using shared prober split
    prober_split = shared_data["prober_split"]
    tr_idx = prober_split["train_idx"]
    clf = LogisticRegression(
        max_iter=cfg["probe_max_iter"],
        class_weight="balanced",
        solver=cfg["probe_solver"],
        n_jobs=-1,
        random_state=SEED,
    )
    clf.fit(trainer["X_train"][tr_idx], trainer["y_train"][tr_idx])

    pred_t = clf.predict(trainer["X_test"])
    proba_t = clf.predict_proba(trainer["X_test"])[:, 1]
    metrics_trainer = compute_metrics(trainer["y_test"], pred_t, proba_t)

    # Train alignment network (tester → trainer)
    align_model, align_info = train_alignment_network(
        alignment["X_tester_train"], alignment["X_trainer_train"],
        alignment["X_tester_val"], alignment["X_trainer_val"],
        cfg,
    )

    # Project tester
    align_model.eval()
    X_tester_scaled = alignment["scaler_tester"].transform(tester["X_test_raw"])
    with torch.no_grad():
        projected = align_model(torch.tensor(X_tester_scaled, dtype=torch.float32, device=DEVICE)).cpu().numpy()

    pred_s = clf.predict(projected)
    proba_s = clf.predict_proba(projected)[:, 1]
    metrics_tester = compute_metrics(tester["y_test"], pred_s, proba_s)

    # Save checkpoint
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            "method": "hybrid",
            "prober": {
                "model_class": "LogisticRegression",
                "model": clf,
                "params": clf.get_params(),
                "n_features": trainer["X_train"].shape[1],
                "n_iter": int(clf.n_iter_[0]),
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
