"""Hybrid method: AlignmentNetwork (neural) + LogisticRegression prober."""

import time

import torch
from sklearn.linear_model import LogisticRegression

from ..config import DEVICE, SEED, HYBRID_CONFIG
from .training import compute_metrics, count_params, train_alignment_network


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
    t0_detector = time.time()
    clf.fit(trainer["X_train"][tr_idx], trainer["y_train"][tr_idx])
    detector_time = time.time() - t0_detector

    detector_params = count_params(clf)
    detector_train_n = int(len(tr_idx))

    pred_t = clf.predict(trainer["X_test"])
    proba_t = clf.predict_proba(trainer["X_test"])[:, 1]
    metrics_trainer = compute_metrics(trainer["y_test"], pred_t, proba_t)

    # Train alignment network (tester → trainer)
    t0_aligner = time.time()
    align_model, align_info = train_alignment_network(
        alignment["X_tester_train"], alignment["X_trainer_train"],
        alignment["X_tester_val"], alignment["X_trainer_val"],
        cfg,
    )
    aligner_time = time.time() - t0_aligner

    aligner_params = count_params(align_model)
    aligner_train_n = int(alignment["X_tester_train"].shape[0])

    # Project tester
    align_model.eval()
    X_tester_scaled = alignment["scaler_tester"].transform(tester["X_test_raw"])
    with torch.no_grad():
        projected = align_model(torch.tensor(X_tester_scaled, dtype=torch.float32, device=DEVICE)).cpu().numpy()

    pred_s = clf.predict(projected)
    proba_s = clf.predict_proba(projected)[:, 1]
    metrics_tester = compute_metrics(tester["y_test"], pred_s, proba_s)

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
