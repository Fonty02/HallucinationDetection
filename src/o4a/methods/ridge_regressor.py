"""RidgeRegressor method: LogisticRegression prober + Ridge alignment."""

import os

import torch
from sklearn.linear_model import LogisticRegression, Ridge

from ..config import SEED, RIDGE_REGRESSOR_CONFIG
from .training import compute_metrics


def run_ridge_regressor(shared_data: dict, config: dict = None, save_dir: str = None) -> dict:
    """
    1. Train LogisticRegression on trainer scaled data.
    2. Evaluate on trainer test (→ trainer metrics).
    3. Train Ridge regressor on concordant alignment data (tester→trainer).
    4. Project tester test through Ridge, evaluate with trainer prober (→ tester metrics).
    """
    cfg = config or RIDGE_REGRESSOR_CONFIG

    trainer = shared_data["trainer"]
    tester = shared_data["tester"]
    alignment = shared_data["alignment"]

    # 1. Trainer prober (shared prober split)
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

    # 2. Trainer eval
    pred_t = clf.predict(trainer["X_test"])
    proba_t = clf.predict_proba(trainer["X_test"])[:, 1]
    metrics_trainer = compute_metrics(trainer["y_test"], pred_t, proba_t)

    # 3. Ridge alignment: tester → trainer space
    aligner = Ridge(alpha=cfg["ridge_alpha"], fit_intercept=False)
    aligner.fit(alignment["X_tester_train"], alignment["X_trainer_train"])

    # 4. Project tester test & evaluate
    X_tester_scaled = alignment["scaler_tester"].transform(tester["X_test_raw"])
    X_tester_proj = aligner.predict(X_tester_scaled)
    pred_s = clf.predict(X_tester_proj)
    proba_s = clf.predict_proba(X_tester_proj)[:, 1]
    metrics_tester = compute_metrics(tester["y_test"], pred_s, proba_s)

    # Save checkpoint
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            "method": "ridge_regressor",
            "prober": {
                "model_class": "LogisticRegression",
                "model": clf,
                "params": clf.get_params(),
                "n_features": trainer["X_train"].shape[1],
                "n_iter": int(clf.n_iter_[0]),
            },
            "aligner": {
                "model_class": "Ridge",
                "model": aligner,
                "params": aligner.get_params(),
                "input_dim": alignment["X_tester_train"].shape[1],
                "output_dim": alignment["X_trainer_train"].shape[1],
            },
            "config": cfg,
        }
        torch.save(checkpoint, os.path.join(save_dir, "checkpoint.pt"))

    return {"trainer": metrics_trainer, "tester": metrics_tester}
