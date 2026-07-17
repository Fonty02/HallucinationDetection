"""Procrustes method: LogisticRegression prober + Procrustes alignment."""

import time

import torch
from sklearn.linear_model import LinearRegression, LogisticRegression

from ..config import SEED, PROCRUSTES_CONFIG
from .training import compute_metrics, count_params


def _fit_procrustes_linear_map(x, y, eps=1e-12):
    """
    Fit a Procrustes map (with scale + translation) from x -> y.
      x: [n, d_x], y: [n, d_y]
    Returns:
      A: [d_x, d_y], b: [d_y], scale: float
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"Procrustes requires same n_samples, got {x.shape[0]} and {y.shape[0]}"
        )

    x_t = torch.as_tensor(x, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    mean_x = x_t.mean(dim=0, keepdim=True)
    mean_y = y_t.mean(dim=0, keepdim=True)
    x_centered = x_t - mean_x
    y_centered = y_t - mean_y

    cross_cov = x_centered.T @ y_centered
    u, s, vh = torch.linalg.svd(cross_cov, full_matrices=False)
    rotation = u @ vh
    scale = s.sum() / (x_centered.pow(2).sum() + eps)

    A = scale * rotation
    b = (mean_y - mean_x @ A).squeeze(0)
    return A, b, float(scale.item())


def _build_linear_regressor_from_map(A, b):
    """Build a sklearn-compatible regressor with .predict() from y = x @ A + b."""
    model = LinearRegression()
    model.coef_ = A.T.detach().cpu().numpy()
    model.intercept_ = b.detach().cpu().numpy()
    model.n_features_in_ = int(A.shape[0])
    return model


def _procrustes_matrix_params(A, b):
    """Return number of learned parameters in the Procrustes transformation matrix."""
    return int(A.numel()) + int(b.numel())


def run_procrustes(shared_data: dict, config: dict = None, save_dir: str = None) -> dict:
    """
    1. Train LogisticRegression on trainer scaled data.
    2. Evaluate on trainer test (-> trainer metrics).
    3. Train Procrustes map on concordant alignment data (tester->trainer).
    4. Project tester test through Procrustes, evaluate with trainer prober (-> tester metrics).
    """
    cfg = config or PROCRUSTES_CONFIG

    trainer = shared_data["trainer"]
    tester = shared_data["tester"]
    alignment = shared_data["alignment"]

    # 1. Trainer prober (trained on all balanced training data)
    clf = LogisticRegression(
        max_iter=cfg["probe_max_iter"],
        class_weight="balanced",
        solver=cfg["probe_solver"],
        n_jobs=-1,
        random_state=SEED,
    )
    t0_detector = time.time()
    clf.fit(trainer["X_train"], trainer["y_train"])
    detector_time = time.time() - t0_detector

    detector_params = count_params(clf)
    detector_train_n = int(len(trainer["y_train"]))

    # 2. Trainer eval
    pred_t = clf.predict(trainer["X_test"])
    proba_t = clf.predict_proba(trainer["X_test"])[:, 1]
    metrics_trainer = compute_metrics(trainer["y_test"], pred_t, proba_t)

    # 3. Procrustes alignment: tester -> trainer space
    t0_aligner = time.time()
    A, b, scale = _fit_procrustes_linear_map(
        alignment["X_tester_train"], alignment["X_trainer_train"]
    )
    aligner = _build_linear_regressor_from_map(A, b)
    aligner_time = time.time() - t0_aligner

    aligner_params = _procrustes_matrix_params(A, b)
    aligner_train_n = int(alignment["X_tester_train"].shape[0])

    # 4. Project tester test & evaluate
    X_tester_scaled = alignment["scaler_tester"].transform(tester["X_test_raw"])
    X_tester_proj = aligner.predict(X_tester_scaled)
    pred_s = clf.predict(X_tester_proj)
    proba_s = clf.predict_proba(X_tester_proj)[:, 1]
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
