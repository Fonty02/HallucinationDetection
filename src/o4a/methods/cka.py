"""CKA method: LogisticRegression prober + CKA-based linear alignment."""

import torch
from sklearn.linear_model import LinearRegression, LogisticRegression

from ..config import SEED, CKA_CONFIG
from .training import compute_metrics


def _fit_cka_linear_map(x, y, eps=1e-12):
    """
    Fit a CKA-inspired linear map from x -> y.
    Uses the Procrustes rotation scaled by linear CKA.

      x: [n, d_x], y: [n, d_y]
    Returns:
      A: [d_x, d_y], b: [d_y], cka: float
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"CKA requires same n_samples, got {x.shape[0]} and {y.shape[0]}"
        )

    x_t = torch.as_tensor(x, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    mean_x = x_t.mean(dim=0, keepdim=True)
    mean_y = y_t.mean(dim=0, keepdim=True)
    x_centered = x_t - mean_x
    y_centered = y_t - mean_y

    cross_cov = x_centered.T @ y_centered
    u, _, vh = torch.linalg.svd(cross_cov, full_matrices=False)
    rotation = u @ vh

    hsic = torch.sum(cross_cov ** 2)
    norm_x = torch.linalg.norm(x_centered.T @ x_centered, ord="fro")
    norm_y = torch.linalg.norm(y_centered.T @ y_centered, ord="fro")
    denom = norm_x * norm_y + eps
    if denom.item() <= eps:
        cka = 0.0
    else:
        cka = float((hsic / denom).item())

    A = cka * rotation
    b = (mean_y - mean_x @ A).squeeze(0)
    return A, b, cka


def _build_linear_regressor_from_map(A, b):
    """Build a sklearn-compatible regressor with .predict() from y = x @ A + b."""
    model = LinearRegression()
    model.coef_ = A.T.detach().cpu().numpy()
    model.intercept_ = b.detach().cpu().numpy()
    model.n_features_in_ = int(A.shape[0])
    return model


def run_cka(shared_data: dict, config: dict = None, save_dir: str = None) -> dict:
    """
    1. Train LogisticRegression on trainer scaled data.
    2. Evaluate on trainer test (-> trainer metrics).
    3. Train CKA-based map on concordant alignment data (tester->trainer).
    4. Project tester test through CKA map, evaluate with trainer prober (-> tester metrics).
    """
    cfg = config or CKA_CONFIG

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

    # 3. CKA alignment: tester -> trainer space
    A, b, cka = _fit_cka_linear_map(
        alignment["X_tester_train"], alignment["X_trainer_train"]
    )
    aligner = _build_linear_regressor_from_map(A, b)

    # 4. Project tester test & evaluate
    X_tester_scaled = alignment["scaler_tester"].transform(tester["X_test_raw"])
    X_tester_proj = aligner.predict(X_tester_scaled)
    pred_s = clf.predict(X_tester_proj)
    proba_s = clf.predict_proba(X_tester_proj)[:, 1]
    metrics_tester = compute_metrics(tester["y_test"], pred_s, proba_s)

    return {"trainer": metrics_trainer, "tester": metrics_tester}
