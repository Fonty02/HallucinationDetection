"""Single-LLM probers: the detector stage of each O4A method, without cross-LLM alignment.

Every prober consumes the dict returned by o4a.data.prepare_single_model_data
(X_train/y_train balanced, X_val/y_val and X_test/y_test imbalanced, all scaled)
and trains exactly as the trainer side of the corresponding method (same names as run_experiments):

- ridge_regressor, procrustes, cka, cca, hybrid:
                     LogisticRegression with each method's probe config (same model as the layer study)
- full_nonlinear:    MLPProber on the full-dimensional input
- reduced_nonlinear: Autoencoder -> MLPProber on the latent space
- one_for_all:       Encoder + ClassificationHead trained jointly (teacher)
"""

import os
import time

import torch
from joblib import dump
from sklearn.linear_model import LogisticRegression

from .config import (
    DEVICE,
    SEED,
    RIDGE_REGRESSOR_CONFIG,
    PROCRUSTES_CONFIG,
    CKA_CONFIG,
    CCA_CONFIG,
    HYBRID_CONFIG,
    FULL_NONLINEAR_CONFIG,
    REDUCED_NONLINEAR_CONFIG,
    ONE_FOR_ALL_CONFIG,
)
from .methods.one_for_all import _train_teacher_pipeline
from .methods.training import (
    compute_metrics, count_params, split_train_val, train_autoencoder, train_mlp_prober,
)


def _eval_logits(logits_fn, X, y):
    """Evaluate a torch model returning logits on numpy X."""
    with torch.no_grad():
        proba = torch.sigmoid(logits_fn(torch.tensor(X, dtype=torch.float32, device=DEVICE))).cpu().numpy()
    return compute_metrics(y, (proba > 0.5).astype(int), proba)


def run_linear_prober(data: dict, config: dict = None, save_dir: str = None) -> dict:
    cfg = config or RIDGE_REGRESSOR_CONFIG
    clf = LogisticRegression(
        max_iter=cfg["probe_max_iter"],
        class_weight="balanced",
        solver=cfg["probe_solver"],
        n_jobs=-1,
        random_state=SEED,
    )
    t0 = time.time()
    clf.fit(data["X_train"], data["y_train"])
    train_time = time.time() - t0

    pred = clf.predict(data["X_test"])
    proba = clf.predict_proba(data["X_test"])[:, 1]
    metrics = compute_metrics(data["y_test"], pred, proba)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        dump(clf, os.path.join(save_dir, "detector.joblib"))

    return {
        "metrics": metrics,
        "_meta": {
            "params": count_params(clf),
            "train_time_s": train_time,
            "train_n": int(len(data["y_train"])),
        },
    }


def run_mlp_prober(data: dict, config: dict = None, save_dir: str = None) -> dict:
    cfg = config or FULL_NONLINEAR_CONFIG
    t0 = time.time()
    prober, _ = train_mlp_prober(
        data["X_train"], data["y_train"], data["X_val"], data["y_val"],
        input_dim=data["X_train"].shape[1], cfg=cfg,
    )
    train_time = time.time() - t0

    prober.eval()
    metrics = _eval_logits(prober, data["X_test"], data["y_test"])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(prober.state_dict(), os.path.join(save_dir, "detector.pt"))

    return {
        "metrics": metrics,
        "_meta": {
            "params": count_params(prober),
            "train_time_s": train_time,
            "train_n": int(len(data["y_train"])),
        },
    }


def run_ae_mlp_prober(data: dict, config: dict = None, save_dir: str = None) -> dict:
    cfg = config or REDUCED_NONLINEAR_CONFIG

    # 85/15 split inside the balanced train set for the autoencoder (as in reduced_nonlinear)
    ae_tr, ae_val = split_train_val(len(data["X_train"]), seed=SEED)
    t0_ae = time.time()
    ae, _ = train_autoencoder(
        data["X_train"][ae_tr], data["X_train"][ae_val], data["X_train"].shape[1], cfg,
    )
    ae_time = time.time() - t0_ae
    ae.eval()

    def encode(X):
        with torch.no_grad():
            return ae.encode(torch.tensor(X, dtype=torch.float32, device=DEVICE)).cpu().numpy()

    z_train, z_val, z_test = encode(data["X_train"]), encode(data["X_val"]), encode(data["X_test"])

    t0 = time.time()
    prober, _ = train_mlp_prober(
        z_train, data["y_train"], z_val, data["y_val"],
        input_dim=cfg["autoencoder_latent_dim"], cfg=cfg,
    )
    train_time = time.time() - t0

    prober.eval()
    metrics = _eval_logits(prober, z_test, data["y_test"])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(ae.state_dict(), os.path.join(save_dir, "ae.pt"))
        torch.save(prober.state_dict(), os.path.join(save_dir, "detector.pt"))

    return {
        "metrics": metrics,
        "_meta": {
            "params": count_params(prober),
            "train_time_s": train_time,
            "train_n": int(z_train.shape[0]),
            "ae_params": count_params(ae),
            "ae_time_s": ae_time,
        },
    }


def run_encoder_head_prober(data: dict, config: dict = None, save_dir: str = None) -> dict:
    cfg = config or ONE_FOR_ALL_CONFIG
    t0 = time.time()
    encoder, head, _ = _train_teacher_pipeline(
        data["X_train"], data["y_train"], data["X_val"], data["y_val"],
        input_dim=data["X_train"].shape[1], cfg=cfg,
    )
    train_time = time.time() - t0

    encoder.eval(); head.eval()
    metrics = _eval_logits(lambda x: head(encoder(x)), data["X_test"], data["y_test"])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(encoder.state_dict(), os.path.join(save_dir, "encoder.pt"))
        torch.save(head.state_dict(), os.path.join(save_dir, "head.pt"))

    return {
        "metrics": metrics,
        "_meta": {
            "params": count_params(encoder) + count_params(head),
            "train_time_s": train_time,
            "train_n": int(len(data["y_train"])),
        },
    }


# Keyed by the method names in run_experiments; each uses its own method's config.
PROBER_REGISTRY = {
    "ridge_regressor": (run_linear_prober, RIDGE_REGRESSOR_CONFIG),
    "procrustes": (run_linear_prober, PROCRUSTES_CONFIG),
    "cka": (run_linear_prober, CKA_CONFIG),
    "cca": (run_linear_prober, CCA_CONFIG),
    "hybrid": (run_linear_prober, HYBRID_CONFIG),
    "full_nonlinear": (run_mlp_prober, FULL_NONLINEAR_CONFIG),
    "reduced_nonlinear": (run_ae_mlp_prober, REDUCED_NONLINEAR_CONFIG),
    "one_for_all": (run_encoder_head_prober, ONE_FOR_ALL_CONFIG),
}
PROBERS = list(PROBER_REGISTRY)
