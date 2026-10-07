"""Reuse models saved by run_experiments instead of retraining them.

run_experiments trains every method on prepare_shared_data(experiment, layer_type) and saves the
weights under saved_models/<experiment>/<layer_type>/<method>/<seed>/. Cross-domain runs train on
exactly the same data (only the test split changes), so those weights are loaded and evaluated on
the new test split with the same evaluation code as each method:

- cross-LLM: every method of <trainer>To<tester>_<train dataset> (detector + aligner / student).
- single-LLM: the trainer side of any <model>To*_<train dataset> experiment, since
  prepare_single_model_data uses the trainer splits and scaler of prepare_shared_data.

A saved model is used only if the matching results/experiments CSV row exists, its layers match
the current selection and the method completed; params/times/train_n are copied from that row.
Any failure returns None, and the caller trains from scratch.
"""

from __future__ import annotations

import csv
import os
import traceback

import torch
from joblib import load

from .config import (
    DEVICE,
    ROOT_DIR,
    RIDGE_REGRESSOR_CONFIG,
    PROCRUSTES_CONFIG,
    CKA_CONFIG,
    CCA_CONFIG,
    HYBRID_CONFIG,
    FULL_NONLINEAR_CONFIG,
    REDUCED_NONLINEAR_CONFIG,
    ONE_FOR_ALL_CONFIG,
)
from .experiments import MODELS, experiment_name
from .methods.training import compute_metrics
from .models import AlignmentNetwork, Autoencoder, ClassificationHead, Encoder, MLPProber
from .probers import _eval_logits

PRETRAINED_MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
PRETRAINED_RESULTS_DIR = os.path.join(ROOT_DIR, "results", "experiments")

LINEAR_METHODS = ["ridge_regressor", "procrustes", "cka", "cca", "hybrid"]
METHOD_CONFIGS = {
    "ridge_regressor": RIDGE_REGRESSOR_CONFIG,
    "procrustes": PROCRUSTES_CONFIG,
    "cka": CKA_CONFIG,
    "cca": CCA_CONFIG,
    "hybrid": HYBRID_CONFIG,
    "full_nonlinear": FULL_NONLINEAR_CONFIG,
    "reduced_nonlinear": REDUCED_NONLINEAR_CONFIG,
    "one_for_all": ONE_FOR_ALL_CONFIG,
}
CROSS_LLM_META = [
    "detector_params", "detector_time_s", "detector_train_n",
    "aligner_params", "aligner_time_s", "aligner_train_n",
    "ae_trainer_params", "ae_trainer_time_s", "ae_tester_params", "ae_tester_time_s",
]


# ------------------------------------------------------------------
# Lookup
# ------------------------------------------------------------------

def _results_row(exp_name: str, layer_type: str, seed: int) -> dict | None:
    path = os.path.join(PRETRAINED_RESULTS_DIR, exp_name, f"seed_{seed}", f"results_{layer_type}.csv")
    if not os.path.exists(path):
        return None
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("experiment") == exp_name and row.get("layer_type") == layer_type
                    and int(row.get("seed", -1)) == seed):
                return row
    return None


def _find_saved(
    exp_name: str, layer_type: str, seed: int, method: str,
    trainer_layers: list[int], tester_layers: list[int] | None = None,
) -> tuple[str, dict] | None:
    """(model_dir, results_row) of a completed run_experiments method with the current layers."""
    row = _results_row(exp_name, layer_type, seed)
    if row is None or row.get("trainer_layers") != str(trainer_layers):
        return None
    if tester_layers is not None and row.get("tester_layers") != str(tester_layers):
        return None
    method_cols = [c for c in row if c.startswith(f"{method}_")]
    if not method_cols or any(row[c] in ("", "ERROR") for c in method_cols):
        return None
    model_dir = os.path.join(PRETRAINED_MODELS_DIR, exp_name, layer_type, method, str(seed))
    if not os.path.isdir(model_dir):
        return None
    return model_dir, row


def _load_torch(module: torch.nn.Module, path: str) -> torch.nn.Module:
    module.load_state_dict(torch.load(path, map_location=DEVICE))
    return module.to(DEVICE).eval()


def _mlp_prober(model_dir: str, input_dim: int, cfg: dict) -> MLPProber:
    prober = MLPProber(input_dim=input_dim, hidden_dim=cfg["prober_hidden_dim"], dropout=cfg["prober_dropout"])
    return _load_torch(prober, os.path.join(model_dir, "detector.pt"))


def _autoencoder(path: str, input_dim: int, cfg: dict) -> Autoencoder:
    ae = Autoencoder(
        input_dim=input_dim,
        latent_dim=cfg["autoencoder_latent_dim"],
        hidden_dim=cfg["autoencoder_hidden_dim"],
        dropout=cfg["autoencoder_dropout"],
    )
    return _load_torch(ae, path)


def _alignment_network(model_dir: str, input_dim: int, output_dim: int, cfg: dict) -> AlignmentNetwork:
    net = AlignmentNetwork(
        input_dim=input_dim, output_dim=output_dim,
        hidden_dim=cfg["alignment_hidden_dim"], dropout=cfg["alignment_dropout"],
    )
    return _load_torch(net, os.path.join(model_dir, "aligner.pt"))


def _encoder(path: str, input_dim: int, cfg: dict) -> Encoder:
    enc = Encoder(input_dim, cfg["encoder_latent_dim"], cfg["encoder_hidden_dim"], cfg["encoder_dropout"])
    return _load_torch(enc, path)


def _head(model_dir: str, cfg: dict) -> ClassificationHead:
    head = ClassificationHead(cfg["encoder_latent_dim"], cfg["head_hidden_dim"], cfg["head_dropout"])
    return _load_torch(head, os.path.join(model_dir, "head.pt"))


def _to_tensor(X) -> torch.Tensor:
    return torch.tensor(X, dtype=torch.float32, device=DEVICE)


# ------------------------------------------------------------------
# Cross-LLM: same evaluation as each method in o4a.methods
# ------------------------------------------------------------------

def _eval_sklearn_detector(clf, X, y) -> dict:
    return compute_metrics(y, clf.predict(X), clf.predict_proba(X)[:, 1])


def _eval_torch_detector(model, X, y) -> dict:
    with torch.no_grad():
        Xt = _to_tensor(X)
        pred = model.predict(Xt).cpu().numpy()
        proba = torch.sigmoid(model(Xt)).cpu().numpy()
    return compute_metrics(y, pred, proba)


def _evaluate_cross_llm(method: str, shared_data: dict, model_dir: str) -> tuple[dict, dict]:
    """Return (trainer_metrics, tester_metrics) of the saved method on shared_data's test splits."""
    cfg = METHOD_CONFIGS[method]
    trainer, tester, alignment = shared_data["trainer"], shared_data["tester"], shared_data["alignment"]
    trainer_dim = trainer["X_train"].shape[1]
    tester_dim = tester["X_train"].shape[1]

    if method in LINEAR_METHODS:
        clf = load(os.path.join(model_dir, "detector.joblib"))
        X_tester_scaled = alignment["scaler_tester"].transform(tester["X_test_raw"])
        if method == "hybrid":
            align_model = _alignment_network(model_dir, tester_dim, trainer_dim, cfg)
            with torch.no_grad():
                X_tester_proj = align_model(_to_tensor(X_tester_scaled)).cpu().numpy()
        else:
            X_tester_proj = load(os.path.join(model_dir, "aligner.joblib")).predict(X_tester_scaled)
        return (_eval_sklearn_detector(clf, trainer["X_test"], trainer["y_test"]),
                _eval_sklearn_detector(clf, X_tester_proj, tester["y_test"]))

    if method == "full_nonlinear":
        prober = _mlp_prober(model_dir, trainer_dim, cfg)
        align_model = _alignment_network(model_dir, tester_dim, trainer_dim, cfg)
        X_tester_scaled = alignment["scaler_tester"].transform(tester["X_test_raw"])
        with torch.no_grad():
            projected = align_model(_to_tensor(X_tester_scaled)).cpu().numpy()
        return (_eval_torch_detector(prober, trainer["X_test"], trainer["y_test"]),
                _eval_torch_detector(prober, projected, tester["y_test"]))

    if method == "reduced_nonlinear":
        latent = cfg["autoencoder_latent_dim"]
        ae_trainer = _autoencoder(os.path.join(model_dir, "ae_trainer.pt"), trainer_dim, cfg)
        ae_tester = _autoencoder(os.path.join(model_dir, "ae_tester.pt"), tester_dim, cfg)
        align_model = _alignment_network(model_dir, latent, latent, cfg)
        prober = _mlp_prober(model_dir, latent, cfg)
        with torch.no_grad():
            z_trainer_test = ae_trainer.encode(_to_tensor(trainer["X_test"])).cpu().numpy()
            z_tester_aligned = align_model(ae_tester.encode(_to_tensor(tester["X_test"]))).cpu().numpy()
        return (_eval_torch_detector(prober, z_trainer_test, trainer["y_test"]),
                _eval_torch_detector(prober, z_tester_aligned, tester["y_test"]))

    if method == "one_for_all":
        enc_trainer = _encoder(os.path.join(model_dir, "enc_trainer.pt"), trainer_dim, cfg)
        enc_tester = _encoder(os.path.join(model_dir, "enc_tester.pt"), tester_dim, cfg)
        head = _head(model_dir, cfg)
        metrics = []
        for enc, side in ((enc_trainer, trainer), (enc_tester, tester)):
            with torch.no_grad():
                Xt = _to_tensor(side["X_test"])
                pred = head.predict(enc(Xt)).cpu().numpy()
                proba = torch.sigmoid(head(enc(Xt))).cpu().numpy()
            metrics.append(compute_metrics(side["y_test"], pred, proba))
        return metrics[0], metrics[1]

    raise ValueError(f"Unknown method: {method}")


def evaluate_pretrained_method(
    method: str, shared_data: dict, trainer: str, tester: str, dataset: str,
    layer_type: str, seed: int, trainer_layers: list[int], tester_layers: list[int],
) -> dict | None:
    """Cross-LLM result dict (as returned by o4a.methods) from saved models, or None."""
    exp_name = experiment_name(trainer, tester, dataset)
    found = _find_saved(exp_name, layer_type, seed, method, trainer_layers, tester_layers)
    if found is None:
        return None
    model_dir, row = found
    try:
        metrics_trainer, metrics_tester = _evaluate_cross_llm(method, shared_data, model_dir)
    except Exception:
        print(f"[PRETRAINED] could not reuse {model_dir}, training instead:")
        traceback.print_exc()
        return None
    return {
        "trainer": metrics_trainer,
        "tester": metrics_tester,
        "_meta": {s: row[f"{method}_{s}"] for s in CROSS_LLM_META if f"{method}_{s}" in row},
        "_source": model_dir,
    }


# ------------------------------------------------------------------
# Single-LLM: trainer side of any experiment with this model as trainer
# ------------------------------------------------------------------

def _evaluate_single_llm(prober: str, data: dict, model_dir: str) -> dict:
    cfg = METHOD_CONFIGS[prober]
    dim = data["X_train"].shape[1]
    if prober in LINEAR_METHODS:
        return _eval_sklearn_detector(load(os.path.join(model_dir, "detector.joblib")), data["X_test"], data["y_test"])
    if prober == "full_nonlinear":
        return _eval_logits(_mlp_prober(model_dir, dim, cfg), data["X_test"], data["y_test"])
    if prober == "reduced_nonlinear":
        ae = _autoencoder(os.path.join(model_dir, "ae_trainer.pt"), dim, cfg)
        with torch.no_grad():
            z_test = ae.encode(_to_tensor(data["X_test"])).cpu().numpy()
        return _eval_logits(_mlp_prober(model_dir, cfg["autoencoder_latent_dim"], cfg), z_test, data["y_test"])
    if prober == "one_for_all":
        enc = _encoder(os.path.join(model_dir, "enc_trainer.pt"), dim, cfg)
        head = _head(model_dir, cfg)
        return _eval_logits(lambda x: head(enc(x)), data["X_test"], data["y_test"])
    raise ValueError(f"Unknown prober: {prober}")


def evaluate_pretrained_prober(
    prober: str, data: dict, model: str, dataset: str, layer_type: str, seed: int, layers: list[int],
) -> dict | None:
    """Single-LLM result dict (as returned by o4a.probers) from saved trainer-side models, or None."""
    for tester in MODELS:
        if tester == model:
            continue
        found = _find_saved(experiment_name(model, tester, dataset), layer_type, seed, prober, layers)
        if found is None:
            continue
        model_dir, row = found
        try:
            metrics = _evaluate_single_llm(prober, data, model_dir)
        except Exception:
            print(f"[PRETRAINED] could not reuse {model_dir}, trying next:")
            traceback.print_exc()
            continue
        meta = {
            "params": row[f"{prober}_detector_params"],
            "train_time_s": row[f"{prober}_detector_time_s"],
            "train_n": row[f"{prober}_detector_train_n"],
        }
        if prober == "reduced_nonlinear":
            meta["ae_params"] = row[f"{prober}_ae_trainer_params"]
            meta["ae_time_s"] = row[f"{prober}_ae_trainer_time_s"]
        return {"metrics": metrics, "_meta": meta, "_source": model_dir}
    return None
