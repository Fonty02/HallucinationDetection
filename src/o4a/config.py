"""Experiment configuration: EXPERIMENTS dict, hyperparameters, constants."""

import os
import torch

# ==================================================================
# PARAMETERS FROM HTC (via environment variables - MANDATORY)
# ==================================================================
# These MUST be set by the HTC job via bash script
# No hardcoded defaults, no fallbacks

_seed_str = os.environ.get("O4A_SEED")
if not _seed_str:
    raise RuntimeError("O4A_SEED environment variable not set. Must be launched via HTC.")
SEED = int(_seed_str)

_device_str = os.environ.get("O4A_DEVICE")
if not _device_str:
    raise RuntimeError("O4A_DEVICE environment variable not set. Must be launched via HTC.")
DEVICE = torch.device(_device_str)

# ==================================================================
# STATIC CONSTANTS (not configurable, same for all jobs)
# ==================================================================

NOTEBOOK_COMPAT = False
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR_NAME = "activation_cache"
LAYER_TYPES = ["attn", "mlp", "hidden"]
METRICS = ["accuracy", "precision", "recall", "f1", "auroc"]
METHODS = ["ridge_regressor", "procrustes", "cka", "cca", "hybrid", "full_nonlinear", "reduced_nonlinear", "one_for_all"]

TRAIN_SPLIT = 0.7
ALIGNMENT_SPLIT = 0.7
PROBER_VAL_SPLIT = 0.15

# Model name aliases (filesystem resolution only)
MODEL_ALIASES = {
    "LLama3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
}

# ==================================================================
# EXPERIMENTS DICTIONARY
#
# Built in o4a/experiments.py from MODELS × DATASETS and the layer selection
# in o4a/selected_layers.json (see scripts/O4A_experiments/select_top_layers.py).
# ==================================================================

from o4a.experiments import EXPERIMENTS  # noqa: E402, F401

# ==================================================================
# HYPERPARAMETERS PER METHOD
# ==================================================================

RIDGE_REGRESSOR_CONFIG = {
    "ridge_alpha": 1000.0,
    "probe_max_iter": 10000,
    "probe_solver": "lbfgs",
}

PROCRUSTES_CONFIG = {
    "ridge_alpha": 1000.0,
    "probe_max_iter": 10000,
    "probe_solver": "lbfgs",
}

CKA_CONFIG = {
    "ridge_alpha": 1000.0,
    "probe_max_iter": 10000,
    "probe_solver": "lbfgs",
}

CCA_CONFIG = {
    "cca_components": 32,
    "cca_max_iter": 100,
    "cca_tol": 1e-04,
    "cca_scale": False,
    "probe_max_iter": 10000,
    "probe_solver": "lbfgs",
}

HYBRID_CONFIG = {
    "alignment_hidden_dim": 128,
    "alignment_dropout": 0.5,
    "alignment_lr": 1e-3,
    "alignment_weight_decay": 0.1,
    "alignment_batch_size": 32,
    "alignment_max_epochs": 1000,
    "alignment_patience": 50,
    "alignment_min_delta": 1e-4,
    "alignment_grad_clip": 1.0,
    "alignment_loss_alpha": 0.01,
    "alignment_loss_beta": 1.0,
    "probe_max_iter": 1000,
    "probe_solver": "lbfgs",
}

FULL_NONLINEAR_CONFIG = {
    "alignment_hidden_dim": 128,
    "alignment_dropout": 0.5,
    "alignment_lr": 1e-3,
    "alignment_weight_decay": 0.1,
    "alignment_batch_size": 32,
    "alignment_max_epochs": 1000,
    "alignment_patience": 50,
    "alignment_min_delta": 1e-4,
    "alignment_grad_clip": 1.0,
    "alignment_loss_alpha": 0.01,
    "alignment_loss_beta": 1.0,
    "prober_hidden_dim": 64,
    "prober_dropout": 0.5,
    "prober_lr": 1e-3,
    "prober_weight_decay": 0.01,
    "prober_batch_size": 64,
    "prober_max_epochs": 200,
    "prober_patience": 30,
    "prober_min_delta": 1e-4,
    "prober_grad_clip": 1.0,
}

REDUCED_NONLINEAR_CONFIG = {
    "autoencoder_latent_dim": 128,
    "autoencoder_hidden_dim": 256,
    "autoencoder_dropout": 0.2,
    "autoencoder_lr": 1e-3,
    "autoencoder_weight_decay": 0.01,
    "autoencoder_batch_size": 64,
    "autoencoder_max_epochs": 300,
    "autoencoder_patience": 30,
    "autoencoder_min_delta": 1e-4,
    "autoencoder_grad_clip": 1.0,
    "alignment_hidden_dim": 256,
    "alignment_dropout": 0.3,
    "alignment_lr": 1e-3,
    "alignment_weight_decay": 0.01,
    "alignment_batch_size": 32,
    "alignment_max_epochs": 500,
    "alignment_patience": 50,
    "alignment_min_delta": 1e-4,
    "alignment_grad_clip": 1.0,
    "alignment_loss_alpha": 0.5,
    "alignment_loss_beta": 0.5,
    "prober_hidden_dim": 64,
    "prober_dropout": 0.3,
    "prober_lr": 1e-3,
    "prober_weight_decay": 0.01,
    "prober_batch_size": 64,
    "prober_max_epochs": 200,
    "prober_patience": 30,
    "prober_min_delta": 1e-4,
    "prober_grad_clip": 1.0,
}

ONE_FOR_ALL_CONFIG = {
    "encoder_latent_dim": 256,
    "encoder_hidden_dim": 512,
    "encoder_dropout": 0.3,
    "head_hidden_dim": 128,
    "head_dropout": 0.3,
    "learning_rate": 1e-3,
    "weight_decay": 1e-2,
    "batch_size": 64,
    "max_epochs": 100,
    "early_stopping_patience": 15,
    "gradient_clip_max_norm": 1.0,
    "val_split": 0.15,
}
