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
# Each entry defines:
#   - dataset:        name of the dataset folder in activation_cache
#   - trainer:        model name used as "teacher" (trains the prober)
#   - tester:         model name used as "student" (cross-model eval)
#   - trainer_layers: dict with attn/mlp/hidden layer indices for trainer
#   - tester_layers:  dict with attn/mlp/hidden layer indices for tester
# ==================================================================

EXPERIMENTS = {
    # ==========================================================
    # belief_bank_constraints (BBC) — 12 combinations
    # ==========================================================
    "QwenToFalcon_BBC": {
        "dataset": "belief_bank_constraints",
        "trainer": "Qwen2.5-7B",
        "tester": "Falcon3-7B-Base",
        "trainer_layers": {
            "attn": [16, 17, 20],
            "mlp": [16, 18, 19],
            "hidden": [18, 19, 20],
        },
        "tester_layers": {
            "attn": [12, 13, 17],
            "mlp": [0, 2, 12],
            "hidden": [0, 1, 2],
        },
    },
    "FalconToQwen_BBC": {
        "dataset": "belief_bank_constraints",
        "trainer": "Falcon3-7B-Base",
        "tester": "Qwen2.5-7B",
        "trainer_layers": {
            "attn": [12, 13, 17],
            "mlp": [0, 2, 12],
            "hidden": [0, 1, 2],
        },
        "tester_layers": {
            "attn": [16, 17, 20],
            "mlp": [16, 18, 19],
            "hidden": [18, 19, 20],
        },
    },
    "QwenToGemma_BBC": {
        "dataset": "belief_bank_constraints",
        "trainer": "Qwen2.5-7B",
        "tester": "gemma-2-9b-it",
        "trainer_layers": {
            "attn": [16, 17, 20],
            "mlp": [16, 18, 19],
            "hidden": [18, 19, 20],
        },
        "tester_layers": {
            "attn": [23, 27, 33],
            "mlp": [24, 25, 26],
            "hidden": [23, 24, 27],
        },
    },
    "GemmaToQwen_BBC": {
        "dataset": "belief_bank_constraints",
        "trainer": "gemma-2-9b-it",
        "tester": "Qwen2.5-7B",
        "trainer_layers": {
            "attn": [23, 27, 33],
            "mlp": [24, 25, 26],
            "hidden": [23, 24, 27],
        },
        "tester_layers": {
            "attn": [16, 17, 20],
            "mlp": [16, 18, 19],
            "hidden": [18, 19, 20],
        },
    },
    "QwenToLlama_BBC": {
        "dataset": "belief_bank_constraints",
        "trainer": "Qwen2.5-7B",
        "tester": "Llama-3.1-8B-Instruct",
        "trainer_layers": {
            "attn": [16, 17, 20],
            "mlp": [16, 18, 19],
            "hidden": [18, 19, 20],
        },
        "tester_layers": {
            "attn": [5, 8, 12],
            "mlp": [13, 14, 15],
            "hidden": [13, 14, 15],
        },
    },
    "LlamaToQwen_BBC": {
        "dataset": "belief_bank_constraints",
        "trainer": "Llama-3.1-8B-Instruct",
        "tester": "Qwen2.5-7B",
        "trainer_layers": {
            "attn": [5, 8, 12],
            "mlp": [13, 14, 15],
            "hidden": [13, 14, 15],
        },
        "tester_layers": {
            "attn": [16, 17, 20],
            "mlp": [16, 18, 19],
            "hidden": [18, 19, 20],
        },
    },
    "FalconToGemma_BBC": {
        "dataset": "belief_bank_constraints",
        "trainer": "Falcon3-7B-Base",
        "tester": "gemma-2-9b-it",
        "trainer_layers": {
            "attn": [12, 13, 17],
            "mlp": [0, 2, 12],
            "hidden": [0, 1, 2],
        },
        "tester_layers": {
            "attn": [23, 27, 33],
            "mlp": [24, 25, 26],
            "hidden": [23, 24, 27],
        },
    },
    "GemmaToFalcon_BBC": {
        "dataset": "belief_bank_constraints",
        "trainer": "gemma-2-9b-it",
        "tester": "Falcon3-7B-Base",
        "trainer_layers": {
            "attn": [23, 27, 33],
            "mlp": [24, 25, 26],
            "hidden": [23, 24, 27],
        },
        "tester_layers": {
            "attn": [12, 13, 17],
            "mlp": [0, 2, 12],
            "hidden": [0, 1, 2],
        },
    },
    "FalconToLlama_BBC": {
        "dataset": "belief_bank_constraints",
        "trainer": "Falcon3-7B-Base",
        "tester": "Llama-3.1-8B-Instruct",
        "trainer_layers": {
            "attn": [12, 13, 17],
            "mlp": [0, 2, 12],
            "hidden": [0, 1, 2],
        },
        "tester_layers": {
            "attn": [5, 8, 12],
            "mlp": [13, 14, 15],
            "hidden": [13, 14, 15],
        },
    },
    "LlamaToFalcon_BBC": {
        "dataset": "belief_bank_constraints",
        "trainer": "Llama-3.1-8B-Instruct",
        "tester": "Falcon3-7B-Base",
        "trainer_layers": {
            "attn": [5, 8, 12],
            "mlp": [13, 14, 15],
            "hidden": [13, 14, 15],
        },
        "tester_layers": {
            "attn": [12, 13, 17],
            "mlp": [0, 2, 12],
            "hidden": [0, 1, 2],
        },
    },
    "GemmaToLlama_BBC": {
        "dataset": "belief_bank_constraints",
        "trainer": "gemma-2-9b-it",
        "tester": "Llama-3.1-8B-Instruct",
        "trainer_layers": {
            "attn": [23, 27, 33],
            "mlp": [24, 25, 26],
            "hidden": [23, 24, 27],
        },
        "tester_layers": {
            "attn": [5, 8, 12],
            "mlp": [13, 14, 15],
            "hidden": [13, 14, 15],
        },
    },
    "LlamaToGemma_BBC": {
        "dataset": "belief_bank_constraints",
        "trainer": "Llama-3.1-8B-Instruct",
        "tester": "gemma-2-9b-it",
        "trainer_layers": {
            "attn": [5, 8, 12],
            "mlp": [13, 14, 15],
            "hidden": [13, 14, 15],
        },
        "tester_layers": {
            "attn": [23, 27, 33],
            "mlp": [24, 25, 26],
            "hidden": [23, 24, 27],
        },
    },
    # ==========================================================
    # belief_bank_facts (BBF) — 12 combinations
    # ==========================================================
    "QwenToFalcon_BBF": {
        "dataset": "belief_bank_facts",
        "trainer": "Qwen2.5-7B",
        "tester": "Falcon3-7B-Base",
        "trainer_layers": {
            "attn": [15, 19, 20],
            "mlp": [15, 20, 22],
            "hidden": [22, 24, 25],
        },
        "tester_layers": {
            "attn": [0, 1, 2],
            "mlp": [10, 12, 14],
            "hidden": [2, 3, 11],
        },
    },
    "FalconToQwen_BBF": {
        "dataset": "belief_bank_facts",
        "trainer": "Falcon3-7B-Base",
        "tester": "Qwen2.5-7B",
        "trainer_layers": {
            "attn": [0, 1, 2],
            "mlp": [10, 12, 14],
            "hidden": [2, 3, 11],
        },
        "tester_layers": {
            "attn": [15, 19, 20],
            "mlp": [15, 20, 22],
            "hidden": [22, 24, 25],
        },
    },
    "QwenToGemma_BBF": {
        "dataset": "belief_bank_facts",
        "trainer": "Qwen2.5-7B",
        "tester": "gemma-2-9b-it",
        "trainer_layers": {
            "attn": [15, 19, 20],
            "mlp": [15, 20, 22],
            "hidden": [22, 24, 25],
        },
        "tester_layers": {
            "attn": [21, 24, 27],
            "mlp": [22, 25, 27],
            "hidden": [23, 26, 34],
        },
    },
    "GemmaToQwen_BBF": {
        "dataset": "belief_bank_facts",
        "trainer": "gemma-2-9b-it",
        "tester": "Qwen2.5-7B",
        "trainer_layers": {
            "attn": [21, 24, 27],
            "mlp": [22, 25, 27],
            "hidden": [23, 26, 34],
        },
        "tester_layers": {
            "attn": [15, 19, 20],
            "mlp": [15, 20, 22],
            "hidden": [22, 24, 25],
        },
    },
    "QwenToLlama_BBF": {
        "dataset": "belief_bank_facts",
        "trainer": "Qwen2.5-7B",
        "tester": "Llama-3.1-8B-Instruct",
        "trainer_layers": {
            "attn": [15, 19, 20],
            "mlp": [15, 20, 22],
            "hidden": [22, 24, 25],
        },
        "tester_layers": {
            "attn": [8, 13, 14],
            "mlp": [14, 15, 21],
            "hidden": [14, 15, 16],
        },
    },
    "LlamaToQwen_BBF": {
        "dataset": "belief_bank_facts",
        "trainer": "Llama-3.1-8B-Instruct",
        "tester": "Qwen2.5-7B",
        "trainer_layers": {
            "attn": [8, 13, 14],
            "mlp": [14, 15, 21],
            "hidden": [14, 15, 16],
        },
        "tester_layers": {
            "attn": [15, 19, 20],
            "mlp": [15, 20, 22],
            "hidden": [22, 24, 25],
        },
    },
    "FalconToGemma_BBF": {
        "dataset": "belief_bank_facts",
        "trainer": "Falcon3-7B-Base",
        "tester": "gemma-2-9b-it",
        "trainer_layers": {
            "attn": [0, 1, 2],
            "mlp": [10, 12, 14],
            "hidden": [2, 3, 11],
        },
        "tester_layers": {
            "attn": [21, 24, 27],
            "mlp": [22, 25, 27],
            "hidden": [23, 26, 34],
        },
    },
    "GemmaToFalcon_BBF": {
        "dataset": "belief_bank_facts",
        "trainer": "gemma-2-9b-it",
        "tester": "Falcon3-7B-Base",
        "trainer_layers": {
            "attn": [21, 24, 27],
            "mlp": [22, 25, 27],
            "hidden": [23, 26, 34],
        },
        "tester_layers": {
            "attn": [0, 1, 2],
            "mlp": [10, 12, 14],
            "hidden": [2, 3, 11],
        },
    },
    "FalconToLlama_BBF": {
        "dataset": "belief_bank_facts",
        "trainer": "Falcon3-7B-Base",
        "tester": "Llama-3.1-8B-Instruct",
        "trainer_layers": {
            "attn": [0, 1, 2],
            "mlp": [10, 12, 14],
            "hidden": [2, 3, 11],
        },
        "tester_layers": {
            "attn": [8, 13, 14],
            "mlp": [14, 15, 21],
            "hidden": [14, 15, 16],
        },
    },
    "LlamaToFalcon_BBF": {
        "dataset": "belief_bank_facts",
        "trainer": "Llama-3.1-8B-Instruct",
        "tester": "Falcon3-7B-Base",
        "trainer_layers": {
            "attn": [8, 13, 14],
            "mlp": [14, 15, 21],
            "hidden": [14, 15, 16],
        },
        "tester_layers": {
            "attn": [0, 1, 2],
            "mlp": [10, 12, 14],
            "hidden": [2, 3, 11],
        },
    },
    "GemmaToLlama_BBF": {
        "dataset": "belief_bank_facts",
        "trainer": "gemma-2-9b-it",
        "tester": "Llama-3.1-8B-Instruct",
        "trainer_layers": {
            "attn": [21, 24, 27],
            "mlp": [22, 25, 27],
            "hidden": [23, 26, 34],
        },
        "tester_layers": {
            "attn": [8, 13, 14],
            "mlp": [14, 15, 21],
            "hidden": [14, 15, 16],
        },
    },
    "LlamaToGemma_BBF": {
        "dataset": "belief_bank_facts",
        "trainer": "Llama-3.1-8B-Instruct",
        "tester": "gemma-2-9b-it",
        "trainer_layers": {
            "attn": [8, 13, 14],
            "mlp": [14, 15, 21],
            "hidden": [14, 15, 16],
        },
        "tester_layers": {
            "attn": [21, 24, 27],
            "mlp": [22, 25, 27],
            "hidden": [23, 26, 34],
        },
    },
    # ==========================================================
    # halu_eval (HE) — 12 combinations
    # ==========================================================
    "QwenToFalcon_HE": {
        "dataset": "halu_eval",
        "trainer": "Qwen2.5-7B",
        "tester": "Falcon3-7B-Base",
        "trainer_layers": {
            "attn": [8, 16, 22],
            "mlp": [5, 22, 26],
            "hidden": [7, 24, 26],
        },
        "tester_layers": {
            "attn": [16, 17, 19],
            "mlp": [1, 2, 19],
            "hidden": [1, 6, 15],
        },
    },
    "FalconToQwen_HE": {
        "dataset": "halu_eval",
        "trainer": "Falcon3-7B-Base",
        "tester": "Qwen2.5-7B",
        "trainer_layers": {
            "attn": [16, 17, 19],
            "mlp": [1, 2, 19],
            "hidden": [1, 6, 15],
        },
        "tester_layers": {
            "attn": [8, 16, 22],
            "mlp": [5, 22, 26],
            "hidden": [7, 24, 26],
        },
    },
    "QwenToGemma_HE": {
        "dataset": "halu_eval",
        "trainer": "Qwen2.5-7B",
        "tester": "gemma-2-9b-it",
        "trainer_layers": {
            "attn": [8, 16, 22],
            "mlp": [5, 22, 26],
            "hidden": [7, 24, 26],
        },
        "tester_layers": {
            "attn": [21, 26, 27],
            "mlp": [23, 24, 28],
            "hidden": [19, 24, 28],
        },
    },
    "GemmaToQwen_HE": {
        "dataset": "halu_eval",
        "trainer": "gemma-2-9b-it",
        "tester": "Qwen2.5-7B",
        "trainer_layers": {
            "attn": [21, 26, 27],
            "mlp": [23, 24, 28],
            "hidden": [19, 24, 28],
        },
        "tester_layers": {
            "attn": [8, 16, 22],
            "mlp": [5, 22, 26],
            "hidden": [7, 24, 26],
        },
    },
    "QwenToLlama_HE": {
        "dataset": "halu_eval",
        "trainer": "Qwen2.5-7B",
        "tester": "Llama-3.1-8B-Instruct",
        "trainer_layers": {
            "attn": [8, 16, 22],
            "mlp": [5, 22, 26],
            "hidden": [7, 24, 26],
        },
        "tester_layers": {
            "attn": [14, 15, 16],
            "mlp": [13, 14, 15],
            "hidden": [14, 15, 16],
        },
    },
    "LlamaToQwen_HE": {
        "dataset": "halu_eval",
        "trainer": "Llama-3.1-8B-Instruct",
        "tester": "Qwen2.5-7B",
        "trainer_layers": {
            "attn": [14, 15, 16],
            "mlp": [13, 14, 15],
            "hidden": [14, 15, 16],
        },
        "tester_layers": {
            "attn": [8, 16, 22],
            "mlp": [5, 22, 26],
            "hidden": [7, 24, 26],
        },
    },
    "FalconToGemma_HE": {
        "dataset": "halu_eval",
        "trainer": "Falcon3-7B-Base",
        "tester": "gemma-2-9b-it",
        "trainer_layers": {
            "attn": [16, 17, 19],
            "mlp": [1, 2, 19],
            "hidden": [1, 6, 15],
        },
        "tester_layers": {
            "attn": [21, 26, 27],
            "mlp": [23, 24, 28],
            "hidden": [19, 24, 28],
        },
    },
    "GemmaToFalcon_HE": {
        "dataset": "halu_eval",
        "trainer": "gemma-2-9b-it",
        "tester": "Falcon3-7B-Base",
        "trainer_layers": {
            "attn": [21, 26, 27],
            "mlp": [23, 24, 28],
            "hidden": [19, 24, 28],
        },
        "tester_layers": {
            "attn": [16, 17, 19],
            "mlp": [1, 2, 19],
            "hidden": [1, 6, 15],
        },
    },
    "FalconToLlama_HE": {
        "dataset": "halu_eval",
        "trainer": "Falcon3-7B-Base",
        "tester": "Llama-3.1-8B-Instruct",
        "trainer_layers": {
            "attn": [16, 17, 19],
            "mlp": [1, 2, 19],
            "hidden": [1, 6, 15],
        },
        "tester_layers": {
            "attn": [14, 15, 16],
            "mlp": [13, 14, 15],
            "hidden": [14, 15, 16],
        },
    },
    "LlamaToFalcon_HE": {
        "dataset": "halu_eval",
        "trainer": "Llama-3.1-8B-Instruct",
        "tester": "Falcon3-7B-Base",
        "trainer_layers": {
            "attn": [14, 15, 16],
            "mlp": [13, 14, 15],
            "hidden": [14, 15, 16],
        },
        "tester_layers": {
            "attn": [16, 17, 19],
            "mlp": [1, 2, 19],
            "hidden": [1, 6, 15],
        },
    },
    "GemmaToLlama_HE": {
        "dataset": "halu_eval",
        "trainer": "gemma-2-9b-it",
        "tester": "Llama-3.1-8B-Instruct",
        "trainer_layers": {
            "attn": [21, 26, 27],
            "mlp": [23, 24, 28],
            "hidden": [19, 24, 28],
        },
        "tester_layers": {
            "attn": [14, 15, 16],
            "mlp": [13, 14, 15],
            "hidden": [14, 15, 16],
        },
    },
    "LlamaToGemma_HE": {
        "dataset": "halu_eval",
        "trainer": "Llama-3.1-8B-Instruct",
        "tester": "gemma-2-9b-it",
        "trainer_layers": {
            "attn": [14, 15, 16],
            "mlp": [13, 14, 15],
            "hidden": [14, 15, 16],
        },
        "tester_layers": {
            "attn": [21, 26, 27],
            "mlp": [23, 24, 28],
            "hidden": [19, 24, 28],
        },
    },
}




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
    "cca_components": 64,
    "cca_max_iter": 500,
    "cca_tol": 1e-06,
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
