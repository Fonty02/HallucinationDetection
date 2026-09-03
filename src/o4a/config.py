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
            "attn": [8, 9, 19],
            "mlp": [10, 19, 21],
            "hidden": [15, 17, 19],
        },
        "tester_layers": {
            "attn": [13, 14, 23],
            "mlp": [16, 20, 24],
            "hidden": [12, 22, 24],
        },
    },
    "FalconToQwen_BBC": {
        "dataset": "belief_bank_constraints",
        "trainer": "Falcon3-7B-Base",
        "tester": "Qwen2.5-7B",
        "trainer_layers": {
            "attn": [13, 14, 23],
            "mlp": [16, 20, 24],
            "hidden": [12, 22, 24],
        },
        "tester_layers": {
            "attn": [8, 9, 19],
            "mlp": [10, 19, 21],
            "hidden": [15, 17, 19],
        },
    },
    "QwenToGemma_BBC": {
        "dataset": "belief_bank_constraints",
        "trainer": "Qwen2.5-7B",
        "tester": "gemma-2-9b-it",
        "trainer_layers": {
            "attn": [8, 9, 19],
            "mlp": [10, 19, 21],
            "hidden": [15, 17, 19],
        },
        "tester_layers": {
            "attn": [24, 28, 39],
            "mlp": [5, 22, 28],
            "hidden": [39, 40, 41],
        },
    },
    "GemmaToQwen_BBC": {
        "dataset": "belief_bank_constraints",
        "trainer": "gemma-2-9b-it",
        "tester": "Qwen2.5-7B",
        "trainer_layers": {
            "attn": [24, 28, 39],
            "mlp": [5, 22, 28],
            "hidden": [39, 40, 41],
        },
        "tester_layers": {
            "attn": [8, 9, 19],
            "mlp": [10, 19, 21],
            "hidden": [15, 17, 19],
        },
    },
    "QwenToLlama_BBC": {
        "dataset": "belief_bank_constraints",
        "trainer": "Qwen2.5-7B",
        "tester": "Llama-3.1-8B-Instruct",
        "trainer_layers": {
            "attn": [8, 9, 19],
            "mlp": [10, 19, 21],
            "hidden": [15, 17, 19],
        },
        "tester_layers": {
            "attn": [12, 13, 15],
            "mlp": [12, 13, 14],
            "hidden": [12, 13, 14],
        },
    },
    "LlamaToQwen_BBC": {
        "dataset": "belief_bank_constraints",
        "trainer": "Llama-3.1-8B-Instruct",
        "tester": "Qwen2.5-7B",
        "trainer_layers": {
            "attn": [12, 13, 15],
            "mlp": [12, 13, 14],
            "hidden": [12, 13, 14],
        },
        "tester_layers": {
            "attn": [8, 9, 19],
            "mlp": [10, 19, 21],
            "hidden": [15, 17, 19],
        },
    },
    "FalconToGemma_BBC": {
        "dataset": "belief_bank_constraints",
        "trainer": "Falcon3-7B-Base",
        "tester": "gemma-2-9b-it",
        "trainer_layers": {
            "attn": [13, 14, 23],
            "mlp": [16, 20, 24],
            "hidden": [12, 22, 24],
        },
        "tester_layers": {
            "attn": [24, 28, 39],
            "mlp": [5, 22, 28],
            "hidden": [39, 40, 41],
        },
    },
    "GemmaToFalcon_BBC": {
        "dataset": "belief_bank_constraints",
        "trainer": "gemma-2-9b-it",
        "tester": "Falcon3-7B-Base",
        "trainer_layers": {
            "attn": [24, 28, 39],
            "mlp": [5, 22, 28],
            "hidden": [39, 40, 41],
        },
        "tester_layers": {
            "attn": [13, 14, 23],
            "mlp": [16, 20, 24],
            "hidden": [12, 22, 24],
        },
    },
    "FalconToLlama_BBC": {
        "dataset": "belief_bank_constraints",
        "trainer": "Falcon3-7B-Base",
        "tester": "Llama-3.1-8B-Instruct",
        "trainer_layers": {
            "attn": [13, 14, 23],
            "mlp": [16, 20, 24],
            "hidden": [12, 22, 24],
        },
        "tester_layers": {
            "attn": [12, 13, 15],
            "mlp": [12, 13, 14],
            "hidden": [12, 13, 14],
        },
    },
    "LlamaToFalcon_BBC": {
        "dataset": "belief_bank_constraints",
        "trainer": "Llama-3.1-8B-Instruct",
        "tester": "Falcon3-7B-Base",
        "trainer_layers": {
            "attn": [12, 13, 15],
            "mlp": [12, 13, 14],
            "hidden": [12, 13, 14],
        },
        "tester_layers": {
            "attn": [13, 14, 23],
            "mlp": [16, 20, 24],
            "hidden": [12, 22, 24],
        },
    },
    "GemmaToLlama_BBC": {
        "dataset": "belief_bank_constraints",
        "trainer": "gemma-2-9b-it",
        "tester": "Llama-3.1-8B-Instruct",
        "trainer_layers": {
            "attn": [24, 28, 39],
            "mlp": [5, 22, 28],
            "hidden": [39, 40, 41],
        },
        "tester_layers": {
            "attn": [12, 13, 15],
            "mlp": [12, 13, 14],
            "hidden": [12, 13, 14],
        },
    },
    "LlamaToGemma_BBC": {
        "dataset": "belief_bank_constraints",
        "trainer": "Llama-3.1-8B-Instruct",
        "tester": "gemma-2-9b-it",
        "trainer_layers": {
            "attn": [12, 13, 15],
            "mlp": [12, 13, 14],
            "hidden": [12, 13, 14],
        },
        "tester_layers": {
            "attn": [24, 28, 39],
            "mlp": [5, 22, 28],
            "hidden": [39, 40, 41],
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
            "attn": [14, 16, 20],
            "mlp": [15, 16, 17],
            "hidden": [13, 15, 27],
        },
        "tester_layers": {
            "attn": [0, 8, 9],
            "mlp": [15, 18, 22],
            "hidden": [14, 18, 20],
        },
    },
    "FalconToQwen_BBF": {
        "dataset": "belief_bank_facts",
        "trainer": "Falcon3-7B-Base",
        "tester": "Qwen2.5-7B",
        "trainer_layers": {
            "attn": [0, 8, 9],
            "mlp": [15, 18, 22],
            "hidden": [14, 18, 20],
        },
        "tester_layers": {
            "attn": [14, 16, 20],
            "mlp": [15, 16, 17],
            "hidden": [13, 15, 27],
        },
    },
    "QwenToGemma_BBF": {
        "dataset": "belief_bank_facts",
        "trainer": "Qwen2.5-7B",
        "tester": "gemma-2-9b-it",
        "trainer_layers": {
            "attn": [14, 16, 20],
            "mlp": [15, 16, 17],
            "hidden": [13, 15, 27],
        },
        "tester_layers": {
            "attn": [18, 22, 23],
            "mlp": [19, 20, 22],
            "hidden": [20, 24, 25],
        },
    },
    "GemmaToQwen_BBF": {
        "dataset": "belief_bank_facts",
        "trainer": "gemma-2-9b-it",
        "tester": "Qwen2.5-7B",
        "trainer_layers": {
            "attn": [18, 22, 23],
            "mlp": [19, 20, 22],
            "hidden": [20, 24, 25],
        },
        "tester_layers": {
            "attn": [14, 16, 20],
            "mlp": [15, 16, 17],
            "hidden": [13, 15, 27],
        },
    },
    "QwenToLlama_BBF": {
        "dataset": "belief_bank_facts",
        "trainer": "Qwen2.5-7B",
        "tester": "Llama-3.1-8B-Instruct",
        "trainer_layers": {
            "attn": [14, 16, 20],
            "mlp": [15, 16, 17],
            "hidden": [13, 15, 27],
        },
        "tester_layers": {
            "attn": [14, 15, 16],
            "mlp": [10, 11, 12],
            "hidden": [12, 13, 15],
        },
    },
    "LlamaToQwen_BBF": {
        "dataset": "belief_bank_facts",
        "trainer": "Llama-3.1-8B-Instruct",
        "tester": "Qwen2.5-7B",
        "trainer_layers": {
            "attn": [14, 15, 16],
            "mlp": [10, 11, 12],
            "hidden": [12, 13, 15],
        },
        "tester_layers": {
            "attn": [14, 16, 20],
            "mlp": [15, 16, 17],
            "hidden": [13, 15, 27],
        },
    },
    "FalconToGemma_BBF": {
        "dataset": "belief_bank_facts",
        "trainer": "Falcon3-7B-Base",
        "tester": "gemma-2-9b-it",
        "trainer_layers": {
            "attn": [0, 8, 9],
            "mlp": [15, 18, 22],
            "hidden": [14, 18, 20],
        },
        "tester_layers": {
            "attn": [18, 22, 23],
            "mlp": [19, 20, 22],
            "hidden": [20, 24, 25],
        },
    },
    "GemmaToFalcon_BBF": {
        "dataset": "belief_bank_facts",
        "trainer": "gemma-2-9b-it",
        "tester": "Falcon3-7B-Base",
        "trainer_layers": {
            "attn": [18, 22, 23],
            "mlp": [19, 20, 22],
            "hidden": [20, 24, 25],
        },
        "tester_layers": {
            "attn": [0, 8, 9],
            "mlp": [15, 18, 22],
            "hidden": [14, 18, 20],
        },
    },
    "FalconToLlama_BBF": {
        "dataset": "belief_bank_facts",
        "trainer": "Falcon3-7B-Base",
        "tester": "Llama-3.1-8B-Instruct",
        "trainer_layers": {
            "attn": [0, 8, 9],
            "mlp": [15, 18, 22],
            "hidden": [14, 18, 20],
        },
        "tester_layers": {
            "attn": [14, 15, 16],
            "mlp": [10, 11, 12],
            "hidden": [12, 13, 15],
        },
    },
    "LlamaToFalcon_BBF": {
        "dataset": "belief_bank_facts",
        "trainer": "Llama-3.1-8B-Instruct",
        "tester": "Falcon3-7B-Base",
        "trainer_layers": {
            "attn": [14, 15, 16],
            "mlp": [10, 11, 12],
            "hidden": [12, 13, 15],
        },
        "tester_layers": {
            "attn": [0, 8, 9],
            "mlp": [15, 18, 22],
            "hidden": [14, 18, 20],
        },
    },
    "GemmaToLlama_BBF": {
        "dataset": "belief_bank_facts",
        "trainer": "gemma-2-9b-it",
        "tester": "Llama-3.1-8B-Instruct",
        "trainer_layers": {
            "attn": [18, 22, 23],
            "mlp": [19, 20, 22],
            "hidden": [20, 24, 25],
        },
        "tester_layers": {
            "attn": [14, 15, 16],
            "mlp": [10, 11, 12],
            "hidden": [12, 13, 15],
        },
    },
    "LlamaToGemma_BBF": {
        "dataset": "belief_bank_facts",
        "trainer": "Llama-3.1-8B-Instruct",
        "tester": "gemma-2-9b-it",
        "trainer_layers": {
            "attn": [14, 15, 16],
            "mlp": [10, 11, 12],
            "hidden": [12, 13, 15],
        },
        "tester_layers": {
            "attn": [18, 22, 23],
            "mlp": [19, 20, 22],
            "hidden": [20, 24, 25],
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
            "attn": [5, 8, 19],
            "mlp": [5, 8, 23],
            "hidden": [5, 6, 11],
        },
        "tester_layers": {
            "attn": [3, 11, 20],
            "mlp": [19, 20, 24],
            "hidden": [0, 3, 19],
        },
    },
    "FalconToQwen_HE": {
        "dataset": "halu_eval",
        "trainer": "Falcon3-7B-Base",
        "tester": "Qwen2.5-7B",
        "trainer_layers": {
            "attn": [3, 11, 20],
            "mlp": [19, 20, 24],
            "hidden": [0, 3, 19],
        },
        "tester_layers": {
            "attn": [5, 8, 19],
            "mlp": [5, 8, 23],
            "hidden": [5, 6, 11],
        },
    },
    "QwenToGemma_HE": {
        "dataset": "halu_eval",
        "trainer": "Qwen2.5-7B",
        "tester": "gemma-2-9b-it",
        "trainer_layers": {
            "attn": [5, 8, 19],
            "mlp": [5, 8, 23],
            "hidden": [5, 6, 11],
        },
        "tester_layers": {
            "attn": [9, 32, 40],
            "mlp": [31, 33, 34],
            "hidden": [2, 13, 41],
        },
    },
    "GemmaToQwen_HE": {
        "dataset": "halu_eval",
        "trainer": "gemma-2-9b-it",
        "tester": "Qwen2.5-7B",
        "trainer_layers": {
            "attn": [9, 32, 40],
            "mlp": [31, 33, 34],
            "hidden": [2, 13, 41],
        },
        "tester_layers": {
            "attn": [5, 8, 19],
            "mlp": [5, 8, 23],
            "hidden": [5, 6, 11],
        },
    },
    "QwenToLlama_HE": {
        "dataset": "halu_eval",
        "trainer": "Qwen2.5-7B",
        "tester": "Llama-3.1-8B-Instruct",
        "trainer_layers": {
            "attn": [5, 8, 19],
            "mlp": [5, 8, 23],
            "hidden": [5, 6, 11],
        },
        "tester_layers": {
            "attn": [1, 16, 30],
            "mlp": [15, 18, 22],
            "hidden": [16, 18, 21],
        },
    },
    "LlamaToQwen_HE": {
        "dataset": "halu_eval",
        "trainer": "Llama-3.1-8B-Instruct",
        "tester": "Qwen2.5-7B",
        "trainer_layers": {
            "attn": [1, 16, 30],
            "mlp": [15, 18, 22],
            "hidden": [16, 18, 21],
        },
        "tester_layers": {
            "attn": [5, 8, 19],
            "mlp": [5, 8, 23],
            "hidden": [5, 6, 11],
        },
    },
    "FalconToGemma_HE": {
        "dataset": "halu_eval",
        "trainer": "Falcon3-7B-Base",
        "tester": "gemma-2-9b-it",
        "trainer_layers": {
            "attn": [3, 11, 20],
            "mlp": [19, 20, 24],
            "hidden": [0, 3, 19],
        },
        "tester_layers": {
            "attn": [9, 32, 40],
            "mlp": [31, 33, 34],
            "hidden": [2, 13, 41],
        },
    },
    "GemmaToFalcon_HE": {
        "dataset": "halu_eval",
        "trainer": "gemma-2-9b-it",
        "tester": "Falcon3-7B-Base",
        "trainer_layers": {
            "attn": [9, 32, 40],
            "mlp": [31, 33, 34],
            "hidden": [2, 13, 41],
        },
        "tester_layers": {
            "attn": [3, 11, 20],
            "mlp": [19, 20, 24],
            "hidden": [0, 3, 19],
        },
    },
    "FalconToLlama_HE": {
        "dataset": "halu_eval",
        "trainer": "Falcon3-7B-Base",
        "tester": "Llama-3.1-8B-Instruct",
        "trainer_layers": {
            "attn": [3, 11, 20],
            "mlp": [19, 20, 24],
            "hidden": [0, 3, 19],
        },
        "tester_layers": {
            "attn": [1, 16, 30],
            "mlp": [15, 18, 22],
            "hidden": [16, 18, 21],
        },
    },
    "LlamaToFalcon_HE": {
        "dataset": "halu_eval",
        "trainer": "Llama-3.1-8B-Instruct",
        "tester": "Falcon3-7B-Base",
        "trainer_layers": {
            "attn": [1, 16, 30],
            "mlp": [15, 18, 22],
            "hidden": [16, 18, 21],
        },
        "tester_layers": {
            "attn": [3, 11, 20],
            "mlp": [19, 20, 24],
            "hidden": [0, 3, 19],
        },
    },
    "GemmaToLlama_HE": {
        "dataset": "halu_eval",
        "trainer": "gemma-2-9b-it",
        "tester": "Llama-3.1-8B-Instruct",
        "trainer_layers": {
            "attn": [9, 32, 40],
            "mlp": [31, 33, 34],
            "hidden": [2, 13, 41],
        },
        "tester_layers": {
            "attn": [1, 16, 30],
            "mlp": [15, 18, 22],
            "hidden": [16, 18, 21],
        },
    },
    "LlamaToGemma_HE": {
        "dataset": "halu_eval",
        "trainer": "Llama-3.1-8B-Instruct",
        "tester": "gemma-2-9b-it",
        "trainer_layers": {
            "attn": [1, 16, 30],
            "mlp": [15, 18, 22],
            "hidden": [16, 18, 21],
        },
        "tester_layers": {
            "attn": [9, 32, 40],
            "mlp": [31, 33, 34],
            "hidden": [2, 13, 41],
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
