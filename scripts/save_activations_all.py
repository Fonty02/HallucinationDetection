import sys
from pathlib import Path
import argparse

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.HallucinationDetection import HallucinationDetection


DEFAULT_PROJECT_DIR = str(PROJECT_ROOT)


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return torch.bfloat16


EXPERIMENT_FALCON_BB_FACTS = {
    "name": "Falcon3-7B / belief_bank facts",
    "project_dir": DEFAULT_PROJECT_DIR,
    "llm_name": "tiiuae/Falcon3-7B-Base",
    "data_name": "belief_bank",
    "belief_bank_data_type": "facts",
    "use_local": False,
    "dtype": "bfloat16",
    "use_device_map": True,
    "use_flash_attn": False,
    "max_samples": None,
    "quantization": True,
    "device": "cuda:2",
}
EXPERIMENT_FALCON_BB_CONSTRAINTS = {
    "name": "Falcon3-7B / belief_bank constraints",
    "project_dir": DEFAULT_PROJECT_DIR,
    "llm_name": "tiiuae/Falcon3-7B-Base",
    "data_name": "belief_bank",
    "belief_bank_data_type": "constraints",
    "use_local": False,
    "dtype": "bfloat16",
    "use_device_map": True,
    "use_flash_attn": False,
    "max_samples": None,
    "quantization": True,
    "device": "cuda:2",
}
EXPERIMENT_FALCON_HALU = {
    "name": "Falcon3-7B / halu_eval",
    "project_dir": DEFAULT_PROJECT_DIR,
    "llm_name": "tiiuae/Falcon3-7B-Base",
    "data_name": "halu_eval",
    "belief_bank_data_type": "facts",
    "use_local": False,
    "dtype": "bfloat16",
    "use_device_map": True,
    "use_flash_attn": False,
    "max_samples": None,
    "quantization": True,
    "device": "cuda:2",
}
EXPERIMENT_QWEN_BB_CONSTRAINTS = {
    "name": "Qwen2.5-7B / belief_bank constraints",
    "project_dir": DEFAULT_PROJECT_DIR,
    "llm_name": "Qwen/Qwen2.5-7B",
    "data_name": "belief_bank",
    "belief_bank_data_type": "constraints",
    "use_local": False,
    "dtype": "bfloat16",
    "use_device_map": True,
    "use_flash_attn": False,
    "max_samples": None,
    "quantization": True,
    "device": "cuda:2",
}
EXPERIMENT_QWEN_BB_FACTS = {
    "name": "Qwen2.5-7B / belief_bank facts",
    "project_dir": DEFAULT_PROJECT_DIR,
    "llm_name": "Qwen/Qwen2.5-7B",
    "data_name": "belief_bank",
    "belief_bank_data_type": "facts",
    "use_local": False,
    "dtype": "bfloat16",
    "use_device_map": True,
    "use_flash_attn": False,
    "max_samples": None,
    "quantization": True,
    "device": "cuda:2",
}
EXPERIMENT_QWEN_HALU = {
    "name": "Qwen2.5-7B / halu_eval",
    "project_dir": DEFAULT_PROJECT_DIR,
    "llm_name": "Qwen/Qwen2.5-7B",
    "data_name": "halu_eval",
    "belief_bank_data_type": "facts",
    "use_local": False,
    "dtype": "bfloat16",
    "use_device_map": True,
    "use_flash_attn": False,
    "max_samples": None,
    "quantization": True,
    "device": "cuda:2",
}
EXPERIMENT_GEMMA_BB_FACTS = {
    "name": "gemma-2-9b-it / belief_bank facts",
    "project_dir": DEFAULT_PROJECT_DIR,
    "llm_name": "google/gemma-2-9b-it",
    "data_name": "belief_bank",
    "belief_bank_data_type": "facts",
    "use_local": False,
    "dtype": "bfloat16",
    "use_device_map": True,
    "use_flash_attn": False,
    "max_samples": None,
    "quantization": True,
    "device": "cuda:2",
}
EXPERIMENT_GEMMA_BB_CONSTRAINTS = {
    "name": "gemma-2-9b-it / belief_bank constraints",
    "project_dir": DEFAULT_PROJECT_DIR,
    "llm_name": "google/gemma-2-9b-it",
    "data_name": "belief_bank",
    "belief_bank_data_type": "constraints",
    "use_local": False,
    "dtype": "bfloat16",
    "use_device_map": True,
    "use_flash_attn": False,
    "max_samples": None,
    "quantization": True,
    "device": "cuda:2",
}
EXPERIMENT_GEMMA_HALU = {
    "name": "gemma-2-9b-it / halu_eval",
    "project_dir": DEFAULT_PROJECT_DIR,
    "llm_name": "google/gemma-2-9b-it",
    "data_name": "halu_eval",
    "belief_bank_data_type": "facts",
    "use_local": False,
    "dtype": "bfloat16",
    "use_device_map": True,
    "use_flash_attn": False,
    "max_samples": None,
    "quantization": True,
    "device": "cuda:2",
}
EXPERIMENT_LLAMA_BB_FACTS = {
    "name": "Llama-3.1-8B-Instruct / belief_bank facts",
    "project_dir": DEFAULT_PROJECT_DIR,
    "llm_name": "meta-llama/Llama-3.1-8B-Instruct",
    "data_name": "belief_bank",
    "belief_bank_data_type": "facts",
    "use_local": False,
    "dtype": "bfloat16",
    "use_device_map": True,
    "use_flash_attn": False,
    "max_samples": None,
    "quantization": True,
    "device": "cuda:2",
}
EXPERIMENT_LLAMA_BB_CONSTRAINTS = {
    "name": "Llama-3.1-8B-Instruct / belief_bank constraints",
    "project_dir": DEFAULT_PROJECT_DIR,
    "llm_name": "meta-llama/Llama-3.1-8B-Instruct",
    "data_name": "belief_bank",
    "belief_bank_data_type": "constraints",
    "use_local": False,
    "dtype": "bfloat16",
    "use_device_map": True,
    "use_flash_attn": False,
    "max_samples": None,
    "quantization": True,
    "device": "cuda:2",
}
EXPERIMENT_LLAMA_HALU = {
    "name": "Llama-3.1-8B-Instruct / halu_eval",
    "project_dir": DEFAULT_PROJECT_DIR,
    "llm_name": "meta-llama/Llama-3.1-8B-Instruct",
    "data_name": "halu_eval",
    "belief_bank_data_type": "facts",
    "use_local": False,
    "dtype": "bfloat16",
    "use_device_map": True,
    "use_flash_attn": False,
    "max_samples": None,
    "quantization": True,
    "device": "cuda:2",
}


EXPERIMENTS = {
    "qwen_bb_facts": EXPERIMENT_QWEN_BB_FACTS,
    "qwen_bb_constraints": EXPERIMENT_QWEN_BB_CONSTRAINTS,
    "qwen_halu": EXPERIMENT_QWEN_HALU,
    "falcon_bb_facts": EXPERIMENT_FALCON_BB_FACTS,
    "falcon_bb_constraints": EXPERIMENT_FALCON_BB_CONSTRAINTS,
    "falcon_halu": EXPERIMENT_FALCON_HALU,
    "gemma_bb_facts": EXPERIMENT_GEMMA_BB_FACTS,
    "gemma_bb_constraints": EXPERIMENT_GEMMA_BB_CONSTRAINTS,
    "gemma_halu": EXPERIMENT_GEMMA_HALU,
    "llama_bb_facts": EXPERIMENT_LLAMA_BB_FACTS,
    "llama_bb_constraints": EXPERIMENT_LLAMA_BB_CONSTRAINTS,
    "llama_halu": EXPERIMENT_LLAMA_HALU,
}


def run_experiment(cfg: dict) -> None:
    dtype = resolve_dtype(cfg["dtype"])
    device = cfg["device"]
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device))

    print("\n" + "=" * 80)
    print(cfg["name"])
    print("=" * 80 + "\n")

    detector = HallucinationDetection(project_dir=cfg["project_dir"])
    detector.save_model_activations(
        llm_name=cfg["llm_name"],
        data_name=cfg["data_name"],
        use_local=cfg["use_local"],
        dtype=dtype,
        use_device_map=cfg["use_device_map"],
        use_flash_attn=cfg["use_flash_attn"],
        max_samples=cfg["max_samples"],
        quantization=cfg["quantization"],
        belief_bank_data_type=cfg["belief_bank_data_type"],
        device=device,
    )


def get_experiment_cfg(experiment_id: str, max_samples: int = None, device: str = None) -> dict:
    if experiment_id not in EXPERIMENTS:
        available = ", ".join(EXPERIMENTS.keys())
        raise ValueError(f"Unknown experiment '{experiment_id}'. Available: {available}")

    cfg = dict(EXPERIMENTS[experiment_id])
    if max_samples is not None:
        cfg["max_samples"] = max_samples
    if device is not None:
        cfg["device"] = device
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run activation-saving experiments sequentially or one-by-one (HTC-friendly)."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        help="Experiment id to run, or 'all' (default). Use --list to show ids.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiment ids and exit.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of samples per experiment.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override (example: cuda:0, cuda:1, cpu).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("Available experiments:")
        for exp_id, cfg in EXPERIMENTS.items():
            print(f"- {exp_id}: {cfg['name']}")
        return

    if args.experiment == "all":
        for exp_id in EXPERIMENTS:
            cfg = get_experiment_cfg(exp_id, max_samples=args.max_samples, device=args.device)
            run_experiment(cfg)
        return

    cfg = get_experiment_cfg(args.experiment, max_samples=args.max_samples, device=args.device)
    run_experiment(cfg)



if __name__ == "__main__":
    main()
