import sys
from pathlib import Path

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


def main():
    run_experiment(EXPERIMENT_QWEN_BB_FACTS)
    run_experiment(EXPERIMENT_QWEN_BB_CONSTRAINTS)
    run_experiment(EXPERIMENT_QWEN_HALU)
    run_experiment(EXPERIMENT_FALCON_BB_FACTS)
    run_experiment(EXPERIMENT_FALCON_BB_CONSTRAINTS)
    run_experiment(EXPERIMENT_FALCON_HALU)
    run_experiment(EXPERIMENT_GEMMA_BB_FACTS)
    run_experiment(EXPERIMENT_GEMMA_BB_CONSTRAINTS)
    run_experiment(EXPERIMENT_GEMMA_HALU)
    run_experiment(EXPERIMENT_LLAMA_BB_FACTS)
    run_experiment(EXPERIMENT_LLAMA_BB_CONSTRAINTS)
    run_experiment(EXPERIMENT_LLAMA_HALU)



if __name__ == "__main__":
    main()
