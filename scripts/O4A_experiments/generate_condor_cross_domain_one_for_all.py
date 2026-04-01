"""
Generate HTCondor submit files for cross-domain OneForAll retraining jobs.

Each job is defined by:
- (trainer, tester) model pair
- encoder dataset (domain of training for encoder/head/adapter)
- activation dataset (domain used for evaluation activations), encoder_dataset != activation_dataset
- seed
"""

import itertools
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().with_name("config_cross_domain_one_for_all.yaml")


def _to_cli_value(value):
    """Convert Python value to CLI string."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def generate_experiment_name(trainer: str, tester: str, dataset: str) -> str:
    """
    Generate experiment name matching o4a.config.EXPERIMENTS naming.
    Example: LlamaToGemma_BBF
    """
    model_short = {
        "Qwen2.5-7B": "Qwen",
        "Falcon3-7B-Base": "Falcon",
        "gemma-2-9b-it": "Gemma",
        "Llama-3.1-8B-Instruct": "Llama",
    }
    dataset_short = {
        "belief_bank_constraints": "BBC",
        "belief_bank_facts": "BBF",
        "halu_eval": "HE",
    }
    trainer_short = model_short.get(trainer, trainer.split("-")[0])
    tester_short = model_short.get(tester, tester.split("-")[0])
    ds_short = dataset_short.get(dataset, dataset[:3].upper())
    return f"{trainer_short}To{tester_short}_{ds_short}"


def main() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    common = cfg["common"]
    htc = cfg["htc"]
    opt = cfg.get("optimization", {})

    output_dir = PROJECT_ROOT / htc["output_dir"]
    logs_dir = PROJECT_ROOT / htc["logs_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    llms = common["llms"]
    datasets = common["datasets"]
    seeds = common["seeds"]
    device = common.get("device", "cuda:0")
    layer_types = common.get("layer_types", []) or []

    submit_lines = []
    job_count = 0

    for trainer, tester in itertools.permutations(llms, 2):
        for encoder_ds, activation_ds in itertools.permutations(datasets, 2):
            for seed in seeds:
                encoder_exp = generate_experiment_name(trainer, tester, encoder_ds)

                args = [
                    encoder_exp,
                    activation_ds,
                    str(seed),
                    _to_cli_value(device),
                    str(opt.get("num_workers", 4)),
                    _to_cli_value(opt.get("pin_memory", True)),
                    str(opt.get("prefetch_factor", 2)),
                    _to_cli_value(opt.get("cudnn_benchmark", True)),
                    _to_cli_value(opt.get("use_amp", True)),
                    _to_cli_value(opt.get("compile_model", False)),
                ]
                if layer_types:
                    args.extend(layer_types)

                submit_lines.append(f'arguments = "{" ".join(args)}"\nqueue\n')
                job_count += 1

    print(f"Total jobs to generate: {job_count}")
    print(f"  - LLM pairs: {len(llms) * (len(llms) - 1)}")
    print(f"  - Domain pairs (encoder->activation): {len(datasets) * (len(datasets) - 1)}")
    print(f"  - Seeds: {len(seeds)}")
    if layer_types:
        print(f"  - Layer types override: {layer_types}")
    else:
        print("  - Layer types override: none (all)")

    max_jobs_per_file = int(htc.get("max_jobs_per_file", 100))
    executable = htc["executable"]

    htc_header_lines = [
        "universe = vanilla",
        f"executable = {executable}",
        f"request_cpus = {htc['request_cpus']}",
        f"request_gpus = {htc['request_gpus']}",
    ]

    if "request_memory" in htc:
        htc_header_lines.append(f"request_memory = {htc['request_memory']}")
    if "request_disk" in htc:
        htc_header_lines.append(f"request_disk = {htc['request_disk']}")
    if "initialdir" in htc:
        htc_header_lines.append(f"initialdir = {htc['initialdir']}")
    if "getenv" in htc:
        htc_header_lines.append(f"getenv = {htc['getenv']}")
    if "requirements" in htc:
        htc_header_lines.append(f"requirements = {htc['requirements']}")

    htc_header_lines.extend([
        f"log = {htc['logs_dir']}/job_$(Cluster)_$(Process).log",
        f"output = {htc['logs_dir']}/job_$(Cluster)_$(Process).out",
        f"error = {htc['logs_dir']}/job_$(Cluster)_$(Process).err",
        "",
    ])

    for idx in range(0, len(submit_lines), max_jobs_per_file):
        subset = submit_lines[idx : idx + max_jobs_per_file]
        file_num = idx // max_jobs_per_file + 1
        submit_path = output_dir / f"run_cross_domain_one_for_all_jobs_{file_num}.htc"

        with open(submit_path, "w", encoding="utf-8") as f:
            f.write("\n".join(htc_header_lines))
            f.writelines(subset)

        print(f"Created submit file: {submit_path} with {len(subset)} jobs")

    submit_all_path = output_dir / "submit_all.sh"
    with open(submit_all_path, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all generated HTCondor job files\n\n")
        f.write('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n\n')
        num_files = (len(submit_lines) + max_jobs_per_file - 1) // max_jobs_per_file
        for i in range(1, num_files + 1):
            f.write(f'condor_submit "$SCRIPT_DIR/run_cross_domain_one_for_all_jobs_{i}.htc"\n')

    print(f"\nCreated submit helper: {submit_all_path}")
    print(f"Run 'bash {submit_all_path.relative_to(PROJECT_ROOT)}' to submit all jobs")


if __name__ == "__main__":
    main()
