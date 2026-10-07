"""
Generate HTCondor submit files for cross-domain prober jobs.

Each single-LLM job is defined by:
- model (single LLM, no trainer/tester pair)
- train dataset (domain the probers are trained on)
- activation dataset (domain used for evaluation), train_dataset != activation_dataset
- seed

Each cross-LLM job (cross-LLM and cross-domain at the same time, all probers) is defined by:
- trainer -> tester model pair
- train dataset, activation dataset (train_dataset != activation_dataset)
- seed
"""

import itertools
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().with_name("config_cross_domain_probers.yaml")


def _to_cli_value(value):
    """Convert Python value to CLI string."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _common_cli_args(cfg: dict) -> list[str]:
    """Runtime/optimization positional arguments shared by every job."""
    opt = cfg.get("optimization", {})
    return [
        _to_cli_value(cfg["common"].get("device", "cuda:0")),
        str(opt.get("num_workers", 4)),
        _to_cli_value(opt.get("pin_memory", True)),
        str(opt.get("prefetch_factor", 2)),
        _to_cli_value(opt.get("cudnn_benchmark", True)),
        _to_cli_value(opt.get("use_amp", True)),
        _to_cli_value(opt.get("compile_model", False)),
    ]


def _model_pairs(cfg: dict) -> list[tuple[str, str]]:
    """(model, tester_model) pairs: tester 'none' for single-LLM, ordered pairs for cross-LLM."""
    common = cfg["common"]
    pairs = []
    if common.get("single_llm", True):
        pairs.extend((model, "none") for model in common["llms"])
    if common.get("cross_llm", False):
        pairs.extend(itertools.permutations(common["llms"], 2))
    return pairs


def build_job_args(cfg: dict) -> list[list[str]]:
    """Positional arguments for run_cross_domain_probers.sh, one list per job."""
    common = cfg["common"]
    probers = common.get("probers", []) or []
    layer_types = common.get("layer_types", []) or []
    runtime_args = _common_cli_args(cfg)
    probers_arg = ",".join(probers) if probers else "all"

    jobs = []
    for model, tester in _model_pairs(cfg):
        for train_ds, activation_ds in itertools.permutations(common["datasets"], 2):
            for seed in common["seeds"]:
                args = [model, tester, train_ds, activation_ds, str(seed), *runtime_args, probers_arg]
                args.extend(layer_types)
                jobs.append(args)
    return jobs


def print_summary(cfg: dict, job_count: int) -> None:
    common = cfg["common"]
    datasets = common["datasets"]
    llms = common["llms"]
    print(f"Total jobs to generate: {job_count}")
    print(f"  - Single-LLM jobs: {'on' if common.get('single_llm', True) else 'off'} ({len(llms)} LLMs)")
    print(f"  - Cross-LLM jobs:  {'on' if common.get('cross_llm', False) else 'off'} "
          f"({len(llms) * (len(llms) - 1)} LLM pairs)")
    print(f"  - Probers: {common.get('probers') or 'all'}")
    print(f"  - Domain pairs (train->activation): {len(datasets) * (len(datasets) - 1)}")
    print(f"  - Seeds: {len(common['seeds'])}")
    print(f"  - Layer types: {common.get('layer_types') or 'all'}")


def main() -> None:
    cfg = load_config()
    htc = cfg["htc"]

    output_dir = PROJECT_ROOT / htc["output_dir"]
    logs_dir = PROJECT_ROOT / htc["logs_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    submit_lines = [f'arguments = "{" ".join(args)}"\nqueue\n' for args in build_job_args(cfg)]
    print_summary(cfg, len(submit_lines))

    max_jobs_per_file = int(htc.get("max_jobs_per_file", 500))

    htc_header_lines = [
        "universe = vanilla",
        f"executable = {htc['executable']}",
        f"request_cpus = {htc['request_cpus']}",
        f"request_gpus = {htc['request_gpus']}",
    ]
    for key in ("request_memory", "request_disk", "initialdir", "getenv", "requirements"):
        if key in htc:
            htc_header_lines.append(f"{key} = {htc[key]}")
    htc_header_lines.extend([
        f"log = {htc['logs_dir']}/job_$(Cluster)_$(Process).log",
        f"output = {htc['logs_dir']}/job_$(Cluster)_$(Process).out",
        f"error = {htc['logs_dir']}/job_$(Cluster)_$(Process).err",
        "",
    ])

    for old_file in output_dir.glob("run_cross_domain_probers_jobs_*.htc"):
        old_file.unlink()

    for idx in range(0, len(submit_lines), max_jobs_per_file):
        subset = submit_lines[idx : idx + max_jobs_per_file]
        file_num = idx // max_jobs_per_file + 1
        submit_path = output_dir / f"run_cross_domain_probers_jobs_{file_num}.htc"

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
            f.write(f'condor_submit "$SCRIPT_DIR/run_cross_domain_probers_jobs_{i}.htc"\n')

    print(f"\nCreated submit helper: {submit_all_path}")
    print(f"Run 'bash {submit_all_path.relative_to(PROJECT_ROOT)}' to submit all jobs")


if __name__ == "__main__":
    main()
