"""Generate HTCondor submit files for layer-wise study jobs."""

from __future__ import annotations

import itertools
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().with_name("config_run_layers_study.yaml")


def _to_cli_value(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _split_seeds_to_csv(split_seeds: list[int]) -> str:
    return ",".join(str(seed) for seed in split_seeds)


def main() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    common = cfg["common"]
    runtime = cfg.get("runtime", {})
    htc = cfg["htc"]

    llms = common["llms"]
    datasets = common["datasets"]
    layer_types = common.get("layer_types", ["attn", "mlp", "hidden"]) or ["attn", "mlp", "hidden"]
    split_seeds = common["split_seeds"]
    test_size = common.get("test_size", 0.3)

    device = runtime.get("device", "cpu")
    max_iter = runtime.get("max_iter", 10000)
    logreg_n_jobs = runtime.get("logreg_n_jobs", -1)
    output_base_dir = runtime.get("output_base_dir", "results/layers_study")

    split_seeds_csv = _split_seeds_to_csv(split_seeds)

    output_dir = PROJECT_ROOT / htc["output_dir"]
    logs_dir = PROJECT_ROOT / htc["logs_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    submit_lines: list[str] = []
    job_count = 0

    for model_name, dataset_name, layer_type in itertools.product(llms, datasets, layer_types):
        args = [
            model_name,
            dataset_name,
            layer_type,
            split_seeds_csv,
            _to_cli_value(test_size),
            _to_cli_value(device),
            _to_cli_value(max_iter),
            _to_cli_value(logreg_n_jobs),
            _to_cli_value(output_base_dir),
        ]
        submit_lines.append(f'arguments = "{" ".join(args)}"\nqueue\n')
        job_count += 1

    print(f"Total jobs to generate: {job_count}")
    print(f"  - LLMs: {len(llms)}")
    print(f"  - Datasets: {len(datasets)}")
    print(f"  - Layer types: {len(layer_types)} ({', '.join(layer_types)})")
    print(f"  - Split seeds per job: {len(split_seeds)} ({split_seeds_csv})")

    max_jobs_per_file = int(htc.get("max_jobs_per_file", 500))
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

    htc_header_lines.extend(
        [
            f"log = {htc['logs_dir']}/job_$(Cluster)_$(Process).log",
            f"output = {htc['logs_dir']}/job_$(Cluster)_$(Process).out",
            f"error = {htc['logs_dir']}/job_$(Cluster)_$(Process).err",
            "",
        ]
    )

    for old_file in output_dir.glob("run_layers_study_jobs_*.htc"):
        old_file.unlink()

    for idx in range(0, len(submit_lines), max_jobs_per_file):
        subset = submit_lines[idx : idx + max_jobs_per_file]
        file_num = idx // max_jobs_per_file + 1
        submit_path = output_dir / f"run_layers_study_jobs_{file_num}.htc"

        with open(submit_path, "w", encoding="utf-8") as file:
            file.write("\n".join(htc_header_lines))
            file.writelines(subset)

        print(f"Created submit file: {submit_path} with {len(subset)} jobs")

    submit_all_path = output_dir / "submit_all.sh"
    with open(submit_all_path, "w", encoding="utf-8") as file:
        file.write("#!/bin/bash\n")
        file.write("# Submit all generated HTCondor layer study files\n\n")
        file.write('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n\n')
        num_files = (len(submit_lines) + max_jobs_per_file - 1) // max_jobs_per_file
        for i in range(1, num_files + 1):
            file.write(f'condor_submit "$SCRIPT_DIR/run_layers_study_jobs_{i}.htc"\n')

    print(f"\nCreated submit helper: {submit_all_path}")
    print(f"Run 'bash {submit_all_path.relative_to(PROJECT_ROOT)}' to submit all jobs")


if __name__ == "__main__":
    main()
