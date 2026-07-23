"""
Generate Leonardo (SLURM) submit files for layer-wise study jobs.

Each job is defined by: (model, dataset, layer_type), internally evaluated
on all split seeds.

Mirrors the job/parameter generation logic of generate_condor_run_layers_study.py,
but emits SLURM sbatch scripts instead of HTCondor submit files.
"""

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
    leonardo = cfg["leonardo"]

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

    output_dir = PROJECT_ROOT / leonardo["output_dir"]
    logs_dir = PROJECT_ROOT / leonardo["logs_dir"]
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
        executable = leonardo["executable"]
        submit_lines.append(f"srun -u bash {executable} {' '.join(args)}\n")
        job_count += 1

    print(f"Total jobs to generate: {job_count}")
    print(f"  - LLMs: {len(llms)}")
    print(f"  - Datasets: {len(datasets)}")
    print(f"  - Layer types: {len(layer_types)} ({', '.join(layer_types)})")
    print(f"  - Split seeds per job: {len(split_seeds)} ({split_seeds_csv})")

    max_jobs_per_file = 2 #int(leonardo.get("max_jobs_per_file", 24))
    job_name = leonardo.get("job_name", "o4a_layers_study")

    # Remove stale files from previous generations.
    for old_file in output_dir.glob("run_layers_study_jobs_*.sh"):
        old_file.unlink()

    # Split into multiple sbatch files if needed
    num_files = (len(submit_lines) + max_jobs_per_file - 1) // max_jobs_per_file
    for idx in range(0, len(submit_lines), max_jobs_per_file):
        subset = submit_lines[idx : idx + max_jobs_per_file]
        file_num = idx // max_jobs_per_file + 1
        submit_path = output_dir / f"run_layers_study_jobs_{file_num}.sh"

        with open(submit_path, "w", encoding="utf-8") as file:
            file.write(
                f"""#!/bin/bash

#SBATCH -A {leonardo['account']}
#SBATCH -p {leonardo['partition']}
#SBATCH --qos {leonardo['qos']}
#SBATCH --time={leonardo['time']}
#SBATCH -N {leonardo['nodes']}
#SBATCH --ntasks={leonardo['ntasks']}
#SBATCH --cpus-per-task={leonardo['cpus_per_task']}
#SBATCH --gpus-per-task={leonardo['gpus_per_task']}
#SBATCH --mem={leonardo['mem']}
#SBATCH --job-name={job_name}_{file_num}
#SBATCH --output={leonardo['logs_dir']}/run_layers_study_{file_num}_%j.out
#SBATCH --error={leonardo['logs_dir']}/run_layers_study_{file_num}_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user={leonardo['mail_user']}


source .venv/bin/activate

"""
            )
            # Write jobs
            file.writelines(subset)

        print(f"Created submit file: {submit_path} with {len(subset)} jobs")

    # Generate a convenience script to submit all files
    submit_all_path = output_dir / "submit_all.sh"
    with open(submit_all_path, "w", encoding="utf-8") as file:
        file.write("#!/bin/bash\n")
        file.write("# Submit all generated SLURM layer study files\n\n")
        file.write('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n\n')
        for i in range(1, num_files + 1):
            file.write(f'sbatch "$SCRIPT_DIR/run_layers_study_jobs_{i}.sh"\n')

    print(f"\nCreated submit helper: {submit_all_path}")
    print(f"Run 'bash {submit_all_path.relative_to(PROJECT_ROOT)}' to submit all jobs")


if __name__ == "__main__":
    main()
