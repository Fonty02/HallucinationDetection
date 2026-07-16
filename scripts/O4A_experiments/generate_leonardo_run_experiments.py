"""
Generate Leonardo (SLURM) submit files for run_experiments.py jobs.

Each job is defined by: (trainer, tester, dataset, seed, layer_type)
Generates all valid combinations (trainer != tester).

Mirrors the job/parameter generation logic of generate_condor_run_experiments.py,
but emits SLURM sbatch scripts instead of HTCondor submit files.
"""

import itertools
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().with_name("config_run_experiments.yaml")


def _to_cli_value(value):
    """Convert Python value to CLI string."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def generate_experiment_name(trainer: str, tester: str, dataset: str) -> str:
    """
    Generate experiment name matching the pattern in config.py.
    E.g., QwenToFalcon_BBC, LlamaToGemma_HE, etc.
    """
    # Short names for models
    model_short = {
        "Qwen2.5-7B": "Qwen",
        "Falcon3-7B-Base": "Falcon",
        "gemma-2-9b-it": "Gemma",
        "Llama-3.1-8B-Instruct": "Llama",
    }
    # Short names for datasets
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
    leonardo = cfg["leonardo"]
    opt = cfg.get("optimization", {})

    output_dir = PROJECT_ROOT / leonardo["output_dir"]
    logs_dir = PROJECT_ROOT / leonardo["logs_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    llms = common["llms"]
    datasets = common["datasets"]
    seeds = common["seeds"]
    layer_types = common.get("layer_types", ["attn", "mlp", "hidden"])
    if not layer_types:
        layer_types = ["attn", "mlp", "hidden"]
    device = common.get("device", "cuda:0")

    # Generate all combinations where trainer != tester
    submit_lines = []
    job_count = 0

    for trainer, tester in itertools.permutations(llms, 2):
        for dataset in datasets:
            for seed in seeds:
                for layer_type in layer_types:
                    exp_name = generate_experiment_name(trainer, tester, dataset)

                    # Same positional arguments run_experiments.sh expects
                    # (experiment_name seed device [optimization flags] layer_type)
                    args = [
                        exp_name,
                        str(seed),
                        _to_cli_value(device),
                        # Optimization flags
                        str(opt.get("num_workers", 4)),
                        _to_cli_value(opt.get("pin_memory", True)),
                        str(opt.get("prefetch_factor", 2)),
                        _to_cli_value(opt.get("cudnn_benchmark", True)),
                        _to_cli_value(opt.get("use_amp", True)),
                        _to_cli_value(opt.get("compile_model", False)),
                        layer_type,
                    ]
                    executable = leonardo["executable"]
                    submit_lines.append(f"srun -u bash {executable} {' '.join(args)}\n")
                    job_count += 1

    print(f"Total jobs to generate: {job_count}")
    print(f"  - LLMs: {len(llms)} -> {len(llms) * (len(llms) - 1)} trainer/tester pairs")
    print(f"  - Datasets: {len(datasets)}")
    print(f"  - Seeds: {len(seeds)}")
    print(f"  - Layer types: {len(layer_types)} ({', '.join(layer_types)})")

    max_jobs_per_file = int(leonardo.get("max_jobs_per_file", 24))
    job_name = leonardo.get("job_name", "o4a_run_exp")

    # Remove stale files from previous generations.
    for old_file in output_dir.glob("run_experiments_jobs_*.sh"):
        old_file.unlink()

    # Split into multiple sbatch files if needed
    num_files = (len(submit_lines) + max_jobs_per_file - 1) // max_jobs_per_file
    for idx in range(0, len(submit_lines), max_jobs_per_file):
        subset = submit_lines[idx : idx + max_jobs_per_file]
        file_num = idx // max_jobs_per_file + 1
        submit_path = output_dir / f"run_experiments_jobs_{file_num}.sh"

        with open(submit_path, "w", encoding="utf-8") as f:
            f.write(
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
#SBATCH --output={leonardo['logs_dir']}/run_experiments_{file_num}_%j.out
#SBATCH --error={leonardo['logs_dir']}/run_experiments_{file_num}_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user={leonardo['mail_user']}


source .venv/bin/activate

"""
            )
            # Write jobs
            f.writelines(subset)

        print(f"Created submit file: {submit_path} with {len(subset)} jobs")

    # Generate a convenience script to submit all files
    submit_all_path = output_dir / "submit_all.sh"
    with open(submit_all_path, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all generated SLURM job files\n\n")
        f.write('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n\n')
        for i in range(1, num_files + 1):
            f.write(f'sbatch "$SCRIPT_DIR/run_experiments_jobs_{i}.sh"\n')

    print(f"\nCreated submit helper: {submit_all_path}")
    print(f"Run 'bash {submit_all_path.relative_to(PROJECT_ROOT)}' to submit all jobs")


if __name__ == "__main__":
    main()
