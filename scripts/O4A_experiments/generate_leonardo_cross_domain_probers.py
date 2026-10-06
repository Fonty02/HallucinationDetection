"""
Generate Leonardo (SLURM) submit files for single-LLM cross-domain prober jobs.

Same jobs as generate_condor_cross_domain_probers.py (model, train dataset,
activation dataset, seed), emitted as sbatch scripts instead of HTCondor submit files.
"""

from pathlib import Path

from generate_condor_cross_domain_probers import build_job_args, load_config, print_summary


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    cfg = load_config()
    leonardo = cfg["leonardo"]

    output_dir = PROJECT_ROOT / leonardo["output_dir"]
    logs_dir = PROJECT_ROOT / leonardo["logs_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    executable = leonardo["executable"]
    submit_lines = [f"srun -u bash {executable} {' '.join(args)}\n" for args in build_job_args(cfg)]
    print_summary(cfg, len(submit_lines))

    max_jobs_per_file = int(leonardo.get("max_jobs_per_file", 10))
    job_name = leonardo.get("job_name", "o4a_cd_probers")

    # Remove stale files from previous generations.
    for old_file in output_dir.glob("run_cross_domain_jobs_*.sh"):
        old_file.unlink()

    num_files = (len(submit_lines) + max_jobs_per_file - 1) // max_jobs_per_file
    for idx in range(0, len(submit_lines), max_jobs_per_file):
        subset = submit_lines[idx : idx + max_jobs_per_file]
        file_num = idx // max_jobs_per_file + 1
        submit_path = output_dir / f"run_cross_domain_jobs_{file_num}.sh"

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
#SBATCH --job-name={file_num}_{job_name}
#SBATCH --output={leonardo['logs_dir']}/run_cross_domain_probers_{file_num}_%j.out
#SBATCH --error={leonardo['logs_dir']}/run_cross_domain_probers_{file_num}_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user={leonardo['mail_user']}


source .venv/bin/activate

"""
            )
            f.writelines(subset)

        print(f"Created submit file: {submit_path} with {len(subset)} jobs")

    submit_all_path = output_dir / "submit_all.sh"
    with open(submit_all_path, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all generated SLURM job files\n\n")
        f.write('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n\n')
        for i in range(1, num_files + 1):
            f.write(f'sbatch "$SCRIPT_DIR/run_cross_domain_jobs_{i}.sh"\n')

    print(f"\nCreated submit helper: {submit_all_path}")
    print(f"Run 'bash {submit_all_path.relative_to(PROJECT_ROOT)}' to submit all jobs")


if __name__ == "__main__":
    main()
