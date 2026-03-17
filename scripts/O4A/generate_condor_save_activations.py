import itertools
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().with_name("config_save_activations.yaml")


def _to_cli_value(value):
    return "None" if value is None else str(value)


def main() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    common = cfg["common"]
    htc = cfg["htc"]

    output_dir = PROJECT_ROOT / htc["output_dir"]
    logs_dir = PROJECT_ROOT / htc["logs_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    common_keys = list(common.keys())
    common_values = [common[k] for k in common_keys]

    submit_lines = []
    for combo in itertools.product(*common_values):
        params = dict(zip(common_keys, combo))
        args = [
            params["experiments"],
            _to_cli_value(params["max_samples"]),
            _to_cli_value(params["device"]),
        ]
        submit_lines.append(f'arguments = "{" ".join(args)}"\nqueue\n')

    max_jobs_per_file = int(htc.get("max_jobs_per_file", 500))
    executable = htc["executable"]

    for idx in range(0, len(submit_lines), max_jobs_per_file):
        subset = submit_lines[idx : idx + max_jobs_per_file]
        file_num = idx // max_jobs_per_file + 1
        submit_path = output_dir / f"save_activations_jobs_{file_num}.htc"

        with open(submit_path, "w", encoding="utf-8") as f:
            f.write(
                "\n".join(
                    [
                        "universe = vanilla",
                        f"executable = {executable}",
                        f"request_cpus = {htc['request_cpus']}",
                        f"request_gpus = {htc['request_gpus']}",
                        f"log = {htc['logs_dir']}/job_$(Cluster)_$(Process).log",
                        f"output = {htc['logs_dir']}/job_$(Cluster)_$(Process).out",
                        f"error = {htc['logs_dir']}/job_$(Cluster)_$(Process).err",
                        "",
                    ]
                )
            )
            f.writelines(subset)

        print(f"Created submit file: {submit_path} with {len(subset)} jobs")


if __name__ == "__main__":
    main()
