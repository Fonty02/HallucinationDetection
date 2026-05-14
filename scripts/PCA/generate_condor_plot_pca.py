import ast
import itertools
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().with_name("config_plot_pca.yaml")


def _to_cli_value(value):
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _as_list(value):
    if isinstance(value, list):
        return value
    return [value]


def _sanitize(text: str) -> str:
    return (
        text.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "")
        .replace(",", "_")
        .replace(":", "-")
    )


def _load_experiments_from_python(config_path: Path) -> dict:
    source = config_path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(config_path))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "EXPERIMENTS":
                    return ast.literal_eval(node.value)
    raise ValueError(f"EXPERIMENTS dictionary not found in {config_path}")


def _build_first_layer_lookup(experiments: dict) -> dict:
    """
    Build lookup:
      (model, dataset, layer_type) -> first layer index (int)
    """
    lookup: dict[tuple[str, str, str], int] = {}
    for exp in experiments.values():
        dataset = exp["dataset"]
        for role in ("trainer", "tester"):
            model = exp[role]
            layers_by_type = exp[f"{role}_layers"]
            for layer_type, layers in layers_by_type.items():
                if not layers:
                    continue
                key = (str(model), str(dataset), str(layer_type))
                first_layer = int(layers[0])
                if key in lookup and lookup[key] != first_layer:
                    raise ValueError(
                        f"Inconsistent first layer for {key}: "
                        f"{lookup[key]} vs {first_layer}"
                    )
                lookup[key] = first_layer
    return lookup


def _build_dataset_layers_spec(
    model: str,
    datasets: list[str],
    layer_type: str,
    first_layer_lookup: dict,
) -> str:
    pairs = []
    missing = []
    for dataset in datasets:
        key = (model, dataset, layer_type)
        layer = first_layer_lookup.get(key)
        if layer is None:
            missing.append(key)
            continue
        pairs.append(f"{dataset}:{layer}")

    if missing:
        missing_s = ", ".join([f"({m},{d},{lt})" for m, d, lt in missing])
        raise ValueError(f"Missing layer mapping for: {missing_s}")

    return ",".join(pairs)


def main() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    common = cfg["common"]
    htc = cfg["htc"]

    output_dir = PROJECT_ROOT / htc["output_dir"]
    logs_dir = PROJECT_ROOT / htc["logs_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    llms = _as_list(common["llms"])
    datasets = _as_list(common.get("datasets", ["belief_bank_constraints", "belief_bank_facts", "halu_eval"]))
    layer_types = _as_list(common.get("layer_types", ["attn", "mlp", "hidden"]))
    max_points_per_class = _as_list(common.get("max_points_per_class", [1500]))
    balance_flags = _as_list(common.get("balance", [True]))
    standardize_flags = _as_list(common.get("standardize", [False]))
    output_roots = _as_list(common.get("output_roots", [Path("results") / "pca_multidataset"]))
    seeds = _as_list(common.get("seeds", [42]))
    cache_dirs = _as_list(common.get("cache_dirs", ["activation_cache"]))
    dataset_layers_specs_override = _as_list(common.get("dataset_layers_specs", []))

    layer_source_cfg = common.get("layer_source_config", "src/o4a/config.py")
    layer_source_cfg_path = (PROJECT_ROOT / layer_source_cfg).resolve()
    experiments = _load_experiments_from_python(layer_source_cfg_path)
    first_layer_lookup = _build_first_layer_lookup(experiments)

    submit_lines = []
    for (
        llm,
        layer_type,
        max_points,
        balance,
        standardize,
        output_root,
        seed,
        cache_dir,
    ) in itertools.product(
        llms,
        layer_types,
        max_points_per_class,
        balance_flags,
        standardize_flags,
        output_roots,
        seeds,
        cache_dirs,
    ):
        if dataset_layers_specs_override:
            # Backward-compatible mode: one job for each explicit spec.
            dataset_layers_specs = dataset_layers_specs_override
        else:
            # Auto mode: build one spec from first layer in O4A config for each dataset.
            dataset_layers_specs = [
                _build_dataset_layers_spec(
                    model=str(llm),
                    datasets=[str(d) for d in datasets],
                    layer_type=str(layer_type),
                    first_layer_lookup=first_layer_lookup,
                )
            ]

        for dataset_layers_spec in dataset_layers_specs:
            spec_tag = _sanitize(dataset_layers_spec)
            figure_name = f"pca_{_sanitize(llm)}_{layer_type}_{spec_tag}_seed{seed}.png"
            args = [
                str(llm),
                str(dataset_layers_spec),
                str(layer_type),
                _to_cli_value(max_points),
                _to_cli_value(balance),
                _to_cli_value(standardize),
                str(output_root),
                figure_name,
                str(seed),
                str(cache_dir),
            ]
            submit_lines.append(f'arguments = "{" ".join(args)}"\nqueue\n')

    max_jobs_per_file = int(htc.get("max_jobs_per_file", 500))
    executable = htc["executable"]

    header_lines = [
        "universe = vanilla",
        f"executable = {executable}",
        f"request_cpus = {htc['request_cpus']}",
        f"request_gpus = {htc['request_gpus']}",
    ]
    if "request_memory" in htc:
        header_lines.append(f"request_memory = {htc['request_memory']}")
    if "request_disk" in htc:
        header_lines.append(f"request_disk = {htc['request_disk']}")
    if "initialdir" in htc:
        header_lines.append(f"initialdir = {htc['initialdir']}")
    if "getenv" in htc:
        header_lines.append(f"getenv = {htc['getenv']}")
    if "requirements" in htc:
        header_lines.append(f"requirements = {htc['requirements']}")
    header_lines.extend(
        [
            f"log = {htc['logs_dir']}/job_$(Cluster)_$(Process).log",
            f"output = {htc['logs_dir']}/job_$(Cluster)_$(Process).out",
            f"error = {htc['logs_dir']}/job_$(Cluster)_$(Process).err",
            "",
        ]
    )

    for old_file in output_dir.glob("plot_pca_jobs_*.htc"):
        old_file.unlink()

    for idx in range(0, len(submit_lines), max_jobs_per_file):
        subset = submit_lines[idx : idx + max_jobs_per_file]
        file_num = idx // max_jobs_per_file + 1
        submit_path = output_dir / f"plot_pca_jobs_{file_num}.htc"
        with open(submit_path, "w", encoding="utf-8") as f:
            f.write("\n".join(header_lines))
            f.writelines(subset)
        print(f"Created submit file: {submit_path} with {len(subset)} jobs")

    submit_all_path = output_dir / "submit_all.sh"
    with open(submit_all_path, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all generated HTCondor job files\n\n")
        f.write('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n\n')
        num_files = (len(submit_lines) + max_jobs_per_file - 1) // max_jobs_per_file
        for i in range(1, num_files + 1):
            f.write(f'condor_submit "$SCRIPT_DIR/plot_pca_jobs_{i}.htc"\n')
    print(f"Created submit helper: {submit_all_path}")
    print(f"Datasets: {datasets}")
    print(f"Layer types: {layer_types} (first layer from {layer_source_cfg})")
    print(f"Total jobs: {len(submit_lines)}")


if __name__ == "__main__":
    main()
