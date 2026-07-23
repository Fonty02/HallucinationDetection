"""Rebuild a (missing/incomplete) hallucination_labels.json from generation_*.json files.

Use this when a label-scoring run was interrupted or crashed, leaving
activation_cache/<model>/<dataset>/generations/hallucination_labels.json
with fewer entries than generation_*.json files in the same folder.

For "facts" and "halu_eval" datasets, gold answers are looked up by instance_id
against a freshly-built dataset (both are fully deterministic to reconstruct).

For "constraints", BeliefBankDataset grounds constraints against a *random*
train/val/test split of calibration facts (src/data/BeliefBankDataset.py,
Facts.get_splits), so re-running it never reproduces the same instance_id ->
fact mapping used originally. Instead we ground constraints against ALL known
calibration facts (no split, so no randomness) and match each generation by
the exact fact text embedded in its prompt, which is stable regardless of how
the constraints happen to be grounded/ordered.

Example:
    python scripts/rebuild_hallucination_labels.py \\
        --model-name Falcon3-7B-Base \\
        --dataset-name belief_bank_constraints \\
        --data-name belief_bank \\
        --belief-bank-data-type constraints
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.BeliefBankDataset import BeliefBankDataset, Constraints, Facts
from src.data.HaluEvalDataset import HaluEvalDataset

GENERATION_FILE_RE = re.compile(r"generation_(\d+)\.json$")
FACT_RE = re.compile(r"Fact:\s*(.+?)\s*\nAnswer:")


def build_constraints_fact_to_answer(project_dir: Path) -> dict[str, str]:
    """fact text -> gold answer, grounded against every known calibration fact.

    Unlike BeliefBankDataset(data_type="constraints"), this uses the complete
    calibration fact set instead of a randomly-sampled train split, so the
    result is the same on every run.
    """
    constraints_path = project_dir / "data" / "beliefbank" / "constraints_v2.json"
    calibration_facts_path = project_dir / "data" / "beliefbank" / "calibration_facts.json"

    constraints = Constraints(project_root=str(project_dir), constraints_path=str(constraints_path), model_type="demo")
    calibration_facts = Facts(project_root=str(project_dir), constraints=constraints, facts_path=str(calibration_facts_path), model_type="demo")
    grounded = constraints.get_grounded_constraints(facts=calibration_facts.get_whole_set(), path=str(constraints_path))

    fact_to_answer = {}
    for row in grounded:
        fact, belief = BeliefBankDataset.get_implication(row)
        fact_to_answer[fact] = "yes" if belief else "no"
        neg_fact, neg_belief = BeliefBankDataset.get_negated_implication(row)
        fact_to_answer[neg_fact] = "yes" if neg_belief else "no"
    return fact_to_answer


def build_id_to_answer(data_name: str, project_dir: Path, belief_bank_data_type: str, use_local: bool) -> dict[int, str]:
    """instance_id -> gold_answer, reconstructed the same way load_dataset builds it."""
    if data_name == "belief_bank":
        dataset = BeliefBankDataset(project_root=str(project_dir), data_type=belief_bank_data_type, recreate_ids=True)
    elif data_name == "halu_eval":
        dataset = HaluEvalDataset(use_local=use_local, recreate_ids=True)
    else:
        raise ValueError(f"Unsupported data_name: {data_name}")

    return {int(instance_id): answer for _, answer, instance_id in dataset}


def rebuild_labels(
    model_name: str,
    dataset_name: str,
    data_name: str,
    belief_bank_data_type: str,
    project_dir: Path,
    use_local: bool,
) -> Path:
    generations_dir = project_dir / "activation_cache" / model_name / dataset_name / "generations"
    if not generations_dir.is_dir():
        raise FileNotFoundError(f"Generations directory not found: {generations_dir}")

    print(f"[rebuild] scanning {generations_dir} ...")
    generation_files = sorted(
        generations_dir.glob("generation_*.json"),
        key=lambda p: int(GENERATION_FILE_RE.match(p.name).group(1)),
    )
    if not generation_files:
        raise RuntimeError(f"No generation_*.json files found in {generations_dir}")
    print(f"[rebuild] found {len(generation_files)} generation_*.json files")

    by_constraint_text = data_name == "belief_bank" and belief_bank_data_type == "constraints"
    if by_constraint_text:
        print("[rebuild] grounding constraints against ALL known calibration facts (deterministic, no split) ...")
        fact_to_answer = build_constraints_fact_to_answer(project_dir)
        id_to_answer = None
        print(f"[rebuild] built fact_to_answer lookup with {len(fact_to_answer)} distinct fact strings")
    else:
        print(f"[rebuild] reconstructing {data_name} dataset to look up gold answers by instance_id ...")
        fact_to_answer = None
        id_to_answer = build_id_to_answer(data_name, project_dir, belief_bank_data_type, use_local)
        print(f"[rebuild] built id_to_answer lookup with {len(id_to_answer)} instance ids")

    labels = []
    unmatched_ids = []
    total = len(generation_files)
    report_every = max(1, total // 20)
    print(f"[rebuild] matching {total} generation files against gold answers ...")
    for i, gen_path in enumerate(generation_files, start=1):
        if i % report_every == 0 or i == total:
            print(f"[rebuild]   {i}/{total} processed, {len(labels)} matched, {len(unmatched_ids)} unmatched so far")
        gen = json.loads(gen_path.read_text(encoding="utf-8"))
        instance_id = int(gen["instance_id"])
        generated_text = gen["generated_output"]

        if by_constraint_text:
            match = FACT_RE.search(gen["input"])
            question = match.group(1) if match else None
            gold_answer = fact_to_answer.get(question) if question else None
        else:
            question = None
            gold_answer = id_to_answer.get(instance_id)

        if gold_answer is None:
            unmatched_ids.append(instance_id)
            continue

        is_hallucination = int(gold_answer.lower().strip() not in generated_text.lower().strip())
        labels.append(
            {
                "instance_id": instance_id,
                "question": question if question is not None else gen.get("input", ""),
                "gold_answer": gold_answer,
                "generated_answer": generated_text,
                "is_hallucination": is_hallucination,
                "evaluation_method": "substring_match_case_insensitive",
            }
        )

    if unmatched_ids:
        preview = unmatched_ids[:10]
        print(
            f"[WARN] {len(unmatched_ids)} generation files had no matching gold answer "
            f"(instance ids: {preview}{'...' if len(unmatched_ids) > 10 else ''}). They were skipped."
        )

    labels.sort(key=lambda item: item["instance_id"])

    labels_path = generations_dir / "hallucination_labels.json"
    labels_path.write_text(json.dumps(labels, indent=4), encoding="utf-8")

    print(f"Wrote {len(labels)} labels to {labels_path} (from {len(generation_files)} generation files)")
    return labels_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", required=True, help="activation_cache folder name, e.g. Falcon3-7B-Base")
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="activation_cache dataset folder, e.g. belief_bank_constraints, belief_bank_facts, halu_eval",
    )
    parser.add_argument("--data-name", required=True, choices=["belief_bank", "halu_eval"], help="dataset loader to use for gold answers")
    parser.add_argument(
        "--belief-bank-data-type",
        default="facts",
        choices=["facts", "constraints"],
        help="only used when --data-name belief_bank",
    )
    parser.add_argument("--project-dir", default=str(PROJECT_ROOT))
    parser.add_argument("--use-local", action="store_true", default=True, help="load HaluEval from local cache (default: True)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rebuild_labels(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        data_name=args.data_name,
        belief_bank_data_type=args.belief_bank_data_type,
        project_dir=Path(args.project_dir),
        use_local=args.use_local,
    )


if __name__ == "__main__":
    main()
