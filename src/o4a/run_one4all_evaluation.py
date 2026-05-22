"""Dedicated OneForAll evaluation runner for a single O4A experiment.

This script:
1) trains only OneForAll for one experiment from o4a.config.EXPERIMENTS,
2) saves model weights,
3) evaluates on trainer/tester test splits,
4) exports two JSON files (one per LLM) with per-instance predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from o4a.config import (  # noqa: E402
    CACHE_DIR_NAME,
    DEVICE,
    EXPERIMENTS,
    LAYER_TYPES,
    MODEL_ALIASES,
    ONE_FOR_ALL_CONFIG,
    ROOT_DIR,
    SEED,
    TRAIN_SPLIT,
)
from o4a.data import get_balanced_indices, set_seed  # noqa: E402
from o4a.methods.one_for_all import _train_student_adapter, _train_teacher_pipeline  # noqa: E402


def _split_train_val(n_samples: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)
    n_val = int(val_ratio * n_samples)
    return perm[n_val:], perm[:n_val]


def _sanitize_name(name: str) -> str:
    return name.replace("/", "_")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float | None]:
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = None
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": auroc,
    }


@torch.no_grad()
def _predict_with_encoder_head(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    logits = head(encoder(x_t))
    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    preds = (probs > 0.5).astype(np.int64)
    return preds, probs


def _to_numpy_float32(tensor_or_array: object) -> np.ndarray:
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.cpu().numpy().astype(np.float32)
    return np.asarray(tensor_or_array, dtype=np.float32)


def _load_single_layer_with_ids_and_labels(
    model_name: str,
    dataset_name: str,
    layer_idx: int,
    layer_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_root = Path(ROOT_DIR) / CACHE_DIR_NAME
    model_dir = _resolve_model_dir(cache_root, model_name)
    activation_dir = cache_root / model_dir / dataset_name / f"activation_{layer_type}"

    hall_dir = activation_dir / "hallucinated"
    non_hall_dir = activation_dir / "not_hallucinated"

    if hall_dir.is_dir() and non_hall_dir.is_dir():
        hall_act = torch.load(hall_dir / f"layer{layer_idx}_activations.pt", map_location="cpu")
        non_hall_act = torch.load(non_hall_dir / f"layer{layer_idx}_activations.pt", map_location="cpu")
        with open(hall_dir / f"layer{layer_idx}_instance_ids.json", "r", encoding="utf-8") as f:
            hall_ids = json.load(f)
        with open(non_hall_dir / f"layer{layer_idx}_instance_ids.json", "r", encoding="utf-8") as f:
            non_hall_ids = json.load(f)

        hall_np = _to_numpy_float32(hall_act)
        non_hall_np = _to_numpy_float32(non_hall_act)

        x = np.vstack([hall_np, non_hall_np]).astype(np.float32)
        y = np.concatenate(
            [
                np.ones(hall_np.shape[0], dtype=np.int64),
                np.zeros(non_hall_np.shape[0], dtype=np.int64),
            ]
        )
        ids = np.asarray(hall_ids + non_hall_ids, dtype=np.int64)
        order = np.argsort(ids)
        return x[order], y[order], ids[order]

    # Old structure: align labels by real instance_id (from layer*_instance_ids.json when present).
    act = torch.load(activation_dir / f"layer{layer_idx}_activations.pt", map_location="cpu")
    x = _to_numpy_float32(act)

    ids_path = activation_dir / f"layer{layer_idx}_instance_ids.json"
    if ids_path.exists():
        with open(ids_path, "r", encoding="utf-8") as f:
            ids_raw = json.load(f)
        ids = np.asarray([int(v) for v in ids_raw], dtype=np.int64)
    else:
        ids = np.arange(x.shape[0], dtype=np.int64)

    labels_by_id = _load_generation_labels(model_name, dataset_name)
    missing_ids = [int(i) for i in ids if int(i) not in labels_by_id]
    if missing_ids:
        raise ValueError(
            f"Missing {len(missing_ids)} instance_id(s) in hallucination_labels for {model_name}/{dataset_name}. "
            f"First missing ids: {missing_ids[:10]}"
        )
    y = np.asarray([int(labels_by_id[int(i)]["is_hallucination"]) for i in ids], dtype=np.int64)

    order = np.argsort(ids)
    return x[order], y[order], ids[order]


def _load_concatenated_layers_with_ids(
    model_name: str,
    dataset_name: str,
    layer_indices: list[int],
    layer_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chunks: list[np.ndarray] = []
    common_y: np.ndarray | None = None
    common_ids: np.ndarray | None = None

    for layer_idx in layer_indices:
        x_l, y_l, ids_l = _load_single_layer_with_ids_and_labels(
            model_name=model_name,
            dataset_name=dataset_name,
            layer_idx=layer_idx,
            layer_type=layer_type,
        )

        if common_y is None:
            common_y = y_l.astype(np.int64)
            common_ids = ids_l
        else:
            if not np.array_equal(common_y, y_l):
                raise ValueError(
                    f"Label mismatch for {model_name} at layer {layer_idx} ({dataset_name}/{layer_type})."
                )
            if not np.array_equal(common_ids, ids_l):
                raise ValueError(
                    f"Instance-id mismatch for {model_name} at layer {layer_idx} ({dataset_name}/{layer_type})."
                )
        chunks.append(x_l.astype(np.float32))

    assert common_y is not None and common_ids is not None
    return np.concatenate(chunks, axis=1), common_y, common_ids


def _build_balanced_split(
    model_name: str,
    dataset_name: str,
    layer_indices: list[int],
    layer_type: str,
    split_seed: int,
) -> dict[str, Any]:
    x_full, y_full, ids_full = _load_concatenated_layers_with_ids(
        model_name=model_name,
        dataset_name=dataset_name,
        layer_indices=layer_indices,
        layer_type=layer_type,
    )

    balanced_idx = get_balanced_indices(y_full, seed=SEED)
    x_bal = x_full[balanced_idx]
    y_bal = y_full[balanced_idx]
    ids_bal = ids_full[balanced_idx]

    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(len(x_bal))
    split_pos = int(TRAIN_SPLIT * len(x_bal))

    tr_idx = perm[:split_pos]
    te_idx = perm[split_pos:]

    x_train_raw = x_bal[tr_idx]
    y_train = y_bal[tr_idx]
    ids_train = ids_bal[tr_idx]

    x_test_raw = x_bal[te_idx]
    y_test = y_bal[te_idx]
    ids_test = ids_bal[te_idx]

    scaler = StandardScaler().fit(x_train_raw)

    return {
        "model_name": model_name,
        "x_train": scaler.transform(x_train_raw).astype(np.float32),
        "x_test": scaler.transform(x_test_raw).astype(np.float32),
        "y_train": y_train.astype(np.int64),
        "y_test": y_test.astype(np.int64),
        "ids_train": ids_train.astype(np.int64),
        "ids_test": ids_test.astype(np.int64),
        "scaler": scaler,
        "n_samples": {
            "full": int(len(x_full)),
            "balanced": int(len(x_bal)),
            "train": int(len(x_train_raw)),
            "test": int(len(x_test_raw)),
        },
    }


def _resolve_model_dir(cache_root: Path, model_name: str) -> str:
    preferred = MODEL_ALIASES.get(model_name, model_name)
    candidates = [preferred, model_name]
    if "/" in preferred:
        candidates.extend([preferred.split("/")[-1], preferred.replace("/", "_"), preferred.replace("/", "-")])
    if "/" in model_name:
        candidates.extend([model_name.split("/")[-1], model_name.replace("/", "_"), model_name.replace("/", "-")])

    for candidate in candidates:
        if (cache_root / candidate).is_dir():
            return candidate

    available = [p.name for p in cache_root.iterdir() if p.is_dir()] if cache_root.exists() else []
    lower_map = {name.lower(): name for name in available}
    for candidate in candidates:
        resolved = lower_map.get(candidate.lower())
        if resolved is not None:
            return resolved

    raise FileNotFoundError(
        f"Model folder for '{model_name}' not found in {cache_root}. "
        f"Available models: {', '.join(sorted(available))}"
    )


def _load_generation_labels(model_name: str, dataset_name: str) -> dict[int, dict[str, Any]]:
    cache_root = Path(ROOT_DIR) / CACHE_DIR_NAME
    model_dir = _resolve_model_dir(cache_root, model_name)
    labels_path = cache_root / model_dir / dataset_name / "generations" / "hallucination_labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Missing generations labels for {model_name}/{dataset_name}: {labels_path}"
        )

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    mapping: dict[int, dict[str, Any]] = {}
    for item in labels:
        if "instance_id" not in item:
            continue
        mapping[int(item["instance_id"])] = item
    return mapping


def _build_record_payload(instance: dict[str, Any] | None, instance_id: int) -> tuple[dict[str, Any], Any, Any]:
    dataset_instance = {"instance_id": int(instance_id)}
    llm_answer = None
    gold_answer = None

    if instance is not None:
        if "question" in instance:
            dataset_instance["question"] = instance["question"]
        for extra_key in ("knowledge", "context", "dialogue_history"):
            if extra_key in instance:
                dataset_instance[extra_key] = instance[extra_key]
        llm_answer = instance.get("generated_answer", instance.get("generated_text"))
        gold_answer = instance.get("gold_answer", instance.get("right_answer"))

    return dataset_instance, llm_answer, gold_answer


def _build_predictions_json(
    model_name: str,
    dataset_name: str,
    layer_type: str,
    ids_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    labels_by_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing = 0
    gt_mismatch_count = 0

    for instance_id, gt, pred, prob in zip(ids_test, y_true, y_pred, y_prob):
        iid = int(instance_id)
        instance = labels_by_id.get(iid)
        if instance is None:
            missing += 1

        dataset_instance, llm_answer, gold_answer = _build_record_payload(instance, iid)
        gt_from_labels = None
        if instance is not None and "is_hallucination" in instance:
            gt_from_labels = int(instance["is_hallucination"])

        gt_split = int(gt)
        gt_final = gt_from_labels if gt_from_labels is not None else gt_split
        if gt_from_labels is not None and gt_from_labels != gt_split:
            gt_mismatch_count += 1

        rows.append(
            {
                "Istanza Dataset": dataset_instance,
                "Risposta data dall LLM": llm_answer,
                "Risposta vera": gold_answer,
                "Predizione One4All": int(pred),
                "GroundTruth per One4All": gt_final,
                "GroundTruth split bilanciato": gt_split,
                "Probabilita One4All": float(prob),
            }
        )

    return {
        "model_name": model_name,
        "dataset": dataset_name,
        "layer_type": layer_type,
        "seed": SEED,
        "num_records": len(rows),
        "missing_generation_records": missing,
        "groundtruth_mismatch_between_split_and_labels": gt_mismatch_count,
        "records": rows,
    }


def run_one_for_all_experiment(experiment_name: str, layer_type: str, output_root: Path) -> dict[str, Any]:
    if layer_type not in LAYER_TYPES:
        raise ValueError(f"Unsupported layer_type '{layer_type}'. Allowed: {LAYER_TYPES}")

    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"Experiment '{experiment_name}' not found in o4a.config.EXPERIMENTS")
    exp_cfg = EXPERIMENTS[experiment_name]
    trainer_model = exp_cfg["trainer"]
    tester_model = exp_cfg["tester"]
    dataset_name = exp_cfg["dataset"]

    trainer_layers = exp_cfg["trainer_layers"]
    tester_layers = exp_cfg["tester_layers"]

    set_seed(SEED)
    trainer_split = _build_balanced_split(
        model_name=trainer_model,
        dataset_name=dataset_name,
        layer_indices=trainer_layers[layer_type],
        layer_type=layer_type,
        split_seed=SEED,
    )
    tester_split = _build_balanced_split(
        model_name=tester_model,
        dataset_name=dataset_name,
        layer_indices=tester_layers[layer_type],
        layer_type=layer_type,
        split_seed=SEED + 1,
    )
    # Fail early if generation metadata is missing (needed for final per-instance JSON export).
    trainer_labels = _load_generation_labels(trainer_model, dataset_name)
    tester_labels = _load_generation_labels(tester_model, dataset_name)

    cfg = ONE_FOR_ALL_CONFIG
    tr_t, val_t = _split_train_val(len(trainer_split["x_train"]), cfg["val_split"], SEED)
    tr_s, val_s = _split_train_val(len(tester_split["x_train"]), cfg["val_split"], SEED + 100)

    teacher_encoder, shared_head, teacher_info = _train_teacher_pipeline(
        trainer_split["x_train"][tr_t],
        trainer_split["y_train"][tr_t],
        trainer_split["x_train"][val_t],
        trainer_split["y_train"][val_t],
        input_dim=trainer_split["x_train"].shape[1],
        cfg=cfg,
    )
    student_encoder, student_info = _train_student_adapter(
        tester_split["x_train"][tr_s],
        tester_split["y_train"][tr_s],
        tester_split["x_train"][val_s],
        tester_split["y_train"][val_s],
        input_dim=tester_split["x_train"].shape[1],
        frozen_head=shared_head,
        cfg=cfg,
    )

    teacher_encoder.eval()
    shared_head.eval()
    student_encoder.eval()

    pred_tr, prob_tr = _predict_with_encoder_head(teacher_encoder, shared_head, trainer_split["x_test"])
    pred_te, prob_te = _predict_with_encoder_head(student_encoder, shared_head, tester_split["x_test"])

    trainer_metrics = _compute_metrics(trainer_split["y_test"], pred_tr, prob_tr)
    tester_metrics = _compute_metrics(tester_split["y_test"], pred_te, prob_te)

    trainer_json = _build_predictions_json(
        model_name=trainer_model,
        dataset_name=dataset_name,
        layer_type=layer_type,
        ids_test=trainer_split["ids_test"],
        y_true=trainer_split["y_test"],
        y_pred=pred_tr,
        y_prob=prob_tr,
        labels_by_id=trainer_labels,
    )
    tester_json = _build_predictions_json(
        model_name=tester_model,
        dataset_name=dataset_name,
        layer_type=layer_type,
        ids_test=tester_split["ids_test"],
        y_true=tester_split["y_test"],
        y_pred=pred_te,
        y_prob=prob_te,
        labels_by_id=tester_labels,
    )

    run_dir = output_root / experiment_name / f"seed_{SEED}" / f"layer_{layer_type}"
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    torch.save(teacher_encoder.state_dict(), weights_dir / "trainer_encoder.pt")
    torch.save(shared_head.state_dict(), weights_dir / "shared_head.pt")
    torch.save(student_encoder.state_dict(), weights_dir / "tester_encoder.pt")

    scaler_payload = {
        "trainer_scaler_mean": trainer_split["scaler"].mean_.tolist(),
        "trainer_scaler_scale": trainer_split["scaler"].scale_.tolist(),
        "tester_scaler_mean": tester_split["scaler"].mean_.tolist(),
        "tester_scaler_scale": tester_split["scaler"].scale_.tolist(),
    }
    with open(weights_dir / "scalers.json", "w", encoding="utf-8") as f:
        json.dump(scaler_payload, f, indent=2, ensure_ascii=False)

    trainer_json_path = run_dir / f"{_sanitize_name(trainer_model)}_test_predictions.json"
    tester_json_path = run_dir / f"{_sanitize_name(tester_model)}_test_predictions.json"
    with open(trainer_json_path, "w", encoding="utf-8") as f:
        json.dump(trainer_json, f, indent=2, ensure_ascii=False)
    with open(tester_json_path, "w", encoding="utf-8") as f:
        json.dump(tester_json, f, indent=2, ensure_ascii=False)

    summary = {
        "experiment_name": experiment_name,
        "trainer_model": trainer_model,
        "tester_model": tester_model,
        "dataset": dataset_name,
        "layer_type": layer_type,
        "seed": SEED,
        "device": str(DEVICE),
        "training": {
            "teacher_pipeline": teacher_info,
            "student_adapter": student_info,
        },
        "metrics": {
            "trainer_test": trainer_metrics,
            "tester_test": tester_metrics,
        },
        "n_samples": {
            "trainer": trainer_split["n_samples"],
            "tester": tester_split["n_samples"],
        },
        "outputs": {
            "run_dir": str(run_dir),
            "weights_dir": str(weights_dir),
            "trainer_predictions_json": str(trainer_json_path),
            "tester_predictions_json": str(tester_json_path),
        },
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train ONLY OneForAll for one configured experiment, "
            "save model weights, and export per-instance JSON predictions."
        )
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment key from o4a.config.EXPERIMENTS (e.g., GemmaToLlama_HE).",
    )
    parser.add_argument(
        "--layer-type",
        required=True,
        choices=LAYER_TYPES,
        help="Layer type to run (attn/mlp/hidden).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/one4all_experiment_details",
        help="Base output directory.",
    )
    args = parser.parse_args()

    print(f"[DEBUG] ROOT_DIR={ROOT_DIR}")
    print(f"[DEBUG] SEED={SEED}")
    print(f"[DEBUG] DEVICE={DEVICE}")
    print(f"[DEBUG] experiment={args.experiment}")
    print(f"[DEBUG] layer_type={args.layer_type}")

    summary = run_one_for_all_experiment(
        experiment_name=args.experiment,
        layer_type=args.layer_type,
        output_root=Path(args.output_dir),
    )

    print("[INFO] Completed OneForAll experiment run.")
    print(f"[INFO] Summary saved in: {summary['outputs']['run_dir']}")
    print(f"[INFO] Trainer JSON: {summary['outputs']['trainer_predictions_json']}")
    print(f"[INFO] Tester JSON: {summary['outputs']['tester_predictions_json']}")


if __name__ == "__main__":
    main()
