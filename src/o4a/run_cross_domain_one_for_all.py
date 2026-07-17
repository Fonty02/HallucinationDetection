"""OneForAll runner: train on one domain, evaluate on another activation domain."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from o4a.config import (  # noqa: E402
    DEVICE,
    EXPERIMENTS,
    LAYER_TYPES,
    ONE_FOR_ALL_CONFIG,
    ROOT_DIR,
    SEED,
)
from o4a.data import (  # noqa: E402
    DataManager,
    prepare_shared_data,
    set_seed,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from o4a.methods.one_for_all import (  # noqa: E402
    _train_student_adapter,
    _train_teacher_pipeline,
)
from o4a.methods.training import count_params  # noqa: E402


def _split_train_val(n_samples: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)
    n_val = int(val_ratio * n_samples)
    return perm[n_val:], perm[:n_val]


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
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
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


@torch.no_grad()
def _predict_with_encoder_head(encoder: torch.nn.Module, head: torch.nn.Module, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    logits = head(encoder(X_t))
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = (probs > 0.5).astype(np.int64)
    return preds, probs


def _load_stratified_test_set(
    model_name: str,
    dataset_name: str,
    layer_indices: list[int],
    layer_type: str,
    test_size: float = 0.15,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a stratified test split from the activation dataset (imbalanced, preserves original class proportions)."""
    X_full, _, _ = DataManager.load_concatenated_layers(model_name, dataset_name, layer_indices, layer_type)
    stats = DataManager.get_stats(model_name, dataset_name, layer_type=layer_type)

    # Build label array in positional order (same approach as get_undersampled_indices_per_model)
    hall_set = set(stats["hallucinated_ids"])
    y = np.array([1 if i in hall_set else 0 for i in range(stats["total"])], dtype=np.int8)

    all_idx = np.arange(stats["total"])
    _, test_idx = train_test_split(
        all_idx, test_size=test_size, random_state=seed, stratify=y,
    )
    return X_full[test_idx], y[test_idx]


def _train_encoder_domain_bundle(exp_cfg: dict[str, Any], layer_type: str, cfg: dict[str, Any]) -> dict[str, Any]:
    shared_data = prepare_shared_data(exp_cfg, layer_type)
    trainer = shared_data["trainer"]
    tester = shared_data["tester"]

    tr_t, val_t = _split_train_val(len(trainer["X_train"]), cfg["val_split"], SEED)
    tr_s, val_s = _split_train_val(len(tester["X_train"]), cfg["val_split"], SEED + 100)

    t0_teacher = time.time()
    teacher_encoder, shared_head, teacher_info = _train_teacher_pipeline(
        trainer["X_train"][tr_t],
        trainer["y_train"][tr_t],
        trainer["X_train"][val_t],
        trainer["y_train"][val_t],
        input_dim=trainer["X_train"].shape[1],
        cfg=cfg,
    )
    teacher_time = time.time() - t0_teacher

    teacher_params = count_params(teacher_encoder) + count_params(shared_head)
    teacher_train_n = int(len(tr_t))

    t0_student = time.time()
    student_encoder, student_info = _train_student_adapter(
        tester["X_train"][tr_s],
        tester["y_train"][tr_s],
        tester["X_train"][val_s],
        tester["y_train"][val_s],
        input_dim=tester["X_train"].shape[1],
        frozen_head=shared_head,
        cfg=cfg,
    )
    student_time = time.time() - t0_student

    student_params = count_params(student_encoder)
    student_train_n = int(len(tr_s))

    return {
        "teacher_encoder": teacher_encoder,
        "student_encoder": student_encoder,
        "shared_head": shared_head,
        "teacher_scaler": trainer["scaler"],
        "student_scaler": tester["scaler"],
        "teacher_test": (trainer["X_test_raw"], trainer["y_test"]),
        "student_test": (tester["X_test_raw"], tester["y_test"]),
        "training_info": {
            "teacher_pipeline": teacher_info,
            "student_adapter": student_info,
        },
        "meta": {
            "detector_params": teacher_params,
            "detector_time_s": teacher_time,
            "detector_train_n": teacher_train_n,
            "aligner_params": student_params,
            "aligner_time_s": student_time,
            "aligner_train_n": student_train_n,
        },
    }


def _validate_experiment_args(encoder_experiment: str, head_experiment: str | None) -> dict[str, Any]:
    if encoder_experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown encoder experiment: {encoder_experiment}")
    if head_experiment is not None and head_experiment != encoder_experiment:
        raise ValueError(
            "This runner now uses single-domain training for encoder and head.\n"
            "If --head-experiment is provided, it must be identical to --encoder-experiment.\n"
            f"Got encoder={encoder_experiment}, head={head_experiment}."
        )
    return EXPERIMENTS[encoder_experiment]


def run_cross_domain_one_for_all(
    encoder_experiment: str,
    layer_types: list[str],
    activation_dataset: str | None,
    head_experiment: str | None = None,
) -> list[dict[str, Any]]:
    enc_cfg = _validate_experiment_args(encoder_experiment, head_experiment)
    activation_dataset = activation_dataset or enc_cfg["dataset"]
    model_cfg = ONE_FOR_ALL_CONFIG

    teacher_model = enc_cfg["trainer"]
    student_model = enc_cfg["tester"]

    results: list[dict[str, Any]] = []

    print(f"[INFO] Encoder experiment: {encoder_experiment} (dataset={enc_cfg['dataset']})")
    print(f"[INFO] Head source:        {encoder_experiment} (same training as encoder domain)")
    print(f"[INFO] Activation dataset: {activation_dataset}")
    print(f"[INFO] Trainer/Tester:     {teacher_model} -> {student_model}")

    for layer_type in layer_types:
        print(f"\n{'=' * 72}")
        print(f"[LAYER] {layer_type}")
        print(f"{'=' * 72}")
        t0 = time.time()

        base_out = {
            "encoder_experiment": encoder_experiment,
            "head_experiment": encoder_experiment,
            "dataset_train": enc_cfg["dataset"],
            "dataset_activation": activation_dataset,
            "dataset_head": enc_cfg["dataset"],
            "layer_type": layer_type,
            "teacher_model": teacher_model,
            "student_model": student_model,
            "seed": SEED,
            "device": str(DEVICE),
            "head_source_mode": "same_as_encoder_domain",
        }

        encoder_bundle = None
        try:
            set_seed(SEED)
            encoder_bundle = _train_encoder_domain_bundle(enc_cfg, layer_type, model_cfg)

            teacher_layers = enc_cfg["trainer_layers"][layer_type]
            student_layers = enc_cfg["tester_layers"][layer_type]

            encoder_bundle["teacher_encoder"].eval()
            encoder_bundle["student_encoder"].eval()
            encoder_bundle["shared_head"].eval()

            if activation_dataset == enc_cfg["dataset"]:
                # Same domain: use proper test split from prepare_shared_data (no leakage)
                X_t_raw, y_t = encoder_bundle["teacher_test"]
                X_s_raw, y_s = encoder_bundle["student_test"]
                X_t = encoder_bundle["teacher_scaler"].transform(X_t_raw).astype(np.float32)
                X_s = encoder_bundle["student_scaler"].transform(X_s_raw).astype(np.float32)
            else:
                # Cross domain: load a stratified test split from activation dataset
                # (preserves original class proportions, no training overlap — different dataset)
                X_t_raw, y_t = _load_stratified_test_set(
                    teacher_model, activation_dataset, teacher_layers, layer_type, seed=SEED,
                )
                X_s_raw, y_s = _load_stratified_test_set(
                    student_model, activation_dataset, student_layers, layer_type, seed=SEED,
                )
                X_t = encoder_bundle["teacher_scaler"].transform(X_t_raw).astype(np.float32)
                X_s = encoder_bundle["student_scaler"].transform(X_s_raw).astype(np.float32)

            pred_t, prob_t = _predict_with_encoder_head(
                encoder_bundle["teacher_encoder"],
                encoder_bundle["shared_head"],
                X_t,
            )
            pred_s, prob_s = _predict_with_encoder_head(
                encoder_bundle["student_encoder"],
                encoder_bundle["shared_head"],
                X_s,
            )

            out = {
                **base_out,
                "status": "ok",
                "encoder_domain_training": encoder_bundle["training_info"],
                "head_domain_training": {
                    "mode": "same_as_encoder_domain",
                    "teacher_pipeline": encoder_bundle["training_info"]["teacher_pipeline"],
                },
                **encoder_bundle["meta"],
                "n_samples": {
                    "teacher_eval": int(len(y_t)),
                    "student_eval": int(len(y_s)),
                },
                "eval": {
                    "teacher_on_eval": _compute_metrics(y_t, pred_t, prob_t),
                    "student_adapter_on_eval": _compute_metrics(y_s, pred_s, prob_s),
                },
                "runtime_seconds": round(time.time() - t0, 3),
            }
            print(
                f"[OK] teacher_acc={out['eval']['teacher_on_eval']['accuracy']:.4f} "
                f"| student_acc={out['eval']['student_adapter_on_eval']['accuracy']:.4f}"
            )
        except Exception:
            out = {
                **base_out,
                "status": "error",
                "error": traceback.format_exc(),
                "runtime_seconds": round(time.time() - t0, 3),
            }
            print("[ERROR] Layer failed:")
            print(out["error"])
        finally:
            if encoder_bundle is not None:
                del encoder_bundle
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results.append(out)

    return results


def _build_csv_header() -> list[str]:
    info_cols = [
        "encoder_experiment",
        "head_experiment",
        "dataset_train",
        "dataset_head",
        "dataset_activation",
        "layer_type",
        "teacher_model",
        "student_model",
        "seed",
        "device",
        "head_source_mode",
        "status",
        "runtime_seconds",
        "teacher_eval_samples",
        "student_eval_samples",
        "detector_params",
        "detector_time_s",
        "detector_train_n",
        "aligner_params",
        "aligner_time_s",
        "aligner_train_n",
    ]
    metric_cols = []
    for role in ("teacher_on_eval", "student_adapter_on_eval"):
        for metric in ("accuracy", "precision", "recall", "f1", "auroc", "confusion_matrix"):
            metric_cols.append(f"{role}_{metric}")
    return info_cols + metric_cols


def _result_to_csv_row(result: dict[str, Any]) -> dict[str, Any]:
    row = {
        "encoder_experiment": result.get("encoder_experiment", ""),
        "head_experiment": result.get("head_experiment", ""),
        "dataset_train": result.get("dataset_train", ""),
        "dataset_head": result.get("dataset_head", ""),
        "dataset_activation": result.get("dataset_activation", ""),
        "layer_type": result.get("layer_type", ""),
        "teacher_model": result.get("teacher_model", ""),
        "student_model": result.get("student_model", ""),
        "seed": result.get("seed", ""),
        "device": result.get("device", ""),
        "head_source_mode": result.get("head_source_mode", ""),
        "status": result.get("status", ""),
        "runtime_seconds": result.get("runtime_seconds", ""),
        "teacher_eval_samples": result.get("n_samples", {}).get("teacher_eval", ""),
        "student_eval_samples": result.get("n_samples", {}).get("student_eval", ""),
        "detector_params": result.get("detector_params", ""),
        "detector_time_s": result.get("detector_time_s", ""),
        "detector_train_n": result.get("detector_train_n", ""),
        "aligner_params": result.get("aligner_params", ""),
        "aligner_time_s": result.get("aligner_time_s", ""),
        "aligner_train_n": result.get("aligner_train_n", ""),
    }

    eval_metrics = result.get("eval", {})
    for role in ("teacher_on_eval", "student_adapter_on_eval"):
        role_metrics = eval_metrics.get(role, {})
        for metric in ("accuracy", "precision", "recall", "f1", "auroc"):
            row[f"{role}_{metric}"] = role_metrics.get(metric, "")
        confusion = role_metrics.get("confusion_matrix", "")
        row[f"{role}_confusion_matrix"] = (
            json.dumps(confusion, separators=(",", ":")) if confusion != "" else ""
        )

    if result.get("status") == "error":
        for role in ("teacher_on_eval", "student_adapter_on_eval"):
            for metric in ("accuracy", "precision", "recall", "f1", "auroc", "confusion_matrix"):
                row[f"{role}_{metric}"] = "ERROR"

    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-domain activation evaluation with OneForAll retraining mode: "
            "trains encoder/head once on encoder experiment, then evaluates on activation dataset."
        )
    )
    parser.add_argument(
        "--encoder-experiment",
        required=True,
        help="Experiment used for encoder/scaler training (must exist in o4a.config.EXPERIMENTS).",
    )
    parser.add_argument(
        "--head-experiment",
        default=None,
        help=(
            "Deprecated compatibility argument. If provided, it must be the same as "
            "--encoder-experiment."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--activation-dataset",
        default=None,
        help=(
            "Dataset used for activations at eval time. "
            "Default: encoder experiment dataset."
        ),
    )
    parser.add_argument(
        "--layer-types",
        nargs="+",
        choices=LAYER_TYPES,
        default=None,
        help="Optional subset of layer types (default: all).",
    )
    args = parser.parse_args()

    selected_layer_types = args.layer_types if args.layer_types else LAYER_TYPES

    print(f"[DEBUG] Root: {ROOT_DIR}")
    print(f"[DEBUG] Seed: {SEED}")
    print(f"[DEBUG] Device: {DEVICE}")
    print(f"[DEBUG] Layer types: {selected_layer_types}")

    results = run_cross_domain_one_for_all(
        encoder_experiment=args.encoder_experiment,
        layer_types=selected_layer_types,
        activation_dataset=args.activation_dataset,
        head_experiment=args.head_experiment,
    )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    header = _build_csv_header()
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for result in results:
            writer.writerow(_result_to_csv_row(result))

    ok_count = sum(1 for r in results if r.get("status") == "ok")
    print(f"\n[INFO] Saved {len(results)} layer result row(s) to CSV: {args.output}")
    print(f"[INFO] Successful layers: {ok_count}/{len(results)}")


if __name__ == "__main__":
    main()
