"""OneForAll: shared ClassificationHead, separate Encoders. No alignment needed."""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from ..config import DEVICE, SEED, ONE_FOR_ALL_CONFIG, NOTEBOOK_COMPAT
from ..data import set_seed, get_generator
from ..models import Encoder, ClassificationHead
from .training import compute_metrics, count_params


def _capture_state_dict(model: nn.Module):
    """Snapshot model weights. In notebook-compat mode this is a shallow copy."""
    if NOTEBOOK_COMPAT:
        return model.state_dict().copy()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _train_teacher_pipeline(X_train, y_train, X_val, y_val, input_dim, cfg):
    """Train encoder + classification head jointly for the trainer."""
    set_seed(SEED)
    encoder = Encoder(input_dim, cfg["encoder_latent_dim"], cfg["encoder_hidden_dim"], cfg["encoder_dropout"]).to(DEVICE)
    head = ClassificationHead(cfg["encoder_latent_dim"], cfg["head_hidden_dim"], cfg["head_dropout"]).to(DEVICE)

    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["max_epochs"])

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos if n_pos > 0 else 1.0], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, generator=get_generator(SEED))
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    best_states, best_acc, patience = None, 0.0, 0
    best_epoch_t, total_epochs_t = 0, 0
    for epoch in range(cfg["max_epochs"]):
        total_epochs_t = epoch + 1
        encoder.train(); head.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(head(encoder(xb)), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["gradient_clip_max_norm"])
            optimizer.step()

        encoder.eval(); head.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = head.predict(encoder(xb.to(DEVICE))).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())
        acc = accuracy_score(all_labels, all_preds)
        scheduler.step()

        if acc > best_acc:
            best_acc, patience = acc, 0
            best_epoch_t = epoch + 1
            best_states = {
                "encoder": _capture_state_dict(encoder),
                "head": _capture_state_dict(head),
            }
        else:
            patience += 1
            if patience >= cfg["early_stopping_patience"]:
                break

    if best_states:
        encoder.load_state_dict(best_states["encoder"])
        head.load_state_dict(best_states["head"])
    training_info = {
        "total_epochs": total_epochs_t,
        "best_epoch": best_epoch_t,
        "best_val_acc": best_acc,
    }
    return encoder, head, training_info


def _train_student_adapter(X_train, y_train, X_val, y_val, input_dim, frozen_head, cfg):
    """Train a new encoder for tester with the frozen head from trainer."""
    frozen_head.eval()
    for p in frozen_head.parameters():
        p.requires_grad = False

    set_seed(SEED)
    encoder = Encoder(input_dim, cfg["encoder_latent_dim"], cfg["encoder_hidden_dim"], cfg["encoder_dropout"]).to(DEVICE)
    optimizer = optim.AdamW(encoder.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["max_epochs"])

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos if n_pos > 0 else 1.0], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, generator=get_generator(SEED))
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    best_state, best_acc, patience = None, 0.0, 0
    best_epoch_s, total_epochs_s = 0, 0
    for epoch in range(cfg["max_epochs"]):
        total_epochs_s = epoch + 1
        encoder.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(frozen_head(encoder(xb)), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg["gradient_clip_max_norm"])
            optimizer.step()

        encoder.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = frozen_head.predict(encoder(xb.to(DEVICE))).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())
        acc = accuracy_score(all_labels, all_preds)
        scheduler.step()

        if acc > best_acc:
            best_acc, patience = acc, 0
            best_epoch_s = epoch + 1
            best_state = _capture_state_dict(encoder)
        else:
            patience += 1
            if patience >= cfg["early_stopping_patience"]:
                break

    if best_state:
        encoder.load_state_dict(best_state)
    training_info = {
        "total_epochs": total_epochs_s,
        "best_epoch": best_epoch_s,
        "best_val_acc": best_acc,
    }
    return encoder, training_info


def run_one_for_all(shared_data: dict, config: dict = None, save_dir: str = None) -> dict:
    """
    OneForAll: train Encoder+Head on trainer, then train a new Encoder
    for tester with the head frozen. No alignment needed.
    Uses the same train/test splits as all other methods.
    """
    cfg = config or ONE_FOR_ALL_CONFIG
    trainer = shared_data["trainer"]
    tester = shared_data["tester"]

    # Phase 1: Train trainer pipeline (shared split: balanced train, imbalanced val)
    t0_detector = time.time()
    enc_trainer, head, teacher_info = _train_teacher_pipeline(
        trainer["X_train"], trainer["y_train"],
        trainer["X_val"], trainer["y_val"],
        input_dim=trainer["X_train"].shape[1], cfg=cfg,
    )
    detector_time = time.time() - t0_detector

    detector_params = count_params(enc_trainer) + count_params(head)
    detector_train_n = int(len(trainer["y_train"]))

    # Eval trainer
    enc_trainer.eval(); head.eval()
    with torch.no_grad():
        X_t = torch.tensor(trainer["X_test"], dtype=torch.float32, device=DEVICE)
        pred_t = head.predict(enc_trainer(X_t)).cpu().numpy()
        proba_t = torch.sigmoid(head(enc_trainer(X_t))).cpu().numpy()
    metrics_trainer = compute_metrics(trainer["y_test"], pred_t, proba_t)

    # Phase 2: Train tester encoder with frozen head
    t0_aligner = time.time()
    enc_tester, student_info = _train_student_adapter(
        tester["X_train"], tester["y_train"],
        tester["X_val"], tester["y_val"],
        input_dim=tester["X_train"].shape[1], frozen_head=head, cfg=cfg,
    )
    aligner_time = time.time() - t0_aligner

    aligner_params = count_params(enc_tester)
    aligner_train_n = int(len(tester["y_train"]))

    # Eval tester
    enc_tester.eval()
    with torch.no_grad():
        X_s = torch.tensor(tester["X_test"], dtype=torch.float32, device=DEVICE)
        pred_s = head.predict(enc_tester(X_s)).cpu().numpy()
        proba_s = torch.sigmoid(head(enc_tester(X_s))).cpu().numpy()
    metrics_tester = compute_metrics(tester["y_test"], pred_s, proba_s)

    # Save models
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(enc_trainer.state_dict(), os.path.join(save_dir, "enc_trainer.pt"))
        torch.save(head.state_dict(), os.path.join(save_dir, "head.pt"))
        torch.save(enc_tester.state_dict(), os.path.join(save_dir, "enc_tester.pt"))

    return {
        "trainer": metrics_trainer,
        "tester": metrics_tester,
        "_meta": {
            "detector_params": detector_params,
            "detector_time_s": detector_time,
            "detector_train_n": detector_train_n,
            "aligner_params": aligner_params,
            "aligner_time_s": aligner_time,
            "aligner_train_n": aligner_train_n,
        },
    }
