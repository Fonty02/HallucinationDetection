"""Shared training routines for alignment, probing, and autoencoder."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from ..config import DEVICE, SEED, NOTEBOOK_COMPAT
from ..data import set_seed, get_generator
from ..models import AlignmentNetwork, Autoencoder, MixedLoss, MLPProber


def count_params(model) -> int:
    """Return number of learned parameters for a torch module or sklearn estimator."""
    import torch
    if isinstance(model, torch.nn.Module):
        return sum(p.numel() for p in model.parameters())
    if hasattr(model, "coef_"):
        n = int(model.coef_.size)
        if hasattr(model, "intercept_") and model.intercept_ is not None:
            intercept = model.intercept_
            if isinstance(intercept, (int, float)):
                n += 1
            else:
                n += int(np.asarray(intercept).size)
        return n
    # CCA-like wrappers: sum parameters of internal components
    total = 0
    for attr in ("cca", "regressor", "encoder", "head", "net"):
        comp = getattr(model, attr, None)
        if comp is not None:
            total += count_params(comp)
    if total > 0:
        return total
    # numpy/torch tensor (e.g. Procrustes/CKA matrix)
    for attr in dir(model):
        if attr.startswith("_"):
            continue
        val = getattr(model, attr, None)
        if isinstance(val, (torch.Tensor,)):
            return int(val.numel())
        if isinstance(val, np.ndarray):
            return int(val.size)
    return 0


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auroc": roc_auc_score(y_true, y_proba),
    }


def split_train_val(n, val_ratio=0.15, seed=SEED):
    """Return (train_idx, val_idx) from a random 85/15 permutation."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    v = int(val_ratio * n)
    return perm[v:], perm[:v]


def _capture_state_dict(model: nn.Module):
    """Snapshot model weights. In notebook-compat mode this is a shallow copy."""
    if NOTEBOOK_COMPAT:
        return model.state_dict().copy()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ------------------------------------------------------------------
# Alignment Network training
# ------------------------------------------------------------------

def train_alignment_network(X_src_train, X_tgt_train, X_src_val, X_tgt_val, cfg):
    """
    Train an AlignmentNetwork (src → tgt) with MixedLoss.

    Expected config keys (all prefixed ``alignment_``):
        alignment_hidden_dim, alignment_dropout,
        alignment_lr, alignment_weight_decay, alignment_batch_size,
        alignment_max_epochs, alignment_patience, alignment_min_delta,
        alignment_grad_clip, alignment_loss_alpha, alignment_loss_beta
    """
    set_seed(SEED)
    model = AlignmentNetwork(
        input_dim=X_src_train.shape[1],
        output_dim=X_tgt_train.shape[1],
        hidden_dim=cfg["alignment_hidden_dim"],
        dropout=cfg["alignment_dropout"],
    ).to(DEVICE)

    criterion = MixedLoss(alpha=cfg["alignment_loss_alpha"], beta=cfg["alignment_loss_beta"])
    optimizer = optim.AdamW(model.parameters(), lr=cfg["alignment_lr"],
                            weight_decay=cfg["alignment_weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["alignment_max_epochs"])

    train_ds = TensorDataset(
        torch.tensor(X_src_train, dtype=torch.float32),
        torch.tensor(X_tgt_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_src_val, dtype=torch.float32),
        torch.tensor(X_tgt_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["alignment_batch_size"],
                              shuffle=True, generator=get_generator(SEED))
    val_loader = DataLoader(val_ds, batch_size=cfg["alignment_batch_size"], shuffle=False)

    best_state, best_loss, patience = None, float("inf"), 0
    best_epoch, total_epochs = 0, 0

    for epoch in range(cfg["alignment_max_epochs"]):
        total_epochs = epoch + 1
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["alignment_grad_clip"])
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item()
        val_loss /= max(len(val_loader), 1)

        if val_loss < best_loss - cfg["alignment_min_delta"]:
            best_loss, patience = val_loss, 0
            best_epoch = epoch + 1
            best_state = _capture_state_dict(model)
        else:
            patience += 1
            if patience >= cfg["alignment_patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    training_info = {
        "total_epochs": total_epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
    }
    return model, training_info


# ------------------------------------------------------------------
# MLP Prober training
# ------------------------------------------------------------------

def train_mlp_prober(X_train, y_train, X_val, y_val, input_dim, cfg):
    """
    Train an MLPProber with BCEWithLogitsLoss (class-balanced).

    Expected config keys (all prefixed ``prober_``):
        prober_hidden_dim, prober_dropout,
        prober_lr, prober_weight_decay, prober_batch_size,
        prober_max_epochs, prober_patience, prober_grad_clip
    """
    set_seed(SEED)
    prober = MLPProber(
        input_dim=input_dim,
        hidden_dim=cfg["prober_hidden_dim"],
        dropout=cfg["prober_dropout"],
    ).to(DEVICE)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos if n_pos > 0 else 1.0], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(prober.parameters(), lr=cfg["prober_lr"],
                            weight_decay=cfg["prober_weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["prober_max_epochs"])

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["prober_batch_size"],
                              shuffle=True, generator=get_generator(SEED))
    val_loader = DataLoader(val_ds, batch_size=cfg["prober_batch_size"], shuffle=False)

    best_state, best_acc, patience = None, 0.0, 0
    best_epoch, total_epochs = 0, 0
    min_delta = cfg.get("prober_min_delta", 0.0)

    for epoch in range(cfg["prober_max_epochs"]):
        total_epochs = epoch + 1
        prober.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(prober(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prober.parameters(), cfg["prober_grad_clip"])
            optimizer.step()
        scheduler.step()

        prober.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = prober.predict(xb.to(DEVICE)).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())
        acc = accuracy_score(all_labels, all_preds)

        if acc > best_acc + min_delta:
            best_acc, patience = acc, 0
            best_epoch = epoch + 1
            best_state = _capture_state_dict(prober)
        else:
            patience += 1
            if patience >= cfg["prober_patience"]:
                break

    if best_state is not None:
        prober.load_state_dict(best_state)
    training_info = {
        "total_epochs": total_epochs,
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
    }
    return prober, training_info


# ------------------------------------------------------------------
# Autoencoder training
# ------------------------------------------------------------------

def train_autoencoder(X_train, X_val, input_dim, cfg):
    """
    Train an Autoencoder with MSELoss (reconstruction).

    Expected config keys (all prefixed ``autoencoder_``):
        autoencoder_latent_dim, autoencoder_hidden_dim, autoencoder_dropout,
        autoencoder_lr, autoencoder_weight_decay, autoencoder_batch_size,
        autoencoder_max_epochs, autoencoder_patience, autoencoder_min_delta,
        autoencoder_grad_clip
    """
    set_seed(SEED)
    ae = Autoencoder(
        input_dim=input_dim,
        latent_dim=cfg["autoencoder_latent_dim"],
        hidden_dim=cfg["autoencoder_hidden_dim"],
        dropout=cfg["autoencoder_dropout"],
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(ae.parameters(), lr=cfg["autoencoder_lr"],
                            weight_decay=cfg["autoencoder_weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["autoencoder_max_epochs"])

    train_t = torch.tensor(X_train, dtype=torch.float32)
    val_t = torch.tensor(X_val, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_t), batch_size=cfg["autoencoder_batch_size"],
                              shuffle=True, generator=get_generator(SEED))
    val_loader = DataLoader(TensorDataset(val_t), batch_size=cfg["autoencoder_batch_size"],
                            shuffle=False)

    best_state, best_loss, patience = None, float("inf"), 0
    best_epoch, total_epochs = 0, 0

    for epoch in range(cfg["autoencoder_max_epochs"]):
        total_epochs = epoch + 1
        ae.train()
        for (xb,) in train_loader:
            xb = xb.to(DEVICE)
            optimizer.zero_grad()
            recon, _ = ae(xb)
            loss = criterion(recon, xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), cfg["autoencoder_grad_clip"])
            optimizer.step()
        scheduler.step()

        ae.eval()
        v_loss = 0.0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(DEVICE)
                recon, _ = ae(xb)
                v_loss += criterion(recon, xb).item()
        v_loss /= max(len(val_loader), 1)

        if v_loss < best_loss - cfg["autoencoder_min_delta"]:
            best_loss, patience = v_loss, 0
            best_epoch = epoch + 1
            best_state = _capture_state_dict(ae)
        else:
            patience += 1
            if patience >= cfg["autoencoder_patience"]:
                break

    if best_state is not None:
        ae.load_state_dict(best_state)
    training_info = {
        "total_epochs": total_epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
    }
    return ae, training_info
