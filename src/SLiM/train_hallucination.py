"""
Training script per SLiM Hallucination Reduction — Contrastive Learning.

Addestra il modulo GeneralSLiMedNet con loss contrastiva (InfoNCE) su coppie
paired (truthful vs hallucinated) per insegnare ai parametri scale/shift
a separare truth da hallucination nello spazio delle rappresentazioni nascoste.

Approccio ispirato a TruthX (ACL 2024), CAA e RepE:
- state=1.0 per ENTRAMBI i campioni: la trasformazione FiLM (s·h + b)
  agisce come selettore di feature (scale) e bias direzionale (shift)
- InfoNCE loss sulle rappresentazioni last-token post-modulazione
- Coppie di training identiche a quelle di TruthX per confronto equo

Uso:
    python -m src.SLiM.train_hallucination \\
        --model_name Qwen/Qwen2.5-7B \\
        --dataset belief_bank_facts \\
        --num_pairs 2000 \\
        --epochs 3 \\
        --batch_size 4 \\
        --lr 5e-4 \\
        --target_layer 15 \\
        --temperature 0.1 \\
        --device cuda:0
"""

import argparse
import json
import os
import sys
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from peft import prepare_model_for_kbit_training
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.utils import create_bnb_config, load_llm, load_tokenizer
from src.SLiM.model_general import GeneralSLiMedNet
from src.SLiM.dataset_hallucination import (
    collate_fn_paired,
    collate_fn_hallucination,
    create_slim_dataset,
)


def print_trainable_parameters(model):
    """Stampa il numero di parametri trainabili vs totali."""
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(
        f"Parametri trainabili: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.4f}%)"
    )
    return trainable


# =============================================================================
# CONTRASTIVE LOSS (InfoNCE — adapted from TruthX Eq. 5)
# =============================================================================


class SLiMContrastiveLoss(nn.Module):
    """
    Contrastive loss (InfoNCE) per SLiM hallucination reduction.

    Ispirata alla truthful_contrastive_loss di TruthX (Eq. 5 del paper):

        L = CTR(h_pos, H_pos, H_neg) + CTR(h_neg, H_neg, H_pos)

    dove CTR(s, S+, S-) = -log[ Σ_{s'∈S+} exp(sim(s,s')/τ) /
                                Σ_{s'∈S+∪S-} exp(sim(s,s')/τ) ]

    L'obiettivo è separare le rappresentazioni dei campioni truthful
    da quelli hallucinated nello spazio FiLM-modulato.

    A differenza di TruthX che opera su attivazioni pre-estratte, qui
    operiamo sulle rappresentazioni last-token catturate dall'hook SLiM
    durante il forward pass.
    """

    def __init__(self, temperature: float = 0.1):
        """
        Args:
            temperature: Temperature τ per InfoNCE (default 0.1, come TruthX).
                         Valori più bassi → distribuzioni più "peaked" → loss più
                         sensibile alle differenze. TruthX usa 0.1.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, h_pos: torch.Tensor, h_neg: torch.Tensor) -> torch.Tensor:
        """
        Calcola la contrastive loss bidirezionale.

        Args:
            h_pos: [N, D] rappresentazioni post-SLiM dei campioni truthful
            h_neg: [N, D] rappresentazioni post-SLiM dei campioni hallucinated

        Returns:
            Scalar loss value (L_pos + L_neg)
        """
        # L2-normalize per usare cosine similarity
        h_pos = F.normalize(h_pos, p=2, dim=1)  # [N, D]
        h_neg = F.normalize(h_neg, p=2, dim=1)  # [N, D]

        # Matrici di similarità (divise per τ)
        sim_pp = torch.mm(h_pos, h_pos.T) / self.temperature  # [N, N]
        sim_pn = torch.mm(h_pos, h_neg.T) / self.temperature  # [N, N]
        sim_nn = torch.mm(h_neg, h_neg.T) / self.temperature  # [N, N]
        sim_np = torch.mm(h_neg, h_pos.T) / self.temperature  # [N, N]

        # Parte 1: CTR(h_pos, H_pos, H_neg)
        # Per ogni h_pos[i]: S+ = tutti h_pos (incluso i), S- = tutti h_neg
        log_num_1 = torch.logsumexp(sim_pp, dim=1)  # [N]
        log_den_1 = torch.logsumexp(
            torch.cat([sim_pp, sim_pn], dim=1), dim=1
        )  # [N]
        loss_pos = -(log_num_1 - log_den_1).mean()

        # Parte 2: CTR(h_neg, H_neg, H_pos)
        # Per ogni h_neg[i]: S+ = tutti h_neg (incluso i), S- = tutti h_pos
        log_num_2 = torch.logsumexp(sim_nn, dim=1)  # [N]
        log_den_2 = torch.logsumexp(
            torch.cat([sim_nn, sim_np], dim=1), dim=1
        )  # [N]
        loss_neg = -(log_num_2 - log_den_2).mean()

        return loss_pos + loss_neg


def save_slim_checkpoint(
    model: GeneralSLiMedNet,
    tokenizer,
    epoch: int,
    loss: float,
    perplexity: float,
    save_path: str,
    training_time: float = 0.0,
    args_dict: dict = None,
    final: bool = False,
):
    """
    Salva un checkpoint SLiM.

    Include solo i pesi dei moduli SLiM (scale, shift, state_proj),
    NON l'intero modello base.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Raccogli solo i parametri SLiM (non il base model)
    slim_state_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            slim_state_dict[name] = param.data.cpu()

    checkpoint = {
        "epoch": epoch,
        "slim_state_dict": slim_state_dict,
        "loss": loss,
        "perplexity": perplexity,
        "training_time_seconds": training_time,
        "args": args_dict or {},
        "arch_name": model.arch_name,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "n_slim_layers": model.n_slim_layers,
        "apply_SLiM_at_layers": model.apply_SLiM_at_layers,
        "target_layer": model.target_layer,
    }

    torch.save(checkpoint, save_path)

    tag = "FINAL" if final else f"epoch_{epoch}"
    print(f"[SLiM] Checkpoint salvato ({tag}): {save_path}")
    print(f"       Loss: {loss:.4f} | Perplexity: {perplexity:.2f}")


def train_slim(
    model: GeneralSLiMedNet,
    dataloader: DataLoader,
    criterion: SLiMContrastiveLoss,
    optimizer: optim.Optimizer,
    scheduler=None,
    device: str = "cuda",
    accumulation_steps: int = 8,
    epochs: int = 3,
    save_path: str = None,
    tokenizer=None,
    args_dict: dict = None,
    verbose: bool = True,
    val_dataloader: DataLoader = None,
    patience: int = 10,
    min_delta: float = 1e-4,
):
    """
    Loop di training contrastivo per SLiM.

    Per ogni batch di coppie (pos, neg):
    1. Forward pass dei campioni pos con state=1.0 → cattura hidden state
    2. Forward pass dei campioni neg con state=1.0 → cattura hidden state
    3. Estrae last-token representation da entrambi
    4. Calcola InfoNCE loss che separa truthful da hallucinated
    5. Backward attraverso entrambi i rami (shared SLiM parameters)

    Il capture mode del hook salva le rappresentazioni con gradients e
    detach l'output per i layer successivi (memory optimization).

    Args:
        model: GeneralSLiMedNet (con base model frozen)
        dataloader: DataLoader paired (pos_ids, pos_mask, neg_ids, neg_mask)
        criterion: SLiMContrastiveLoss
        optimizer: Ottimizzatore (AdamW)
        scheduler: Learning rate scheduler (opzionale)
        device: Device target
        accumulation_steps: Passi di accumulo gradiente
        epochs: Numero di epoche massimo
        save_path: Percorso base per il salvataggio
        tokenizer: Tokenizer (per eventuali valutazioni)
        args_dict: Dizionario argomenti per il checkpoint
        verbose: Se stampare progresso
        val_dataloader: DataLoader di validazione paired (opzionale)
        patience: Epoche senza miglioramento prima di early stopping
        min_delta: Miglioramento minimo considerato significativo
    """
    device_type = "cuda" if "cuda" in device else "cpu"

    model.train()
    model.set_capture_mode(True)

    step = 0
    total_loss = 0
    training_start = time.time()

    # Early stopping state
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state_dict = None
    best_epoch = None
    stopped_early = False

    epoch_bar = tqdm(range(epochs), desc="Epoche", unit="ep", position=0)

    for epoch in epoch_bar:
        # ── TRAINING ──────────────────────────────────────────────────────────
        model.train()
        model.set_capture_mode(True)
        running_loss = 0
        n_batches = 0
        optimizer.zero_grad()

        batch_bar = tqdm(
            dataloader,
            desc=f"Train {epoch + 1}/{epochs}",
            unit="batch",
            position=1,
            leave=False,
        )

        for i, (pos_ids, pos_masks, neg_ids, neg_masks) in enumerate(batch_bar):
            pos_ids = pos_ids.to(device)
            pos_masks = pos_masks.to(device)
            neg_ids = neg_ids.to(device)
            neg_masks = neg_masks.to(device)

            batch_size = pos_ids.size(0)
            # state=1.0 per entrambi: stessa trasformazione FiLM
            state = torch.ones(batch_size, 1, device=device)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                # Forward pos: hook cattura hidden state con gradients
                _ = model(
                    input_ids=pos_ids,
                    state_tensor=state,
                    attention_mask=pos_masks,
                )
                h_pos_full = model.get_captured_representation()  # [N, S_pos, H]
                h_pos = GeneralSLiMedNet.extract_last_token(
                    h_pos_full, pos_masks
                )  # [N, H]

                # Forward neg: hook cattura nuovo hidden state
                _ = model(
                    input_ids=neg_ids,
                    state_tensor=state,
                    attention_mask=neg_masks,
                )
                h_neg_full = model.get_captured_representation()  # [N, S_neg, H]
                h_neg = GeneralSLiMedNet.extract_last_token(
                    h_neg_full, neg_masks
                )  # [N, H]

                # Contrastive loss (InfoNCE)
                loss = criterion(h_pos, h_neg)
                loss = loss / accumulation_steps

            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                step += 1
                total_loss += loss.item() * accumulation_steps

            running_loss += loss.item() * accumulation_steps
            n_batches += 1

            avg_so_far = running_loss / n_batches
            batch_bar.set_postfix(
                loss=f"{loss.item() * accumulation_steps:.4f}",
                avg=f"{avg_so_far:.4f}",
            )

        batch_bar.close()

        epoch_loss = running_loss / max(n_batches, 1)
        training_time = time.time() - training_start

        # ── VALIDATION ────────────────────────────────────────────────────────
        val_loss = None
        if val_dataloader is not None:
            model.eval()
            model.set_capture_mode(True)
            val_running_loss = 0
            val_n_batches = 0

            val_bar = tqdm(
                val_dataloader,
                desc=f"Val   {epoch + 1}/{epochs}",
                unit="batch",
                position=1,
                leave=False,
            )

            with torch.no_grad():
                for pos_ids, pos_masks, neg_ids, neg_masks in val_bar:
                    pos_ids = pos_ids.to(device)
                    pos_masks = pos_masks.to(device)
                    neg_ids = neg_ids.to(device)
                    neg_masks = neg_masks.to(device)

                    batch_size = pos_ids.size(0)
                    state = torch.ones(batch_size, 1, device=device)

                    with torch.autocast(
                        device_type=device_type, dtype=torch.bfloat16
                    ):
                        _ = model(
                            input_ids=pos_ids,
                            state_tensor=state,
                            attention_mask=pos_masks,
                        )
                        h_pos_full = model.get_captured_representation()
                        h_pos = GeneralSLiMedNet.extract_last_token(
                            h_pos_full, pos_masks
                        )

                        _ = model(
                            input_ids=neg_ids,
                            state_tensor=state,
                            attention_mask=neg_masks,
                        )
                        h_neg_full = model.get_captured_representation()
                        h_neg = GeneralSLiMedNet.extract_last_token(
                            h_neg_full, neg_masks
                        )

                        v_loss = criterion(h_pos, h_neg)

                    val_running_loss += v_loss.item()
                    val_n_batches += 1
                    val_bar.set_postfix(val_loss=f"{v_loss.item():.4f}")

            val_bar.close()
            val_loss = val_running_loss / max(val_n_batches, 1)

            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_state_dict = {
                    k: v.clone()
                    for k, v in model.state_dict().items()
                    if any(
                        p.requires_grad
                        for n, p in model.named_parameters()
                        if n == k
                    )
                }
                # Salva il miglior checkpoint
                if save_path:
                    best_path = save_path.replace(".pth", "_best.pth")
                    save_slim_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        epoch=epoch + 1,
                        loss=val_loss,
                        perplexity=0.0,  # Non applicabile per contrastive
                        save_path=best_path,
                        training_time=training_time,
                        args_dict=args_dict,
                        final=False,
                    )
                best_epoch = epoch + 1
            else:
                epochs_no_improve += 1

        # ── AGGIORNA BARRA EPOCHE ─────────────────────────────────────────────
        postfix = {
            "tr_loss": f"{epoch_loss:.4f}",
            "t": f"{training_time:.0f}s",
        }
        if val_loss is not None:
            postfix["val_loss"] = f"{val_loss:.4f}"
            postfix["no_imp"] = epochs_no_improve
        epoch_bar.set_postfix(**postfix)

        if verbose:
            msg = f"\n  Epoca {epoch+1} | tr_loss: {epoch_loss:.4f}"
            if val_loss is not None:
                msg += f" | val_loss: {val_loss:.4f}"
                if epochs_no_improve == 0:
                    msg += " ✓ best"
                else:
                    msg += f" (no imp {epochs_no_improve}/{patience})"
            msg += f" | tempo: {training_time:.1f}s"
            tqdm.write(msg)

        # ── EARLY STOPPING ────────────────────────────────────────────────────
        if val_dataloader is not None and epochs_no_improve >= patience:
            tqdm.write(
                f"\n[SLiM] Early stopping a epoca {epoch+1} "
                f"(nessun miglioramento per {patience} epoche). "
                f"Best val_loss: {best_val_loss:.4f} @ epoca {best_epoch}"
            )
            stopped_early = True
            break

    # Ripristina i pesi migliori se disponibili
    if best_state_dict is not None:
        slim_state = {k: v for k, v in model.state_dict().items()}
        slim_state.update(best_state_dict)
        model.load_state_dict(slim_state, strict=False)
        tqdm.write(
            f"[SLiM] Pesi migliori ripristinati "
            f"(epoca {best_epoch}, val_loss={best_val_loss:.4f})"
        )

    final_loss = total_loss / max(step, 1)
    total_time = time.time() - training_start

    # Fallback: se non c'è validation, salva un unico checkpoint finale come best.
    if val_dataloader is None and save_path:
        best_path = save_path.replace(".pth", "_best.pth")
        save_slim_checkpoint(
            model=model,
            tokenizer=tokenizer,
            epoch=epochs,
            loss=final_loss,
            perplexity=0.0,
            save_path=best_path,
            training_time=total_time,
            args_dict=args_dict,
            final=True,
        )

    # Disabilita capture mode
    model.set_capture_mode(False)

    tqdm.write("\n[SLiM] Training contrastivo completato!")
    tqdm.write(f"  Contrastive loss finale (train): {final_loss:.4f}")
    if best_val_loss != float("inf"):
        tqdm.write(
            f"  Miglior val_loss:    {best_val_loss:.4f} (epoca {best_epoch})"
        )
    tqdm.write(f"  Tempo totale:        {total_time:.1f}s")
    if stopped_early:
        tqdm.write(f"  Early stopping dopo {best_epoch} epoche utili.")

    return {
        "final_loss": final_loss,
        "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "training_time_seconds": total_time,
        "total_steps": step,
    }


# =============================================================================
# GENERATIVE TRAINING (Cross-Entropy Loss)
# =============================================================================


def train_slim_generative(
    model: GeneralSLiMedNet,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler=None,
    device: str = "cuda",
    accumulation_steps: int = 8,
    epochs: int = 3,
    save_path: str = None,
    tokenizer=None,
    args_dict: dict = None,
    verbose: bool = True,
    val_dataloader: DataLoader = None,
    patience: int = 10,
    min_delta: float = 1e-4,
):
    """
    Generative training loop for SLiM with cross-entropy loss.

    Unlike contrastive training which only separates representations,
    this directly optimizes SLiM to produce better next-token predictions.

    For each sample (input_ids, target_ids, attention_mask, state):
    1. state=1.0 for truthful samples, state=0.0 for hallucinated
    2. Forward pass through full model (hook applies FiLM at target layer)
    3. CE loss on output logits vs target tokens
    4. Gradients flow: logits → frozen layers → steered_output → SLiM params

    At inference with state=1.0, the learned FiLM transform steers toward
    truthful token predictions.
    """
    device_type = "cuda" if "cuda" in device else "cpu"

    model.train()
    # No capture mode: gradients must flow through subsequent frozen layers
    # to reach the logits (via steered_output → frozen layers → lm_head)
    model.set_capture_mode(False)

    step = 0
    total_loss = 0
    training_start = time.time()

    # Early stopping state
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state_dict = None
    best_epoch = None
    stopped_early = False

    epoch_bar = tqdm(range(epochs), desc="Epoche", unit="ep", position=0)

    for epoch in epoch_bar:
        # ── TRAINING ──────────────────────────────────────────────────────────
        model.train()
        model.set_capture_mode(False)
        running_loss = 0
        n_batches = 0
        optimizer.zero_grad()

        batch_bar = tqdm(
            dataloader,
            desc=f"Train {epoch + 1}/{epochs}",
            unit="batch",
            position=1,
            leave=False,
        )

        for i, (input_ids, target_ids, attention_mask, states) in enumerate(batch_bar):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            attention_mask = attention_mask.to(device)
            states = states.to(device)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                # Forward: hook applies state-conditioned FiLM modulation
                logits = model(
                    input_ids=input_ids,
                    state_tensor=states,
                    attention_mask=attention_mask,
                )
                # Cross-entropy loss on next-token prediction
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                    ignore_index=-100,
                )
                loss = loss / accumulation_steps

            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                step += 1
                total_loss += loss.item() * accumulation_steps

            running_loss += loss.item() * accumulation_steps
            n_batches += 1

            avg_so_far = running_loss / n_batches
            ppl_so_far = torch.exp(torch.tensor(avg_so_far)).item()
            batch_bar.set_postfix(
                loss=f"{loss.item() * accumulation_steps:.4f}",
                avg=f"{avg_so_far:.4f}",
                ppl=f"{ppl_so_far:.2f}",
            )

        batch_bar.close()

        epoch_loss = running_loss / max(n_batches, 1)
        epoch_ppl = torch.exp(torch.tensor(epoch_loss)).item()
        training_time = time.time() - training_start

        # ── VALIDATION ────────────────────────────────────────────────────────
        val_loss = None
        if val_dataloader is not None:
            model.eval()
            model.set_capture_mode(False)
            val_running_loss = 0
            val_n_batches = 0

            val_bar = tqdm(
                val_dataloader,
                desc=f"Val   {epoch + 1}/{epochs}",
                unit="batch",
                position=1,
                leave=False,
            )

            with torch.no_grad():
                for input_ids, target_ids, attention_mask, states in val_bar:
                    input_ids = input_ids.to(device)
                    target_ids = target_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    states = states.to(device)

                    with torch.autocast(
                        device_type=device_type, dtype=torch.bfloat16
                    ):
                        logits = model(
                            input_ids=input_ids,
                            state_tensor=states,
                            attention_mask=attention_mask,
                        )
                        v_loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            target_ids.reshape(-1),
                            ignore_index=-100,
                        )

                    val_running_loss += v_loss.item()
                    val_n_batches += 1
                    val_bar.set_postfix(val_loss=f"{v_loss.item():.4f}")

            val_bar.close()
            val_loss = val_running_loss / max(val_n_batches, 1)

            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_state_dict = {
                    k: v.clone()
                    for k, v in model.state_dict().items()
                    if any(
                        p.requires_grad
                        for n, p in model.named_parameters()
                        if n == k
                    )
                }
                if save_path:
                    best_path = save_path.replace(".pth", "_best.pth")
                    save_slim_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        epoch=epoch + 1,
                        loss=val_loss,
                        perplexity=torch.exp(torch.tensor(val_loss)).item(),
                        save_path=best_path,
                        training_time=training_time,
                        args_dict=args_dict,
                        final=False,
                    )
                best_epoch = epoch + 1
            else:
                epochs_no_improve += 1

        # ── AGGIORNA BARRA EPOCHE ─────────────────────────────────────────────
        postfix = {
            "tr_loss": f"{epoch_loss:.4f}",
            "ppl": f"{epoch_ppl:.2f}",
            "t": f"{training_time:.0f}s",
        }
        if val_loss is not None:
            postfix["val_loss"] = f"{val_loss:.4f}"
            postfix["no_imp"] = epochs_no_improve
        epoch_bar.set_postfix(**postfix)

        if verbose:
            msg = f"\n  Epoca {epoch+1} | tr_loss: {epoch_loss:.4f} | ppl: {epoch_ppl:.2f}"
            if val_loss is not None:
                val_ppl = torch.exp(torch.tensor(val_loss)).item()
                msg += f" | val_loss: {val_loss:.4f} | val_ppl: {val_ppl:.2f}"
                if epochs_no_improve == 0:
                    msg += " ✓ best"
                else:
                    msg += f" (no imp {epochs_no_improve}/{patience})"
            msg += f" | tempo: {training_time:.1f}s"
            tqdm.write(msg)

        # ── EARLY STOPPING ────────────────────────────────────────────────────
        if val_dataloader is not None and epochs_no_improve >= patience:
            tqdm.write(
                f"\n[SLiM] Early stopping a epoca {epoch+1} "
                f"(nessun miglioramento per {patience} epoche). "
                f"Best val_loss: {best_val_loss:.4f} @ epoca {best_epoch}"
            )
            stopped_early = True
            break

    # Ripristina i pesi migliori se disponibili
    if best_state_dict is not None:
        slim_state = {k: v for k, v in model.state_dict().items()}
        slim_state.update(best_state_dict)
        model.load_state_dict(slim_state, strict=False)
        tqdm.write(
            f"[SLiM] Pesi migliori ripristinati "
            f"(epoca {best_epoch}, val_loss={best_val_loss:.4f})"
        )

    final_loss = total_loss / max(step, 1)
    total_time = time.time() - training_start

    # Fallback: se non c'è validation, salva checkpoint finale come best
    if val_dataloader is None and save_path:
        best_path = save_path.replace(".pth", "_best.pth")
        save_slim_checkpoint(
            model=model,
            tokenizer=tokenizer,
            epoch=epochs,
            loss=final_loss,
            perplexity=torch.exp(torch.tensor(final_loss)).item(),
            save_path=best_path,
            training_time=total_time,
            args_dict=args_dict,
            final=True,
        )

    tqdm.write("\n[SLiM] Training generativo completato!")
    tqdm.write(f"  CE loss finale (train):  {final_loss:.4f}")
    tqdm.write(f"  Perplexity finale:       {torch.exp(torch.tensor(final_loss)).item():.2f}")
    if best_val_loss != float("inf"):
        tqdm.write(
            f"  Miglior val_loss:        {best_val_loss:.4f} (epoca {best_epoch})"
        )
    tqdm.write(f"  Tempo totale:            {total_time:.1f}s")
    if stopped_early:
        tqdm.write(f"  Early stopping dopo {best_epoch} epoche utili.")

    return {
        "final_loss": final_loss,
        "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "training_time_seconds": total_time,
        "total_steps": step,
    }


def get_save_dir(project_root: str, model_name: str, dataset_name: str) -> str:
    """Genera il percorso di salvataggio per i checkpoint SLiM."""
    model_safe = model_name.replace("/", "_")
    return os.path.join(
        project_root, "SteeringVectors", "SLiM", model_safe, dataset_name
    )


def main():
    parser = argparse.ArgumentParser(
        description="SLiM Hallucination Reduction — Contrastive Training"
    )

    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Nome modello HuggingFace (es. google/gemma-2-9b-it)"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["belief_bank_facts", "belief_bank_constraints", "halu_eval"],
        help="Dataset di training"
    )
    parser.add_argument("--num_pairs", type=int, default=2000, help="Numero di coppie")
    parser.add_argument("--epochs", type=int, default=3, help="Numero di epoche")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (coppie per batch)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=0, help="Lunghezza massima sequenza (0 = no troncamento)")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Passi di warmup")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--project_root", type=str, default=".", help="Root del progetto")
    parser.add_argument("--use_local_halueval", action="store_true", help="Usa HaluEval locale")
    parser.add_argument("--state_dim", type=int, default=1, help="Dimensione stato (1 = scalare)")
    parser.add_argument("--val_split", type=float, default=0.2, help="Frazione dataset usata per validation")
    parser.add_argument("--patience", type=int, default=10, help="Epoche senza miglioramento prima di early stopping")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Miglioramento minimo val_loss")
    parser.add_argument("--target_layer", type=int, required=True, help="Indice del layer transformer su cui applicare SLiM")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature τ per InfoNCE (solo contrastive)")
    parser.add_argument("--loss_type", type=str, default="generative",
                        choices=["contrastive", "generative"],
                        help="Tipo di loss: 'generative' (CE, default) o 'contrastive' (InfoNCE)")

    args = parser.parse_args()
    project_root = os.path.abspath(args.project_root)

    loss_label = "Cross-Entropy (generative)" if args.loss_type == "generative" else f"InfoNCE τ={args.temperature}"
    print(f"\n{'='*60}")
    print(f"  SLiM Hallucination Reduction — {args.loss_type.capitalize()} Training")
    print(f"{'='*60}")
    print(f"  Modello:      {args.model_name}")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Coppie:       {args.num_pairs}")
    print(f"  Epoche:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  LR:           {args.lr}")
    print(f"  Device:       {args.device}")
    print(f"  Target layer: {args.target_layer}")
    print(f"  Loss:         {loss_label}")
    print(f"{'='*60}\n")

    # 1. Carica tokenizer
    print("[1/5] Caricamento tokenizer...")
    tokenizer = load_tokenizer(args.model_name)

    # 2. Carica modello base con quantizzazione 4-bit
    print("[2/5] Caricamento modello con quantizzazione 4-bit...")
    bnb_config = create_bnb_config()
    base_model = load_llm(
        args.model_name,
        bnb_config,
        device=args.device,
    )
    base_model = prepare_model_for_kbit_training(base_model)

    # Freeze tutti i parametri del modello base
    for param in base_model.parameters():
        param.requires_grad = False

    # 3. Crea GeneralSLiMedNet
    print("[3/5] Creazione GeneralSLiMedNet...")
    slim_model = GeneralSLiMedNet(
        model=base_model,
        state_embed_dim=args.state_dim,
        target_layer=args.target_layer,
        dtype=torch.bfloat16,  # match base model dtype → risparmio ~2.8GB VRAM
    )

    # Sposta i moduli SLiM su device (già in bfloat16 dalla __init__)
    slim_model.state_proj = slim_model.state_proj.to(args.device)
    slim_model.SLiM_scale = slim_model.SLiM_scale.to(args.device)
    slim_model.SLiM_shift = slim_model.SLiM_shift.to(args.device)

    trainable_params = print_trainable_parameters(slim_model)

    # 4. Crea dataset
    use_paired = (args.loss_type == "contrastive")
    mode_label = "paired (contrastive)" if use_paired else "flat (generative)"
    print(f"[4/5] Creazione dataset {mode_label}...")
    dataset = create_slim_dataset(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        project_root=project_root,
        num_pairs=args.num_pairs,
        max_length=args.max_length,
        use_local_halueval=args.use_local_halueval,
        paired=use_paired,
    )

    # Train / Validation split
    val_split = max(0.0, min(args.val_split, 0.9))
    if val_split > 0.0:
        num_val = int(len(dataset) * val_split)
        num_train = len(dataset) - num_val
        train_dataset, val_dataset = random_split(
            dataset,
            [num_train, num_val],
            generator=torch.Generator().manual_seed(42),
        )
        print(
            f"  Split: {num_train} train / {num_val} val "
            f"({int((1-val_split)*100)}% / {int(val_split*100)}%)"
        )
    else:
        train_dataset = dataset
        val_dataset = None
        print("  Nessun validation split (val_split=0.0)")

    # Select collate function based on loss type
    collate_fn = collate_fn_paired if use_paired else collate_fn_hallucination

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )

    # 5. Setup training
    print(f"[5/5] Setup training {args.loss_type}...")
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, slim_model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    total_steps = (len(dataloader) // args.accumulation_steps) * args.epochs
    print(f"  Loss: {loss_label}")
    print(f"  Patience early stopping: {args.patience} epoche (min_delta={args.min_delta})")
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, total_steps // 5),
        num_training_steps=total_steps,
    )

    # Percorso di salvataggio
    save_dir = get_save_dir(project_root, args.model_name, args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    save_filename = (
        f"slim_{args.dataset}_pairs{args.num_pairs}_"
        f"layer{args.target_layer}_"
        f"{args.loss_type}_"
        f"lr{args.lr}_bs{args.batch_size}_ep{args.epochs}.pth"
    )
    save_path = os.path.join(save_dir, save_filename)

    # Dizionario argomenti
    args_dict = vars(args).copy()
    args_dict["trainable_params"] = trainable_params
    args_dict["total_training_pairs"] = len(dataset)
    args_dict["num_train_pairs"] = len(train_dataset)
    args_dict["num_val_pairs"] = len(val_dataset) if val_dataset is not None else 0

    # Salva config di training
    config_path = os.path.join(save_dir, save_filename.replace(".pth", "_config.json"))
    with open(config_path, "w") as f:
        json.dump(args_dict, f, indent=2)
    print(f"Config salvata: {config_path}")

    # Training
    print(f"\nInizio training {args.loss_type} ({total_steps} step stimati)...\n")

    if args.loss_type == "contrastive":
        criterion = SLiMContrastiveLoss(temperature=args.temperature)
        result = train_slim(
            model=slim_model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=args.device,
            accumulation_steps=args.accumulation_steps,
            epochs=args.epochs,
            save_path=save_path,
            tokenizer=tokenizer,
            args_dict=args_dict,
            val_dataloader=val_dataloader,
            patience=args.patience,
            min_delta=args.min_delta,
        )
    else:  # generative
        result = train_slim_generative(
            model=slim_model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=args.device,
            accumulation_steps=args.accumulation_steps,
            epochs=args.epochs,
            save_path=save_path,
            tokenizer=tokenizer,
            args_dict=args_dict,
            val_dataloader=val_dataloader,
            patience=args.patience,
            min_delta=args.min_delta,
        )

    # Salva risultato finale
    result["args"] = args_dict
    result_path = os.path.join(save_dir, save_filename.replace(".pth", "_result.json"))
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nRisultati salvati: {result_path}")

    # Cleanup
    slim_model.remove_hooks()
    del slim_model, base_model
    torch.cuda.empty_cache()

    print("\n[SLiM] Training completato con successo!")


if __name__ == "__main__":
    main()
