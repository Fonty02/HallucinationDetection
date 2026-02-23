"""
Training script per SLiM Hallucination Reduction.

Addestra il modello GeneralSLiMedNet su dataset paired (BBF, BBC, HaluEval)
con quantizzazione 4-bit BitsAndBytes, mixed precision training,
gradient accumulation, e linear warmup scheduler.

Salva i checkpoint in SteeringVectors/SLiM/.
Uso:
    python -m src.SLiM.train_hallucination \
        --model_name Qwen/Qwen2.5-7B \
        --dataset belief_bank_facts \
        --num_pairs 2000 \
        --epochs 3 \
        --batch_size 2 \
        --lr 5e-4 \
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
    criterion: nn.Module,
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
    Loop di training per SLiM.

    Segue la procedura del paper SLiM (Sezione 3.2):
    - Forward pass con stato condizionante
    - Cross-entropy loss sulle predizioni next-token
    - autocast bfloat16 (no GradScaler: bf16 ha stesso range esponente di fp32)
    - Gradient accumulation ogni `accumulation_steps` passi

    Se `val_dataloader` è fornito, calcola la validation loss a ogni epoca e applica
    early stopping con `patience` epoche senza miglioramento >= `min_delta`.
    Il checkpoint migliore (minima val loss) viene salvato separatamente.

    Args:
        model: GeneralSLiMedNet (con base model frozen)
        dataloader: DataLoader di training con (input_ids, target_ids, attention_mask, state)
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Ottimizzatore (AdamW)
        scheduler: Learning rate scheduler (opzionale)
        device: Device target
        accumulation_steps: Passi di accumulo gradiente
        epochs: Numero di epoche massimo
        save_path: Percorso base per il salvataggio
        tokenizer: Tokenizer (per eventuali valutazioni)
        args_dict: Dizionario argomenti per il checkpoint
        verbose: Se stampare progresso
        val_dataloader: DataLoader di validazione (opzionale)
        patience: Epoche senza miglioramento prima di early stopping
        min_delta: Miglioramento minimo considerato significativo
    """
    device_type = "cuda" if "cuda" in device else "cpu"

    model.train()
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

        for i, (input_ids, target_ids, attention_mask, state) in enumerate(batch_bar):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            attention_mask = attention_mask.to(device)
            state = state.to(device)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits = model(
                    input_ids=input_ids,
                    state_tensor=state,
                    attention_mask=attention_mask,
                )
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                )

            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                step += 1
                total_loss += loss.item()

            running_loss += loss.item()
            n_batches += 1

            avg_so_far = running_loss / n_batches
            batch_bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_so_far:.4f}")

        batch_bar.close()

        epoch_loss = running_loss / max(n_batches, 1)
        epoch_ppx = torch.exp(torch.tensor(epoch_loss)).item()
        training_time = time.time() - training_start

        # ── VALIDATION ────────────────────────────────────────────────────────
        val_loss = None
        val_ppx = None
        if val_dataloader is not None:
            model.eval()
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
                for input_ids, target_ids, attention_mask, state in val_bar:
                    input_ids = input_ids.to(device)
                    target_ids = target_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    state = state.to(device)

                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits = model(
                            input_ids=input_ids,
                            state_tensor=state,
                            attention_mask=attention_mask,
                        )
                        v_loss = criterion(
                            logits.view(-1, logits.size(-1)),
                            target_ids.view(-1),
                        )

                    val_running_loss += v_loss.item()
                    val_n_batches += 1
                    val_bar.set_postfix(val_loss=f"{v_loss.item():.4f}")

            val_bar.close()
            val_loss = val_running_loss / max(val_n_batches, 1)
            val_ppx = torch.exp(torch.tensor(val_loss)).item()

            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_state_dict = {
                    k: v.clone() for k, v in model.state_dict().items()
                    if any(p.requires_grad for n, p in model.named_parameters() if n == k)
                }
                # Salva il miglior checkpoint
                if save_path:
                    best_path = save_path.replace(".pth", "_best.pth")
                    save_slim_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        epoch=epoch + 1,
                        loss=val_loss,
                        perplexity=val_ppx,
                        save_path=best_path,
                        training_time=training_time,
                        args_dict=args_dict,
                        final=False,
                    )
                best_epoch = epoch + 1
            else:
                epochs_no_improve += 1

        # ── AGGIORNA BARRA EPOCHE ─────────────────────────────────────────────
        postfix = {"tr_loss": f"{epoch_loss:.4f}", "ppl": f"{epoch_ppx:.2f}", "t": f"{training_time:.0f}s"}
        if val_loss is not None:
            postfix["val_loss"] = f"{val_loss:.4f}"
            postfix["no_imp"] = epochs_no_improve
        epoch_bar.set_postfix(**postfix)

        if verbose:
            msg = (
                f"\n  Epoca {epoch+1} | tr_loss: {epoch_loss:.4f} | ppl: {epoch_ppx:.2f}"
            )
            if val_loss is not None:
                msg += f" | val_loss: {val_loss:.4f} | val_ppl: {val_ppx:.2f}"
                if epochs_no_improve == 0:
                    msg += " ✓ best"
                else:
                    msg += f" (no imp {epochs_no_improve}/{patience})"
            msg += f" | tempo: {training_time:.1f}s"
            tqdm.write(msg)

        # Checkpoint per-epoch disabilitati: salviamo solo il best.

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
        tqdm.write(f"[SLiM] Pesi migliori ripristinati (epoca {best_epoch}, val_loss={best_val_loss:.4f})")

    final_loss = total_loss / max(step, 1)
    final_ppx = torch.exp(torch.tensor(final_loss)).item()
    total_time = time.time() - training_start

    # Fallback: se non c'è validation, salva un unico checkpoint finale come best.
    if val_dataloader is None and save_path:
        best_path = save_path.replace(".pth", "_best.pth")
        save_slim_checkpoint(
            model=model,
            tokenizer=tokenizer,
            epoch=epochs,
            loss=final_loss,
            perplexity=final_ppx,
            save_path=best_path,
            training_time=total_time,
            args_dict=args_dict,
            final=True,
        )

    tqdm.write("\n[SLiM] Training completato!")
    tqdm.write(f"  Loss finale (train): {final_loss:.4f}")
    tqdm.write(f"  Perplexity finale:   {final_ppx:.2f}")
    if best_val_loss != float('inf'):
        tqdm.write(f"  Miglior val_loss:    {best_val_loss:.4f} (epoca {best_epoch})")
    tqdm.write(f"  Tempo totale:        {total_time:.1f}s")
    if stopped_early:
        tqdm.write(f"  Early stopping dopo {best_epoch} epoche utili.")

    return {
        "final_loss": final_loss,
        "final_perplexity": final_ppx,
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
    parser = argparse.ArgumentParser(description="SLiM Hallucination Reduction Training")

    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Nome modello HuggingFace (es. Qwen/Qwen2.5-7B, tiiuae/Falcon3-7B-Base)"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["belief_bank_facts", "belief_bank_constraints", "halu_eval"],
        help="Dataset di training"
    )
    parser.add_argument("--num_pairs", type=int, default=2000, help="Numero di coppie")
    parser.add_argument("--epochs", type=int, default=3, help="Numero di epoche")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=0, help="Lunghezza massima sequenza (0 = usa lunghezza effettiva dell'input, senza troncamento)")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Passi di warmup")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--project_root", type=str, default=".", help="Root del progetto")
    parser.add_argument("--use_local_halueval", action="store_true", help="Usa HaluEval locale")
    parser.add_argument("--state_dim", type=int, default=1, help="Dimensione stato (1 = scalare)")
    parser.add_argument("--val_split", type=float, default=0.2, help="Frazione dataset usata per validation (0.0 = nessuna validation)")
    parser.add_argument("--patience", type=int, default=10, help="Epoche senza miglioramento prima di early stopping")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Miglioramento minimo val_loss per resettare patience")
    parser.add_argument("--target_layer", type=int, required=True, help="Indice del layer transformer su cui applicare SLiM")

    args = parser.parse_args()
    project_root = os.path.abspath(args.project_root)

    print(f"\n{'='*60}")
    print("  SLiM Hallucination Reduction - Training")
    print(f"{'='*60}")
    print(f"  Modello:      {args.model_name}")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Coppie:       {args.num_pairs}")
    print(f"  Epoche:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  LR:           {args.lr}")
    print(f"  Device:       {args.device}")
    print(f"  Target layer: {args.target_layer}")
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
    print("[4/5] Creazione dataset...")
    dataset = create_slim_dataset(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        project_root=project_root,
        num_pairs=args.num_pairs,
        max_length=args.max_length,
        use_local_halueval=args.use_local_halueval,
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

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_hallucination,
        num_workers=0,
        pin_memory=True,
    )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_hallucination,
            num_workers=0,
            pin_memory=True,
        )

    # 5. Setup training
    print("[5/5] Setup training...")
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, slim_model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    total_steps = (len(dataloader) // args.accumulation_steps) * args.epochs
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
        f"lr{args.lr}_bs{args.batch_size}_ep{args.epochs}.pth"
    )
    save_path = os.path.join(save_dir, save_filename)

    # Dizionario argomenti
    args_dict = vars(args).copy()
    args_dict["trainable_params"] = trainable_params
    args_dict["total_training_samples"] = len(dataset)
    args_dict["num_train_samples"] = len(train_dataset)
    args_dict["num_val_samples"] = len(val_dataset) if val_dataset is not None else 0

    # Salva config di training
    config_path = os.path.join(save_dir, save_filename.replace(".pth", "_config.json"))
    with open(config_path, "w") as f:
        json.dump(args_dict, f, indent=2)
    print(f"Config salvata: {config_path}")

    # Training
    print(f"\nInizio training ({total_steps} step stimati)...\n")

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
