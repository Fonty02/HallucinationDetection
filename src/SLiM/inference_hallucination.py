"""
Inference e valutazione per SLiM Hallucination Reduction.

Esegue esperimenti di tipo:
- Baseline: modello senza SLiM
- SLiM: modello con SLiM (same-dataset)
- CrossDataset: modello con SLiM trainato su un dataset, valutato su un altro

Salva i risultati in SLiMExperiments.csv e i dettagli per campione in InferenceResults/.

Uso:
    python -m src.SLiM.inference_hallucination \
        --model_name Qwen/Qwen2.5-7B \
        --dataset belief_bank_facts \
        --slim_checkpoint SteeringVectors/SLiM/.../slim_xxx.pth \
        --top_k 10 \
        --state_value 1.0 \
        --device cuda:0
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
import warnings

import torch
from tqdm import tqdm
from peft import prepare_model_for_kbit_training

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.utils import create_bnb_config, load_llm, load_tokenizer
from src.model.prompts import PROMPT_QA, PROMPT_HALU
from src.SLiM.model_general import GeneralSLiMedNet


# =============================================================================
# CSV Columns
# =============================================================================

CSV_COLUMNS = [
    "experiment_id",
    "type",
    "model",
    "dataset_eval",
    "dataset_train",
    "num_samples_evaluated",
    "num_hallucinations",
    "hallucination_rate",
    "slim_checkpoint",
    "num_pairs_train",
    "state_value",
    "top_k_layers",
    "num_layers_total",
    "gate_values",
    "slim_training_time_seconds",
    "slim_inference_time_seconds",
    "slim_trainable_params",
    "learning_rate",
    "batch_size",
    "num_epochs",
    "highlight",
]


def save_to_csv(csv_path: str, result: dict):
    """Append un risultato al CSV. Crea il file con header se non esiste."""
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()

        # Filtra solo le colonne previste
        row = {k: result.get(k, None) for k in CSV_COLUMNS}
        writer.writerow(row)

    print(f"[CSV] Risultato salvato in: {csv_path}")


def load_dataset_for_eval(dataset_name: str, project_root: str, num_samples: int = -1):
    """
    Carica il dataset per la valutazione (non paired, tutto il dataset).

    Returns:
        dataset: Dataset object con __getitem__ che ritorna (fact, label, instance_id)
        valid_indices: Lista di indici da valutare
        is_halu_eval: Bool, True se HaluEval
    """
    is_halu_eval = (dataset_name == "halu_eval")

    if is_halu_eval:
        from src.data.HaluEvalDataset import HaluEvalDataset
        # label=0 = risposte corrette (per valutare se il modello genera la corretta)
        dataset = HaluEvalDataset(label=0, use_local=False)
    elif dataset_name in ("belief_bank_facts", "belief_bank_constraints"):
        from src.data.BeliefBankDataset import BeliefBankDataset
        data_type = dataset_name.replace("belief_bank_", "")
        dataset = BeliefBankDataset(
            project_root=project_root,
            model_type="demo",
            recreate_ids=True,
            data_type=data_type,
        )
    else:
        raise ValueError(f"Dataset sconosciuto: {dataset_name}")

    total = len(dataset)
    if num_samples > 0:
        valid_indices = list(range(min(num_samples, total)))
    else:
        valid_indices = list(range(total))

    return dataset, valid_indices, is_halu_eval


def evaluate_hallucination(
    generated_text: str,
    expected_label: str,
    is_halu_eval: bool,
) -> bool:
    """
    Determina se una risposta generata è un'allucinazione.

    Per BeliefBank: verifica se "yes"/"no" è contenuto nella risposta.
    Per HaluEval: contenimento bidirezionale tra risposta generata e attesa.

    Returns:
        True se è un'allucinazione
    """
    gen_lower = generated_text.lower().strip()
    exp_lower = expected_label.lower().strip()

    if is_halu_eval:
        return (exp_lower not in gen_lower) and (gen_lower not in exp_lower)
    else:
        return exp_lower not in gen_lower


def run_inference(
    experiment_id: str,
    experiment_type: str,
    model_name: str,
    dataset_name: str,
    dataset_train: str,
    slim_checkpoint: str,
    state_value: float,
    top_k: int,
    num_samples: int,
    project_root: str,
    device: str,
    output_csv: str,
):
    """
    Esegue un singolo esperimento di inferenza.

    Args:
        experiment_id: ID univoco dell'esperimento
        experiment_type: "Baseline", "SLiM", o "CrossDataset"
        model_name: Nome modello HuggingFace
        dataset_name: Dataset di valutazione
        dataset_train: Dataset di training (per SLiM/CrossDataset)
        slim_checkpoint: Path al checkpoint SLiM (None per baseline)
        state_value: Valore dello stato (1.0 = truthful steering)
        top_k: Numero di layer Top-K (-1 = tutti)
        num_samples: Numero di campioni da valutare (-1 = tutti)
        project_root: Root del progetto
        device: Device target
        output_csv: Path al CSV di output

    Returns:
        Dict con i risultati
    """
    print(f"\n{'='*60}")
    print(f"  Esperimento: {experiment_id}")
    print(f"  Tipo: {experiment_type}")
    print(f"  Modello: {model_name}")
    print(f"  Dataset eval: {dataset_name}")
    print(f"  Dataset train: {dataset_train}")
    print(f"  Checkpoint: {slim_checkpoint}")
    print(f"  Top-K: {top_k}")
    print(f"  State: {state_value}")
    print(f"{'='*60}\n")

    # 1. Carica tokenizer e modello
    print("[1/4] Caricamento modello...")
    tokenizer = load_tokenizer(model_name)
    bnb_config = create_bnb_config()
    base_model = load_llm(model_name, bnb_config, device=device)
    base_model = prepare_model_for_kbit_training(base_model)

    for param in base_model.parameters():
        param.requires_grad = False

    # 2. Setup SLiM (se non baseline)
    slim_model = None
    active_layers = None
    gate_values_str = None
    checkpoint_data = {}
    training_time = None
    trainable_params = None
    lr = None
    batch_size = None
    num_epochs_train = None
    num_pairs_train = None

    use_slim = experiment_type != "Baseline" and slim_checkpoint is not None

    if use_slim:
        print("[2/4] Caricamento SLiM...")

        # Carica checkpoint
        checkpoint_data = torch.load(slim_checkpoint, map_location="cpu")
        slim_args = checkpoint_data.get("args", {})

        training_time = checkpoint_data.get("training_time_seconds")
        trainable_params = slim_args.get("trainable_params")
        lr = slim_args.get("lr")
        batch_size = slim_args.get("batch_size")
        num_epochs_train = slim_args.get("epochs")
        num_pairs_train = slim_args.get("num_pairs")

        state_dim = slim_args.get("state_dim", 1)
        slim_rank = checkpoint_data.get("slim_rank", slim_args.get("slim_rank", 64))

        # Crea SLiMedNet e carica pesi
        slim_model = GeneralSLiMedNet(
            model=base_model,
            state_embed_dim=state_dim,
            slim_rank=slim_rank,
        )

        # Carica solo i pesi SLiM
        slim_state_dict = checkpoint_data.get("slim_state_dict", {})
        missing, unexpected = slim_model.load_state_dict(slim_state_dict, strict=False)

        # Sposta moduli SLiM su device
        slim_model.state_proj = slim_model.state_proj.to(device)
        slim_model.gate = slim_model.gate.to(device)
        slim_model.SLiM_scale = slim_model.SLiM_scale.to(device)
        slim_model.SLiM_shift = slim_model.SLiM_shift.to(device)

        print(f"  Pesi SLiM caricati. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

        # Calcola gate values e Top-K
        state_tensor = torch.FloatTensor([state_value]).to(device)
        gate_dict = slim_model.get_gate_values(state_tensor)
        gate_values_str = json.dumps(
            {str(k): round(v, 4) for k, v in gate_dict.items()}
        )

        ranking = slim_model.get_layer_ranking(state_tensor)
        print("\n  Gate values (top 5):")
        for layer_idx, gv in ranking[:5]:
            print(f"    Layer {layer_idx}: {gv:.4f}")

        # Seleziona Top-K layers
        if top_k > 0:
            active_layers = slim_model.get_top_k_layers(state_tensor, top_k)
            active_physical = [slim_model.apply_SLiM_at_layers[i] for i in active_layers]
            print(f"\n  Top-{top_k} layers attivi (fisici): {active_physical}")
        else:
            print(f"\n  Tutti i {slim_model.n_slim_layers} layers attivi")

        slim_model.eval()
    else:
        print("[2/4] Baseline - nessun SLiM")

    # 3. Carica dataset per valutazione
    print("[3/4] Caricamento dataset di valutazione...")
    dataset, valid_indices, is_halu_eval = load_dataset_for_eval(
        dataset_name, project_root, num_samples
    )

    if is_halu_eval:
        prompt_template = PROMPT_HALU
        max_new_tokens = 100
        print(f"  Modalità HaluEval ({len(valid_indices)} campioni)")
    else:
        prompt_template = PROMPT_QA
        max_new_tokens = 5
        print(f"  Modalità BeliefBank ({len(valid_indices)} campioni)")

    # 4. Loop di inferenza
    print("[4/4] Inferenza...")
    n_hallucinations = 0
    total_evaluated = 0
    inference_results = []

    # Prepara il tensore di stato
    state_tensor_batch = None
    if use_slim:
        state_tensor_batch = torch.FloatTensor([[state_value]]).to(device)

    inference_start_time = time.time()

    for idx in tqdm(valid_indices, desc=f"{experiment_id}"):
        fact, label, instance_id = dataset[idx]
        question = prompt_template.format(question=fact)

        inputs = tokenizer(question, return_tensors="pt").to(device)

        with torch.no_grad():
            if use_slim and slim_model is not None:
                outputs = slim_model.generate(
                    input_ids=inputs["input_ids"],
                    state_tensor=state_tensor_batch,
                    attention_mask=inputs.get("attention_mask"),
                    active_layers=active_layers,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            else:
                outputs = base_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        is_hallucinated = evaluate_hallucination(generated_text, label, is_halu_eval)

        if is_hallucinated:
            n_hallucinations += 1
        total_evaluated += 1

        # Debug primi 5 campioni
        if total_evaluated <= 5:
            print(f"\n{'─'*60}")
            print(f"  [{total_evaluated}/5] Prompt: {fact[:100]}...")
            print(f"  Generato: {generated_text}")
            print(f"  Atteso:   {label}")
            print(f"  Allucinazione: {'SÌ ❌' if is_hallucinated else 'NO ✅'}")
            print(f"{'─'*60}")

        inference_results.append({
            "instance_id": instance_id,
            "prompt": fact,
            "generated_answer": generated_text,
            "expected_answer": label,
            "is_hallucination": int(is_hallucinated),
        })

    inference_time = time.time() - inference_start_time
    rate = n_hallucinations / total_evaluated if total_evaluated > 0 else 0

    print(f"\n{'='*40}")
    print(f"  Risultati: {experiment_id}")
    print(f"  Valutati: {total_evaluated}")
    print(f"  Allucinazioni: {n_hallucinations} ({rate*100:.2f}%)")
    print(f"  Tempo: {inference_time:.2f}s")
    print(f"{'='*40}")

    # Salva dettagli per campione in JSON
    results_dir = os.path.join(project_root, "InferenceResults", "SLiM")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{experiment_id}_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(inference_results, f, indent=2, ensure_ascii=False)
    print(f"Dettagli salvati: {results_path}")

    # Costruisci result dict
    result = {
        "experiment_id": experiment_id,
        "type": experiment_type,
        "model": model_name,
        "dataset_eval": dataset_name,
        "dataset_train": dataset_train,
        "num_samples_evaluated": total_evaluated,
        "num_hallucinations": n_hallucinations,
        "hallucination_rate": rate,
        "slim_checkpoint": slim_checkpoint,
        "num_pairs_train": num_pairs_train,
        "state_value": state_value,
        "top_k_layers": top_k,
        "num_layers_total": slim_model.num_layers if slim_model else None,
        "gate_values": gate_values_str,
        "slim_training_time_seconds": training_time,
        "slim_inference_time_seconds": inference_time,
        "slim_trainable_params": trainable_params,
        "learning_rate": lr,
        "batch_size": batch_size,
        "num_epochs": num_epochs_train,
        "highlight": None,
    }

    # Salva nel CSV
    save_to_csv(output_csv, result)

    # Cleanup
    if slim_model:
        slim_model.remove_hooks()
        del slim_model
    del base_model
    torch.cuda.empty_cache()

    return result


# =============================================================================
# EXPERIMENTS DEFINITION
# =============================================================================

def build_experiments(
    models: list,
    datasets: list,
    checkpoint_dir: str,
    top_k_values: list = [-1],
    state_value: float = 1.0,
) -> dict:
    """
    Costruisce il dizionario esperimenti per tutti i modelli e dataset.

    Genera automaticamente:
    1. Baseline per ogni (model, dataset)
    2. SLiM per ogni (model, dataset, top_k) — same dataset train/eval
    3. CrossDataset per ogni (model, dataset_train ≠ dataset_eval, top_k)

    Args:
        models: Lista nomi modelli
        datasets: Lista nomi dataset
        checkpoint_dir: Directory base dei checkpoint SLiM
        top_k_values: Lista di valori Top-K da testare
        state_value: Valore dello stato per il steering

    Returns:
        Dict {experiment_id: config_dict}
    """
    experiments = {}

    for model_name in models:
        model_safe = model_name.replace("/", "_")

        for ds in datasets:
            # --- Baseline ---
            exp_id = f"Baseline_{model_safe}_{ds}"
            experiments[exp_id] = {
                "type": "Baseline",
                "model": model_name,
                "dataset_eval": ds,
                "dataset_train": None,
                "slim_checkpoint": None,
                "top_k": -1,
                "state_value": state_value,
            }

            # --- SLiM (same dataset) ---
            for top_k in top_k_values:
                # Cerca il checkpoint più recente
                ckpt_dir = os.path.join(checkpoint_dir, model_safe, ds)
                ckpt = find_latest_checkpoint(ckpt_dir)

                if ckpt:
                    tk_str = f"top{top_k}" if top_k > 0 else "allLayers"
                    exp_id = f"SLiM_{model_safe}_{ds}_{tk_str}"
                    experiments[exp_id] = {
                        "type": "SLiM",
                        "model": model_name,
                        "dataset_eval": ds,
                        "dataset_train": ds,
                        "slim_checkpoint": ckpt,
                        "top_k": top_k,
                        "state_value": state_value,
                    }

            # --- CrossDataset ---
            for ds_train in datasets:
                if ds_train == ds:
                    continue

                for top_k in top_k_values:
                    ckpt_dir = os.path.join(checkpoint_dir, model_safe, ds_train)
                    ckpt = find_latest_checkpoint(ckpt_dir)

                    if ckpt:
                        tk_str = f"top{top_k}" if top_k > 0 else "allLayers"
                        exp_id = f"CrossDS_{model_safe}_train{ds_train}_eval{ds}_{tk_str}"
                        experiments[exp_id] = {
                            "type": "CrossDataset",
                            "model": model_name,
                            "dataset_eval": ds,
                            "dataset_train": ds_train,
                            "slim_checkpoint": ckpt,
                            "top_k": top_k,
                            "state_value": state_value,
                        }

    return experiments


def find_latest_checkpoint(directory: str) -> str:
    """Trova il checkpoint più recente (ultima epoca) in una directory."""
    if not os.path.isdir(directory):
        return None

    checkpoints = [
        f for f in os.listdir(directory)
        if f.endswith(".pth") and "epoch" in f
    ]

    if not checkpoints:
        # Fallback: qualsiasi .pth
        checkpoints = [f for f in os.listdir(directory) if f.endswith(".pth")]

    if not checkpoints:
        return None

    # Ordina per numero di epoca (o per data di modifica)
    checkpoints.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)
    return os.path.join(directory, checkpoints[0])


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SLiM Hallucination Reduction - Inference")

    # Modalità singolo esperimento
    parser.add_argument("--model_name", type=str, default=None, help="Modello da valutare")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset di valutazione")
    parser.add_argument("--slim_checkpoint", type=str, default=None, help="Path checkpoint SLiM")
    parser.add_argument("--dataset_train", type=str, default=None, help="Dataset di training (per CrossDataset)")
    parser.add_argument("--top_k", type=int, default=-1, help="Top-K layers (-1 = tutti)")
    parser.add_argument("--state_value", type=float, default=1.0, help="Valore dello stato")
    parser.add_argument("--experiment_type", type=str, default="SLiM",
                        choices=["Baseline", "SLiM", "CrossDataset"])

    # Modalità batch (tutti gli esperimenti)
    parser.add_argument("--run_all", action="store_true", help="Esegui tutti gli esperimenti")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Lista modelli per modalità batch")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Lista dataset per modalità batch")
    parser.add_argument("--top_k_values", nargs="+", type=int, default=[-1],
                        help="Lista valori Top-K per batch")

    # Opzioni comuni
    parser.add_argument("--num_samples", type=int, default=-1, help="Campioni da valutare (-1 = tutti)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--project_root", type=str, default=".", help="Root progetto")
    parser.add_argument("--output_csv", type=str, default="SLiMExperiments.csv", help="File CSV output")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory base checkpoint (default: SteeringVectors/SLiM)")
    parser.add_argument("--failure_log", type=str, default="slim_log.txt", help="Log fallimenti")

    args = parser.parse_args()
    project_root = os.path.abspath(args.project_root)
    output_csv = os.path.join(project_root, args.output_csv) if not os.path.isabs(args.output_csv) else args.output_csv
    failure_log = os.path.join(project_root, args.failure_log) if not os.path.isabs(args.failure_log) else args.failure_log
    checkpoint_dir = args.checkpoint_dir or os.path.join(project_root, "SteeringVectors", "SLiM")

    if args.run_all:
        # ============================
        # Modalità batch
        # ============================
        models = args.models or ["Qwen/Qwen2.5-7B", "tiiuae/Falcon3-7B-Base"]
        datasets = args.datasets or [
            "belief_bank_facts",
            "belief_bank_constraints",
            "halu_eval",
        ]

        experiments = build_experiments(
            models=models,
            datasets=datasets,
            checkpoint_dir=checkpoint_dir,
            top_k_values=args.top_k_values,
            state_value=args.state_value,
        )

        print(f"Trovati {len(experiments)} esperimenti da eseguire.\n")

        completed = 0
        failed = 0

        for exp_id, exp_config in experiments.items():
            try:
                run_inference(
                    experiment_id=exp_id,
                    experiment_type=exp_config["type"],
                    model_name=exp_config["model"],
                    dataset_name=exp_config["dataset_eval"],
                    dataset_train=exp_config.get("dataset_train"),
                    slim_checkpoint=exp_config.get("slim_checkpoint"),
                    state_value=exp_config.get("state_value", 1.0),
                    top_k=exp_config.get("top_k", -1),
                    num_samples=args.num_samples,
                    project_root=project_root,
                    device=args.device,
                    output_csv=output_csv,
                )
                completed += 1
            except Exception as e:
                tb = traceback.format_exc()
                failed += 1
                print(f"\n[ERRORE] {exp_id}: {e}")
                with open(failure_log, "a") as f:
                    f.write(f"\n{'='*60}\n{exp_id}\n{tb}\n")
                continue

        print(f"\nCompletati: {completed} | Falliti: {failed}")

    else:
        # ============================
        # Modalità singolo esperimento
        # ============================
        if not args.model_name or not args.dataset:
            parser.error("--model_name e --dataset sono richiesti in modalità singola")

        model_safe = args.model_name.replace("/", "_")
        ds_train = args.dataset_train or args.dataset
        tk_str = f"top{args.top_k}" if args.top_k > 0 else "allLayers"
        exp_id = f"{args.experiment_type}_{model_safe}_{args.dataset}_{tk_str}"

        run_inference(
            experiment_id=exp_id,
            experiment_type=args.experiment_type,
            model_name=args.model_name,
            dataset_name=args.dataset,
            dataset_train=ds_train,
            slim_checkpoint=args.slim_checkpoint,
            state_value=args.state_value,
            top_k=args.top_k,
            num_samples=args.num_samples,
            project_root=project_root,
            device=args.device,
            output_csv=output_csv,
        )


if __name__ == "__main__":
    main()
