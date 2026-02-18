import os
import json
import torch
import numpy as np
import argparse
import csv
import sys
import re
from tqdm import tqdm
from typing import List, Tuple
import time
import traceback
from glob import glob
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add project root to Python path before importing local modules
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from truthx_model import TruthX, LLMArchitectureDetector  # noqa: E402
from MLP import MLP  # noqa: E402
from src.model.HallucinationDetection import HallucinationDetection  # noqa: E402
from src.data.HaluEvalDataset import HaluEvalDataset  # noqa: E402


# =============================================================================
# MLP ALIGNER WRAPPER (per compatibilità con interfaccia .predict() di sklearn)
# =============================================================================

class MLPAlignerWrapper:
    """
    Wrapper per un MLP aligner salvato come checkpoint PyTorch (.pt),
    che espone l'interfaccia .predict(X_numpy) compatibile con sklearn Ridge.
    """
    def __init__(self, checkpoint_path, device="cpu"):
        ckpt = torch.load(checkpoint_path, map_location=device)
        input_size = ckpt["input_size"]
        output_size = ckpt["output_size"]
        # Infer hidden_size from actual weight shapes instead of trusting metadata,
        # because some checkpoints (alignment_cache_mlp) store a wrong hidden_size.
        state_dict = ckpt["model_state_dict"]
        hidden_size = state_dict["fc1.weight"].shape[0]  # fc1: Linear(input, hidden)
        self.model = MLP(input_size, hidden_size, output_size)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def predict(self, X):
        """Interfaccia compatibile con sklearn: numpy in, numpy out."""
        with torch.no_grad():
            X_t = torch.from_numpy(X).float().to(self.device)
            out = self.model(X_t)
        return out.cpu().numpy()


class ExperimentSkippedError(Exception):
    """Eccezione usata per skippare esperimenti con prerequisiti mancanti."""


def has_bidirectional_directional_aligners(aligner_dir):
    """
    True se nella cartella sono presenti entrambe le direzioni esplicite:
    - target_to_trainer
    - trainer_to_target
    Supporta naming sia 'procrustes_*' sia 'ridge_regressor_*'.
    """
    directional_filenames = {
        "target_to_trainer": [
            "procrustes_target_to_trainer.pkl",
            "ridge_regressor_target_to_trainer.pkl",
        ],
        "trainer_to_target": [
            "procrustes_trainer_to_target.pkl",
            "ridge_regressor_trainer_to_target.pkl",
        ],
    }

    for _, names in directional_filenames.items():
        found = any(os.path.exists(os.path.join(aligner_dir, n)) for n in names)
        if not found:
            return False
    return True


def load_aligner(
    aligner_dir,
    device="cpu",
    preferred_direction="trainer_to_target",
    strict_direction=False
):
    """
    Carica un aligner dalla directory data, rilevando automaticamente
    il tipo di file presente:
      - procrustes_trainer_to_target.pkl / procrustes_target_to_trainer.pkl
      - ridge_regressor_trainer_to_target.pkl / ridge_regressor_target_to_trainer.pkl
      - procrustes.pkl / ridge_regressor.pkl  → legacy sklearn (solo trainer->target)
      - mlp_regressor.pt     → MLP PyTorch (wrapper)
    
    Returns:
        aligner object con metodo .predict(), oppure None se non trovato.
    """
    if preferred_direction not in {"target_to_trainer", "trainer_to_target"}:
        raise ValueError(f"Unsupported preferred_direction: {preferred_direction}")

    directional_paths = {
        "target_to_trainer": [
            os.path.join(aligner_dir, "procrustes_target_to_trainer.pkl"),
            os.path.join(aligner_dir, "ridge_regressor_target_to_trainer.pkl"),
        ],
        "trainer_to_target": [
            os.path.join(aligner_dir, "procrustes_trainer_to_target.pkl"),
            os.path.join(aligner_dir, "ridge_regressor_trainer_to_target.pkl"),
        ],
    }

    legacy_paths = [
        os.path.join(aligner_dir, "procrustes.pkl"),
        os.path.join(aligner_dir, "ridge_regressor.pkl"),
    ]

    mlp_path = os.path.join(aligner_dir, "mlp_regressor.pt")
    for model_path in directional_paths[preferred_direction]:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                aligner = pickle.load(f)
            print(f"  ✓ Aligner caricato ({preferred_direction}): {model_path}")
            return aligner

    if strict_direction:
        print(f"  ✗ Aligner direzionale '{preferred_direction}' non trovato in: {aligner_dir}")
        return None

    # Fallback legacy/undirected.
    if preferred_direction == "trainer_to_target":
        for model_path in legacy_paths:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    aligner = pickle.load(f)
                print(f"  ✓ Aligner legacy caricato (trainer_to_target): {model_path}")
                return aligner

    if os.path.exists(mlp_path):
        aligner = MLPAlignerWrapper(mlp_path, device=device)
        print(f"  ✓ Aligner MLP caricato: {mlp_path}")
        return aligner

    print(f"  ✗ Nessun aligner trovato in: {aligner_dir}")
    print(
        "    (cercati procrustes_*.pkl, ridge_regressor_*.pkl, "
        "procrustes.pkl, ridge_regressor.pkl, mlp_regressor.pt)"
    )
    return None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
# =============================================================================

EXPERIMENTS= { 
    "Gemma_to_Llama_BBF_PROCRUSTES" : {
        "type" : "CrossModel",
        "model_to_evaluate" : "meta-llama/Llama-3.1-8B-Instruct",
        "dataset_to_evaluate" : "belief_bank_facts",
        "useTruthX" : True,
        "truthX_autoencoder": "AutoEncoder/belief_bank_facts/autoencoder_google_gemma-2-9b-it_pairs6500_cw1.0_tw1.0_ew1.0_rw0.0_mlp.pt",
        "truthX_steering_vectors": "SteeringVectors/belief_bank_facts/steering_vectors_google_gemma-2-9b-it_pairs6500_cw1.0_tw1.0_ew1.0_rw0.0_mlp.pt",
        "truthX_dataset": "belief_bank_facts",
        "aligner_cache": "alignment_cache_procrustes"
    },
    "Gemma_to_Llama_BBC_PROCRUSTES" : {
        "type" : "CrossModel",
        "model_to_evaluate" : "meta-llama/Llama-3.1-8B-Instruct",
        "dataset_to_evaluate" : "belief_bank_constraints",
        "useTruthX" : True,
        "truthX_autoencoder": "AutoEncoder/belief_bank_constraints/autoencoder_google_gemma-2-9b-it_pairs6500_cw1.0_tw1.0_ew1.0_rw0.0_mlp.pt",
        "truthX_steering_vectors": "SteeringVectors/belief_bank_constraints/steering_vectors_google_gemma-2-9b-it_pairs6500_cw1.0_tw1.0_ew1.0_rw0.0_mlp.pt",
        "truthX_dataset": "belief_bank_constraints",
        "aligner_cache": "alignment_cache_procrustes"
    },
    "Gemma_to_Llama_HE_PROCRUSTES" : {
        "type" : "CrossModel",
        "model_to_evaluate" : "meta-llama/Llama-3.1-8B-Instruct",
        "dataset_to_evaluate" : "halu_eval",
        "useTruthX" : True,
        "truthX_autoencoder": "AutoEncoder/halu_eval/autoencoder_google_gemma-2-9b-it_pairs2500_cw1.0_tw1.0_ew1.0_rw0.0_mlp.pt",
        "truthX_steering_vectors": "SteeringVectors/halu_eval/steering_vectors_google_gemma-2-9b-it_pairs2500_cw1.0_tw1.0_ew1.0_rw0.0_mlp.pt",
        "truthX_dataset": "halu_eval",
        "aligner_cache": "alignment_cache_procrustes"
    },
        "Llama_to_Gemma_BBF_MLP2" : {
        "type" : "CrossModel",
        "model_to_evaluate" : "google/gemma-2-9b-it",
        "dataset_to_evaluate" : "belief_bank_facts",
        "useTruthX" : True,
        "truthX_autoencoder": "AutoEncoder/belief_bank_facts/autoencoder_meta-llama_Llama-3.1-8B-Instruct_pairs6500_cw1.0_tw1.0_ew1.0_rw0.0_mlp.pt",
        "truthX_steering_vectors": "SteeringVectors/belief_bank_facts/steering_vectors_meta-llama_Llama-3.1-8B-Instruct_pairs6500_cw1.0_tw1.0_ew1.0_rw0.0_mlp.pt",
        "truthX_dataset": "belief_bank_facts",
        "aligner_cache": "alignment_cache_mlp2"
    },
        "Llama_to_Gemma_BBC_MLP2" : {
        "type" : "CrossModel",
        "model_to_evaluate" : "google/gemma-2-9b-it",
        "dataset_to_evaluate" : "belief_bank_constraints",
        "useTruthX" : True,
        "truthX_autoencoder": "AutoEncoder/belief_bank_constraints/autoencoder_meta-llama_Llama-3.1-8B-Instruct_pairs6500_cw1.0_tw1.0_ew1.0_rw0.0_mlp.pt",
        "truthX_steering_vectors": "SteeringVectors/belief_bank_constraints/steering_vectors_meta-llama_Llama-3.1-8B-Instruct_pairs6500_cw1.0_tw1.0_ew1.0_rw0.0_mlp.pt",
        "truthX_dataset": "belief_bank_constraints",
        "aligner_cache": "alignment_cache_mlp2"
    },
    "Llama_to_Gemma_HE_MLP" : {
        "type" : "CrossModel",
        "model_to_evaluate" : "google/gemma-2-9b-it",
        "dataset_to_evaluate" : "halu_eval",
        "useTruthX" : True,
        "truthX_autoencoder": "AutoEncoder/halu_eval/autoencoder_meta-llama_Llama-3.1-8B-Instruct_pairs2500_cw1.0_tw1.0_ew1.0_rw0.0_mlp.pt",
        "truthX_steering_vectors": "SteeringVectors/halu_eval/steering_vectors_meta-llama_Llama-3.1-8B-Instruct_pairs2500_cw1.0_tw1.0_ew1.0_rw0.0_mlp.pt",
        "truthX_dataset": "halu_eval",
        "aligner_cache": "alignment_cache_mlp"
    },
    "Llama_to_Gemma_HE_MLP2" : {
        "type" : "CrossModel",
        "model_to_evaluate" : "google/gemma-2-9b-it",
        "dataset_to_evaluate" : "halu_eval",
        "useTruthX" : True,
        "truthX_autoencoder": "AutoEncoder/halu_eval/autoencoder_meta-llama_Llama-3.1-8B-Instruct_pairs2500_cw1.0_tw1.0_ew1.0_rw0.0_mlp.pt",
        "truthX_steering_vectors": "SteeringVectors/halu_eval/steering_vectors_meta-llama_Llama-3.1-8B-Instruct_pairs2500_cw1.0_tw1.0_ew1.0_rw0.0_mlp.pt",
        "truthX_dataset": "halu_eval",
        "aligner_cache": "alignment_cache_mlp2"
    }   
}

def save_to_csv(filename, data_dict):
    """
    Salva una riga di risultati nel CSV. Crea l'header se il file non esiste.
    """
    file_exists = os.path.isfile(filename)
    
    fieldnames = [
        "experiment_id", 
        "type", 
        "model", 
        "dataset", 
        "num_samples_evaluated", 
        "num_hallucinations", 
        "hallucination_rate",
        # TruthX paths
        "truthX_autoencoder",
        "truthX_steering_vectors",
        # Additional metadata
        "num_pairs",
        "contrastive_weight",
        "editing_weight",
        "reconstruction_weight",
        "temperature",
        "edit_strength",
        "top_k_modules",
        # Timing info
        "ae_training_time_seconds",
        "inference_time_seconds"
    ]

    # If file already exists but misses some columns, update file to include new headers
    if file_exists:
        try:
            with open(filename, mode='r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                existing_fieldnames = reader.fieldnames or []
                missing = [fn for fn in fieldnames if fn not in existing_fieldnames]
                if missing:
                    # Read all rows and rewrite file with new header including missing fields
                    rows = list(reader)
                    with open(filename, mode='w', newline='', encoding='utf-8') as writefile:
                        writer = csv.DictWriter(writefile, fieldnames=fieldnames)
                        writer.writeheader()
                        for r in rows:
                            for m in missing:
                                r[m] = ''
                            writer.writerow(r)
                    print(f"Aggiornato header di {filename} aggiungendo colonne: {missing}")
        except Exception as e:
            print(f"Warning: impossibile aggiornare header CSV esistente: {e}")

    # Append the new row
    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # If file didn't exist, write header
        if not file_exists:
            writer.writeheader()
        writer.writerow(data_dict)
    print(f"Risultati salvati in {filename}")


def append_failure_log(log_path, experiment_id, reason, traceback_text=None):
    """
    Appende su log.txt un failure di esperimento con timestamp e motivo.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, mode='a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] EXPERIMENT FAILED: {experiment_id}\n")
        f.write(f"Reason: {reason}\n")
        if traceback_text:
            f.write("Traceback:\n")
            f.write(traceback_text.rstrip() + "\n")
        f.write("-" * 80 + "\n")


def append_skip_log(log_path, experiment_id, reason):
    """Appende su log.txt uno skip di esperimento con timestamp e motivo."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, mode='a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] EXPERIMENT SKIPPED: {experiment_id}\n")
        f.write(f"Reason: {reason}\n")
        f.write("-" * 80 + "\n")


def resolve_truthx_artifact_path(project_dir, configured_path, artifact_kind):
    """
    Risolve robustamente il path di un artifact TruthX.

    Strategia:
    1) Usa il path configurato (relativo a project_dir) se esiste.
    2) Se manca, cerca nella stessa cartella un file con stesso suffisso
       `_pairs...pt`, ignorando differenze nel token del modello.

    Returns:
        (resolved_path, resolution_note)
    """
    if not configured_path:
        return None, f"{artifact_kind}: path non configurato"

    configured_abs = configured_path if os.path.isabs(configured_path) else os.path.join(project_dir, configured_path)
    configured_abs = os.path.abspath(configured_abs)

    if os.path.exists(configured_abs):
        return configured_abs, "configured"

    search_dir = os.path.dirname(configured_abs)
    base_name = os.path.basename(configured_abs)

    if not os.path.isdir(search_dir):
        return None, f"{artifact_kind}: directory non trovata ({search_dir})"

    # Esempi attesi:
    # autoencoder_<model_token>_pairs6500_..._mlp.pt
    # steering_vectors_<model_token>_pairs6500_..._mlp.pt
    # Catturiamo prefisso + suffisso pairs..., ignorando il token modello.
    match = re.match(r"^(autoencoder|steering_vectors)_(.+?)(_pairs\d+.*\.pt)$", base_name)
    if match:
        prefix = match.group(1)
        suffix = match.group(3)
        pattern = os.path.join(search_dir, f"{prefix}_*{suffix}")
        candidates = glob(pattern)
        if candidates:
            resolved = max(candidates, key=os.path.getmtime)
            return resolved, f"auto-resolved via pattern {pattern}"

    # Fallback finale: prende il più recente con prefisso corretto nella cartella.
    generic_prefix = "autoencoder_" if artifact_kind == "Autoencoder" else "steering_vectors_"
    generic_pattern = os.path.join(search_dir, f"{generic_prefix}*.pt")
    generic_candidates = glob(generic_pattern)
    if generic_candidates:
        resolved = max(generic_candidates, key=os.path.getmtime)
        return resolved, f"auto-resolved generic via pattern {generic_pattern}"

    return None, f"{artifact_kind}: nessun match trovato in {search_dir}"


def get_model_layers_info(model) -> Tuple[List, int, str, dict]:
    arch_name, arch_config = LLMArchitectureDetector.detect_architecture(model)
    layers = LLMArchitectureDetector.get_layers(model, arch_config)
    num_layers = len(layers)
    return layers, num_layers, arch_name, arch_config

# =============================================================================
# EVALUATION LOGIC
# =============================================================================

def load_training_config(project_dir, model_name, dataset_name):
    """
    Carica il file di configurazione del training per ottenere gli indici originali
    del dataset BeliefBank usati nel train/val set.
    
    Il training config salva instance_ids 0-1599 per 800 coppie, ma questi 
    corrispondono agli indici originali 0-799 (prima metà) e 12500-13299 (seconda metà)
    del dataset BeliefBank completo.
    
    Returns:
        set: Set di indici originali BeliefBank da escludere, oppure None se il file non esiste
    """
    model_name_safe = model_name.replace("/", "_")
    config_dir = os.path.join(project_dir, "ConfigTraining")
    
    # Try to find a config file with pattern (newer format with pairs suffix)
    config_pattern = os.path.join(config_dir, f"{model_name_safe}_{dataset_name}_pairs*.json")
    config_matches = glob(config_pattern)
    config_path = None
    
    if config_matches:
        # Pick the most recently modified config file if multiple exist
        config_path = max(config_matches, key=os.path.getmtime)
        print(f"Found training config: {config_path}")
    else:
        # Fallback to legacy format (without pairs suffix)
        legacy_config = os.path.join(config_dir, f"{model_name_safe}_{dataset_name}.json")
        if os.path.exists(legacy_config):
            config_path = legacy_config
        else:
            print(f"Warning: Training config non trovato: {config_dir}")
            print("Procedendo senza filtrare gli instance_ids.")
            return None, None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            training_config = json.load(f)
        
        total_pairs = training_config.get("total_pairs", 0)
        
        # Per HaluEval, gli instance_ids sono sequenziali (0..N-1) e le coppie
        # usano gli stessi indici per label=0 e label=1, quindi basta escludere 0..(total_pairs-1)
        if dataset_name == "halu_eval":
            excluded_original_indices = set(range(total_pairs))
            print(f"Training config caricato (HaluEval): {total_pairs} coppie totali")
            print(f"Indici HaluEval esclusi: {len(excluded_original_indices)} elementi (0-{total_pairs-1})")
            return excluded_original_indices, training_config
        
        # Carica il dataset BeliefBank per conoscere la dimensione totale
        # Questo è necessario per mappare gli indici delle coppie agli indici originali
        from src.data.BeliefBankDataset import BeliefBankDataset
        # Determine the proper BeliefBank data_type based on the dataset being evaluated
        bb_data_type = "facts"
        if dataset_name == "belief_bank_constraints":
            bb_data_type = "constraints"
        elif dataset_name == "belief_bank_facts":
            bb_data_type = "facts"
        print(f"Using BeliefBank data_type='{bb_data_type}' for dataset '{dataset_name}'")
        full_dataset = BeliefBankDataset(
            project_root=project_dir,
            model_type="demo",
            recreate_ids=True,
            data_type=bb_data_type
        )
        
        total_samples = len(full_dataset)
        half = total_samples // 2
        
        # Gli indici originali usati nel training sono:
        # - Prima metà: 0 fino a (total_pairs - 1)
        # - Seconda metà: half fino a (half + total_pairs - 1)
        excluded_original_indices = set()
        
        for i in range(total_pairs):
            excluded_original_indices.add(i)           # Prima metà
            excluded_original_indices.add(i + half)    # Seconda metà
        
        print(f"Training config caricato: {total_pairs} coppie totali")
        print(f"Indici originali esclusi: {len(excluded_original_indices)} elementi")
        print(f"  - Prima metà: 0-{total_pairs-1}")
        print(f"  - Seconda metà: {half}-{half + total_pairs - 1}")
        
        # Return both the excluded indices set and the loaded training config dict
        return excluded_original_indices, training_config
    except Exception as e:
        print(f"Errore nel caricamento training config: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def detect_cache_structure_type(project_dir, cache_dir_name, model_name, dataset_name):
    """
    Rileva automaticamente se la struttura della cache è vecchia o nuova.
    
    Returns:
        'new': struttura con cartelle hallucinated/ e not_hallucinated/
        'old': struttura con hallucination_labels.json
    """
    # Costruisci il percorso base della cache
    model_name_short = model_name.split("/")[-1]
    results_dir = os.path.join(project_dir, cache_dir_name)
    
    # Check per la nuova struttura con activation_attn/hallucinated/
    base_path = os.path.join(results_dir, model_name_short, dataset_name, "activation_attn")
    hallucinated_path = os.path.join(base_path, "hallucinated")
    
    if os.path.isdir(hallucinated_path):
        return 'new'
    return 'old'


def load_labels_from_old_structure(project_dir, cache_dir_name, model_name, dataset_name):
    """
    Carica labels dalla vecchia struttura con hallucination_labels.json
    
    Returns:
        list: lista di dict con 'instance_id' e 'is_hallucination'
    """
    model_name_short = model_name.split("/")[-1]
    results_dir = os.path.join(project_dir, cache_dir_name)
    gen_dir = os.path.join(results_dir, model_name_short, dataset_name, "generations")
    hallucination_labels_path = os.path.join(gen_dir, "hallucination_labels.json")
    
    if not os.path.exists(hallucination_labels_path):
        raise FileNotFoundError(f"hallucination_labels.json non trovato: {hallucination_labels_path}")
    
    with open(hallucination_labels_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def load_labels_from_new_structure(project_dir, cache_dir_name, model_name, dataset_name):
    """
    Carica labels dalla nuova struttura con cartelle hallucinated/ e not_hallucinated/
    
    Returns:
        list: lista di dict con 'instance_id' e 'is_hallucination'
    """
    model_name_short = model_name.split("/")[-1]
    results_dir = os.path.join(project_dir, cache_dir_name)
    base_path = os.path.join(results_dir, model_name_short, dataset_name, "activation_attn")
    hallucinated_path = os.path.join(base_path, "hallucinated")
    not_hallucinated_path = os.path.join(base_path, "not_hallucinated")
    
    hall_ids_path = os.path.join(hallucinated_path, "layer0_instance_ids.json")
    not_hall_ids_path = os.path.join(not_hallucinated_path, "layer0_instance_ids.json")
    
    if not os.path.exists(hall_ids_path):
        raise FileNotFoundError(f"File instance_ids non trovato: {hall_ids_path}")
    if not os.path.exists(not_hall_ids_path):
        raise FileNotFoundError(f"File instance_ids non trovato: {not_hall_ids_path}")
    
    with open(hall_ids_path, 'r', encoding='utf-8') as f:
        hallucinated_ids = json.load(f)
    with open(not_hall_ids_path, 'r', encoding='utf-8') as f:
        not_hallucinated_ids = json.load(f)
    
    # Concatena gli ID e le label
    ids_concat = np.array(hallucinated_ids + not_hallucinated_ids)
    labels_concat = np.concatenate([
        np.ones(len(hallucinated_ids), dtype=int),
        np.zeros(len(not_hallucinated_ids), dtype=int)
    ])
    
    # IMPORTANTE: Ordina per instance_id per mantenere la distribuzione corretta
    sort_indices = np.argsort(ids_concat)
    ids_sorted = ids_concat[sort_indices]
    labels_sorted = labels_concat[sort_indices]
    
    # Converti in formato compatibile con il resto del codice
    labels_data = []
    for instance_id, label in zip(ids_sorted, labels_sorted):
        labels_data.append({
            'instance_id': int(instance_id),
            'is_hallucination': int(label)
        })
    
    return labels_data


def evaluate_baseline_from_cache(experiment_id, config, num_samples, project_dir):
    """
    Recupera i risultati per la baseline direttamente dalla cache esistente
    senza eseguire inferenza.
    ESCLUDE gli instance_ids usati nel training/validation per evitare data leakage.
    Supporta sia il formato vecchio (hallucination_labels.json) che il nuovo
    (cartelle hallucinated/ e not_hallucinated/).
    """
    print(f"[{experiment_id}] Valutazione Baseline (da Cache)...")
    
    model_name = config.get("model_to_evaluate")
    dataset_name = config.get("dataset_to_evaluate")
    cache_dir_name = "activation_cache"  # La baseline usa activation_cache, non activation_cache_truthx
    
    try:
        # Rileva automaticamente la struttura e carica i dati
        structure_type = detect_cache_structure_type(project_dir, cache_dir_name, model_name, dataset_name)
        print(f"Rilevata struttura cache: {structure_type}")
        
        if structure_type == 'new':
            labels_data = load_labels_from_new_structure(project_dir, cache_dir_name, model_name, dataset_name)
        else:
            labels_data = load_labels_from_old_structure(project_dir, cache_dir_name, model_name, dataset_name)
        
        # DEBUG: Conta prima del filtraggio
        total_before = len(labels_data)
        hall_before = sum(item.get("is_hallucination", 0) for item in labels_data)
        print(f"Prima del filtraggio: {total_before} campioni totali, {hall_before} allucinazioni ({hall_before/total_before*100:.2f}%)")
        
        # Carica training config per escludere instance_ids usati nel train/val
        excluded_ids, training_cfg = load_training_config(
            project_dir,
            model_name,
            dataset_name
        )
        
        # Filtra gli instance_ids usati nel training/validation
        original_count = len(labels_data)
        if excluded_ids is not None:
            labels_data = [
                item for item in labels_data
                if item.get("instance_id") not in excluded_ids
            ]
            print(f"Filtrati {original_count - len(labels_data)} campioni usati nel training/validation")
            print(f"Rimangono {len(labels_data)} campioni per la valutazione")
            # DEBUG: Conta dopo il filtraggio train/val
            hall_after_filter = sum(item.get("is_hallucination", 0) for item in labels_data)
            print(f"Dopo filtro train/val: {len(labels_data)} campioni, {hall_after_filter} allucinazioni ({hall_after_filter/len(labels_data)*100:.2f}%)")
        
        # Filtra per num_samples se specificato
        if num_samples > 0:
            labels_data = labels_data[:num_samples]
            # DEBUG: Conta dopo il limite num_samples
            hall_after_limit = sum(item.get("is_hallucination", 0) for item in labels_data)
            print(f"Dopo limite num_samples: {len(labels_data)} campioni, {hall_after_limit} allucinazioni ({hall_after_limit/len(labels_data)*100:.2f}%)")
        
        evaluated_samples = len(labels_data)
        # Conta il numero di allucinazioni (is_hallucination = 1)
        n_hallucinations = sum(item.get("is_hallucination", 0) for item in labels_data)
        
        rate = n_hallucinations / evaluated_samples if evaluated_samples > 0 else 0
        
        print(f"Valutati {evaluated_samples} campioni, {n_hallucinations} allucinazioni ({rate*100:.2f}%)")
        
        # Prova a recuperare metadati del training (se disponibili)
        model_name_safe = model_name.replace("/", "_")
        num_pairs = None
        ae_args = {}
        if 'training_cfg' in locals() and training_cfg is not None:
            num_pairs = training_cfg.get("total_pairs")
        
        # Try to locate an autoencoder file matching the naming convention
        ae_training_time_seconds = None
        ae_search_dir = os.path.join(project_dir, "AutoEncoder", dataset_name)
        ae_pattern = os.path.join(ae_search_dir, f"autoencoder_{model_name_safe}_*.pt")
        ae_matches = glob(ae_pattern)
        ae_default = None
        if ae_matches:
            # pick the most recently modified autoencoder
            ae_default = max(ae_matches, key=os.path.getmtime)
            print(f"Found autoencoder file: {ae_default}")
            try:
                saved = torch.load(ae_default, map_location="cpu")
                ae_args = saved.get("args", {}) or {}
                ae_training_time_seconds = saved.get("training_time_seconds")
            except Exception:
                ae_args = {}
        else:
            # Fallback to legacy name
            legacy = os.path.join(ae_search_dir, f"autoencoder_{model_name_safe}.pt")
            if os.path.exists(legacy):
                ae_default = legacy
                try:
                    saved = torch.load(ae_default, map_location="cpu")
                    ae_args = saved.get("args", {}) or {}
                    ae_training_time_seconds = saved.get("training_time_seconds")
                except Exception:
                    ae_args = {}
            else:
                print(f"No autoencoder found in {ae_search_dir}")
        
        return {
            "experiment_id": experiment_id,
            "type": config.get("type"),
            "model": model_name,
            "dataset": dataset_name,
            "num_samples_evaluated": evaluated_samples,
            "num_hallucinations": n_hallucinations,
            "hallucination_rate": rate,
            "truthX_autoencoder": None,
            "truthX_steering_vectors": None,
            "num_pairs": None,
            "contrastive_weight": None,
            "editing_weight": None,
            "temperature": None,
            "ae_training_time_seconds": None,
            "edit_strength": None,
            "top_k_modules": None,
            "inference_time_seconds": None,
        }

    except FileNotFoundError as fnf_error:
        print(f"Errore: {fnf_error}")
        raise
    except Exception as e:
        print(f"Errore nel recupero cache baseline: {e}")
        import traceback
        traceback.print_exc()
        raise


def evaluate_truthx_experiment(experiment_id, config, num_samples, project_dir, device, quantization=True):
    """
    Esegue l'inferenza con TruthX attivo. Non salva attivazioni pesanti, 
    calcola solo se l'output è allucinato.
    ESCLUDE gli instance_ids usati nel training/validation per evitare data leakage.
    """
    print(f"[{experiment_id}] Valutazione TruthX (Inferenza attiva)...")
    print(f"Using device: {device}")
    
    model_name = config.get("model_to_evaluate")
    dataset_name = config.get("dataset_to_evaluate")
    autoencoder_path = config.get("truthX_autoencoder")
    steering_vectors_path = config.get("truthX_steering_vectors")
    
    # 1. Caricamento Modello e Tokenizer
    print(f"Caricamento modello: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configurazione quantizzazione se richiesta
    if quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        print("Usando quantizzazione 4-bit")
    else:
        quantization_config = None
        print("Modello in float16 senza quantizzazione")
    
    # Usa device_map="auto" oppure carica su dispositivo specifico
    if device.startswith("cuda"):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map={"": device},
            quantization_config=quantization_config
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            quantization_config=quantization_config
        )
        model = model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Caricamento Dataset
    print(f"Caricamento dataset: {dataset_name}")
    hd = HallucinationDetection(project_dir, cache_dir_name="activation_cache_truthx")
    
    if dataset_name == "belief_bank_facts":
        hd.load_dataset(dataset_name="belief_bank", belief_bank_data_type="facts")
    elif dataset_name == "belief_bank_constraints":
        hd.load_dataset(dataset_name="belief_bank", belief_bank_data_type="constraints")
    else:
        hd.load_dataset(dataset_name=dataset_name)
    
    dataset = hd.dataset
    
    # Carica training config per escludere instance_ids usati nel train/val
    excluded_ids, training_cfg = load_training_config(project_dir, model_name, dataset_name)
    
    # Crea una lista di indici validi (esclusi quelli nel train/val)
    valid_indices = []
    for idx in range(len(dataset)):
        fact, label, instance_id = dataset[idx]
        if excluded_ids is None or instance_id not in excluded_ids:
            valid_indices.append(idx)
    
    print(f"Dataset originale: {len(dataset)} esempi")
    print(f"Dopo filtraggio train/val: {len(valid_indices)} esempi validi")
    
    # Filtra per num_samples se specificato
    if num_samples > 0:
        valid_indices = valid_indices[:num_samples]
    
    print(f"Esempi da valutare: {len(valid_indices)}")

    # 3. Inizializzazione TruthX
    hidden_size = LLMArchitectureDetector.get_hidden_size(model)
    
    truthx_editor = None
    ae_full_path = None
    sv_full_path = None
    truthx_enabled = bool(config.get("useTruthX", False))

    if truthx_enabled:
        ae_full_path, ae_note = resolve_truthx_artifact_path(
            project_dir,
            autoencoder_path,
            "Autoencoder"
        )
        sv_full_path, sv_note = resolve_truthx_artifact_path(
            project_dir,
            steering_vectors_path,
            "Steering vectors"
        )

        if ae_note != "configured":
            print(f"[PathResolver] Autoencoder: {ae_note}")
        if sv_note != "configured":
            print(f"[PathResolver] Steering vectors: {sv_note}")

        missing_resources = []
        if not ae_full_path or not os.path.exists(ae_full_path):
            missing_resources.append(
                f"Autoencoder non trovato: {autoencoder_path}"
            )
        if not sv_full_path or not os.path.exists(sv_full_path):
            missing_resources.append(
                f"Steering vectors non trovati: {steering_vectors_path}"
            )
        if missing_resources:
            skip_reason = (
                "Missing TruthX artifacts: "
                + " | ".join(missing_resources)
                + f" | project_dir: {project_dir}"
            )
            raise ExperimentSkippedError(skip_reason)

        print(f"Caricamento Autoencoder TruthX da: {ae_full_path}")
        print(f"Caricamento Steering Vectors da: {sv_full_path}")
        
        # Load aligner for CrossModel experiments
        pre_aligner = None
        post_aligner = None
        
        if config.get("type") == "CrossModel":
            print("\n[CrossModel] Caricamento aligner per trasformazione cross-model...")

            aligner_cache_dir = config.get("aligner_cache", "alignment_cache2")
            print(f"  Aligner cache directory: {aligner_cache_dir}")

            # Preferred mode: single directory per esperimento con entrambe le direzioni.
            single_aligner_dir = os.path.join(project_dir, aligner_cache_dir, experiment_id)
            print(f"  Candidate single aligner dir: {single_aligner_dir}")

            if has_bidirectional_directional_aligners(single_aligner_dir):
                print("  ✓ Rilevato aligner bidirezionale nella stessa cartella.")
                pre_aligner = load_aligner(
                    single_aligner_dir,
                    device=device,
                    preferred_direction="target_to_trainer",
                    strict_direction=True
                )
                post_aligner = load_aligner(
                    single_aligner_dir,
                    device=device,
                    preferred_direction="trainer_to_target",
                    strict_direction=True
                )
            else:
                print("  ! Aligner bidirezionale non trovato, fallback legacy a due cartelle.")

                # Legacy fallback:
                # pre  = reverse experiment (trainer_to_target in quella cartella)
                # post = current experiment (trainer_to_target in questa cartella)
                exp_parts = experiment_id.split("_to_")
                if len(exp_parts) == 2:
                    trainer_name = exp_parts[0]
                    target_and_dataset = exp_parts[1].split("_")
                    target_name = target_and_dataset[0]
                    dataset_suffix = "_".join(target_and_dataset[1:])

                    post_aligner_name = experiment_id
                    pre_aligner_name = f"{target_name}_to_{trainer_name}_{dataset_suffix}"

                    pre_aligner_dir = os.path.join(project_dir, aligner_cache_dir, pre_aligner_name)
                    post_aligner_dir = os.path.join(project_dir, aligner_cache_dir, post_aligner_name)

                    print(f"  Legacy pre-aligner dir:  {pre_aligner_dir}")
                    print(f"  Legacy post-aligner dir: {post_aligner_dir}")

                    pre_aligner = load_aligner(
                        pre_aligner_dir,
                        device=device,
                        preferred_direction="trainer_to_target",
                        strict_direction=False
                    )
                    post_aligner = load_aligner(
                        post_aligner_dir,
                        device=device,
                        preferred_direction="trainer_to_target",
                        strict_direction=False
                    )
                else:
                    raise FileNotFoundError(
                        f"Cannot parse experiment_id '{experiment_id}' for aligner names"
                    )

            if pre_aligner is None:
                raise FileNotFoundError(
                    f"Pre-aligner non trovato per esperimento {experiment_id} in cache '{aligner_cache_dir}'"
                )
            if post_aligner is None:
                raise FileNotFoundError(
                    f"Post-aligner non trovato per esperimento {experiment_id} in cache '{aligner_cache_dir}'"
                )
        
        # Initialize TruthX with aligners
        truthx_editor = TruthX(
            autoencoder_path=ae_full_path,
            steering_vectors_path=sv_full_path,
            hidden_size=hidden_size,
            edit_strength=1,  # α = 1.0 come da paper per open-ended
            top_layers=1,  # k = 1 come configurato
            pre_aligner=pre_aligner,
            post_aligner=post_aligner
        )
        print("TruthX Editor inizializzato con successo.")
        print(f"Top-{truthx_editor.top_layers} layers da editare: {truthx_editor.rank[:truthx_editor.top_layers]}")
    else:
        print("useTruthX=False: si procede senza editing TruthX.")
    
    # 4. Applica hooks al modello se TruthX è attivo
    hooks = []
    if truthx_editor:
        layers, num_layers, arch_name, arch_config = get_model_layers_info(model)
        
        # Ottieni i top-k virtual layer selezionati dal probing
        top_k_virtual_indices = truthx_editor.rank[:truthx_editor.top_layers]
        virtual_layer_info = truthx_editor.virtual_layer_info
        
        print(f"\nApplicando hooks solo ai top-{truthx_editor.top_layers} moduli selezionati:")
        
        # Crea dizionario per raggruppare moduli per layer fisico
        # {layer_idx: [module_types]}
        layers_to_hook = {}
        for virtual_idx in top_k_virtual_indices:
            physical_layer, module_type = virtual_layer_info[virtual_idx]
            if physical_layer not in layers_to_hook:
                layers_to_hook[physical_layer] = []
            layers_to_hook[physical_layer].append(module_type)
            print(f"  Virtual {virtual_idx}: Layer {physical_layer}, {module_type}")
        
        def make_mlp_hook(layer_idx):
            """Hook per l'output del MLP"""
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                modified = truthx_editor.edit(hidden_states, layer_idx, 'mlp')
                
                if isinstance(output, tuple):
                    return (modified,) + output[1:]
                else:
                    return modified
            return hook_fn
        
        def make_attn_hook(layer_idx):
            """Hook per l'output dell'attention"""
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                modified = truthx_editor.edit(hidden_states, layer_idx, 'attn')
                
                if isinstance(output, tuple):
                    return (modified,) + output[1:]
                else:
                    return modified
            return hook_fn
        
        # Registra hooks SOLO sui layer e moduli selezionati
        for physical_layer_idx, module_types in layers_to_hook.items():
            layer = layers[physical_layer_idx]
            
            if 'attn' in module_types:
                attn_module = LLMArchitectureDetector.get_attention_module(layer, arch_config)
                hook_attn = attn_module.register_forward_hook(make_attn_hook(physical_layer_idx))
                hooks.append(hook_attn)
            
            if 'mlp' in module_types or 'ffn' in module_types:
                mlp_module = LLMArchitectureDetector.get_mlp_module(layer, arch_config)
                hook_mlp = mlp_module.register_forward_hook(make_mlp_hook(physical_layer_idx))
                hooks.append(hook_mlp)
        
        print(f"\nHooks registrati: {len(hooks)} moduli (solo top-{truthx_editor.top_layers})")

    # 5. Loop di Inferenza
    n_hallucinations = 0
    total_evaluated = 0
    inference_results = []  # Raccogliere tutti i risultati
    
    print(f"Avvio inferenza su {len(valid_indices)} esempi...")
    
    from src.model.prompts import PROMPT_QA, PROMPT_HALU

    # Determina se il dataset è HaluEval (QA aperta) o BeliefBank (yes/no)
    is_halu_eval = (dataset_name == "halu_eval")
    
    if is_halu_eval:
        current_prompt_template = PROMPT_HALU
        current_max_new_tokens = 100  # HaluEval richiede risposte più lunghe
        print("Modalità HaluEval")
    else:
        current_prompt_template = PROMPT_QA
        current_max_new_tokens = 5  # BeliefBank: risposte yes/no
        print("Modalità BeliefBank")

    inference_start_time = time.time()
    for idx in tqdm(valid_indices):
        fact, label, instance_id = dataset[idx]
        
        # Costruisci il prompt in base al dataset
        question = current_prompt_template.format(question=fact)
        
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        
        # Genera la risposta
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=current_max_new_tokens,
                do_sample=False
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        generated_text = generated_text.strip()
        
        # Valuta se è allucinato
        if is_halu_eval:
            # Per HaluEval: confronto semantico tra risposta generata e expected answer
            # Normalizziamo entrambe e verifichiamo se la risposta attesa è contenuta
            # nella risposta generata (o viceversa)
            gen_lower = generated_text.lower().strip()
            exp_lower = label.lower().strip()
            
            # Verifica contenimento bidirezionale (la risposta può essere più lunga o più corta)
            is_hallucinated = (exp_lower not in gen_lower) and (gen_lower not in exp_lower)
            
            # Fallback: controlla se le prime parole significative coincidono
            """if is_hallucinated:
                gen_words = set(gen_lower.split())
                exp_words = set(exp_lower.split())
                if exp_words and gen_words:
                    overlap = len(gen_words & exp_words) / len(exp_words)
                    if overlap >= 0.5:  # almeno 50% delle parole della risposta attesa presenti
                        is_hallucinated = False"""
        else:
            # Per BeliefBank: verifica se yes/no è contenuto nella risposta
            gen_lower = generated_text.lower()
            expected_answer = label.lower()
            is_hallucinated = expected_answer not in gen_lower
        
        if is_hallucinated:
            n_hallucinations += 1
        total_evaluated += 1
        
        # DEBUG: stampa i primi 5 elementi
        if total_evaluated <= 5:
            print(f"\n{'─'*60}")
            print(f"  [DEBUG] Esempio {total_evaluated}/5")
            print(f"  Input LLM:          {question[:200]}{'...' if len(question) > 200 else ''}")
            print(f"  Risposta Generata:  {generated_text}")
            print(f"  Risposta Giusta:    {label}")
            print(f"  Allucinazione:      {'SÌ ❌' if is_hallucinated else 'NO ✅'}")
            print(f"{'─'*60}")
        
        # Salva il risultato
        inference_results.append({
            "instance_id": instance_id,
            "prompt": fact,
            "generated_answer": generated_text,
            "expected_answer": label,
            "is_hallucination": int(is_hallucinated)
        })

    inference_time_seconds = None
    if 'inference_start_time' in locals():
        inference_time_seconds = time.time() - inference_start_time
        print(f"Inferenza completata in {inference_time_seconds:.2f} secondi (totale)")

    # 6. Rimuovi hooks
    for hook in hooks:
        hook.remove()
    
    # Prova a recuperare metadati del training (se disponibili)
    model_name_safe = model_name.replace("/", "_")
    num_pairs = None
    ae_args = {}
    training_cfg_exists = 'training_cfg' in locals() and training_cfg is not None
    if training_cfg_exists:
        num_pairs = training_cfg.get("total_pairs")

    # Load autoencoder args if not already loaded
    if ae_full_path and os.path.exists(ae_full_path):
        try:
            saved = torch.load(ae_full_path, map_location="cpu")
            ae_args = saved.get("args", {}) or {}
        except Exception:
            ae_args = {}
    
    # Salva i risultati in JSON
    results_dir = os.path.join(project_dir, "InferenceResults")
    os.makedirs(results_dir, exist_ok=True)

    # Build suffix with num_pairs and contrastive weight (if available)
    num_pairs_val = num_pairs if num_pairs is not None else 'NA'
    cw_val = ae_args.get('contrastive_weight') if isinstance(ae_args, dict) else None
    cw_str = cw_val if cw_val is not None else 'NA'
    ew_val = ae_args.get('editing_weight') if isinstance(ae_args, dict) else None
    ew_str = ew_val if ew_val is not None else 'NA'
    use_residual = ae_args.get('residual', False) if isinstance(ae_args, dict) else False
    arch_tag = "res" if use_residual else "mlp"

    suffix = f"pairs{num_pairs_val}_cw{cw_str}_ew{ew_str}_{arch_tag}"

    results_filename = f"{experiment_id}_{suffix}_results.json"
    results_path = os.path.join(results_dir, results_filename)

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(inference_results, f, indent=2, ensure_ascii=False)

    print(f"\nRisultati salvati in: {results_path}")
    
    # Pulisci memoria
    del model
    torch.cuda.empty_cache()
    
    rate = n_hallucinations / total_evaluated if total_evaluated > 0 else 0
    
    print(f"Valutati {total_evaluated} campioni, {n_hallucinations} allucinazioni ({rate*100:.2f}%)")

    edit_strength_val = None
    top_k_val = None
    if truthx_editor:
        edit_strength_val = getattr(truthx_editor, 'edit_strength', None)
        top_k_val = getattr(truthx_editor, 'top_layers', None)

    # Try to get ae training time if available
    ae_training_time_seconds = None
    if ae_full_path and os.path.exists(ae_full_path):
        try:
            saved = torch.load(ae_full_path, map_location="cpu")
            ae_training_time_seconds = saved.get("training_time_seconds")
        except Exception:
            ae_training_time_seconds = None

    # We only record total inference time (per-request averaging removed as requested)

    # Extract individual parameters from ae_args dict
    contrastive_weight_val = ae_args.get('contrastive_weight') if isinstance(ae_args, dict) else None
    editing_weight_val = ae_args.get('editing_weight') if isinstance(ae_args, dict) else None
    reconstruction_weight_val = ae_args.get('reconstruction_weight') if isinstance(ae_args, dict) else None
    temperature_val = ae_args.get('temperature') if isinstance(ae_args, dict) else None
    
    # Get TruthX autoencoder and steering vectors paths from config
    truthx_autoencoder_path = config.get("truthX_autoencoder")
    truthx_steering_vectors_path = config.get("truthX_steering_vectors")

    return {
        "experiment_id": experiment_id,
        "type": config.get("type"),
        "model": model_name,
        "dataset": dataset_name,
        "num_samples_evaluated": total_evaluated,
        "num_hallucinations": n_hallucinations,
        "hallucination_rate": rate,
        "truthX_autoencoder": truthx_autoencoder_path,
        "truthX_steering_vectors": truthx_steering_vectors_path,
        "num_pairs": num_pairs,
        "contrastive_weight": contrastive_weight_val,
        "editing_weight": editing_weight_val,
        "reconstruction_weight": reconstruction_weight_val,
        "temperature": temperature_val,
        "ae_training_time_seconds": ae_training_time_seconds,
        "inference_time_seconds": inference_time_seconds,
        "edit_strength": edit_strength_val,
        "top_k_modules": top_k_val,
    }

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TruthX Inference & Experiments Runner")
    
    parser.add_argument("--experiments_file", type=str, default="EXPERIMENTS.txt", 
                        help="Path al file di configurazione esperimenti")
    parser.add_argument("--project_dir", type=str, default=".", 
                        help="Directory radice del progetto")
    parser.add_argument("--output_csv", type=str, default="truthXEperiments.csv", 
                        help="File CSV di output")
    parser.add_argument("--num_samples", type=int, default=-1, 
                        help="Numero di esempi da valutare (-1 per tutti)")
    parser.add_argument("--specific_experiment", type=str, default=None, 
                        help="Nome di un esperimento specifico da eseguire (opzionale)")
    parser.add_argument("--device", type=str, default="cuda:1", 
                        help="Device to use for inference (e.g., 'cuda:0', 'cuda:1', 'cpu')")
    parser.add_argument("--failure_log", type=str, default="log.txt",
                        help="File di log per esperimenti falliti/skippati")
    
    args = parser.parse_args()
    project_dir = os.path.abspath(args.project_dir)
    output_csv_path = args.output_csv if os.path.isabs(args.output_csv) else os.path.join(project_dir, args.output_csv)
    failure_log_path = args.failure_log if os.path.isabs(args.failure_log) else os.path.join(project_dir, args.failure_log)
    output_csv_path = os.path.abspath(output_csv_path)
    failure_log_path = os.path.abspath(failure_log_path)
    
    # 1. Carica configurazione esperimenti
    all_experiments = EXPERIMENTS
    if not all_experiments:
        print("Nessun esperimento trovato. Uscita.")
        exit()

    # Filtra se richiesto un esperimento specifico
    if args.specific_experiment:
        if args.specific_experiment in all_experiments:
            all_experiments = {args.specific_experiment: all_experiments[args.specific_experiment]}
        else:
            print(f"Esperimento '{args.specific_experiment}' non trovato nel file.")
            exit()

    print(f"Trovati {len(all_experiments)} esperimenti da eseguire.")
    completed_count = 0
    skipped_count = 0
    failed_count = 0

    # 2. Ciclo sugli esperimenti
    for exp_id, exp_config in all_experiments.items():
        print(f"\n{'='*40}")
        print(f"RUNNING: {exp_id}")
        print(f"{'='*40}")
        try:
            result_row = None

            # Caso A: Baseline (Usa Cache)
            if exp_config.get("type") == "baseline" or exp_config.get("useTruthX") is False:
                result_row = evaluate_baseline_from_cache(
                    exp_id, exp_config, args.num_samples, project_dir
                )

            # Caso B: TruthX (Inferenza attiva)
            else:
                result_row = evaluate_truthx_experiment(
                    exp_id, exp_config, args.num_samples, project_dir, args.device
                )

            # 3. Salvataggio Risultati
            if result_row:
                save_to_csv(output_csv_path, result_row)
                completed_count += 1
            else:
                raise RuntimeError("Risultato esperimento vuoto (result_row is None)")

        except ExperimentSkippedError as e:
            reason = str(e)
            skipped_count += 1
            print(f"Esperimento {exp_id} SKIPPATO: {reason}")
            append_skip_log(failure_log_path, exp_id, reason)
            print(f"Failure log aggiornato: {failure_log_path}")
            print("Passo al prossimo esperimento...")
            continue

        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()
            failed_count += 1
            print(f"Esperimento {exp_id} FALLITO: {reason}")
            append_failure_log(failure_log_path, exp_id, reason, tb)
            print(f"Failure log aggiornato: {failure_log_path}")
            print("Passo al prossimo esperimento...")
            continue

    print(
        f"\nCompletati: {completed_count} | Skippati: {skipped_count} | Falliti: {failed_count}"
    )
