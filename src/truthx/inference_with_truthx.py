import os
import json
import torch
import argparse
import csv
import sys
from tqdm import tqdm
from typing import List, Tuple
import time
from glob import glob
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to Python path before importing local modules
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from truthx_model import TruthX, LLMArchitectureDetector  # noqa: E402
from src.model.HallucinationDetection import HallucinationDetection  # noqa: E402

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
# =============================================================================

EXPERIMENTS= {
    "Qwen_baseline_BBF" : {
        "type" : "baseline",
        "model_to_evaluate" : "Qwen/Qwen2.5-7B",
        "dataset_to_evaluate" : "belief_bank_facts",
        "useTruthX" : False,
        "truthX_autoencoder": None,
        "truthX_steering_vectors": None,
        "truthX_dataset": None
    },
    "Qwen_TruthX_BBF" : {
        "type" : "TruthX_Standard",
        "model_to_evaluate" : "Qwen/Qwen2.5-7B",
        "dataset_to_evaluate" : "belief_bank_facts",
        "useTruthX" : True,
        "truthX_autoencoder": "AutoEncoder/belief_bank_facts/autoencoder_Qwen_Qwen2.5-7B_pairs2000_cw1.0.pt",
        "truthX_steering_vectors": "SteeringVectors/belief_bank_facts/steering_vectors_Qwen_Qwen2.5-7B_pairs2000_cw1.0.pt",
        "truthX_dataset": "belief_bank_facts"
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
        
        # Carica il dataset BeliefBank per conoscere la dimensione totale
        # Questo è necessario per mappare gli indici delle coppie agli indici originali
        from src.data.BeliefBankDataset import BeliefBankDataset
        full_dataset = BeliefBankDataset(
            project_root=project_dir,
            model_type="demo",
            recreate_ids=True,
            data_type="facts"  # Assumiamo facts, ma potrebbe essere diverso
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


def evaluate_baseline_from_cache(experiment_id, config, num_samples, project_dir):
    """
    Recupera i risultati per la baseline direttamente dalla cache esistente
    senza eseguire inferenza.
    ESCLUDE gli instance_ids usati nel training/validation per evitare data leakage.
    """
    print(f"[{experiment_id}] Valutazione Baseline (da Cache)...")
    
    # Inizializza la classe di supporto per trovare i percorsi corretti
    # La baseline usa activation_cache, non activation_cache_truthx
    hd = HallucinationDetection(project_dir, cache_dir_name="activation_cache")
    hd.llm_name = config.get("model_to_evaluate")
    hd.dataset_name = config.get("dataset_to_evaluate")
    hd._create_folders_if_not_exists()
    
    gen_dir = hd.generation_save_dir
    hallucination_labels_path = os.path.join(gen_dir, "hallucination_labels.json")
    
    print(f"Cercando file labels in: {hallucination_labels_path}")
    
    if not os.path.exists(hallucination_labels_path):
        print(f"File hallucination_labels.json non trovato: {hallucination_labels_path}")
        return None
    
    try:
        # Carica il file hallucination_labels.json
        with open(hallucination_labels_path, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)
        
        # Carica training config per escludere instance_ids usati nel train/val
        excluded_ids, training_cfg = load_training_config(
            project_dir,
            config.get("model_to_evaluate"),
            config.get("dataset_to_evaluate")
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
        
        # Filtra per num_samples se specificato
        if num_samples > 0:
            labels_data = labels_data[:num_samples]
        
        evaluated_samples = len(labels_data)
        # Conta il numero di allucinazioni (is_hallucination = 1)
        n_hallucinations = sum(item.get("is_hallucination", 0) for item in labels_data)
        
        rate = n_hallucinations / evaluated_samples if evaluated_samples > 0 else 0
        
        print(f"Valutati {evaluated_samples} campioni, {n_hallucinations} allucinazioni ({rate*100:.2f}%)")
        
        # Prova a recuperare metadati del training (se disponibili)
        model_name_safe = config.get("model_to_evaluate").replace("/", "_")
        dataset_name = config.get("dataset_to_evaluate")
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
            "model": config.get("model_to_evaluate"),
            "dataset": config.get("dataset_to_evaluate"),
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

    except Exception as e:
        print(f"Errore nel recupero cache baseline: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_truthx_experiment(experiment_id, config, num_samples, project_dir):
    """
    Esegue l'inferenza con TruthX attivo. Non salva attivazioni pesanti, 
    calcola solo se l'output è allucinato.
    ESCLUDE gli instance_ids usati nel training/validation per evitare data leakage.
    """
    print(f"[{experiment_id}] Valutazione TruthX (Inferenza attiva)...")
    
    model_name = config.get("model_to_evaluate")
    dataset_name = config.get("dataset_to_evaluate")
    autoencoder_path = config.get("truthX_autoencoder")
    steering_vectors_path = config.get("truthX_steering_vectors")
    
    # 1. Caricamento Modello e Tokenizer
    print(f"Caricamento modello: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Caricamento Dataset
    print(f"Caricamento dataset: {dataset_name}")
    hd = HallucinationDetection(project_dir, cache_dir_name="activation_cache_truthx")
    
    if dataset_name == "belief_bank_facts":
        hd.load_dataset(dataset_name="belief_bank", belief_bank_data_type="facts")
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
    
    # If config provided explicit paths, prefer them. Otherwise try to locate steering vectors similarly to AE discovery
    sv_full_path = None
    if autoencoder_path and steering_vectors_path:
        ae_full_path = os.path.join(project_dir, autoencoder_path)
        sv_full_path = os.path.join(project_dir, steering_vectors_path)
    else:
        # Try to find steering vectors in SteeringVectors/<dataset_name>/ matching model_name_safe
        model_name_safe = model_name.replace("/", "_")
        sv_search_dir = os.path.join(project_dir, "SteeringVectors", dataset_name)
        sv_pattern = os.path.join(sv_search_dir, f"steering_vectors_{model_name_safe}_*.pt")
        sv_matches = glob(sv_pattern)
        if sv_matches:
            sv_full_path = max(sv_matches, key=os.path.getmtime)
            print(f"Found steering vectors file: {sv_full_path}")
        else:
            legacy_sv = os.path.join(sv_search_dir, f"steering_vectors_{model_name_safe}.pt")
            if os.path.exists(legacy_sv):
                sv_full_path = legacy_sv
            else:
                sv_full_path = None
        
        # Also try to find autoencoder
        ae_search_dir = os.path.join(project_dir, "AutoEncoder", dataset_name)
        ae_pattern = os.path.join(ae_search_dir, f"autoencoder_{model_name_safe}_*.pt")
        ae_matches = glob(ae_pattern)
        if ae_matches:
            ae_full_path = max(ae_matches, key=os.path.getmtime)
            print(f"Found autoencoder file: {ae_full_path}")
        else:
            legacy_ae = os.path.join(ae_search_dir, f"autoencoder_{model_name_safe}.pt")
            if os.path.exists(legacy_ae):
                ae_full_path = legacy_ae

    if ae_full_path and sv_full_path and os.path.exists(ae_full_path) and os.path.exists(sv_full_path):
        print(f"Caricamento Autoencoder TruthX da: {ae_full_path}")
        print(f"Caricamento Steering Vectors da: {sv_full_path}")
        truthx_editor = TruthX(
            autoencoder_path=ae_full_path,
            steering_vectors_path=sv_full_path,
            hidden_size=hidden_size,
            edit_strength=1,  # α = 1.0 come da paper per open-ended
            top_layers=1  # k = 10 come da paper
        )
        print("TruthX Editor inizializzato con successo.")
        print(f"Top-{truthx_editor.top_layers} layers da editare: {truthx_editor.rank[:truthx_editor.top_layers]}")
    else:
        if ae_full_path and not os.path.exists(ae_full_path):
            print(f"Warning: Autoencoder non trovato: {ae_full_path}")
        if sv_full_path and not os.path.exists(sv_full_path):
            print(f"Warning: Steering vectors non trovati: {sv_full_path}")
        print("Si procede senza editing TruthX (baseline).")
    
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
    
    from src.model.prompts import PROMPT_QA

    inference_start_time = time.time()
    for idx in tqdm(valid_indices):
        fact, label, instance_id = dataset[idx]
        
        # Costruisci il prompt come in HallucinationDetection
        question = PROMPT_QA.format(question=fact)
        
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        
        # Genera la risposta
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=5,
                do_sample=False
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        generated_text = generated_text.strip().lower()
        
        # Valuta se è allucinato
        expected_answer = label.lower()
        
        is_hallucinated = expected_answer not in generated_text
        
        if is_hallucinated:
            n_hallucinations += 1
        total_evaluated += 1
        
        # Salva il risultato
        inference_results.append({
            "instance_id": instance_id,
            "prompt": fact,
            "generated_answer": generated_text,
            "expected_answer": expected_answer,
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

    suffix = f"pairs{num_pairs_val}_cw{cw_str}"

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
    
    args = parser.parse_args()
    
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

    # 2. Ciclo sugli esperimenti
    for exp_id, exp_config in all_experiments.items():
        print(f"\n{'='*40}")
        print(f"RUNNING: {exp_id}")
        print(f"{'='*40}")
        
        result_row = None
        
        # Caso A: Baseline (Usa Cache)
        if exp_config.get("type") == "baseline" or exp_config.get("useTruthX") is False:
            result_row = evaluate_baseline_from_cache(
                exp_id, exp_config, args.num_samples, args.project_dir
            )
            # Se la cache fallisce, si potrebbe voler eseguire l'inferenza normale senza TruthX
            if result_row is None:
                print("Cache non trovata. Esecuzione inferenza standard (senza TruthX)...")
                result_row = evaluate_truthx_experiment(
                    exp_id, exp_config, args.num_samples, args.project_dir
                )
        
        # Caso B: TruthX (Inferenza attiva)
        else:
            result_row = evaluate_truthx_experiment(
                exp_id, exp_config, args.num_samples, args.project_dir
            )
            
        # 3. Salvataggio Risultati
        if result_row:
            save_to_csv(args.output_csv, result_row)