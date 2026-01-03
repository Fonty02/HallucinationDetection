"""
Universal Prober for LLM - Linear Analysis
Logic: Identical to FullLinear.ipynb (Hardcoded paths, Concatenation of layers).
"""

import json
import os
import gc
import random
import traceback
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split

# ==================================================================
# CONFIGURATION
# ==================================================================
class Config:
    # ROOT_DIR punta alla cartella principale del progetto (2 livelli sopra lo script)
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CACHE_DIR_NAME = "activation_cache"
    
    SEED = 42
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Configurazione identica al Notebook
    LAYER_CONFIG = {
        "belief_bank_facts": {
            "Qwen2.5-7B": {
                "attn": [14, 15, 17],
                "mlp": [14, 23, 25],
                "hidden": [15, 16, 17]
            },
            "Falcon3-7B-Base": {
                "attn": [18, 19, 26],
                "mlp": [18, 19, 20],
                "hidden": [17, 18, 21]
            },
            "save_dir": "Qwen_Falcon_BBF_Linear_Results"
        }
    }

# ==================================================================
# REPRODUCIBILITY & UTILS
# ==================================================================
def set_seed(seed=Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_balanced_indices(y, seed=Config.SEED):
    """
    Funzione di bilanciamento (undersampling) identica al notebook.
    """
    rng = np.random.RandomState(seed)
    unique_classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    
    selected_indices = []
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        if len(cls_indices) > min_count:
            sampled = rng.choice(cls_indices, size=min_count, replace=False)
            selected_indices.extend(sampled)
        else:
            selected_indices.extend(cls_indices)
            
    return np.sort(np.array(selected_indices))


def get_concordant_indices_and_undersample(y_a, y_b, seed=Config.SEED):
    """
    Trova campioni dove ENTRAMBI i modelli concordano sulla label,
    poi applica undersampling per bilanciare le classi.
    Identica alla logica del notebook.
    """
    assert len(y_a) == len(y_b), "Le label dei due modelli devono avere la stessa lunghezza"
    
    # Trova campioni concordanti
    concordant_mask = (y_a == y_b)
    concordant_indices = np.where(concordant_mask)[0]
    concordant_labels = y_a[concordant_mask]
    
    n_hall = np.sum(concordant_labels == 1)
    n_non_hall = np.sum(concordant_labels == 0)
    
    print(f"   Concordant samples: {len(concordant_indices)} (Hall: {n_hall}, Non-Hall: {n_non_hall})")
    
    # Undersampling
    min_count = min(n_hall, n_non_hall)
    rng = np.random.RandomState(seed)
    
    hall_idx = concordant_indices[concordant_labels == 1]
    non_hall_idx = concordant_indices[concordant_labels == 0]
    
    hall_sampled = rng.choice(hall_idx, size=min_count, replace=False)
    non_hall_sampled = rng.choice(non_hall_idx, size=min_count, replace=False)
    
    balanced_indices = np.concatenate([hall_sampled, non_hall_sampled])
    balanced_labels = np.concatenate([np.ones(min_count, dtype=int), np.zeros(min_count, dtype=int)])
    
    # Shuffle
    shuffle_idx = rng.permutation(len(balanced_indices))
    balanced_indices = balanced_indices[shuffle_idx]
    balanced_labels = balanced_labels[shuffle_idx]
    
    print(f"   After undersampling: {len(balanced_indices)} samples ({min_count} per class)")
    
    return balanced_indices, balanced_labels

def plot_confusion_matrix(cm, layer_type, model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Non-Hall', 'Hall'],
                yticklabels=['Non-Hall', 'Hall'])
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(f'CM - {layer_type.upper()} ({model_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"cm_{model_name}_{layer_type}.png"))
    plt.close()

# ==================================================================
# DATA MANAGEMENT (LOGICA NOTEBOOK)
# ==================================================================
class DataManager:
    @staticmethod
    def detect_structure_type(model_name, dataset_name, layer_type):
        """Rileva se la struttura ha cartelle hallucinated/not_hallucinated."""
        base_path = os.path.join(
            Config.ROOT_DIR, Config.CACHE_DIR_NAME, 
            model_name, dataset_name, f"activation_{layer_type}"
        )
        hallucinated_path = os.path.join(base_path, "hallucinated")
        return 'new' if os.path.isdir(hallucinated_path) else 'old'

    @staticmethod
    def load_activations_and_labels(model_name, dataset_name, layer, layer_type):
        """
        Carica attivazioni e label per un singolo layer.
        Supporta sia struttura vecchia che nuova (con hallucinated/not_hallucinated).
        Path reale: activation_cache/model_name/dataset_name/activation_{layer_type}/
        """
        structure = DataManager.detect_structure_type(model_name, dataset_name, layer_type)
        base_path = os.path.join(
            Config.ROOT_DIR, Config.CACHE_DIR_NAME, 
            model_name, dataset_name, f"activation_{layer_type}"
        )
        
        if structure == 'new':
            # Nuova struttura con cartelle separate
            hall_act_path = os.path.join(base_path, "hallucinated", f"layer{layer}_activations.pt")
            hall_ids_path = os.path.join(base_path, "hallucinated", f"layer{layer}_instance_ids.json")
            not_hall_act_path = os.path.join(base_path, "not_hallucinated", f"layer{layer}_activations.pt")
            not_hall_ids_path = os.path.join(base_path, "not_hallucinated", f"layer{layer}_instance_ids.json")
            
            hall_activations = torch.load(hall_act_path, map_location=Config.DEVICE)
            not_hall_activations = torch.load(not_hall_act_path, map_location=Config.DEVICE)
            
            with open(hall_ids_path, 'r') as f:
                hall_ids = json.load(f)
            with open(not_hall_ids_path, 'r') as f:
                not_hall_ids = json.load(f)
            
            if isinstance(hall_activations, torch.Tensor):
                hall_activations = hall_activations.cpu().numpy().astype(np.float32)
            if isinstance(not_hall_activations, torch.Tensor):
                not_hall_activations = not_hall_activations.cpu().numpy().astype(np.float32)
            
            X_concat = np.vstack([hall_activations, not_hall_activations])
            y_concat = np.concatenate([
                np.ones(hall_activations.shape[0], dtype=int),
                np.zeros(not_hall_activations.shape[0], dtype=int)
            ])
            ids_concat = np.array(hall_ids + not_hall_ids)
            
            # Ordina per instance_id
            sort_indices = np.argsort(ids_concat)
            X = X_concat[sort_indices]
            y = y_concat[sort_indices]
            instance_ids = ids_concat[sort_indices]
            
            return X, y, instance_ids
        else:
            # Vecchia struttura: file singoli + labels da generations/
            file_path = os.path.join(base_path, f"layer{layer}_activations.pt")
            activations = torch.load(file_path, map_location=Config.DEVICE)
            
            if isinstance(activations, torch.Tensor):
                X = activations.cpu().numpy().astype(np.float32)
            else:
                X = activations.astype(np.float32)
            
            labels_path = os.path.join(
                Config.ROOT_DIR, Config.CACHE_DIR_NAME, 
                model_name, dataset_name, "generations", "hallucination_labels.json"
            )
            with open(labels_path, 'r') as f:
                labels_data = json.load(f)
            
            y = np.array([item['is_hallucination'] for item in labels_data], dtype=int)
            instance_ids = np.arange(len(y))
            
            return X, y, instance_ids

    @staticmethod
    def load_concatenated_layers(dataset_name, model_name, layer_type, layer_indices):
        """
        Carica i layer specificati e li concatena (feature stacking),
        replicando la logica del notebook.
        """
        combined_X = []
        common_y = None
        
        print(f"   Loading {layer_type} layers {layer_indices} for {model_name}...")

        for idx in layer_indices:
            try:
                X_layer, y_layer, _ = DataManager.load_activations_and_labels(
                    model_name, dataset_name, idx, layer_type
                )
                
                # Controllo coerenza label tra layer diversi
                if common_y is None:
                    common_y = y_layer
                else:
                    if not np.array_equal(common_y, y_layer):
                        raise ValueError(f"Mismatch label nel layer {idx} per {model_name}")

                combined_X.append(X_layer)

            except FileNotFoundError as e:
                print(f"     [WARN] Layer {idx} non trovato: {e}. Skipping.")
                continue
            except Exception as e:
                print(f"     [ERROR] Errore caricamento layer {idx}: {e}")
                raise e

        if not combined_X:
            raise FileNotFoundError(f"Nessun dato trovato per {model_name} {layer_type}")

        # Concatenazione feature (asse 1)
        X_final = np.concatenate(combined_X, axis=1)
        print(f"   -> Final Combined Shape: {X_final.shape}")
        
        return X_final, common_y

# ==================================================================
# MAIN PIPELINE
# ==================================================================
def run_experiment_pipeline(X_t_train, y_t_train, X_t_test, y_t_test, 
                           X_s_test_raw, y_s_test,
                           X_align_t, X_align_s, scaler_s,
                           teacher_name, student_name, layer_type):
    """
    Pipeline identica al notebook:
    - Prober: addestrato sul dataset bilanciato del teacher
    - Allineamento: addestrato sui dati concordanti
    - Test cross-model: dati student → proiettati → valutati con prober teacher
    """
    # 1. Train Teacher Prober
    clf = LogisticRegression(max_iter=10000, class_weight='balanced', solver='lbfgs', n_jobs=-1, random_state=Config.SEED)
    clf.fit(X_t_train, y_t_train)
    
    # Eval Teacher
    pred_t = clf.predict(X_t_test)
    proba_t = clf.predict_proba(X_t_test)[:, 1]
    
    metrics_teacher = {
        "accuracy": accuracy_score(y_t_test, pred_t),
        "precision": precision_score(y_t_test, pred_t),
        "recall": recall_score(y_t_test, pred_t),
        "f1": f1_score(y_t_test, pred_t),
        "auroc": roc_auc_score(y_t_test, proba_t)
    }
    
    # 2. Train Alignment su campioni concordanti
    aligner = Ridge(alpha=1000.0, fit_intercept=False)
    aligner.fit(X_align_s, X_align_t)  # Student -> Teacher
    
    # 3. Cross-Model Eval
    X_s_proj = aligner.predict(scaler_s.transform(X_s_test_raw))
    pred_c = clf.predict(X_s_proj)
    proba_c = clf.predict_proba(X_s_proj)[:, 1]
    
    metrics_cross = {
        "accuracy": accuracy_score(y_s_test, pred_c),
        "precision": precision_score(y_s_test, pred_c),
        "recall": recall_score(y_s_test, pred_c),
        "f1": f1_score(y_s_test, pred_c),
        "auroc": roc_auc_score(y_s_test, proba_c)
    }
    
    print(f"     Teacher: Acc={metrics_teacher['accuracy']:.4f}, F1={metrics_teacher['f1']:.4f}, AUROC={metrics_teacher['auroc']:.4f}")
    print(f"     Cross:   Acc={metrics_cross['accuracy']:.4f}, F1={metrics_cross['f1']:.4f}, AUROC={metrics_cross['auroc']:.4f}")
    
    return {
        "layer_type": layer_type,
        "teacher_name": teacher_name,
        "student_name": student_name,
        "teacher": metrics_teacher,
        "cross": metrics_cross,
        "confusion_matrix_cross": confusion_matrix(y_s_test, pred_c).tolist()
    }


def main():
    set_seed(Config.SEED)
    print("=== Starting Analysis (Notebook Logic) ===")
    print(f"Root Dir: {os.path.abspath(Config.ROOT_DIR)}")
    
    for d_name, d_config in Config.LAYER_CONFIG.items():
        save_dir_name = d_config.get("save_dir", f"results_{d_name}")
        save_dir = os.path.join(os.path.dirname(__file__), save_dir_name)  # Cartella dove si trova il file Python
        os.makedirs(save_dir, exist_ok=True)
        
        models = [k for k in d_config.keys() if k != "save_dir"]
        if len(models) < 2:
            print("Serve una coppia Teacher-Student. Trovati:", models)
            continue
            
        model_A, model_B = models[0], models[1]
        print(f"\n{'='*60}")
        print(f"Dataset: {d_name} | Models: {model_A} vs {model_B}")
        print(f"{'='*60}")
        
        all_results = []
        layer_types = ["attn", "mlp", "hidden"]
        
        for l_type in layer_types:
            print(f"\n--- Processing {l_type.upper()} ---")
            
            try:
                # 1. Caricamento dati COMPLETI
                X_a_full, y_a = DataManager.load_concatenated_layers(d_name, model_A, l_type, d_config[model_A][l_type])
                X_b_full, y_b = DataManager.load_concatenated_layers(d_name, model_B, l_type, d_config[model_B][l_type])
                
                # 2. Trova campioni CONCORDANTI per allineamento
                print("\n   Finding concordant samples for alignment...")
                align_indices, align_labels = get_concordant_indices_and_undersample(y_a, y_b, Config.SEED)
                
                # Split alignment data (70/30, solo train usato per alignment)
                rng = np.random.RandomState(Config.SEED)
                n_align = len(align_indices)
                shuffled_align = rng.permutation(n_align)
                split_align = int(0.7 * n_align)
                align_train_local = shuffled_align[:split_align]
                
                X_align_a = X_a_full[align_indices][align_train_local]
                X_align_b = X_b_full[align_indices][align_train_local]
                
                # 3. Undersampling SEPARATO per ogni modello (per probing)
                print("\n   Balancing datasets for probing...")
                idx_a_bal = get_balanced_indices(y_a, Config.SEED)
                idx_b_bal = get_balanced_indices(y_b, Config.SEED)  # STESSO SEED come nel notebook
                
                X_a_bal, y_a_bal = X_a_full[idx_a_bal], y_a[idx_a_bal]
                X_b_bal, y_b_bal = X_b_full[idx_b_bal], y_b[idx_b_bal]
                
                print(f"   {model_A} balanced: {len(idx_a_bal)} ({np.sum(y_a_bal==1)} hall, {np.sum(y_a_bal==0)} non-hall)")
                print(f"   {model_B} balanced: {len(idx_b_bal)} ({np.sum(y_b_bal==1)} hall, {np.sum(y_b_bal==0)} non-hall)")
                
                # 4. Split train/test per ogni modello
                rng_a = np.random.RandomState(Config.SEED)
                rng_b = np.random.RandomState(Config.SEED + 1)
                
                shuffled_a = rng_a.permutation(len(X_a_bal))
                shuffled_b = rng_b.permutation(len(X_b_bal))
                
                split_a = int(0.7 * len(X_a_bal))
                split_b = int(0.7 * len(X_b_bal))
                
                X_a_train, X_a_test = X_a_bal[shuffled_a[:split_a]], X_a_bal[shuffled_a[split_a:]]
                y_a_train, y_a_test = y_a_bal[shuffled_a[:split_a]], y_a_bal[shuffled_a[split_a:]]
                
                X_b_train, X_b_test = X_b_bal[shuffled_b[:split_b]], X_b_bal[shuffled_b[split_b:]]
                y_b_train, y_b_test = y_b_bal[shuffled_b[:split_b]], y_b_bal[shuffled_b[split_b:]]
                
                # 5. Scaling
                scaler_a = StandardScaler().fit(X_a_train)
                scaler_b = StandardScaler().fit(X_b_train)
                scaler_align_a = StandardScaler().fit(X_align_a)
                scaler_align_b = StandardScaler().fit(X_align_b)
                
                X_a_train_sc = scaler_a.transform(X_a_train)
                X_a_test_sc = scaler_a.transform(X_a_test)
                X_b_train_sc = scaler_b.transform(X_b_train)
                X_b_test_sc = scaler_b.transform(X_b_test)
                
                X_align_a_sc = scaler_align_a.transform(X_align_a)
                X_align_b_sc = scaler_align_b.transform(X_align_b)
                
                # 6. Scenario A -> B
                print(f"\n   Scenario: {model_A} (Teacher) -> {model_B} (Student)")
                res_ab = run_experiment_pipeline(
                    X_a_train_sc, y_a_train, X_a_test_sc, y_a_test,
                    X_b_test, y_b_test,  # raw test per student
                    X_align_a_sc, X_align_b_sc, scaler_align_b,
                    model_A, model_B, l_type
                )
                all_results.append(res_ab)
                plot_confusion_matrix(np.array(res_ab["confusion_matrix_cross"]), l_type, f"{model_B}_on_{model_A}", save_dir)
                
                # 7. Scenario B -> A
                print(f"\n   Scenario: {model_B} (Teacher) -> {model_A} (Student)")
                res_ba = run_experiment_pipeline(
                    X_b_train_sc, y_b_train, X_b_test_sc, y_b_test,
                    X_a_test, y_a_test,
                    X_align_b_sc, X_align_a_sc, scaler_align_a,
                    model_B, model_A, l_type
                )
                all_results.append(res_ba)
                plot_confusion_matrix(np.array(res_ba["confusion_matrix_cross"]), l_type, f"{model_A}_on_{model_B}", save_dir)
                
                # Cleanup
                del X_a_full, X_b_full, X_a_bal, X_b_bal
                gc.collect()

            except Exception as e:
                print(f"!!! Error processing {l_type}: {e}")
                traceback.print_exc()
        
        # Salva risultati
        json_path = os.path.join(save_dir, f"full_linear_results_{d_name}.json")
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Risultati salvati in: {json_path}")

if __name__ == "__main__":
    main()