def load_and_split_layers(model_name, dataset_name, layer_indices, type_layer, stats, train_indices, test_indices, memmap_dir="memmap_cache"):
    """
    Versione OTTIMIZZATA 2025: zero picco RAM usando np.memmap + float16
    Non tocca nessun'altra parte del tuo pipeline.
    """
    print(f" Caricamento MEMMAP ottimizzato {model_name} [{type_layer}]: layers {layer_indices}...")

    os.makedirs(memmap_dir, exist_ok=True)

    total_samples = stats['total']
    hallucinated_set = set(stats['hallucinated_items'])

    # Label (una volta sola)
    y_full = np.zeros(total_samples, dtype=np.int8)
    y_full[list(hallucinated_set)] = 1
    y_train = y_full[train_indices]
    y_test  = y_full[test_indices]

    n_train = len(train_indices)
    n_test  = len(test_indices)

    # Trova layer effettivamente esistenti + calcola dimensione totale
    valid_layers = []
    total_dim = 0
    first_shape = None

    print(" Prima passata: rilevamento layer e calcolo dimensione...")
    for layer_idx in layer_indices:
        file_path = os.path.join(PROJECT_ROOT, CACHE_DIR_NAME, model_name, dataset_name,
                                 "activation_"+type_layer, f"layer{layer_idx}_activations.pt")
        if not os.path.exists(file_path):
            print(f" Warning: Layer {layer_idx} non trovato. Salto.")
            continue

        acts = torch.load(file_path, map_location='cpu')
        acts = acts[:total_samples] if acts.shape[0] > total_samples else acts

        if first_shape is None:
            first_shape = acts.shape[1:]

        # Calcola dimensione dopo flatten
        if acts.ndim > 2:
            dim = np.prod(acts.shape[1:])
        else:
            dim = acts.shape[1]
        total_dim += dim
        valid_layers.append((layer_idx, file_path))

        del acts
        gc.collect()

    if not valid_layers:
        raise ValueError(f"Nessun layer valido trovato per {model_name}")

    print(f" Trovati {len(valid_layers)} layer validi → {total_dim:,} feature totali per sample")

    # Crea memmap su disco in float16 (4× meno spazio)
    dtype = np.float16

    train_path = os.path.join(memmap_dir, f"{model_name}_{dataset_name}_{type_layer}_train.dat")
    test_path  = os.path.join(memmap_dir, f"{model_name}_{dataset_name}_{type_layer}_test.dat")

    X_train_mm = np.memmap(train_path, dtype=dtype, mode='w+', shape=(n_train, total_dim))
    X_test_mm  = np.memmap(test_path,  dtype=dtype, mode='w+', shape=(n_test,  total_dim))

    # Seconda passata: scrivi direttamente nei memmap
    print(" Seconda passata: scrittura su disco...")
    offset = 0

    for layer_idx, file_path in valid_layers:
        print(f"  Processing layer {layer_idx}...", end=" ")

        acts = torch.load(file_path, map_location='cpu')
        if acts.shape[0] > total_samples:
            acts = acts[:total_samples]

        # Converti subito in float16 e numpy
        if isinstance(acts, torch.Tensor):
            X_layer = acts.cpu().half().numpy()   # <--- float16!
        else:
            X_layer = acts.astype(np.float16)

        # Flatten se necessario (es. per attn maps o cls tokens con seq_len)
        if X_layer.ndim > 2:
            X_layer = X_layer.reshape(X_layer.shape[0], -1)

        layer_dim = X_layer.shape[1]
        end = offset + layer_dim

        # Scrivi direttamente nei memmap (zero copia in RAM!)
        X_train_mm[:, offset:end] = X_layer[train_indices]
        X_test_mm[:,  offset:end]  = X_layer[test_indices]

        print(f"done ({layer_dim} dims → offset {offset}:{end})")

        offset = end
        del X_layer, acts
        gc.collect()

    # Forza scrittura su disco
    X_train_mm.flush()
    X_test_mm.flush()

    print(f" Completato! Memmap salvati:")
    print(f"   Train → {train_path}  ({X_train_mm.shape}, {X_train_mm.nbytes/1e9:.2f} GB)")
    print(f"   Test  → {test_path}   ({X_test_mm.shape}, {X_test_mm.nbytes/1e9:.2f} GB)")

    return X_train_mm, X_test_mm, y_train, y_test

# ==================================================================
# 4. Pipeline 
# ==================================================================
def run_experiment_pipeline_cached(X_teacher, y_teacher, teacher_name,
                                   X_student, y_student, student_name, layer_type, config_name,
                                   patience=100, min_delta=1e-4):
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {layer_type.upper()} → {teacher_name} ← {student_name}")
    print(f"{'='*60}")

    # Dati già splittati
    X_A_train_full, X_A_test = X_teacher['X_train'], X_teacher['X_test']
    y_A_train_full, y_A_test = y_teacher['y_train'], y_teacher['y_test']
    X_B_train_full, X_B_test = X_student['X_train'], X_student['X_test']
    y_B_train_full, y_B_test = y_student['y_train'], y_student['y_test']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------------------------------
    # 1. Teacher Probing
    # --------------------------------------------------
    print("1. Training teacher probe...")
    probe_teacher = LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs', n_jobs=-1)
    probe_teacher.fit(X_A_train_full, y_A_train_full)
    
    # --- METRICHE TEACHER ---
    y_pred_teacher = probe_teacher.predict(X_A_test)
    cm_teacher = confusion_matrix(y_A_test, y_pred_teacher)
    acc_teacher = accuracy_score(y_A_test, y_pred_teacher)
    prec_teacher = precision_score(y_A_test, y_pred_teacher)
    rec_teacher = recall_score(y_A_test, y_pred_teacher)
    f1_teacher = f1_score(y_A_test, y_pred_teacher)
    print(f"   Teacher F1: {f1_teacher:.4f}")

    # --------------------------------------------------
    # 2. Alignment Training (Student → Teacher space)
    # --------------------------------------------------
    print("2. Training alignment network...")
    
    # Create Validation Split (10%)
    num_train = len(X_B_train_full)
    indices = np.arange(num_train)
    np.random.shuffle(indices)
    val_size = int(num_train * 0.1)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    X_B_train = X_B_train_full[train_indices]
    X_A_train = X_A_train_full[train_indices]
    
    X_B_val = X_B_train_full[val_indices]
    X_A_val = X_A_train_full[val_indices]

    train_dataset = AlignmentDataset(X_B_train, X_A_train)
    val_dataset = AlignmentDataset(X_B_val, X_A_val)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    
    aligner = AlignmentNetwork(
        input_dim=X_B_train.shape[1],
        output_dim=X_A_train.shape[1],
        dropout=0.1
    ).to(device)
    
    optimizer = optim.AdamW(aligner.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    epochs = 10000
    
    # Early Stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        aligner.train()
        epoch_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            projected = aligner(data)
            
            loss = nn.MSELoss(reduction='mean')(projected, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(aligner.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation
        aligner.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                projected = aligner(data)
                loss = nn.MSELoss(reduction='mean')(projected, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0 or epoch < 4:
            print(f"   Epoch {epoch+1:2d}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
        # Early Stopping Check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = aligner.state_dict()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"   Early stopping triggered at epoch {epoch+1}. Best Val Loss: {best_val_loss:.6f}")
            break
            
    # Load best model
    if best_model_state is not None:
        aligner.load_state_dict(best_model_state)
    
    # Save the best alignment network to disk
    model_save_dir = os.path.join("alignment_models", layer_type)
    os.makedirs(model_save_dir, exist_ok=True)
    model_filename = os.path.join(model_save_dir, f"{config_name}_aligner_{student_name}_to_{teacher_name}.pt")
    
    torch.save({
        'model_state_dict': aligner.state_dict(),
        'input_dim': X_B_train.shape[1],
        'output_dim': X_A_train.shape[1],
        'dropout': 0.1,
        'best_val_loss': best_val_loss,
        'layer_type': layer_type,
        'student_model': student_name,
        'teacher_model': teacher_name,
    }, model_filename)
    print(f"   ✓ Alignment network saved: {model_filename}")

    # --------------------------------------------------
    # 3. Evaluation: Student projected → Teacher probe
    # --------------------------------------------------
    print("3. Projecting student test set & evaluating...")
    aligner.eval()
    with torch.no_grad():
        X_B_test_tensor = torch.tensor(X_B_test, dtype=torch.float32).to(device)
        X_B_projected = aligner(X_B_test_tensor).cpu().numpy()
    
    y_pred_cross = probe_teacher.predict(X_B_projected)
    
    # --- METRICHE CROSS-MODEL ---
    cm_cross = confusion_matrix(y_B_test, y_pred_cross)
    acc_cross = accuracy_score(y_B_test, y_pred_cross)
    prec_cross = precision_score(y_B_test, y_pred_cross)
    rec_cross = recall_score(y_B_test, y_pred_cross)
    f1_cross = f1_score(y_B_test, y_pred_cross)
    
    print(f"\nFINAL RESULT:")
    print(f"   Teacher F1         : {f1_teacher:.4f}")
    print(f"   Student → Teacher F1: {f1_cross:.4f}")
    print(f"   Transfer gap       : {f1_teacher - f1_cross:.4f}")

    # --------------------------------------------------
    # Ritorna tutto
    # --------------------------------------------------
    return {
        "type": layer_type,
        "teacher_name": teacher_name,
        "student_name": student_name,
        "teacher": {
            "accuracy": acc_teacher,
            "precision": prec_teacher,
            "recall": rec_teacher,
            "f1": f1_teacher,
            "confusion_matrix": cm_teacher.tolist()
        },
        "student_on_teacher": {
            "accuracy": acc_cross,
            "precision": prec_cross,
            "recall": rec_cross,
            "f1": f1_cross,
            "confusion_matrix": cm_cross.tolist()
        }
    }


def plot_confusion_matrix(cm, layer_type, model_name="", save_dir="confusion_matrices"):
    """
    Plotta e salva la confusion matrix come immagine.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
                xticklabels=['Non-Hallucinated', 'Hallucinated'],
                yticklabels=['Non-Hallucinated', 'Hallucinated'])
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    title = f'Confusion Matrix - {layer_type.upper()} Layers'
    if model_name:
        title += f' ({model_name})'
    ax.set_title(title)
    
    plt.tight_layout()
    filename = os.path.join(save_dir, f'confusion_matrix_{layer_type}_{model_name}.png' if model_name else f'confusion_matrix_{layer_type}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Salvato: {filename}")