import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
import torch.nn.functional as F

# Add project root to Python path to enable absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.HallucinationDetection import HallucinationDetection
from truthx_model import MLPAE


def create_balanced_beliefbank_subset(project_root, num_positive=500, num_negative=500):
    """
    Crea un subset bilanciato di BeliefBankFacts.

    Args:
        project_root: Root directory del progetto
        num_positive: Numero di esempi positivi (label "yes")
        num_negative: Numero di esempi negativi (label "no")

    Returns:
        List di esempi bilanciati con etichette
    """
    from src.data.BeliefBankDataset import BeliefBankDataset

    dataset = BeliefBankDataset(
        project_root=project_root,
        model_type="demo",
        recreate_ids=True,
        data_type="facts",
    )

    positive_samples = []
    negative_samples = []

    print(f"Scanning {len(dataset)} samples...")

    for idx in tqdm(range(len(dataset)), desc="Filtering samples"):
        fact, label, instance_id = dataset[idx]

        if label == "yes":
            if len(positive_samples) < num_positive:
                positive_samples.append(
                    {
                        "question": fact,
                        "answer": "True",
                        "instance_id": instance_id,
                        "label": 1,
                    }
                )
        elif label == "no":
            if len(negative_samples) < num_negative:
                negative_samples.append(
                    {
                        "question": fact,
                        "answer": "False",
                        "instance_id": instance_id,
                        "label": 0,
                    }
                )

        if (
            len(positive_samples) >= num_positive
            and len(negative_samples) >= num_negative
        ):
            break

    balanced_subset = positive_samples + negative_samples

    import random

    random.shuffle(balanced_subset)

    print(
        f"Created balanced subset: {len(positive_samples)} positive, {len(negative_samples)} negative"
    )

    return balanced_subset


def save_subset_as_jsonl(subset, output_path):
    """Salva il subset come file JSONL."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for item in subset:
            f.write(json.dumps(item) + "\n")

    print(f"Subset saved to {output_path}")


def extract_activations_for_subset(
    project_root, llm_name, subset, activation_type="attn", quantization=False
):
    """
    Estrae le attivazioni per il subset usando HallucinationDetection.
    """
    print("\n" + "=" * 50)
    print("EXTRACTING ACTIVATIONS FOR BELIEFBANK SUBSET")
    print("=" * 50)

    detector = HallucinationDetection(project_dir=project_root)

    class BeliefBankSubset:
        def __init__(self, subset):
            self.subset = subset

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            item = self.subset[idx]
            return item["question"], item["answer"], item["instance_id"]

    detector.dataset = BeliefBankSubset(subset)
    detector.dataset_name = "belief_bank_subset"

    detector.load_llm(llm_name, quantization=quantization)

    model_name_safe = llm_name.replace("/", "_")

    detector.generation_save_dir = os.path.join(
        detector.CACHE_DIR_NAME, model_name_safe, "belief_bank_subset", "generations"
    )
    detector.mlp_save_dir = os.path.join(
        detector.CACHE_DIR_NAME, model_name_safe, "belief_bank_subset", "activation_mlp"
    )
    detector.attn_save_dir = os.path.join(
        detector.CACHE_DIR_NAME,
        model_name_safe,
        "belief_bank_subset",
        "activation_attn",
    )
    detector.hidden_save_dir = os.path.join(
        detector.CACHE_DIR_NAME,
        model_name_safe,
        "belief_bank_subset",
        "activation_hidden",
    )

    for path in [
        detector.generation_save_dir,
        detector.mlp_save_dir,
        detector.attn_save_dir,
        detector.hidden_save_dir,
    ]:
        os.makedirs(path, exist_ok=True)

    labels_path = os.path.join(
        detector.generation_save_dir, "hallucination_labels.json"
    )
    labels_to_save = [
        {
            "instance_id": item["instance_id"],
            "question": item["question"],
            "gold_answer": item["answer"],
            "is_hallucination": 1 - item["label"],
        }
        for item in subset
    ]
    with open(labels_path, "w") as f:
        json.dump(labels_to_save, f, indent=4)

    print(f"\nExtracting activations for {len(subset)} samples...")

    from src.model.InspectOutputContext import InspectOutputContext
    from src.model.prompts import PROMPT_QA as prompt
    import src.model.utils as ut

    target_layers = list(range(0, detector.llm.config.num_hidden_layers))

    module_names = []
    module_names += [f"model.layers.{idx}" for idx in target_layers]
    module_names += [f"model.layers.{idx}.self_attn" for idx in target_layers]
    module_names += [f"model.layers.{idx}.mlp" for idx in target_layers]

    for item in tqdm(subset, desc="Extracting activations"):
        question, instance_id = (
            item["question"],
            item["instance_id"],
        )

        model_input = prompt.format(question=question)
        tokens = detector.tokenizer(model_input, return_tensors="pt")
        attention_mask = (
            tokens["attention_mask"].to("cuda") if "attention_mask" in tokens else None
        )

        with InspectOutputContext(
            detector.llm,
            module_names,
            save_generation=True,
            save_dir=detector.generation_save_dir,
        ) as inspect:
            output = detector.llm.generate(
                input_ids=tokens["input_ids"].to("cuda"),
                max_new_tokens=detector.MAX_NEW_TOKENS,
                attention_mask=attention_mask,
                do_sample=False,
                top_p=0.95,
                temperature=0.1,
                pad_token_id=detector.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )

            generated_ids = output.sequences[0][tokens["input_ids"].shape[1] :]
            generated_text = detector.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            ut.save_generation_output(
                generated_text, model_input, instance_id, detector.generation_save_dir
            )

        for module, ac in inspect.catcher.items():
            ac_last = ac[0, -1].float().cpu()
            layer_idx = int(module.split(".")[2])

            save_name = f"layer{layer_idx}-id{instance_id}.pt"
            if "mlp" in module:
                save_path = os.path.join(detector.mlp_save_dir, save_name)
            elif "self_attn" in module:
                save_path = os.path.join(detector.attn_save_dir, save_name)
            else:
                save_path = os.path.join(detector.hidden_save_dir, save_name)

            torch.save(ac_last, save_path)
            del ac_last

        del tokens, output, generated_ids
        if attention_mask is not None:
            del attention_mask
        torch.cuda.empty_cache()
        import gc

        gc.collect()

    print("\nActivations saved to:")
    print(f"  - {detector.hidden_save_dir}")
    print(f"  - {detector.mlp_save_dir}")
    print(f"  - {detector.attn_save_dir}")
    print(f"  - {detector.generation_save_dir}")


def load_activations_for_virtual_layer(
    cache_dir, model_name, physical_layer_idx, activation_type
):
    """
    Carica le attivazioni per un singolo virtual layer.

    Seguendo il paper TruthX, ogni layer fisico genera 2 virtual layers:
    - Virtual layer 2*i: attivazioni attention del layer i
    - Virtual layer 2*i+1: attivazioni ffn/mlp del layer i

    Args:
        cache_dir: Directory della cache
        model_name: Nome del modello
        physical_layer_idx: Indice del layer fisico
        activation_type: Tipo di attivazione ("attn" o "mlp")

    Returns:
        hall_acts: Tensor delle attivazioni hallucinate
        not_hall_acts: Tensor delle attivazioni non hallucinate
    """
    labels_path = os.path.join(
        cache_dir,
        model_name,
        "belief_bank_subset",
        "generations",
        "hallucination_labels.json",
    )

    with open(labels_path, "r") as f:
        labels = json.load(f)

    hall_acts_list = []
    not_hall_acts_list = []

    base_path = os.path.join(
        cache_dir, model_name, "belief_bank_subset", f"activation_{activation_type}"
    )

    for label_info in labels:
        instance_id = label_info["instance_id"]
        activation_path = os.path.join(
            base_path, f"layer{physical_layer_idx}-id{instance_id}.pt"
        )

        if os.path.exists(activation_path):
            activation = torch.load(activation_path)

            if label_info["is_hallucination"] == 1:
                hall_acts_list.append(activation)
            else:
                not_hall_acts_list.append(activation)

    hall_acts = torch.stack(hall_acts_list) if hall_acts_list else torch.empty(0)
    not_hall_acts = (
        torch.stack(not_hall_acts_list) if not_hall_acts_list else torch.empty(0)
    )

    return hall_acts, not_hall_acts


def contrastive_loss_info_nce(representations, labels, temperature=0.1):
    """
    Implementazione completa di InfoNCE loss come da paper (Eq. 4-7).
    Garantisce che ci siano sempre coppie positive e negative nel batch.

    Args:
        representations: Tensor [N, D] delle rappresentazioni
        labels: Tensor [N] delle etichette (1 = truthful, 0 = hallucinated)
        temperature: Temperatura per scaling della similarità (default 0.1)

    Returns:
        Loss contrastiva media
    """
    device = representations.device
    n = representations.shape[0]

    # Verifica che ci siano entrambe le classi nel batch
    unique_labels = labels.unique()
    if len(unique_labels) < 2:
        # Se manca una classe, ritorna zero loss (non dovrebbe succedere con batch bilanciati)
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Normalizza le rappresentazioni
    representations = torch.nn.functional.normalize(representations, p=2, dim=1)

    # Calcola la matrice di similarità scalata per temperatura
    similarity_matrix = torch.matmul(representations, representations.T) / temperature

    # Maschera per escludere self-similarity
    mask_self = torch.eye(n, device=device).bool()

    total_loss = 0.0
    count = 0

    for i in range(n):
        # Trova campioni con la stessa etichetta (positivi) e etichetta diversa (negativi)
        same_label_mask = (labels == labels[i]) & ~mask_self[i]
        diff_label_mask = labels != labels[i]

        # Se non ci sono positivi o negativi, usa un fallback
        num_positives = same_label_mask.sum().item()
        num_negatives = diff_label_mask.sum().item()

        if num_positives == 0 or num_negatives == 0:
            # Fallback: usa self come ancora e tutti gli altri come negativi
            # Questo non dovrebbe succedere con batch bilanciati
            continue

        pos_sim = similarity_matrix[i][same_label_mask]
        neg_sim = similarity_matrix[i][diff_label_mask]

        # InfoNCE: log(sum(exp(pos))) - log(sum(exp(pos)) + sum(exp(neg)))
        numerator = torch.logsumexp(pos_sim, dim=0)
        denominator = torch.logsumexp(torch.cat([pos_sim, neg_sim]), dim=0)

        total_loss += -(numerator - denominator)
        count += 1

    # Se nessun campione è stato processato, ritorna zero
    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / count


def semantic_contrastive_loss(
    semantic_reps, truthful_labels, batch_indices, temperature=0.1
):
    """
    Contrastive loss per semantic space (Eq. 6).
    Per lo spazio semantico, vogliamo che rappresentazioni con lo stesso significato
    siano vicine indipendentemente dalla veridicità.

    Args:
        semantic_reps: Tensor [N, D] delle rappresentazioni semantiche
        truthful_labels: Tensor [N] delle etichette di veridicità
        batch_indices: Tensor [N] degli indici originali nel batch
        temperature: Temperatura per scaling della similarità (default 0.1)

    Returns:
        Loss contrastiva per lo spazio semantico
    """
    device = semantic_reps.device
    n = semantic_reps.shape[0]

    # Verifica che ci siano entrambe le classi
    unique_labels = truthful_labels.unique()
    if len(unique_labels) < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    semantic_reps = torch.nn.functional.normalize(semantic_reps, p=2, dim=1)
    similarity_matrix = torch.matmul(semantic_reps, semantic_reps.T) / temperature

    # Maschera per escludere self-similarity
    mask_self = torch.eye(n, device=device).bool()

    total_loss = 0.0
    count = 0

    for i in range(n):
        # Per semantic space: positivi = stessa semantica (stesso token), diversa veridicità
        # negativi = diversa semantica (diverso token), stessa veridicità
        same_token_mask = (batch_indices == batch_indices[i]) & (
            truthful_labels != truthful_labels[i]
        ) & ~mask_self[i]
        diff_token_mask = (batch_indices != batch_indices[i]) & (
            truthful_labels == truthful_labels[i]
        )

        num_positives = same_token_mask.sum().item()
        num_negatives = diff_token_mask.sum().item()

        if num_positives == 0 or num_negatives == 0:
            # Fallback: usa rappresentazioni con stessa etichetta come positivi
            # e diverse etichette come negativi
            same_label_mask = (truthful_labels == truthful_labels[i]) & ~mask_self[i]
            diff_label_mask = truthful_labels != truthful_labels[i]

            if same_label_mask.sum() == 0 or diff_label_mask.sum() == 0:
                continue

            pos_sim = similarity_matrix[i][same_label_mask]
            neg_sim = similarity_matrix[i][diff_label_mask]
        else:
            pos_sim = similarity_matrix[i][same_token_mask]
            neg_sim = similarity_matrix[i][diff_token_mask]

        numerator = torch.logsumexp(pos_sim, dim=0)
        denominator = torch.logsumexp(torch.cat([pos_sim, neg_sim]), dim=0)

        total_loss += -(numerator - denominator)
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / count


def train_truthx_on_beliefbank(args):
    """
    Training del modello TruthX usando BeliefBankFacts.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 50)
    print("STEP 1: CREATING BALANCED BELIEFBANK SUBSET")
    print("=" * 50)

    subset = create_balanced_beliefbank_subset(
        project_root=args.project_root,
        num_positive=args.num_positive,
        num_negative=args.num_negative,
    )

    subset_path = os.path.join(
        args.project_root, "data", "beliefbank", "beliefbank_subset_1000.jsonl"
    )
    save_subset_as_jsonl(subset, subset_path)

    # Controllo cache: se le attivazioni per questo modello e subset
    # esistono già nella cache, salta l'estrazione anche se
    # l'utente ha impostato il flag. Questo evita overhead e ripetizioni.
    def _activations_exist_in_cache(cache_dir, model_name, subset_len, activation_type):
        model_name_safe = model_name.replace("/", "_")
        base = os.path.join(cache_dir, model_name_safe, "belief_bank_subset")

        labels_path = os.path.join(base, "generations", "hallucination_labels.json")
        if not os.path.exists(labels_path):
            return False

        types_to_check = []
        if activation_type == "all":
            types_to_check = ["attn", "mlp"]
        else:
            types_to_check = [activation_type]

        for t in types_to_check:
            act_dir = os.path.join(base, f"activation_{t}")
            if not os.path.isdir(act_dir):
                return False

            # Raccogli gli instance_id trovati nei file (layer{idx}-id{instance}.pt)
            ids = set()
            for fn in os.listdir(act_dir):
                if not fn.endswith('.pt') or not fn.startswith('layer'):
                    continue
                try:
                    # formato previsto: layer{layer_idx}-id{instance_id}.pt
                    part = fn.split('-id')[-1]
                    instance_id = int(part.split('.pt')[0])
                    ids.add(instance_id)
                except Exception:
                    continue

            # Se non troviamo almeno tante istanze quante nel subset, ritorna False
            if len(ids) < subset_len:
                return False

        return True

    should_extract = args.extract_activations
    # Se la cache contiene già le attivazioni, non estrarre.
    if should_extract:
        subset_len = len(subset)
        if _activations_exist_in_cache(args.cache_dir, args.model_name, subset_len, args.activation_type):
            print("Activations already present in cache; skipping extraction.")
            should_extract = False

    if should_extract:
        print("\n" + "=" * 50)
        print("STEP 2: EXTRACTING ACTIVATIONS")
        print("=" * 50)

        extract_activations_for_subset(
            project_root=args.project_root,
            llm_name=args.model_name,
            subset=subset,
            activation_type=args.activation_type,
            quantization=args.quantization,
        )

    print("\n" + "=" * 50)
    print("STEP 3: LOADING ACTIVATIONS")
    print("=" * 50)

    all_pos_acts = []  # Lista di attivazioni positive per ogni virtual layer
    all_neg_acts = []  # Lista di attivazioni negative per ogni virtual layer
    virtual_layer_info = []  # Tiene traccia di (physical_layer, type) per ogni virtual layer

    model_name_safe = args.model_name.replace("/", "_")

    # Determina i target layers se non specificati
    if args.target_layers is None:
        # Prova a determinare il numero di layers dal modello
        # Legge i file di attivazione per scoprire quali layers esistono
        base_path = os.path.join(
            args.cache_dir, model_name_safe, "belief_bank_subset", "activation_attn"
        )
        if os.path.exists(base_path):
            layer_files = [f for f in os.listdir(base_path) if f.startswith("layer")]
            layer_indices = set()
            for f in layer_files:
                # Estrae l'indice del layer dal nome del file (es. "layer0-id123.pt" -> 0)
                try:
                    layer_idx = int(f.split("-")[0].replace("layer", ""))
                    layer_indices.add(layer_idx)
                except (ValueError, IndexError):
                    pass
            args.target_layers = sorted(list(layer_indices))
            print(f"Auto-detected {len(args.target_layers)} layers from activation cache")
        else:
            # Fallback: usa 32 layers (comune per molti modelli)
            args.target_layers = list(range(32))
            print(f"Using default {len(args.target_layers)} layers")

    # Determina i tipi di attivazione da usare
    # Per seguire il paper, usiamo attn e mlp come virtual layers separati
    if args.activation_type == "all":
        # Paper usa attn e ffn (mlp) come virtual layers separati
        activation_types = ["attn", "mlp"]
    elif args.activation_type == "hidden":
        # Se si vuole usare solo hidden, trattalo come un singolo tipo
        activation_types = ["hidden"]
    else:
        activation_types = [args.activation_type]

    num_physical_layers = len(args.target_layers)
    num_virtual_layers = num_physical_layers * len(activation_types)

    print(f"Physical layers: {num_physical_layers}")
    print(f"Activation types: {activation_types}")
    print(f"Virtual layers: {num_virtual_layers} (physical_layers x activation_types)")
    print("\nVirtual layer mapping (following TruthX paper):")
    print("  Virtual layer 2*i = physical layer i, attn")
    print("  Virtual layer 2*i+1 = physical layer i, ffn/mlp")

    # Carica le attivazioni per ogni virtual layer
    # Ordine: (layer0, attn), (layer0, mlp), (layer1, attn), (layer1, mlp), ...
    for physical_layer_idx in tqdm(args.target_layers, desc="Loading activations"):
        for act_type in activation_types:
            hall_acts, not_hall_acts = load_activations_for_virtual_layer(
                args.cache_dir, model_name_safe, physical_layer_idx, act_type
            )
            all_neg_acts.append(hall_acts)
            all_pos_acts.append(not_hall_acts)
            virtual_layer_info.append((physical_layer_idx, act_type))

            virtual_idx = len(virtual_layer_info) - 1
            print(f"  Virtual layer {virtual_idx}: physical layer {physical_layer_idx}, {act_type} - "
                  f"{len(hall_acts)} hall, {len(not_hall_acts)} not_hall")

    # Verifica che abbiamo caricato attivazioni valide
    if not all_pos_acts or all_pos_acts[0].numel() == 0:
        raise ValueError(
            "No activations found. Please run with --extract_activations first."
        )

    hidden_size = all_pos_acts[0].shape[-1]
    num_virtual_layers = len(all_pos_acts)  # Numero totale di virtual layers

    print(f"\nHidden size: {hidden_size}")
    print(f"Num virtual layers: {num_virtual_layers}")
    print(f"Positive samples per virtual layer: {all_pos_acts[0].shape[0]}")
    print(f"Negative samples per virtual layer: {all_neg_acts[0].shape[0]}")

    print("\n" + "=" * 50)
    print("STEP 4: INITIALIZING TRUTHX MODEL")
    print("=" * 50)

    # Parse hidden dims seguendo l'implementazione ufficiale di TruthX
    # Se la stringa è vuota, usa una lista vuota (nessun hidden layer)
    # Altrimenti, split e converti in int
    semantic_hidden_dims = (
        [int(x) for x in args.semantic_hidden_dims.split(",")]
        if args.semantic_hidden_dims != ""
        else []
    )
    truthful_hidden_dims = (
        [int(x) for x in args.truthful_hidden_dims.split(",")]
        if args.truthful_hidden_dims != ""
        else []
    )
    decoder_hidden_dims = (
        [int(x) for x in args.decoder_hidden_dims.split(",")]
        if args.decoder_hidden_dims != ""
        else []
    )

    print(f"Semantic hidden dims: {semantic_hidden_dims}")
    print(f"Truthful hidden dims: {truthful_hidden_dims}")
    print(f"Decoder hidden dims: {decoder_hidden_dims}")

    model = MLPAE(
        in_channels=hidden_size,
        semantic_latent_dim=args.semantic_latent_dim,
        truthful_latent_dim=args.truthful_latent_dim,
        semantic_hidden_dims=semantic_hidden_dims,
        truthful_hidden_dims=truthful_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("\n" + "=" * 50)
    print("STEP 5: TRAINING")
    print("=" * 50)

    print("Starting training...")
    for epoch in range(args.num_epochs):
        total_recon_loss = 0
        total_contrastive_loss = 0
        total_editing_loss = 0
        total_batches = 0

        # Itera su tutti i virtual layers (attn e mlp separatamente)
        for virtual_layer_idx in range(num_virtual_layers):
            pos_acts = all_pos_acts[virtual_layer_idx].to(device)
            neg_acts = all_neg_acts[virtual_layer_idx].to(device)

            all_acts = torch.cat([pos_acts, neg_acts], dim=0)
            labels = torch.cat(
                [torch.ones(pos_acts.shape[0]), torch.zeros(neg_acts.shape[0])]
            ).to(device)

            batch_indices = torch.arange(all_acts.shape[0]).to(device)

            perm = torch.randperm(all_acts.shape[0])
            all_acts = all_acts[perm]
            labels = labels[perm]
            batch_indices = batch_indices[perm]

            for i in range(0, all_acts.shape[0], args.batch_size):
                batch = all_acts[i : i + args.batch_size]
                batch_labels = labels[i : i + args.batch_size]
                batch_idx = batch_indices[i : i + args.batch_size]

                optimizer.zero_grad()

                output, input_x, semantic_rep, truthful_rep = model(batch)

                recon_loss = model.loss_function(output, input_x)

                contrastive_loss_truth = contrastive_loss_info_nce(
                    truthful_rep, batch_labels, temperature=args.temperature
                )

                contrastive_loss_sem = semantic_contrastive_loss(
                    semantic_rep, batch_labels, batch_idx, temperature=args.temperature
                )

                contrastive_loss = contrastive_loss_truth + contrastive_loss_sem

                editing_loss = torch.tensor(0.0, device=device)
                pos_mask = batch_labels == 1
                neg_mask = batch_labels == 0

                if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                    pos_truthful = truthful_rep[pos_mask]
                    neg_truthful = truthful_rep[neg_mask]

                    min_pos = min(pos_mask.sum(), neg_mask.sum())

                    if min_pos > 0:
                        pos_batch = batch[pos_mask][:min_pos]
                        neg_batch = batch[neg_mask][:min_pos]
                        pos_truth = pos_truthful[:min_pos]
                        neg_truth = neg_truthful[:min_pos]

                        recon_pos_to_neg = model(pos_batch, neg_truth)[0]
                        recon_neg_to_pos = model(neg_batch, pos_truth)[0]

                        editing_loss = F.mse_loss(
                            recon_pos_to_neg, neg_batch
                        ) + F.mse_loss(recon_neg_to_pos, pos_batch)

                loss = (
                    recon_loss
                    + args.contrastive_weight * contrastive_loss
                    + args.editing_weight * editing_loss
                )

                loss.backward()
                optimizer.step()

                total_recon_loss += recon_loss.item()
                total_contrastive_loss += contrastive_loss.item()
                total_editing_loss += editing_loss.item()
                total_batches += 1

        avg_recon = total_recon_loss / total_batches
        avg_contrast = total_contrastive_loss / total_batches
        avg_edit = total_editing_loss / total_batches

        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"  Recon Loss: {avg_recon:.4f}")
        print(f"  Contrastive Loss: {avg_contrast:.4f}")
        print(f"  Editing Loss: {avg_edit:.4f}")
        print(
            f"  Total Loss: {avg_recon + args.contrastive_weight * avg_contrast + args.editing_weight * avg_edit:.4f}"
        )

    print("\n" + "=" * 50)
    print("STEP 6: CALCULATING VIRTUAL LAYER CENTERS AND RANKING")
    print("=" * 50)

    # Calcola i centri per ogni virtual layer (come nel paper TruthX)
    pos_centers = []
    neg_centers = []

    with torch.no_grad():
        for virtual_layer_idx in range(num_virtual_layers):
            pos_acts = all_pos_acts[virtual_layer_idx].to(device)
            neg_acts = all_neg_acts[virtual_layer_idx].to(device)

            pos_truthful = model.encode_truthful(pos_acts).mean(dim=0)
            neg_truthful = model.encode_truthful(neg_acts).mean(dim=0)

            pos_centers.append(pos_truthful)
            neg_centers.append(neg_truthful)

    pos_centers = torch.stack(pos_centers)
    neg_centers = torch.stack(neg_centers)

    # Calcola il ranking basato sulla separabilità tra centri positivi e negativi
    rank = []
    for i in range(num_virtual_layers):
        dist = torch.norm(pos_centers[i] - neg_centers[i]).item()
        rank.append((i, dist))

    rank = sorted(rank, key=lambda x: -x[1])
    rank_indices = [i for i, _ in rank]

    print("\nVirtual layer ranking (by separability):")
    print("(Following TruthX: virtual_idx = 2*physical_layer + (0 for attn, 1 for mlp))")
    for idx, (virtual_layer_idx, dist) in enumerate(rank[:20]):
        physical_layer, act_type = virtual_layer_info[virtual_layer_idx]
        print(f"  Rank {idx}: Virtual layer {virtual_layer_idx} "
              f"(physical {physical_layer}, {act_type}), Distance: {dist:.4f}")
    for idx, (layer_idx, dist) in enumerate(rank[:10]):
        print(f"  Rank {idx}: Layer {layer_idx}, Distance: {dist:.4f}")

    print("\n" + "=" * 50)
    print("STEP 7: SAVING MODEL")
    print("=" * 50)

    checkpoint = {
        "state_dict": model.state_dict(),
        "pos_center": pos_centers,
        "neg_center": neg_centers,
        "rank": rank_indices,
        "virtual_layer_info": virtual_layer_info,  # Mapping virtual_idx -> (physical_layer, act_type)
        "num_virtual_layers": num_virtual_layers,
        "args": args,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "truthx_beliefbank_model.pt")
    torch.save(checkpoint, output_path)
    print(f"\nModel saved to {output_path}")

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_root", type=str, default=".", help="Project root directory"
    )
    parser.add_argument("--cache_dir", type=str, default="activation_cache_truthx")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="LLM model name",
    )
    parser.add_argument(
        "--activation_type",
        type=str,
        default="all",
        choices=["hidden", "mlp", "attn", "all"],
        help="Type of activations to use. Use 'all' to concatenate attn, mlp and hidden.",
    )
    parser.add_argument("--output_dir", type=str, default="truthx_models")

    parser.add_argument(
        "--num_positive",
        type=int,
        default=500,
        help="Number of positive (truthful) samples",
    )
    parser.add_argument(
        "--num_negative",
        type=int,
        default=500,
        help="Number of negative (hallucinated) samples",
    )
    parser.add_argument(
        "--extract_activations",
        action="store_true",
        default=False,
        help="Whether to extract activations",
    )
    parser.add_argument(
        "--quantization", action="store_true", help="Use 4-bit quantization", default=True
    )

    parser.add_argument("--semantic_latent_dim", type=int, default=1024)
    parser.add_argument("--truthful_latent_dim", type=int, default=1024)
    parser.add_argument(
        "--semantic_hidden_dims",
        type=str,
        default="",
        help="Comma-separated hidden dims for semantic encoder (e.g., '2048,1024'). Empty string means no hidden layers.",
    )
    parser.add_argument(
        "--truthful_hidden_dims",
        type=str,
        default="",
        help="Comma-separated hidden dims for truthful encoder (e.g., '2048,1024'). Empty string means no hidden layers.",
    )
    parser.add_argument(
        "--decoder_hidden_dims",
        type=str,
        default="",
        help="Comma-separated hidden dims for decoder (e.g., '1024,2048'). Empty string means no hidden layers.",
    )

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--contrastive_weight", type=float, default=1.0)
    parser.add_argument("--editing_weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--target_layers",
        type=str,
        default=None,
        help="Comma-separated list of layer indices (e.g., '0,1,2'). If None, uses all layers.",
    )

    args = parser.parse_args()
    # target_layers verrà determinato dopo il caricamento del modello se non specificato
    if args.target_layers is not None:
        args.target_layers = [int(x) for x in args.target_layers.split(",")]

    train_truthx_on_beliefbank(args)