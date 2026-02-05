import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import logging
import copy
import time

# Add project root to Python path to enable absolute imports
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.HallucinationDetection import HallucinationDetection
from truthx_model import MLPAE


# =============================================================================
# PAIRED ACTIVATIONS DATASET
# =============================================================================


class PairedActivationsDataset(Dataset):
    """
    Dataset che carica coppie di attivazioni (truthful, hallucinated).

    Ogni item restituisce una tupla (truthful_vector, hallucinated_vector)
    dove entrambi i vettori derivano dalla stessa domanda/contesto ma con
    valori di verità opposti.

    Questo è fondamentale per il Disentangled Representation Learning di TruthX:
    - Il Semantic Encoder deve apprendere che il contenuto è lo stesso
      indipendentemente dalla veridicità
    - Il Truthful Encoder deve distinguere tra risposte veritiere e false
    """

    def __init__(
        self,
        cache_dir: str,
        model_name: str,
        layer_idx: int,
        activation_type: str,
        pair_mapping: dict = None,
    ):
        """
        Args:
            cache_dir: Directory della cache delle attivazioni
            model_name: Nome del modello (safe version, con / sostituiti)
            layer_idx: Indice del layer fisico
            activation_type: Tipo di attivazione ("attn", "mlp", "hidden")
            pair_mapping: Dizionario che mappa instance_id_pos -> instance_id_neg
                         per creare le coppie. Se None, viene inferito dai dati.
        """
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.layer_idx = layer_idx
        self.activation_type = activation_type

        # Carica le etichette per determinare quali campioni sono positivi/negativi
        labels_path = os.path.join(
            cache_dir,
            model_name,
            "belief_bank_subset",
            "generations",
            "hallucination_labels.json",
        )

        with open(labels_path, "r") as f:
            self.labels = json.load(f)

        # Costruisci il mapping tra coppie positive e negative
        self.pairs = self._build_pairs(pair_mapping)

        # Path base per le attivazioni
        self.activation_path = os.path.join(
            cache_dir, model_name, "belief_bank_subset", f"activation_{activation_type}"
        )

    def _build_pairs(self, pair_mapping: dict = None) -> list:
        """
        Costruisce le coppie (pos_instance_id, neg_instance_id).

        Nel BeliefBank, fact e negated_fact hanno lo stesso contenuto semantico
        ma verità opposta. Le coppie sono naturalmente allineate dall'ordine
        in cui vengono create dal BeliefBankDataset.

        Se non viene fornito un mapping esplicito, assumiamo che i primi N/2
        campioni siano truthful e i secondi N/2 siano le loro negazioni
        (come creato da extend_with_negated_facts).
        """
        if pair_mapping is not None:
            # Usa il mapping esplicito fornito
            return [(pos_id, neg_id) for pos_id, neg_id in pair_mapping.items()]

        # Separa campioni positivi (truthful) e negativi (hallucinated).
        # USIAMO SOLO INFORMAZIONI PROVENIENTI DA BELIEF BANK:
        # - Preferiamo il campo esplicito 'label' (creato in create_paired_beliefbank_subset).
        # - Se non ci sono 'label', proviamo a inferire dalle instance_id (parity):
        #   i positivi dovrebbero avere id pari e i negativi id dispari (pattern usato)
        # Non usiamo 'is_hallucination' né campi generici 'answer' per determinare pos/neg.
        positive_ids = []
        negative_ids = []

        for label_info in self.labels:
            instance_id = label_info.get("instance_id")

            # 1) Campo esplicito 'label' (1 = truthful / positive)
            if "label" in label_info:
                try:
                    if int(label_info["label"]) == 1:
                        positive_ids.append(instance_id)
                    else:
                        negative_ids.append(instance_id)
                    continue
                except Exception:
                    logging.getLogger(__name__).warning(
                        f"Invalid 'label' for instance {instance_id}; skipping."
                    )
                    continue

        # Se non abbiamo trovato label esplicite, proviamo ad inferire dalla parity degli id
        if len(positive_ids) == 0 and len(negative_ids) == 0:
            ids = []
            for info in self.labels:
                iid = info.get("instance_id")
                try:
                    ids.append(int(iid))
                except Exception:
                    pass

            if not ids:
                raise ValueError(
                    "Could not determine pos/neg samples: no 'label' field and instance ids are not numeric. Please pass explicit pair_mapping."
                )

            evens = sorted([i for i in ids if i % 2 == 0])
            odds = sorted([i for i in ids if i % 2 == 1])

            if len(evens) == len(odds) and len(evens) > 0:
                positive_ids = evens
                negative_ids = odds
                logging.getLogger(__name__).info(
                    f"Inferred pos/neg by id parity: {len(positive_ids)} pairs found."
                )
            else:
                raise ValueError(
                    "Could not deterministically infer pos/neg from instance ids parity. Please provide an explicit pair_mapping."
                )

        # Crea coppie matchando per indice (deterministico): positive_ids[i] ↔ negative_ids[i]
        min_len = min(len(positive_ids), len(negative_ids))

        if min_len == 0:
            raise ValueError(
                "No paired samples found. Ensure labels contain 'label' or pass explicit pair_mapping."
            )

        if len(positive_ids) != len(negative_ids):
            logging.getLogger(__name__).warning(
                f"Number of positives ({len(positive_ids)}) and negatives ({len(negative_ids)}) differ. Truncating to {min_len} pairs."
            )

        pairs = list(zip(positive_ids[:min_len], negative_ids[:min_len]))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Restituisce una coppia (truthful_activation, hallucinated_activation).
        """
        pos_id, neg_id = self.pairs[idx]

        pos_path = os.path.join(
            self.activation_path, f"layer{self.layer_idx}-id{pos_id}.pt"
        )
        neg_path = os.path.join(
            self.activation_path, f"layer{self.layer_idx}-id{neg_id}.pt"
        )

        pos_activation = torch.load(pos_path, map_location="cpu")
        neg_activation = torch.load(neg_path, map_location="cpu")

        return pos_activation, neg_activation, pos_id, neg_id


class AggregatedPairedDataset(Dataset):
    """
    Dataset che carica coppie di attivazioni pre-aggregate in tensori.
    Più efficiente per il training quando le attivazioni sono già in memoria.
    """

    def __init__(self, pos_activations: torch.Tensor, neg_activations: torch.Tensor):
        """
        Args:
            pos_activations: Tensor [N, D] delle attivazioni positive (truthful)
            neg_activations: Tensor [N, D] delle attivazioni negative (hallucinated)

        Nota: Si assume che pos_activations[i] e neg_activations[i] siano
              semanticamente correlati (stessa domanda, risposta opposta).
        """
        assert pos_activations.shape[0] == neg_activations.shape[0], (
            "Positive and negative activations must have the same number of samples"
        )

        self.pos_activations = pos_activations
        self.neg_activations = neg_activations

    def __len__(self):
        return self.pos_activations.shape[0]

    def __getitem__(self, idx):
        return self.pos_activations[idx], self.neg_activations[idx]


# =============================================================================
# LOSS FUNCTIONS (Following TruthX Paper)
# =============================================================================


class TruthXLoss(nn.Module):
    """
    Implementazione delle loss di TruthX come da paper (Eq. 3-11).

    L = L_recon + L_ctr + L_edit

    dove:
    - L_recon: Reconstruction loss (Eq. 3)
    - L_ctr = L_truth + L_sem: Contrastive losses (Eq. 5-7)
    - L_edit: Editing loss (Eq. 10)
    """

    def __init__(self, temperature: float = 0.1, num_hard_negatives: int = -1):
        super().__init__()
        self.temperature = temperature
        self.num_hard_negatives = num_hard_negatives
        self.cosine_loss = nn.CosineEmbeddingLoss(margin=0.0, reduction="mean")

    def reconstruction_loss(
        self, x_recon: torch.Tensor, x_orig: torch.Tensor
    ) -> torch.Tensor:
        """
        L_recon = MSE(x', x) - Eq. 3
        """
        return F.mse_loss(x_recon, x_orig)

    def semantic_loss(
        self, h_sem_pos: torch.Tensor, h_sem_neg: torch.Tensor
    ) -> torch.Tensor:
        """
        Semantic Loss (parte di L_ctr, Eq. 6):

        Il Semantic Encoder deve produrre rappresentazioni simili per
        coppie (pos, neg) che hanno lo stesso contenuto semantico ma
        veridicità opposta.

        Usiamo CosineEmbeddingLoss con target=1 per forzare similarità.

        Args:
            h_sem_pos: [N, D] rappresentazioni semantiche dei campioni positivi
            h_sem_neg: [N, D] rappresentazioni semantiche dei campioni negativi
                       (allineati: h_sem_pos[i] e h_sem_neg[i] sono una coppia)
        """
        # Target = 1 significa che vogliamo che le rappresentazioni siano simili
        target = torch.ones(h_sem_pos.size(0), device=h_sem_pos.device)

        return self.cosine_loss(h_sem_pos, h_sem_neg, target)

    def truthful_contrastive_loss(
        self,
        h_truth_pos: torch.Tensor,
        h_truth_neg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Truthful Contrastive Loss (L_truth, Eq. 5 del paper TruthX):

        L_truth = CTR(h_truth^pos, H_truth^pos, H_truth^neg)
                + CTR(h_truth^neg, H_truth^neg, H_truth^pos)

        Dove CTR(s, S+, S-) è definita in Eq. 4:
        CTR(s, S+, S-) = -log[ Σ_{s'∈S+} exp(sim(s,s')/τ) /
                               Σ_{s'∈(S+∪S-)} exp(sim(s,s')/τ) ]

        IMPORTANTE: Per Eq. 5, S+ INCLUDE il campione stesso!
        - Per h_truth_pos[i]: S+ = TUTTI h_truth_pos (incluso i), S- = TUTTI h_truth_neg
        - Per h_truth_neg[i]: S+ = TUTTI h_truth_neg (incluso i), S- = TUTTI h_truth_pos

        Args:
            h_truth_pos: [N, D] rappresentazioni truthful dei campioni positivi
            h_truth_neg: [N, D] rappresentazioni truthful dei campioni negativi
        """
        # Normalizza le rappresentazioni
        h_pos = F.normalize(h_truth_pos, p=2, dim=1)  # [N, D]
        h_neg = F.normalize(h_truth_neg, p=2, dim=1)  # [N, D]

        # Calcola tutte le similarità con operazioni matriciali
        # sim_pos_pos[i,j] = sim(h_pos[i], h_pos[j])
        sim_pos_pos = torch.mm(h_pos, h_pos.T) / self.temperature  # [N, N]
        # sim_pos_neg[i,j] = sim(h_pos[i], h_neg[j])
        sim_pos_neg = torch.mm(h_pos, h_neg.T) / self.temperature  # [N, N]
        # sim_neg_neg[i,j] = sim(h_neg[i], h_neg[j])
        sim_neg_neg = torch.mm(h_neg, h_neg.T) / self.temperature  # [N, N]
        # sim_neg_pos[i,j] = sim(h_neg[i], h_pos[j])
        sim_neg_pos = torch.mm(h_neg, h_pos.T) / self.temperature  # [N, N]

        # === Parte 1: CTR(h_truth^pos, H_truth^pos, H_truth^neg) ===
        # Per ogni h_pos[i]: S+ = tutti h_pos, S- = tutti h_neg
        # numerator = logsumexp(sim con tutti h_pos)
        # denominator = logsumexp(sim con tutti h_pos E tutti h_neg)

        # Numeratore: sum over S+ (tutti i positivi)
        log_sum_pos_1 = torch.logsumexp(sim_pos_pos, dim=1)  # [N]
        # Denominatore: sum over S+ ∪ S- (tutti)
        all_sims_1 = torch.cat([sim_pos_pos, sim_pos_neg], dim=1)  # [N, 2N]
        log_sum_all_1 = torch.logsumexp(all_sims_1, dim=1)  # [N]
        # Loss per la prima parte
        loss_1 = -(log_sum_pos_1 - log_sum_all_1).mean()

        # === Parte 2: CTR(h_truth^neg, H_truth^neg, H_truth^pos) ===
        # Per ogni h_neg[i]: S+ = tutti h_neg, S- = tutti h_pos

        # Numeratore: sum over S+ (tutti i negativi)
        log_sum_neg_2 = torch.logsumexp(sim_neg_neg, dim=1)  # [N]
        # Denominatore: sum over S+ ∪ S- (tutti)
        all_sims_2 = torch.cat([sim_neg_neg, sim_neg_pos], dim=1)  # [N, 2N]
        log_sum_all_2 = torch.logsumexp(all_sims_2, dim=1)  # [N]
        # Loss per la seconda parte
        loss_2 = -(log_sum_neg_2 - log_sum_all_2).mean()

        return (loss_1 + loss_2) 

    def semantic_contrastive_loss(
        self,
        h_sem_pos: torch.Tensor,
        h_sem_neg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Semantic Contrastive Loss (L_sem, Eq. 6 del paper TruthX).

        L_sem = CTR(h_sem^pos, h_sem^neg, H_sem^pos \ h_sem^pos)
              + CTR(h_sem^neg, h_sem^pos, H_sem^neg \ h_sem^neg)

        Implementazione migliorata:
        - supporto per hard negatives (top-k) tramite `self.num_hard_negatives`
        - gestione dei casi limite (N<2) -> loss = 0
        - normalizzazione finale (media delle due parti)

        Per ogni h_sem_pos[i]:
        - S+ = {h_sem_neg[i]} (la controparte con stessa semantica)
        - S- = {h_sem_pos[j] | j ≠ i} (altri pos con semantica DIVERSA)

        Args:
            h_sem_pos: [N, D] rappresentazioni semantiche positive
            h_sem_neg: [N, D] rappresentazioni semantiche negative (allineate)
        """
        # Normalizza le rappresentazioni
        h_pos = F.normalize(h_sem_pos, p=2, dim=1)  # [N, D]
        h_neg = F.normalize(h_sem_neg, p=2, dim=1)  # [N, D]

        n = h_pos.size(0)
        device = h_pos.device

        # Se non ci sono abbastanza campioni, la loss semantica è zero
        if n < 2:
            return torch.tensor(0.0, device=device)

        # Calcola tutte le similarità con operazioni matriciali
        # sim_pos_neg[i,j] = sim(h_pos[i], h_neg[j]) / τ
        sim_pos_neg = torch.mm(h_pos, h_neg.T) / self.temperature  # [N, N]
        # sim_pos_pos[i,j] = sim(h_pos[i], h_pos[j]) / τ
        sim_pos_pos = torch.mm(h_pos, h_pos.T) / self.temperature  # [N, N]
        # sim_neg_pos[i,j] = sim(h_neg[i], h_pos[j]) / τ
        sim_neg_pos = torch.mm(h_neg, h_pos.T) / self.temperature  # [N, N]
        # sim_neg_neg[i,j] = sim(h_neg[i], h_neg[j]) / τ
        sim_neg_neg = torch.mm(h_neg, h_neg.T) / self.temperature  # [N, N]

        # Maschera per escludere il campione stesso (diagonale)
        mask_diag = torch.eye(n, dtype=torch.bool, device=device)

        # Helper: ottieni denominatore logsumexp usando top-k hard negatives se richiesto
        def _log_denom_with_hard_negatives(pos_sims: torch.Tensor, neg_matrix: torch.Tensor):
            """Ritorna il logsumexp di [pos, top-k(neg_matrix_row)] per ogni riga.

            pos_sims: [N]
            neg_matrix: [N, N] con diagonale già impostata a -inf
            """
            # Decidi quanti hard negatives usare
            max_neg = neg_matrix.size(1)
            # max_neg == n (di cui una è -inf sulla diagonale)
            effective_negatives = n - 1
            if self.num_hard_negatives is None or self.num_hard_negatives <= 0:
                topk = effective_negatives
            else:
                topk = min(self.num_hard_negatives, effective_negatives)

            if topk <= 0:
                # Nessun negativo: il denominatore è solo pos
                all_sims = pos_sims.unsqueeze(1)
            elif topk >= effective_negatives:
                # Usa tutti i negativi (esclusa la diagonale)
                all_sims = torch.cat([pos_sims.unsqueeze(1), neg_matrix], dim=1)
            else:
                topk_vals, _ = torch.topk(neg_matrix, k=topk, dim=1)
                all_sims = torch.cat([pos_sims.unsqueeze(1), topk_vals], dim=1)

            return torch.logsumexp(all_sims, dim=1)

        # === Parte 1: CTR(h_sem^pos, h_sem^neg, H_sem^pos \ h_sem^pos) ===
        pos_sims_1 = torch.diag(sim_pos_neg)  # [N]
        neg_sims_1 = sim_pos_pos.masked_fill(mask_diag, float("-inf"))  # [N, N]
        log_denom_1 = _log_denom_with_hard_negatives(pos_sims_1, neg_sims_1)  # [N]
        loss_1 = (-pos_sims_1 + log_denom_1).mean()

        # === Parte 2: CTR(h_sem^neg, h_sem^pos, H_sem^neg \ h_sem^neg) ===
        pos_sims_2 = torch.diag(sim_neg_pos)  # [N]
        neg_sims_2 = sim_neg_neg.masked_fill(mask_diag, float("-inf"))  # [N, N]
        log_denom_2 = _log_denom_with_hard_negatives(pos_sims_2, neg_sims_2)  # [N]
        loss_2 = (-pos_sims_2 + log_denom_2).mean()

        # Media delle due parti per rendere coerente con L_truth
        return (loss_1 + loss_2) 


    def editing_loss(
        self,
        model: nn.Module,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
        h_sem_pos: torch.Tensor,
        h_sem_neg: torch.Tensor,
        h_truth_pos: torch.Tensor,
        h_truth_neg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Editing Loss (L_edit, Eq. 10):

        x^{pos→neg} = Dec(h_sem^neg + Attn(h_sem^pos, h_truth^neg))
        x^{neg→pos} = Dec(h_sem^pos + Attn(h_sem^neg, h_truth^pos))

        L_edit = MSE(x^neg, x^{pos→neg}) + MSE(x^pos, x^{neg→pos})

        Quando scambiamo le rappresentazioni truthful, la ricostruzione
        dovrebbe approssimare la controparte con veridicità opposta.
        """
        # x^{pos→neg}: ricostruisci usando semantic di pos con truthful di neg
        # Il risultato dovrebbe essere simile a x_neg
        z_pos_to_neg = h_sem_pos + model.attention(h_sem_pos, h_truth_neg, h_truth_neg)
        x_pos_to_neg = model.decode(z_pos_to_neg)

        # x^{neg→pos}: ricostruisci usando semantic di neg con truthful di pos
        # Il risultato dovrebbe essere simile a x_pos
        z_neg_to_pos = h_sem_neg + model.attention(h_sem_neg, h_truth_pos, h_truth_pos)
        x_neg_to_pos = model.decode(z_neg_to_pos)

        loss = F.mse_loss(x_pos_to_neg, x_neg) + F.mse_loss(x_neg_to_pos, x_pos)

        return loss


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================


def create_paired_beliefbank_subset(
    project_root: str,
    data_type: str = "facts",
    num_pairs: int = 500,
):
    """
    Crea un subset bilanciato di BeliefBank con coppie esplicite (fact, negated_fact).

    Ogni coppia ha:
    - Un fatto truthful (belief=1)
    - Il suo negato (belief=0, stessa semantica)

    Args:
        project_root: Root directory del progetto
        data_type: Tipo di dati BeliefBank ("facts" o "constraints")
        num_pairs: Numero di coppie da creare

    Returns:
        List di coppie, dove ogni coppia è un dict con:
        {
            "positive": {"question": ..., "answer": "True", "instance_id": ...},
            "negative": {"question": ..., "answer": "False", "instance_id": ...}
        }
    """
    from src.data.BeliefBankDataset import BeliefBankDataset

    dataset = BeliefBankDataset(
        project_root=project_root,
        model_type="demo",
        recreate_ids=True,
        data_type=data_type,
    )

    # Il dataset di BeliefBank è già strutturato con coppie fact/negated_fact
    # extend_with_negated_facts concatena [facts, negated_facts]
    # Quindi dataset[i] e dataset[i + len(facts)] sono una coppia

    total_samples = len(dataset)
    half = total_samples // 2

    pairs = []
    pair_id = 0

    for i in range(min(num_pairs, half)):
        # Campione originale (prima metà)
        fact_pos, label_pos, _ = dataset[i]
        # Campione negato (seconda metà, stesso indice relativo)
        fact_neg, label_neg, _ = dataset[i + half]

        # Determina quale è truthful e quale hallucinated
        if label_pos == "yes":  # belief=1 è truthful
            positive = {
                "question": fact_pos,
                "answer": "True",
                "instance_id": pair_id * 2,
                "label": 1,
            }
            negative = {
                "question": fact_neg,
                "answer": "False",
                "instance_id": pair_id * 2 + 1,
                "label": 0,
            }
        else:
            positive = {
                "question": fact_neg,
                "answer": "True",
                "instance_id": pair_id * 2,
                "label": 1,
            }
            negative = {
                "question": fact_pos,
                "answer": "False",
                "instance_id": pair_id * 2 + 1,
                "label": 0,
            }

        pairs.append({"positive": positive, "negative": negative, "pair_id": pair_id})
        pair_id += 1

    print(f"Created {len(pairs)} paired samples from BeliefBank")
    return pairs


def save_paired_subset_as_jsonl(pairs: list, output_path: str):
    """Salva le coppie come file JSONL."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Salva in formato flat per compatibilità con il resto della pipeline
    flat_samples = []
    pair_mapping = {}  # pos_instance_id -> neg_instance_id

    for pair in pairs:
        flat_samples.append(pair["positive"])
        flat_samples.append(pair["negative"])
        pair_mapping[pair["positive"]["instance_id"]] = pair["negative"]["instance_id"]

    with open(output_path, "w") as f:
        for item in flat_samples:
            f.write(json.dumps(item) + "\n")

    # Salva anche il mapping delle coppie
    mapping_path = output_path.replace(".jsonl", "_pair_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(pair_mapping, f, indent=2)

    print(f"Subset saved to {output_path}")
    print(f"Pair mapping saved to {mapping_path}")

    return pair_mapping


def extract_activations_for_pairs(
    project_root: str, llm_name: str, pairs: list, quantization: bool = False
):
    """
    Estrae le attivazioni per le coppie usando HallucinationDetection.
    """
    print("\n" + "=" * 50)
    print("EXTRACTING ACTIVATIONS FOR PAIRED SAMPLES")
    print("=" * 50)

    detector = HallucinationDetection(
        project_dir=project_root, cache_dir_name="activation_cache_truthx"
    )

    # Flatten pairs to list of samples
    flat_samples = []
    for pair in pairs:
        flat_samples.append(pair["positive"])
        flat_samples.append(pair["negative"])

    class BeliefBankPairedSubset:
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            item = self.samples[idx]
            return item["question"], item["answer"], item["instance_id"]

    detector.dataset = BeliefBankPairedSubset(flat_samples)
    detector.dataset_name = "belief_bank_subset"

    detector.load_llm(llm_name, quantization=quantization)

    model_name_safe = llm_name.replace("/", "_")

    detector.generation_save_dir = os.path.join(
        detector.cache_dir_name, model_name_safe, "belief_bank_subset", "generations"
    )
    detector.mlp_save_dir = os.path.join(
        detector.cache_dir_name, model_name_safe, "belief_bank_subset", "activation_mlp"
    )
    detector.attn_save_dir = os.path.join(
        detector.cache_dir_name,
        model_name_safe,
        "belief_bank_subset",
        "activation_attn",
    )
    detector.hidden_save_dir = os.path.join(
        detector.cache_dir_name,
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

    # Salva labels
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
        for item in flat_samples
    ]
    with open(labels_path, "w") as f:
        json.dump(labels_to_save, f, indent=4)

    print(f"\nExtracting activations for {len(flat_samples)} samples...")

    from src.model.InspectOutputContext import InspectOutputContext
    from src.model.prompts import PROMPT_TRUTHX as prompt

    target_layers = list(range(0, detector.llm.config.num_hidden_layers))

    module_names = []
    module_names += [f"model.layers.{idx}" for idx in target_layers]
    module_names += [f"model.layers.{idx}.self_attn" for idx in target_layers]
    module_names += [f"model.layers.{idx}.mlp" for idx in target_layers]

    for item in tqdm(flat_samples, desc="Extracting activations"):
        question, instance_id = item["question"], item["instance_id"]

        model_input = prompt.format(question=question)
        tokens = detector.tokenizer(model_input, return_tensors="pt")
        input_length = tokens["input_ids"].shape[1]  # Lunghezza dell'input prompt
        attention_mask = (
            tokens["attention_mask"].to("cuda") if "attention_mask" in tokens else None
        )

        with InspectOutputContext(
            detector.llm,
            module_names,
            save_generation=False,  # Non salviamo la generazione
            save_dir=detector.generation_save_dir,
        ) as inspect:
            # Esegui solo un forward pass sull'input (senza generazione autoregressiva)
            with torch.no_grad():
                output = detector.llm(
                    input_ids=tokens["input_ids"].to("cuda"),
                    attention_mask=attention_mask,
                )

        # Salva SOLO l'attivazione dell'ultimo token dell'input prompt
        # inspect.catcher contiene solo le attivazioni dell'input (nessuna generazione)
        for module, ac in inspect.catcher.items():
            # ac ha shape (batch, seq_len, hidden_dim) dove seq_len = input_length
            # Prendi solo l'attivazione dell'ultimo token dell'input prompt
            ac_prompt_only = ac[0, -1, :].float().cpu()  # Ultimo token del prompt
            layer_idx = int(module.split(".")[2])

            save_name = f"layer{layer_idx}-id{instance_id}.pt"
            if "mlp" in module:
                save_path = os.path.join(detector.mlp_save_dir, save_name)
            elif "self_attn" in module:
                save_path = os.path.join(detector.attn_save_dir, save_name)
            else:
                save_path = os.path.join(detector.hidden_save_dir, save_name)

            torch.save(ac_prompt_only, save_path)
            del ac_prompt_only

        del tokens, output
        if attention_mask is not None:
            del attention_mask
        torch.cuda.empty_cache()
        import gc

        gc.collect()

    print("\nActivations saved to:")
    print(f"  - {detector.hidden_save_dir}")
    print(f"  - {detector.mlp_save_dir}")
    print(f"  - {detector.attn_save_dir}")


def load_paired_activations(
    cache_dir: str,
    model_name: str,
    layer_idx: int,
    activation_type: str,
    pair_mapping: dict,
) -> tuple:
    """
    Carica le attivazioni in modo paired: (pos_acts, neg_acts) allineati.

    Args:
        cache_dir: Directory della cache
        model_name: Nome del modello (safe)
        layer_idx: Indice del layer fisico
        activation_type: Tipo di attivazione
        pair_mapping: Dict {pos_id: neg_id}

    Returns:
        (pos_activations, neg_activations) come tensori allineati
    """
    base_path = os.path.join(
        cache_dir, model_name, "belief_bank_subset", f"activation_{activation_type}"
    )

    pos_acts = []
    neg_acts = []

    for pos_id, neg_id in pair_mapping.items():
        pos_path = os.path.join(base_path, f"layer{layer_idx}-id{pos_id}.pt")
        neg_path = os.path.join(base_path, f"layer{layer_idx}-id{neg_id}.pt")

        if os.path.exists(pos_path) and os.path.exists(neg_path):
            pos_acts.append(torch.load(pos_path, map_location="cpu"))
            neg_acts.append(torch.load(neg_path, map_location="cpu"))

    if not pos_acts:
        return torch.empty(0), torch.empty(0)

    return torch.stack(pos_acts), torch.stack(neg_acts)


# =============================================================================
# PROBING-BASED LAYER SELECTION (Sezione 3.3 e Eq. 15 del paper)
# =============================================================================


def calculate_probing_accuracy(
    model: nn.Module,
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    pos_center: torch.Tensor,
    neg_center: torch.Tensor,
    device: str = "cuda",
) -> float:
    """
    Calcola la probing accuracy come da Eq. 15 del paper TruthX.

    Probe(x) = pos  se sim(h_truth, H̄_truth^pos) ≥ sim(h_truth, H̄_truth^neg)
             = neg  altrimenti

    dove h_truth = TruthEnc(x), H̄_truth^pos e H̄_truth^neg sono i centri.

    Args:
        model: MLPAE trainato
        pos_acts: [N, D] attivazioni truthful (campioni positivi)
        neg_acts: [N, D] attivazioni hallucinated (campioni negativi)
        pos_center: [D] centro delle rappresentazioni truthful positive
        neg_center: [D] centro delle rappresentazioni truthful negative
        device: dispositivo per il calcolo

    Returns:
        accuracy: float in [0, 1]
    """
    model.eval()
    pos_acts = pos_acts.to(device)
    neg_acts = neg_acts.to(device)
    pos_center = pos_center.to(device)
    neg_center = neg_center.to(device)

    with torch.no_grad():
        # Codifica nel truthful space
        h_truth_pos = model.encode_truthful(pos_acts)  # [N, D]
        h_truth_neg = model.encode_truthful(neg_acts)  # [N, D]

        # Normalizza per calcolo similarità coseno
        h_truth_pos = F.normalize(h_truth_pos, p=2, dim=1)
        h_truth_neg = F.normalize(h_truth_neg, p=2, dim=1)
        pos_center_norm = F.normalize(pos_center.unsqueeze(0), p=2, dim=1)
        neg_center_norm = F.normalize(neg_center.unsqueeze(0), p=2, dim=1)

        # Similarità con i centri per i campioni positivi (dovrebbero predire "pos")
        sim_pos_to_pos_center = torch.mm(
            h_truth_pos, pos_center_norm.T
        ).squeeze()  # [N]
        sim_pos_to_neg_center = torch.mm(
            h_truth_pos, neg_center_norm.T
        ).squeeze()  # [N]
        pred_pos = (sim_pos_to_pos_center >= sim_pos_to_neg_center).float()
        correct_pos = pred_pos.sum().item()

        # Similarità con i centri per i campioni negativi (dovrebbero predire "neg")
        sim_neg_to_pos_center = torch.mm(
            h_truth_neg, pos_center_norm.T
        ).squeeze()  # [N]
        sim_neg_to_neg_center = torch.mm(
            h_truth_neg, neg_center_norm.T
        ).squeeze()  # [N]
        pred_neg = (sim_neg_to_neg_center > sim_neg_to_pos_center).float()
        correct_neg = pred_neg.sum().item()

        total = pos_acts.size(0) + neg_acts.size(0)
        accuracy = (correct_pos + correct_neg) / total

    return accuracy


def calculate_probing_accuracy_all_modules(
    model: nn.Module,
    all_pos_acts: list,
    all_neg_acts: list,
    pos_centers: torch.Tensor,
    neg_centers: torch.Tensor,
    virtual_layer_info: list,
    device: str = "cuda",
) -> list:
    """
    Calcola probing accuracy per TUTTI i moduli (attn + ffn per ogni layer).

    Come da paper Sezione 3.3:
    "TruthX edits the LLM's internal representations on the selected top k layers
    from all attention and FFN layers based on the probing accuracy of each layer"

    Args:
        model: MLPAE trainato
        all_pos_acts: Lista di tensori [N, D] per ogni virtual layer
        all_neg_acts: Lista di tensori [N, D] per ogni virtual layer
        pos_centers: [num_layers, D] centri positivi
        neg_centers: [num_layers, D] centri negativi
        virtual_layer_info: Lista di tuple (physical_layer, module_type)
        device: dispositivo

    Returns:
        Lista di dict con {physical_layer, module_type, probing_accuracy}
    """
    results = []

    print(f"\nCalculating probing accuracy for {len(virtual_layer_info)} modules...")

    for virtual_idx, (physical_layer, module_type) in enumerate(
        tqdm(virtual_layer_info, desc="Probing")
    ):
        pos_acts = all_pos_acts[virtual_idx]
        neg_acts = all_neg_acts[virtual_idx]
        pos_center = pos_centers[virtual_idx]
        neg_center = neg_centers[virtual_idx]

        accuracy = calculate_probing_accuracy(
            model, pos_acts, neg_acts, pos_center, neg_center, device
        )

        results.append(
            {
                "virtual_layer_idx": virtual_idx,
                "physical_layer": physical_layer,
                "module_type": module_type,
                "probing_accuracy": accuracy,
            }
        )

    return results


def select_top_k_modules(probing_results: list, k: int = -1) -> list:
    """
    Seleziona i top-k moduli basandosi su probing accuracy.

    Come da paper:
    "For instance, for a 32-layer LLM and k = 10, TruthX selects the top 10 modules
    with the highest probing accuracy out of the total 64 modules"

    Args:
        probing_results: Lista di dict con probing accuracy per ogni modulo
        k: Numero di moduli da selezionare

    Returns:
        Lista dei top-k moduli ordinati per accuracy decrescente
    """
    # Ordina per probing accuracy decrescente
    sorted_results = sorted(
        probing_results, key=lambda x: x["probing_accuracy"], reverse=True
    )

    # Prendi i top k
    top_k = sorted_results[:k]

    # Aggiungi rank
    for i, mod in enumerate(top_k):
        mod["rank"] = i + 1

    return top_k


def save_layer_selection_config(
    top_k_modules: list,
    all_probing_results: list,
    virtual_layer_info: list,
    model_name: str,
    dataset_name: str,
    output_path: str,
):
    """
    Salva la configurazione della layer selection per l'inference.

    Questo file JSON sarà usato durante l'inference per sapere
    quali moduli editare.
    """
    import datetime

    # Conta moduli attn vs ffn nei top-k
    num_attn = sum(1 for m in top_k_modules if m["module_type"] == "attn")
    num_ffn = sum(1 for m in top_k_modules if m["module_type"] in ["mlp", "ffn"])

    # Crea dizionario con tutte le accuracy
    all_accuracies = {}
    for result in all_probing_results:
        key = f"{result['physical_layer']}_{result['module_type']}"
        all_accuracies[key] = result["probing_accuracy"]

    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "num_total_modules": len(all_probing_results),
        "num_selected_modules": len(top_k_modules),
        "top_k_modules": top_k_modules,
        "summary": {
            "num_attn_modules_selected": num_attn,
            "num_ffn_modules_selected": num_ffn,
        },
        "all_modules_accuracies": all_accuracies,
        "virtual_layer_info": [
            {"idx": i, "physical_layer": p, "module_type": t}
            for i, (p, t) in enumerate(virtual_layer_info)
        ],
        "timestamp": datetime.datetime.now().isoformat(),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    return config


# =============================================================================
# TRAINING FUNCTION
# =============================================================================


def train_truthx_on_beliefbank(args):
    """
    Training del modello TruthX usando BeliefBank con paired samples.

    Implementa correttamente il Disentangled Representation Learning:
    1. Crea coppie (truthful, hallucinated) semanticamente correlate
    2. Forma batch strutturati: torch.cat([batch_pos, batch_neg], dim=0)
    3. Calcola le loss corrette:
       - L_sem: forza h_sem(pos) ≈ h_sem(neg) per coppie allineate
       - L_truth: separa h_truth(pos) da h_truth(neg)
       - L_recon: ricostruzione dell'input
       - L_edit: cross-reconstruction con swap delle rappresentazioni truthful
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    print("\n" + "=" * 50)
    print("STEP 1: CREATING PAIRED BELIEFBANK SUBSET")
    print("=" * 50)

    pairs = create_paired_beliefbank_subset(
        project_root=args.project_root,
        data_type=args.beliefbank_data_type,
        num_pairs=args.num_pairs,
    )

    subset_suffix = f"pairs{args.num_pairs}"
    subset_path = os.path.join(
        args.project_root, "data", "beliefbank", f"beliefbank_paired_subset_{subset_suffix}.jsonl"
    )
    pair_mapping = save_paired_subset_as_jsonl(pairs, subset_path)
    print(f"Paired subset saved to {subset_path}")

    # Converti pair_mapping keys a int
    pair_mapping = {int(k): int(v) for k, v in pair_mapping.items()}

    # Check cache per attivazioni esistenti
    model_name_safe = args.model_name.replace("/", "_")

    def _activations_exist_in_cache(cache_dir, model_name, num_pairs):
        base = os.path.join(cache_dir, model_name, "belief_bank_subset")
        labels_path = os.path.join(base, "generations", "hallucination_labels.json")
        if not os.path.exists(labels_path):
            return False

        for act_type in ["attn", "mlp"]:
            act_dir = os.path.join(base, f"activation_{act_type}")
            if not os.path.isdir(act_dir):
                return False

            files = [f for f in os.listdir(act_dir) if f.endswith(".pt")]
            if len(files) < num_pairs * 2:
                return False

        return True

    should_extract = args.extract_activations
    if should_extract:
        print("\n" + "=" * 50)
        print("STEP 2: EXTRACTING ACTIVATIONS")
        print("=" * 50)

        extract_activations_for_pairs(
            project_root=args.project_root,
            llm_name=args.model_name,
            pairs=pairs,
            quantization=args.quantization,
        )
    elif _activations_exist_in_cache(args.cache_dir, model_name_safe, args.num_pairs):
        print("Using existing activations from cache.")
    else:
        raise ValueError(
            "Activations not found in cache. Please run with --extract_activations."
        )

    print("\n" + "=" * 50)
    print("STEP 3: LOADING PAIRED ACTIVATIONS")
    print("=" * 50)

    # Dataset name per salvataggio
    dataset_name = f"belief_bank_{args.beliefbank_data_type}"

    # =========================================================================
    # LAYER SELECTION: Carica TUTTI i layer fisici (attn + ffn)
    # Come da paper Sezione 3.3:
    # "TruthX probes all internal representations instead of only attention or FFN"
    # Quindi carichiamo TUTTI i moduli, poi selezioneremo i top-k con probing
    # =========================================================================

    if args.target_layers is None:
        # Auto-detect tutti i layer disponibili dalla cache
        base_path = os.path.join(
            args.cache_dir, model_name_safe, "belief_bank_subset", "activation_attn"
        )
        if os.path.exists(base_path):
            layer_files = [f for f in os.listdir(base_path) if f.startswith("layer")]
            layer_indices = set()
            for f in layer_files:
                try:
                    layer_idx = int(f.split("-")[0].replace("layer", ""))
                    layer_indices.add(layer_idx)
                except (ValueError, IndexError):
                    pass
            args.target_layers = sorted(list(layer_indices))
            print(f"Auto-detected {len(args.target_layers)} physical layers from cache")
        else:
            raise ValueError(
                "No activations found in cache to auto-detect layers. Please specify --target_layers."
            )

    # Tipi di attivazione: SOLO attn e mlp (ffn) - NO hidden (non nel paper!)
    # Come da paper: "32 attention modules + 32 FFN modules = 64 total modules"
    activation_types = ["attn", "mlp"]

    # Carica attivazioni paired per ogni virtual layer (modulo)
    all_pos_acts = []
    all_neg_acts = []
    virtual_layer_info = []

    print(
        f"\nLoading paired activations for {len(args.target_layers)} physical layers x {len(activation_types)} types..."
    )
    print(f"Total modules to load: {len(args.target_layers) * len(activation_types)}")

    for layer_idx in tqdm(args.target_layers, desc="Loading layers"):
        for act_type in activation_types:
            pos_acts, neg_acts = load_paired_activations(
                args.cache_dir, model_name_safe, layer_idx, act_type, pair_mapping
            )

            if pos_acts.numel() > 0:
                all_pos_acts.append(pos_acts)
                all_neg_acts.append(neg_acts)
                virtual_layer_info.append((layer_idx, act_type))

                print(
                    f"  Virtual layer {len(virtual_layer_info) - 1}: "
                    f"physical {layer_idx}, {act_type} - {pos_acts.shape[0]} pairs"
                )

    if not all_pos_acts:
        raise ValueError("No activations found. Run with --extract_activations first.")

    hidden_size = all_pos_acts[0].shape[-1]
    num_virtual_layers = len(all_pos_acts)
    num_pairs_loaded = all_pos_acts[0].shape[0]

    print(f"\nHidden size: {hidden_size}")
    print(f"Num virtual layers: {num_virtual_layers}")
    print(f"Pairs per virtual layer: {num_pairs_loaded}")

    # ============ TRAIN/VAL SPLIT ============
    val_split = getattr(args, "val_split", 0.2)  # Default 20% validation
    num_train_pairs = int(num_pairs_loaded * (1 - val_split))
    
    print(f"\nTrain/Val split: {int((1-val_split)*100)}% train, {int(val_split*100)}% val")
    print(f"Train pairs per layer: {num_train_pairs}")
    print(f"Val pairs per layer: {num_pairs_loaded - num_train_pairs}")
    
    # Usa lo STESSO shuffle per tutti i layer (consistente)
    # Questo garantisce che la coppia (pos, neg) dello stesso sample 
    # sia sempre nello stesso set (train o val)
    global_indices = torch.randperm(num_pairs_loaded)
    train_pair_indices = global_indices[:num_train_pairs]
    val_pair_indices = global_indices[num_train_pairs:]
    
    # Converti gli indici delle coppie in instance_ids
    # pair_mapping è {pos_instance_id: neg_instance_id}
    pair_list = list(pair_mapping.items())  # [(pos_id, neg_id), ...]
    
    train_instance_ids = []
    for idx in train_pair_indices.tolist():
        pos_id, neg_id = pair_list[idx]
        train_instance_ids.extend([pos_id, neg_id])
    
    val_instance_ids = []
    for idx in val_pair_indices.tolist():
        pos_id, neg_id = pair_list[idx]
        val_instance_ids.extend([pos_id, neg_id])
    
    # Split ogni layer's activations usando gli stessi indici
    all_pos_acts_train = []
    all_neg_acts_train = []
    all_pos_acts_val = []
    all_neg_acts_val = []
    
    for pos_acts, neg_acts in zip(all_pos_acts, all_neg_acts):
        all_pos_acts_train.append(pos_acts[train_pair_indices])
        all_neg_acts_train.append(neg_acts[train_pair_indices])
        all_pos_acts_val.append(pos_acts[val_pair_indices])
        all_neg_acts_val.append(neg_acts[val_pair_indices])

    print("\n" + "=" * 50)
    print("STEP 4: INITIALIZING TRUTHX MODEL")
    print("=" * 50)

    # Parse hidden dims (default architecture: input_dim -> 2048 -> latent_dim)
    semantic_hidden_dims = (
        [int(x) for x in args.semantic_hidden_dims.split(",")]
        if args.semantic_hidden_dims
        else [2048]  # Default: input_dim -> 2048 -> 1024 latent
    )
    truthful_hidden_dims = (
        [int(x) for x in args.truthful_hidden_dims.split(",")]
        if args.truthful_hidden_dims
        else [2048]  # Default: input_dim -> 2048 -> 1024 latent
    )
    decoder_hidden_dims = (
        [int(x) for x in args.decoder_hidden_dims.split(",")]
        if args.decoder_hidden_dims
        else [2048]  # Default: 1024 latent -> 2048 -> input_dim
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = TruthXLoss(temperature=args.temperature)

    print("\n" + "=" * 50)
    print("STEP 5: TRAINING WITH PAIRED SAMPLES")
    print("=" * 50)

    print("Starting training with correct paired batch structure...")
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state_dict = None
    best_epoch = None
    # Start training timer
    training_start_time = time.time()
    for epoch in tqdm(range(args.num_epochs), desc="Training Epochs", unit="epoch"): 
        # ========== TRAINING PHASE ==========
        epoch_recon_loss = 0.0
        epoch_sem_loss = 0.0
        epoch_truth_loss = 0.0
        epoch_edit_loss = 0.0
        epoch_batches = 0

        pbar_layers = tqdm(
            range(num_virtual_layers),
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            leave=False,
        )

        for virtual_layer_idx in pbar_layers:
            pos_acts = all_pos_acts_train[virtual_layer_idx].to(device)
            neg_acts = all_neg_acts_train[virtual_layer_idx].to(device)

            # Crea dataset e dataloader per questo virtual layer
            paired_dataset = AggregatedPairedDataset(pos_acts, neg_acts)
            
            dataloader = DataLoader(
                paired_dataset,
                batch_size=min(args.batch_size, len(paired_dataset)),  # Adatta batch size se dataset è troppo piccolo
                shuffle=True,  # Shuffle delle coppie, non interno al batch
                drop_last=False,  # Includi ultimo batch anche se piccolo
            )

            # Track last batch loss for progress bar (handle empty dataloader)
            last_total_loss = None
            for i, (batch_pos, batch_neg) in enumerate(dataloader):
                """
                BATCH STRUCTURE (FONDAMENTALE):
                - batch_pos[i] e batch_neg[i] sono una coppia semantica
                - Hanno lo stesso contenuto ma veridicità opposta
                - NON si shuffla internamente al batch!

                Per la forward pass, concateniamo:
                X = [batch_pos; batch_neg]  shape: [2N, D]

                Dove i primi N sono positivi e gli ultimi N sono negativi.
                Per ogni indice i in [0, N-1], la sua controparte semantica
                è esattamente a i+N.
                """
                batch_pos = batch_pos.to(device)
                batch_neg = batch_neg.to(device)
                
                # Validazione: batch non è vuoto e contiene valori reali
                if batch_pos.numel() == 0 or batch_neg.numel() == 0:
                    continue
                
                if torch.isnan(batch_pos).any() or torch.isnan(batch_neg).any():
                    continue
                
                if (batch_pos.abs().max() == 0 and batch_neg.abs().max() == 0):
                    # Continua comunque per vedere l'errore
                    pass

                # Debug: inspect first batch of each virtual layer
                if getattr(args, "debug", False) and i == 0:
                    local_logger = logging.getLogger(__name__)
                    local_logger.debug(
                        f"[Layer {virtual_layer_idx}] Batch shapes: pos={batch_pos.shape}, neg={batch_neg.shape}"
                    )
                    with torch.no_grad():
                        pos_norm = F.normalize(batch_pos, p=2, dim=1)
                        neg_norm = F.normalize(batch_neg, p=2, dim=1)
                        pair_sims = (pos_norm * neg_norm).sum(dim=1)
                        local_logger.debug(
                            f"[Layer {virtual_layer_idx}] Pair cosine sim stats: mean={pair_sims.mean().item():.4f}, std={pair_sims.std().item():.4f}, sample={pair_sims[:5].cpu().tolist()}"
                        )

                optimizer.zero_grad()

                # Forward pass per entrambe le metà
                output_pos, x_pos, h_sem_pos, h_truth_pos = model(batch_pos)
                output_neg, x_neg, h_sem_neg, h_truth_neg = model(batch_neg)
                
                # Verifica che le rappresentazioni abbiano varianza
                sem_pos_std = h_sem_pos.std(dim=0).mean().item()
                truth_pos_std = h_truth_pos.std(dim=0).mean().item()
                
                if torch.isnan(h_sem_pos).any() or torch.isnan(h_truth_pos).any():
                    continue

                if getattr(args, "debug", False) and i == 0:
                    local_logger.debug(
                        f"[Layer {virtual_layer_idx}] h_sem_pos mean={h_sem_pos.mean().item():.6f}, std={h_sem_pos.std().item():.6f}"
                    )
                    local_logger.debug(
                        f"[Layer {virtual_layer_idx}] h_sem_neg mean={h_sem_neg.mean().item():.6f}, std={h_sem_neg.std().item():.6f}"
                    )
                    local_logger.debug(
                        f"[Layer {virtual_layer_idx}] h_truth_pos mean={h_truth_pos.mean().item():.6f}, std={h_truth_pos.std().item():.6f}"
                    )
                    local_logger.debug(
                        f"[Layer {virtual_layer_idx}] h_truth_neg mean={h_truth_neg.mean().item():.6f}, std={h_truth_neg.std().item():.6f}"
                    )

                # =====================================================
                # LOSS CALCULATION (Following TruthX Paper)
                # =====================================================

                # 1. Reconstruction Loss (Eq. 3)
                recon_loss = (
                    loss_fn.reconstruction_loss(output_pos, x_pos)
                    + loss_fn.reconstruction_loss(output_neg, x_neg)
                ) / 2

                # 2. Semantic Contrastive Loss (parte di L_ctr)
                sem_ctr_loss = loss_fn.semantic_contrastive_loss(h_sem_pos, h_sem_neg)

                # 3. Truthful Contrastive Loss (L_truth, Eq. 5)
                truth_loss = loss_fn.truthful_contrastive_loss(h_truth_pos, h_truth_neg)

                # 4. Editing Loss (L_edit, Eq. 10)
                edit_loss = loss_fn.editing_loss(
                    model,
                    batch_pos,
                    batch_neg,
                    h_sem_pos,
                    h_sem_neg,
                    h_truth_pos,
                    h_truth_neg,
                )

                # Total loss (Eq. 11)
                total_loss = (
                    recon_loss
                    + args.contrastive_weight * sem_ctr_loss + truth_loss
                    + args.editing_weight * edit_loss
                )

                # Save last batch scalar for progress reporting
                last_total_loss = total_loss.item()

                if getattr(args, "debug", False) and i == 0:
                    local_logger.debug(
                        f"[Layer {virtual_layer_idx}] losses: recon={recon_loss.item():.6f}, sem_ctr={sem_ctr_loss.item():.6f}, truth={truth_loss.item():.6f}, edit={edit_loss.item():.6f}, total={last_total_loss:.6f}"
                    )

                total_loss.backward()
                optimizer.step()

                epoch_recon_loss += recon_loss.item()
                epoch_sem_loss += sem_ctr_loss.item() *args.contrastive_weight
                epoch_truth_loss += truth_loss.item()
                epoch_edit_loss += edit_loss.item()
                epoch_batches += 1

            pbar_layers.set_postfix({"loss": f"{last_total_loss:.4f}" if last_total_loss is not None else "n/a"})

        # Log epoch metrics
        avg_recon = epoch_recon_loss / max(epoch_batches, 1)
        avg_sem = epoch_sem_loss / max(epoch_batches, 1)
        avg_truth = epoch_truth_loss / max(epoch_batches, 1)
        avg_edit = epoch_edit_loss / max(epoch_batches, 1)

        train_total = (
            avg_recon
            + args.contrastive_weight * (avg_sem + avg_truth)
            + args.editing_weight * avg_edit
        )

        # ========== VALIDATION PHASE ==========
        model.eval()
        val_recon_loss = 0.0
        val_sem_loss = 0.0
        val_truth_loss = 0.0
        val_edit_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for virtual_layer_idx in range(num_virtual_layers):
                pos_acts_val = all_pos_acts_val[virtual_layer_idx].to(device)
                neg_acts_val = all_neg_acts_val[virtual_layer_idx].to(device)

                paired_dataset_val = AggregatedPairedDataset(pos_acts_val, neg_acts_val)
                
                if len(paired_dataset_val) == 0:
                    continue
                
                dataloader_val = DataLoader(
                    paired_dataset_val,
                    batch_size=min(args.batch_size, len(paired_dataset_val)),
                    shuffle=False,
                    drop_last=False,
                )

                for batch_pos, batch_neg in dataloader_val:
                    batch_pos = batch_pos.to(device)
                    batch_neg = batch_neg.to(device)
                    
                    if batch_pos.numel() == 0 or batch_neg.numel() == 0:
                        continue

                    output_pos, x_pos, h_sem_pos, h_truth_pos = model(batch_pos)
                    output_neg, x_neg, h_sem_neg, h_truth_neg = model(batch_neg)

                    val_recon_loss += (
                        loss_fn.reconstruction_loss(output_pos, x_pos)
                        + loss_fn.reconstruction_loss(output_neg, x_neg)
                    ) / 2
                    
                    val_sem_loss += loss_fn.semantic_contrastive_loss(h_sem_pos, h_sem_neg) * args.contrastive_weight
                    val_truth_loss += loss_fn.truthful_contrastive_loss(h_truth_pos, h_truth_neg)
                    val_edit_loss += loss_fn.editing_loss(
                        model, batch_pos, batch_neg, h_sem_pos, h_sem_neg, h_truth_pos, h_truth_neg
                    ) * args.editing_weight
                    
                    val_batches += 1

        model.train()

        avg_val_recon = val_recon_loss / max(val_batches, 1)
        avg_val_sem = val_sem_loss / max(val_batches, 1)
        avg_val_truth = val_truth_loss / max(val_batches, 1)
        avg_val_edit = val_edit_loss / max(val_batches, 1)

        val_total = avg_val_recon + avg_val_sem + avg_val_truth + avg_val_edit

        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print(f"  Train | Recon: {avg_recon:.4f}, Semantic: {avg_sem:.4f}, Truthful: {avg_truth:.4f}, Editing: {avg_edit:.4f}, Total: {train_total:.4f}")
        print(f"  Val   | Recon: {avg_val_recon:.4f}, Semantic: {avg_val_sem:.4f}, Truthful: {avg_val_truth:.4f}, Editing: {avg_val_edit:.4f}, Total: {val_total:.4f}")

        # === Early Stopping (on validation loss) ===
        try:
            patience = int(args.early_stopping_patience)
        except Exception:
            patience = 10
        min_delta = getattr(args, "early_stopping_min_delta", 1e-4)

        if val_total + min_delta < best_val_loss:
            best_val_loss = val_total
            epochs_no_improve = 0
            # Save best state dict so we can restore it later
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            print(f"  New best val_loss: {best_val_loss:.6f} at epoch {best_epoch}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{patience} epochs")

        if epochs_no_improve >= patience:
            print(
                f"Early stopping triggered (no improvement for {patience} epochs). Stopping training."
            )
            break

    # Restore best model if available before computing centers and saving
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Loaded best model from epoch {best_epoch} (val_loss={best_val_loss:.6f}) before calculating centers and saving.")

    # Compute training elapsed time
    training_time_seconds = None
    if 'training_start_time' in locals():
        training_time_seconds = time.time() - training_start_time
        print(f"Training completed in {training_time_seconds:.2f} seconds")

    print("\n" + "=" * 50)
    print("STEP 6: CALCULATING TRUTHFUL SPACE CENTERS (δ)")
    print("=" * 50)

    # Calcola i centri per ogni virtual layer (Eq. 12)
    # δ = H̄_truth^pos - H̄_truth^neg
    pos_centers = []
    neg_centers = []
    deltas = []

    model.eval()
    with torch.no_grad():
        for virtual_layer_idx in range(num_virtual_layers):
            pos_acts = all_pos_acts[virtual_layer_idx].to(device)
            neg_acts = all_neg_acts[virtual_layer_idx].to(device)

            h_truth_pos = model.encode_truthful(pos_acts)
            h_truth_neg = model.encode_truthful(neg_acts)

            pos_center = h_truth_pos.mean(dim=0)
            neg_center = h_truth_neg.mean(dim=0)

            pos_centers.append(pos_center.cpu())
            neg_centers.append(neg_center.cpu())
            deltas.append((pos_center - neg_center).cpu())

    pos_centers = torch.stack(pos_centers)
    neg_centers = torch.stack(neg_centers)
    deltas = torch.stack(deltas)

    print("\n" + "=" * 50)
    print("STEP 7: SAVING MODEL AND STEERING VECTORS")
    print("=" * 50)

    dataset_name = f"belief_bank_{args.beliefbank_data_type}"

    # Salva autoencoder (includi num_pairs e contrastive weight nel nome)
    autoencoder_dir = os.path.join(args.project_root, "AutoEncoder", dataset_name)
    os.makedirs(autoencoder_dir, exist_ok=True)
    ae_suffix = f"pairs{args.num_pairs}_cw{args.contrastive_weight}"
    autoencoder_path = os.path.join(
        autoencoder_dir, f"autoencoder_{model_name_safe}_{ae_suffix}.pt"
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "virtual_layer_info": virtual_layer_info,
            "num_virtual_layers": num_virtual_layers,
            "args": vars(args),
            "training_time_seconds": training_time_seconds,
        },
        autoencoder_path,
    )

    print(f"\nAutoencoder saved to {autoencoder_path}")

    # Salva configurazione training set (per evitare data leakage durante inference)
    config_training_dir = os.path.join(args.project_root, "ConfigTraining")
    os.makedirs(config_training_dir, exist_ok=True)
    config_suffix = f"pairs{args.num_pairs}"
    config_training_path = os.path.join(
        config_training_dir, f"{model_name_safe}_{dataset_name}_{config_suffix}.json"
    )
    
    training_config = {
        "model_name": args.model_name,
        "dataset_name": dataset_name,
        "train_instance_ids": sorted(train_instance_ids),
        "val_instance_ids": sorted(val_instance_ids),
        "num_train_samples": len(train_instance_ids),
        "num_val_samples": len(val_instance_ids),
        "num_train_pairs": num_train_pairs,
        "val_split": val_split,
        "total_pairs": num_pairs_loaded,
        "training_time_seconds": training_time_seconds,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
    }
    
    with open(config_training_path, "w") as f:
        json.dump(training_config, f, indent=2)
    
    print(f"Training config saved to {config_training_path}")
    print(f"  - {len(train_instance_ids)} instance_ids in training set")
    print(f"  - {len(val_instance_ids)} instance_ids in validation set")

    # Salva steering vectors (δ per ogni virtual layer)
    # Il rank verrà aggiunto dopo il probing
    steering_dir = os.path.join(args.project_root, "SteeringVectors", dataset_name)
    os.makedirs(steering_dir, exist_ok=True)
    sv_suffix = f"pairs{args.num_pairs}_cw{args.contrastive_weight}"
    steering_path = os.path.join(steering_dir, f"steering_vectors_{model_name_safe}_{sv_suffix}.pt")

    # =========================================================================
    # STEP 8: PROBING-BASED LAYER SELECTION (Eq. 15 del paper)
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 8: PROBING-BASED LAYER SELECTION")
    print("=" * 50)

    probing_results = calculate_probing_accuracy_all_modules(
        model=model,
        all_pos_acts=all_pos_acts,
        all_neg_acts=all_neg_acts,
        pos_centers=pos_centers,
        neg_centers=neg_centers,
        virtual_layer_info=virtual_layer_info,
        device=device,
    )

    # Seleziona top-k moduli
    top_k = args.top_k_modules
    top_k_modules = select_top_k_modules(probing_results, k=top_k)

    print(f"\nTop-{top_k} modules by probing accuracy:")
    for i, mod in enumerate(top_k_modules):
        print(
            f"  Rank {i + 1}: Layer {mod['physical_layer']}, {mod['module_type']}, "
            f"Accuracy: {mod['probing_accuracy']:.4f}"
        )

    # Crea rank_indices basato su probing accuracy
    rank_indices = [mod["virtual_layer_idx"] for mod in top_k_modules]

    # Ora salva steering vectors con il rank corretto
    torch.save(
        {
            "pos_center": pos_centers,
            "neg_center": neg_centers,
            "delta": deltas,  # δ = pos_center - neg_center (Eq. 12)
            "rank": rank_indices,  # Rank basato su probing accuracy
            "probing_results": probing_results,
            "top_k_modules": top_k_modules,
            "virtual_layer_info": virtual_layer_info,
            "num_virtual_layers": num_virtual_layers,
            "model_name": args.model_name,
            "dataset_name": dataset_name,
        },
        steering_path,
    )

    print(f"\nSteering vectors saved to {steering_path}")

    # Salva configurazione layer selection
    sv_suffix = f"pairs{args.num_pairs}_cw{args.contrastive_weight}"
    layer_config_path = os.path.join(
        steering_dir, f"layer_selection_{model_name_safe}_{sv_suffix}.json"
    )
    save_layer_selection_config(
        top_k_modules=top_k_modules,
        all_probing_results=probing_results,
        virtual_layer_info=virtual_layer_info,
        model_name=args.model_name,
        dataset_name=dataset_name,
        output_path=layer_config_path,
    )

    print(f"\nLayer selection config saved to {layer_config_path}")

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TruthX on BeliefBank with paired samples"
    )

    # Paths
    parser.add_argument(
        "--project_root", type=str, default=".", help="Project root directory"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="activation_cache_truthx",
        help="Directory for cached activations",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="truthx_models",
        help="Directory for output models",
    )

    # Model
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2.5-7B", help="LLM model name"
    )

    # Data
    parser.add_argument(
        "--beliefbank_data_type",
        type=str,
        default="facts",
        choices=["facts", "constraints"],
        help="BeliefBank data type",
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=2000,
        help="Number of (truthful, hallucinated) pairs",
    )
    parser.add_argument(
        "--extract_activations",
        action="store_true",
        default=False,
        help="Whether to extract activations",
    )
    parser.add_argument(
        "--quantization",
        action="store_true",
        default=True,
        help="Use 4-bit quantization for LLM",
    )

    # Model architecture
    parser.add_argument(
        "--semantic_latent_dim",
        type=int,
        default=1024,
        help="Dimension of semantic latent space",
    )
    parser.add_argument(
        "--truthful_latent_dim",
        type=int,
        default=1024,
        help="Dimension of truthful latent space",
    )
    parser.add_argument(
        "--semantic_hidden_dims",
        type=str,
        default="2048",
        help="Hidden dims for semantic encoder (comma-separated). Use '2048' to get input->2048->latent (1024)",
    )
    parser.add_argument(
        "--truthful_hidden_dims",
        type=str,
        default="2048",
        help="Hidden dims for truthful encoder (comma-separated). Use '2048' to get input->2048->latent (1024)",
    )
    parser.add_argument(
        "--decoder_hidden_dims",
        type=str,
        default="2048",
        help="Hidden dims for decoder (comma-separated). Use '2048' to get latent(1024)->2048->input",
    )
    # Early stopping
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs)",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=1e-4,
        help="Minimum improvement to reset early stopping patience",
    )

    # Training
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for training"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split ratio (0.2 = 20% validation)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1000, help="Number of training epochs"
    )
    parser.add_argument(
        "--contrastive_weight",
        type=float,
        default=1.0,
        help="Weight for contrastive losses",
    )
    parser.add_argument(
        "--editing_weight", type=float, default=1.0, help="Weight for editing loss"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for contrastive loss",
    )
    parser.add_argument(
        "--target_layers",
        type=str,
        default=None,
        help="Comma-separated list of layer indices (default: all layers)",
    )
    parser.add_argument(
        "--top_k_modules",
        type=int,
        default=-1,
        help="Number of top modules to select based on probing accuracy",
    )

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging for batch contents and loss values",
    )

    args = parser.parse_args()

    # Configure logging based on debug flag
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Parse target_layers
    if args.target_layers is not None:
        args.target_layers = [int(x) for x in args.target_layers.split(",")]

    train_truthx_on_beliefbank(args)
