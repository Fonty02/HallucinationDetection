"""
Dataset per SLiM Hallucination Reduction.

Supporta due modalità:
1. SLiMHallucinationDataset: flat dataset con (input_ids, target_ids, mask, state)
   per training con CrossEntropy (backward-compatible).
2. SLiMPairedDataset: dataset paired (pos_tokens, neg_tokens)
   per training contrastivo (InfoNCE). Ogni coppia ha:
   - pos: campione truthful (text = prompt + answer)
   - neg: campione hallucinated (text = prompt + answer)
   Entrambi vengono processati con state=1.0 — il SLiM module deve apprendere
   scale/shift che separano truthful da hallucinated nello spazio nascosto.

Riutilizza le funzioni create_paired_beliefbank_subset() e
create_paired_halueval_subset() da train_truthx.py.
"""

import os
import sys
import torch
from torch.utils.data import Dataset
from typing import List, Dict

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


# =============================================================================
# Pair creation helpers (extracted from src/truthx/train_truthx.py to avoid
# importing that module, which has a relative `from truthx_model import ...`
# that only works when run directly from its own directory)
# =============================================================================

def create_paired_halueval_subset(num_pairs: int = 500, use_local: bool = False) -> List[Dict]:
    """
    Crea un subset paired da HaluEval.
    Ogni coppia: positive=right_response, negative=hallucinated_response (stesso contesto).
    """
    from src.data.HaluEvalDataset import HaluEvalDataset

    dataset_right = HaluEvalDataset(label=0, use_local=use_local)
    dataset_hal = HaluEvalDataset(label=1, use_local=use_local)

    total_samples = min(len(dataset_right), num_pairs)
    pairs = []

    for pair_id in range(total_samples):
        question_right, answer_right, instance_id = dataset_right[pair_id]
        question_hal, answer_hal, _ = dataset_hal[pair_id]

        assert question_right == question_hal, f"Context mismatch at index {pair_id}"

        pairs.append({
            "positive": {"question": question_right, "answer": answer_right,
                         "instance_id": pair_id * 2, "label": 1},
            "negative": {"question": question_hal, "answer": answer_hal,
                         "instance_id": pair_id * 2 + 1, "label": 0},
            "pair_id": pair_id,
        })

        if pair_id == 0:
            print(f"  [debug] pair 0 HaluEval: {question_right[:80]}...")

    print(f"Creati {len(pairs)} coppie paired da HaluEval")
    return pairs


def create_paired_beliefbank_subset(
    project_root: str,
    data_type: str = "facts",
    num_pairs: int = 500,
) -> List[Dict]:
    """
    Crea un subset bilanciato di BeliefBank con coppie (fact, negated_fact).
    data_type: "facts" o "constraints"
    """
    from src.data.BeliefBankDataset import BeliefBankDataset

    dataset = BeliefBankDataset(
        project_root=project_root,
        model_type="demo",
        recreate_ids=True,
        data_type=data_type,
    )

    total_samples = len(dataset)
    pairs = []
    pair_id = 0

    if data_type == "constraints":
        for i in range(0, min(num_pairs * 2, total_samples), 2):
            fact_pos, _, _ = dataset[i]
            fact_neg, _, _ = dataset[i + 1]
            pairs.append({
                "positive": {"question": fact_pos, "answer": "yes",
                             "instance_id": pair_id * 2, "label": 1},
                "negative": {"question": fact_neg, "answer": "no",
                             "instance_id": pair_id * 2 + 1, "label": 0},
                "pair_id": pair_id,
            })
            if pair_id == 0:
                # Mostra entrambe le parti (positive + negative) per il primo esempio
                print(f"  [debug] pair 0 constraints (pos / neg):\n  POS: {fact_pos[:80]}\n  NEG: {fact_neg[:80]}")
            pair_id += 1
    else:  # facts
        half = total_samples // 2
        for i in range(min(num_pairs, half)):
            fact_pos, label_pos, _ = dataset[i]
            fact_neg, _, _ = dataset[i + half]

            if label_pos == "yes":
                positive = {"question": fact_pos, "answer": "yes",
                            "instance_id": pair_id * 2, "label": 1}
                negative = {"question": fact_neg, "answer": "no",
                            "instance_id": pair_id * 2 + 1, "label": 0}
            else:
                positive = {"question": fact_neg, "answer": "yes",
                            "instance_id": pair_id * 2, "label": 1}
                negative = {"question": fact_pos, "answer": "no",
                            "instance_id": pair_id * 2 + 1, "label": 0}

            pairs.append({"positive": positive, "negative": negative, "pair_id": pair_id})
            if pair_id == 0:
                print(f"  [debug] pair 0 facts: {positive['question'][:80]}\n{negative['question'][:80]}")
            pair_id += 1

    print(f"Creati {len(pairs)} coppie paired da BeliefBank ({data_type})")
    return pairs


def collate_fn_hallucination(batch):
    """
    Collate function per SLiM hallucination dataset.

    Gestisce sequenze di lunghezza variabile con padding dinamico.

    Args:
        batch: List of (input_ids, target_ids, attention_mask, state)
    Returns:
        Tuple of stacked tensors
    """
    input_ids_list, target_ids_list, attention_mask_list, states = zip(*batch)

    # Padding a lunghezza massima nel batch
    max_len = max(ids.size(0) for ids in input_ids_list)

    padded_input = []
    padded_target = []
    padded_mask = []

    for inp, tgt, mask in zip(input_ids_list, target_ids_list, attention_mask_list):
        pad_len = max_len - inp.size(0)
        if pad_len > 0:
            padded_input.append(torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)]))
            padded_target.append(torch.cat([tgt, torch.full((pad_len,), -100, dtype=torch.long)]))
            padded_mask.append(torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)]))
        else:
            padded_input.append(inp)
            padded_target.append(tgt)
            padded_mask.append(mask)

    return (
        torch.stack(padded_input),
        torch.stack(padded_target),
        torch.stack(padded_mask),
        torch.stack(list(states)),
    )


class SLiMHallucinationDataset(Dataset):
    """
    Dataset per il training di SLiM su hallucination reduction.

    Converte le coppie paired (positive/negative) dal formato TruthX
    al formato SLiM con stato scalare [0.0] o [1.0].

    Ogni campione contiene:
    - question: la domanda/fatto
    - answer: la risposta attesa
    - state: [1.0] per truthful, [0.0] per hallucinated
    - input_ids: tokens della sequenza "prompt + answer" (troncata)
    - target_ids: shifted per language modeling
    - attention_mask: mask di attenzione
    """

    def __init__(
        self,
        pairs: List[Dict],
        tokenizer,
        prompt_template: str,
        max_length: int = 0,  # 0 = no truncation (use actual input length)
        include_answer_in_input: bool = True,
        train_on_answer_only: bool = True,
    ):
        """
        Args:
            pairs: Lista di coppie dal formato TruthX
                   [{"positive": {..., "label": 1}, "negative": {..., "label": 0}}]
            tokenizer: Tokenizer HuggingFace
            prompt_template: Template del prompt (es. PROMPT_QA, PROMPT_HALU)
            max_length: Lunghezza massima della sequenza tokenizzata
            include_answer_in_input: Se True, concatena la risposta al prompt per il training
            train_on_answer_only: Se True, la CE viene calcolata solo sui token
                                  della risposta (prompt masked con -100 nei target)
        """
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.samples = []

        for pair in pairs:
            for key, state_value in [("positive", 1.0), ("negative", 0.0)]:
                sample = pair[key]
                question = sample["question"]
                answer = sample.get("answer", "")

                # Costruisci il testo completo per il training
                prompt = prompt_template.format(question=question)
                if include_answer_in_input and answer:
                    full_text = f"{prompt} {answer}"
                else:
                    full_text = prompt

                # Tokenizza — se max_length <= 0 non troncare (usa lunghezza reale dell'input)
                if self.max_length is None or self.max_length <= 0:
                    encoded = tokenizer(
                        full_text,
                        truncation=False,
                        return_tensors="pt",
                        padding=False,
                    )
                else:
                    encoded = tokenizer(
                        full_text,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                        padding=False,
                    )

                input_ids = encoded["input_ids"].squeeze(0)
                attention_mask = encoded["attention_mask"].squeeze(0)

                # Per LM: input = tokens[:-1], target = tokens[1:].
                # In modalità answer-only maskiamo i token target del prompt.
                if input_ids.size(0) > 1:
                    target_ids = input_ids[1:].clone()

                    if train_on_answer_only and include_answer_in_input and answer:
                        if self.max_length is None or self.max_length <= 0:
                            prompt_encoded = tokenizer(
                                prompt,
                                truncation=False,
                                return_tensors="pt",
                                padding=False,
                            )
                        else:
                            prompt_encoded = tokenizer(
                                prompt,
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors="pt",
                                padding=False,
                            )

                        prompt_ids = prompt_encoded["input_ids"].squeeze(0)
                        # Prefix length robusta: in caso di tokenizzazione non perfettamente allineata
                        # tra prompt e prompt+answer, usa il massimo prefisso comune.
                        prefix_len = 0
                        limit = min(prompt_ids.size(0), input_ids.size(0))
                        while prefix_len < limit and prompt_ids[prefix_len] == input_ids[prefix_len]:
                            prefix_len += 1

                        # target[t] predice input_ids[t+1], quindi per includere il primo token
                        # della risposta (indice prefix_len) bisogna tenere da t=prefix_len-1.
                        supervise_from = max(prefix_len - 1, 0)
                        if supervise_from > 0:
                            target_ids[:supervise_from] = -100

                    # Se dopo il masking non resta alcun target valido, salta il campione
                    if torch.all(target_ids == -100):
                        continue

                    self.samples.append({
                        "input_ids": input_ids[:-1],
                        "target_ids": target_ids,
                        "attention_mask": attention_mask[:-1],
                        "state": torch.FloatTensor([state_value]),
                        "question": question,
                        "answer": answer,
                        "label": sample.get("label", int(state_value)),
                    })

        print(f"[SLiM Dataset] Creati {len(self.samples)} campioni da {len(pairs)} coppie")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["input_ids"], s["target_ids"], s["attention_mask"], s["state"]


# =============================================================================
# PAIRED DATASET FOR CONTRASTIVE TRAINING
# =============================================================================


class SLiMPairedDataset(Dataset):
    """
    Dataset per contrastive training di SLiM (InfoNCE).

    Restituisce coppie (pos_tokens, neg_tokens) mantenendo l'allineamento.
    Ogni campione include il testo completo (prompt + answer) per garantire
    che le rappresentazioni differiscano anche per HaluEval dove le domande
    sono identiche per pos e neg.

    NOTA: state=1.0 per ENTRAMBI (pos e neg). La stessa trasformazione FiLM
    viene applicata a entrambi: scale agisce come selettore di feature e
    shift come bias direzionale, forzando lo spazio nascosto a separare
    truth da hallucination in base al CONTENUTO, non allo stato.
    """

    def __init__(
        self,
        pairs: List[Dict],
        tokenizer,
        prompt_template: str,
        max_length: int = 0,
    ):
        """
        Args:
            pairs: Lista di coppie dal formato TruthX
                   [{"positive": {...}, "negative": {...}}]
            tokenizer: Tokenizer HuggingFace
            prompt_template: Template del prompt (es. PROMPT_TRUTHX)
            max_length: Lunghezza massima della sequenza (0 = no truncation)
        """
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.samples = []

        for pair in pairs:
            pos = pair["positive"]
            neg = pair["negative"]

            pos_prompt = self._build_prompt(pos)
            neg_prompt = self._build_prompt(neg)
            pos_text = self._build_text(pos_prompt, pos.get("answer", ""))
            neg_text = self._build_text(neg_prompt, neg.get("answer", ""))

            pos_enc = self._tokenize(pos_text)
            neg_enc = self._tokenize(neg_text)
            pos_prompt_enc = self._tokenize(pos_prompt)
            neg_prompt_enc = self._tokenize(neg_prompt)
            pos_answer_start = self._common_prefix_len(
                pos_prompt_enc["input_ids"], pos_enc["input_ids"]
            )
            neg_answer_start = self._common_prefix_len(
                neg_prompt_enc["input_ids"], neg_enc["input_ids"]
            )

            # Verifica che i token non siano vuoti
            if pos_enc["input_ids"].size(0) > 0 and neg_enc["input_ids"].size(0) > 0:
                self.samples.append({
                    "pos_input_ids": pos_enc["input_ids"],
                    "pos_attention_mask": pos_enc["attention_mask"],
                    "neg_input_ids": neg_enc["input_ids"],
                    "neg_attention_mask": neg_enc["attention_mask"],
                    "pos_answer_start": pos_answer_start,
                    "neg_answer_start": neg_answer_start,
                })

        print(f"[SLiM Paired] Creati {len(self.samples)} coppie da {len(pairs)} pairs")

    def _build_prompt(self, sample: Dict) -> str:
        """Costruisce il prompt con lo stesso template usato in inferenza."""
        return self.prompt_template.format(question=sample["question"])

    def _build_text(self, prompt: str, answer: str) -> str:
        """Costruisce il testo completo (prompt + answer)."""
        if answer:
            return f"{prompt} {answer}"
        return prompt

    @staticmethod
    def _common_prefix_len(a: torch.Tensor, b: torch.Tensor) -> int:
        """Lunghezza del massimo prefisso comune tra due sequenze token."""
        prefix_len = 0
        limit = min(a.size(0), b.size(0))
        while prefix_len < limit and a[prefix_len] == b[prefix_len]:
            prefix_len += 1
        return int(prefix_len)

    def _tokenize(self, text: str) -> Dict:
        """Tokenizza il testo con troncamento opzionale."""
        if self.max_length is None or self.max_length <= 0:
            encoded = self.tokenizer(
                text, truncation=False, return_tensors="pt", padding=False
            )
        else:
            encoded = self.tokenizer(
                text, truncation=True, max_length=self.max_length,
                return_tensors="pt", padding=False
            )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s["pos_input_ids"], s["pos_attention_mask"],
            s["neg_input_ids"], s["neg_attention_mask"],
            s["pos_answer_start"], s["neg_answer_start"],
        )


def collate_fn_paired(batch):
    """
    Collate function per SLiMPairedDataset.

    Padda pos e neg separatamente (possono avere lunghezze diverse) e
    restituisce tensori pronti per il training contrastivo.

    Args:
        batch: List of
            (pos_input_ids, pos_mask, neg_input_ids, neg_mask, pos_answer_start, neg_answer_start)

    Returns:
        (pos_input_ids, pos_attention_mask, neg_input_ids, neg_attention_mask,
         pos_answer_start, neg_answer_start)
        tutti padded a lunghezza massima nel rispettivo gruppo.
    """
    (
        pos_ids_list,
        pos_masks_list,
        neg_ids_list,
        neg_masks_list,
        pos_answer_starts,
        neg_answer_starts,
    ) = zip(*batch)

    def _pad_sequences(ids_list, masks_list, pad_value=0):
        max_len = max(x.size(0) for x in ids_list)
        padded_ids = []
        padded_masks = []
        for ids, mask in zip(ids_list, masks_list):
            pad_len = max_len - ids.size(0)
            if pad_len > 0:
                padded_ids.append(
                    torch.cat([ids, torch.full((pad_len,), pad_value, dtype=ids.dtype)])
                )
                padded_masks.append(
                    torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
                )
            else:
                padded_ids.append(ids)
                padded_masks.append(mask)
        return torch.stack(padded_ids), torch.stack(padded_masks)

    pos_ids, pos_masks = _pad_sequences(pos_ids_list, pos_masks_list)
    neg_ids, neg_masks = _pad_sequences(neg_ids_list, neg_masks_list)

    return (
        pos_ids,
        pos_masks,
        neg_ids,
        neg_masks,
        torch.tensor(pos_answer_starts, dtype=torch.long),
        torch.tensor(neg_answer_starts, dtype=torch.long),
    )


def create_slim_dataset(
    dataset_name: str,
    tokenizer,
    project_root: str,
    num_pairs: int = 500,
    max_length: int = 0,  # 0 = no truncation (use actual input length)
    use_local_halueval: bool = False,
    paired: bool = False,
    train_on_answer_only: bool = True,
):
    """
    Factory function: crea un SLiMHallucinationDataset o SLiMPairedDataset.

    Args:
        dataset_name: "belief_bank_facts", "belief_bank_constraints", o "halu_eval"
        tokenizer: Tokenizer HuggingFace
        project_root: Root del progetto (per BeliefBank)
        num_pairs: Numero di coppie
        max_length: Lunghezza massima sequenza
        use_local_halueval: Se usare HaluEval locale
        paired: Se True, restituisce SLiMPairedDataset per contrastive training.
                Se False, restituisce SLiMHallucinationDataset (flat, per CE loss).
        train_on_answer_only: Se True (solo flat/generative), la loss CE è
                              calcolata sui token risposta e non sul prompt.

    Returns:
        SLiMPairedDataset o SLiMHallucinationDataset
    """
    from src.model.prompts import PROMPT_QA, PROMPT_HALU, PROMPT_TRUTHX

    if dataset_name == "halu_eval":
        pairs = create_paired_halueval_subset(
            num_pairs=num_pairs, use_local=use_local_halueval
        )
        # Use PROMPT_HALU to match inference prompt template
        prompt_template = PROMPT_HALU
    elif dataset_name in ("belief_bank_facts", "belief_bank_constraints"):
        data_type = dataset_name.replace("belief_bank_", "")
        pairs = create_paired_beliefbank_subset(
            project_root=project_root,
            data_type=data_type,
            num_pairs=num_pairs,
        )
        # Use PROMPT_QA to match inference prompt template
        prompt_template = PROMPT_QA
    else:
        raise ValueError(
            f"Dataset sconosciuto: {dataset_name}. "
            f"Usa 'belief_bank_facts', 'belief_bank_constraints', o 'halu_eval'"
        )

    if paired:
        return SLiMPairedDataset(
            pairs=pairs,
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_length=max_length,
        )

    return SLiMHallucinationDataset(
        pairs=pairs,
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        max_length=max_length,
        train_on_answer_only=train_on_answer_only,
    )
