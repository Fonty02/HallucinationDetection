"""
Dataset per SLiM Hallucination Reduction.

Converte i paired dataset di TruthX (BeliefBank Facts, BeliefBank Constraints,
HaluEval) nel formato SLiM: (tokenized_text, state), dove:
- state = [1.0] per campioni truthful (positive)
- state = [0.0] per campioni hallucinated (negative)

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
                "positive": {"question": fact_pos, "answer": "True",
                             "instance_id": pair_id * 2, "label": 1},
                "negative": {"question": fact_neg, "answer": "False",
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
                positive = {"question": fact_pos, "answer": "True",
                            "instance_id": pair_id * 2, "label": 1}
                negative = {"question": fact_neg, "answer": "False",
                            "instance_id": pair_id * 2 + 1, "label": 0}
            else:
                positive = {"question": fact_neg, "answer": "True",
                            "instance_id": pair_id * 2, "label": 1}
                negative = {"question": fact_pos, "answer": "False",
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
    ):
        """
        Args:
            pairs: Lista di coppie dal formato TruthX
                   [{"positive": {..., "label": 1}, "negative": {..., "label": 0}}]
            tokenizer: Tokenizer HuggingFace
            prompt_template: Template del prompt (es. PROMPT_QA, PROMPT_HALU)
            max_length: Lunghezza massima della sequenza tokenizzata
            include_answer_in_input: Se True, concatena la risposta al prompt per il training
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

                # Per il language modeling: input = tokens[:-1], target = tokens[1:]
                if input_ids.size(0) > 1:
                    self.samples.append({
                        "input_ids": input_ids[:-1],
                        "target_ids": input_ids[1:],
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


def create_slim_dataset(
    dataset_name: str,
    tokenizer,
    project_root: str,
    num_pairs: int = 500,
    max_length: int = 0,  # 0 = no truncation (use actual input length)
    use_local_halueval: bool = False,
) -> SLiMHallucinationDataset:
    """
    Factory function: crea un SLiMHallucinationDataset dal nome del dataset.

    Args:
        dataset_name: "belief_bank_facts", "belief_bank_constraints", o "halu_eval"
        tokenizer: Tokenizer HuggingFace
        project_root: Root del progetto (per BeliefBank)
        num_pairs: Numero di coppie
        max_length: Lunghezza massima sequenza
        use_local_halueval: Se usare HaluEval locale

    Returns:
        SLiMHallucinationDataset pronto per il DataLoader
    """
    from src.model.prompts import PROMPT_QA, PROMPT_HALU, PROMPT_TRUTHX

    if dataset_name == "halu_eval":
        pairs = create_paired_halueval_subset(
            num_pairs=num_pairs, use_local=use_local_halueval
        )
        prompt_template = PROMPT_TRUTHX
    elif dataset_name in ("belief_bank_facts", "belief_bank_constraints"):
        data_type = dataset_name.replace("belief_bank_", "")
        pairs = create_paired_beliefbank_subset(
            project_root=project_root,
            data_type=data_type,
            num_pairs=num_pairs,
        )
        prompt_template = PROMPT_TRUTHX
    else:
        raise ValueError(
            f"Dataset sconosciuto: {dataset_name}. "
            f"Usa 'belief_bank_facts', 'belief_bank_constraints', o 'halu_eval'"
        )

    return SLiMHallucinationDataset(
        pairs=pairs,
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        max_length=max_length,
    )
