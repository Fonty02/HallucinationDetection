"""
Valuta WaveSteer su un intero dataset.
Utile per Cross-Domain testing (es. train su BeliefBankFacts, test su BeliefBankConstraints).

Uso:
  python scripts/evaluate_wavesteer.py \
    --model_name Qwen/Qwen2.5-7B \
    --dataset belief_bank_constraints \
    --ckpt SteeringVectors/WaveSteer/Qwen2.5-7B/belief_bank_facts/wavesteer_best.pt \
    --limit 200
"""

import argparse
import os
import sys

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.WaveSteer.wavesteer_model import WaveSteer
from src.WaveSteer.inference import WaveSteerInferenceWrapper
from src.model.prompts import PROMPT_QA
from src.data.BeliefBankDataset import BeliefBankDataset


def evaluate_hallucination(generated_text: str, expected_label: str) -> bool:
    gen_lower = generated_text.lower().strip()
    exp_lower = expected_label.lower().strip()
    return exp_lower not in gen_lower


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--dataset", type=str, default="belief_bank_constraints")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to wavesteer_best.pt")
    parser.add_argument("--limit", type=int, default=100000000, help="Max campioni da valutare")
    parser.add_argument("--alpha", type=float, default=1.0, help="Steering strength")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # 1. Carica pesi WaveSteer
    if not os.path.exists(args.ckpt):
        print(f"Errore: Checkpoint {args.ckpt} non trovato.")
        return

    print("Caricamento WaveSteer...")
    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt["config"]
    
    wavesteer = WaveSteer(
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dwt_level=config["dwt_level"],
        bottleneck=config.get("bottleneck", 64),
    )
    wavesteer.load_state_dict(ckpt["state_dict"])
    wavesteer.to(device)
    wavesteer.eval()

    # 2. Carica Modello HF
    print(f"\nCaricamento base LLM ({args.model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()

    # 3. Wrapper
    wrapper = WaveSteerInferenceWrapper(
        model=model,
        wavesteer=wavesteer,
        target_layers=list(range(config["num_layers"])),
        activation_type="self_attn"
    )
    wrapper.register_hooks()

    # 4. Carica Dataset tramite BeliefBankDataset
    # supporta anche altre tipologie come in SLiM
    data_type = "facts"
    if "constraints" in args.dataset:
        data_type = "constraints"

    print(f"\nCaricamento Dataset {args.dataset}...")
    dataset = BeliefBankDataset(
        project_root=project_root,
        model_type="demo",
        recreate_ids=True,
        data_type=data_type,
    )
    
    num_samples = min(args.limit, len(dataset))
    print(f"Valutazione su {num_samples} campioni (limit={args.limit})")
    
    hallucinations_baseline = 0
    hallucinations_wavesteer = 0
    failures = 0

    pbar = tqdm(range(num_samples))
    for i in pbar:
        fact, label_yes_no, instance_id = dataset[i]
        prompt = PROMPT_QA.format(question=fact)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generazione Baseline
        wrapper.set_mode("off")
        with torch.no_grad():
            base_out = model.generate(**inputs, max_new_tokens=20, do_sample=False, temperature=0.1)
        base_text = tokenizer.decode(base_out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        is_hal_base = evaluate_hallucination(base_text, label_yes_no)
        if is_hal_base:
            hallucinations_baseline += 1

        # Generazione WaveSteer (Two-Pass)
        wrapper.clear()
        wrapper.set_mode("record")
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1)
            
        has_signal = wrapper.compute_edits(alpha=args.alpha)
        if has_signal:
            wrapper.set_mode("edit")
            with torch.no_grad():
                steered_out = model.generate(**inputs, max_new_tokens=20, do_sample=False, temperature=0.1)
            steered_text = tokenizer.decode(steered_out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            is_hal_steer = evaluate_hallucination(steered_text, label_yes_no)
            if is_hal_steer:
                hallucinations_wavesteer += 1
        else:
            failures += 1
            if is_hal_base:
                hallucinations_wavesteer += 1

        pbar.set_postfix(
            BaseHR=f"{hallucinations_baseline/(i+1):.2%}",
            WaveHR=f"{hallucinations_wavesteer/(i+1):.2%}"
        )
        
    wrapper.remove_hooks()
    
    print("\n" + "="*50)
    print("RISULTATI VALUTAZIONE CROSS-DOMAIN WAVESTEER")
    print(f"Modello LLM: {args.model_name}")
    print(f"Dataset Test: {args.dataset}")
    print(f"Alpha/Steering: {args.alpha}")
    print(f"Campioni usati: {num_samples} (failures a.k.a empty signal: {failures})")
    print("-" * 50)
    hr_base = hallucinations_baseline / num_samples
    hr_wave = hallucinations_wavesteer / num_samples
    print(f"[BASELINE]  Allucinazioni: {hallucinations_baseline} -> HR = {hr_base:.2%}")
    print(f"[WAVESTEER] Allucinazioni: {hallucinations_wavesteer} -> HR = {hr_wave:.2%}")
    print("="*50)


if __name__ == "__main__":
    main()
