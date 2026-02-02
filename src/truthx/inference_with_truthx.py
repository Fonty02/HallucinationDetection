import os
import json
import torch
import argparse
from tqdm import tqdm
from typing import List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from truthx_model import TruthX, LLMArchitectureDetector


def get_model_layers_info(model) -> Tuple[List, int, str, dict]:
    """
    Ottiene informazioni sui layer del modello in modo agnostico.

    Returns:
        (layers, num_layers, arch_name, arch_config)
    """
    arch_name, arch_config = LLMArchitectureDetector.detect_architecture(model)
    layers = LLMArchitectureDetector.get_layers(model, arch_config)
    num_layers = len(layers)

    print(f"Detected architecture: {arch_name} with {num_layers} layers")

    return layers, num_layers, arch_name, arch_config


def apply_truthx_hook(
    model, truthx_editor: TruthX, target_layers: List[int], arch_config: dict = None
):
    """
    Applica hooks al modello per editare le rappresentazioni durante l'inferenza.
    Supporta diverse architetture LLM.

    Args:
        model: Modello LLM (es. LlamaForCausalLM, GPT2LMHeadModel, ecc.)
        truthx_editor: Istanza di TruthX per l'editing
        target_layers: Lista di layer indices da editare
        arch_config: Configurazione architettura (se None, viene rilevata automaticamente)
    """
    hooks = []

    if arch_config is None:
        _, arch_config = LLMArchitectureDetector.detect_architecture(model)

    layers = LLMArchitectureDetector.get_layers(model, arch_config)

    def create_hook(layer_name: str):
        def hook_fn(module, input, output):
            # output può essere un tensore o una tupla
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Imposta il layer corrente per TruthX
            truthx_editor.cur_layer_id = layer_name

            # Applica TruthX editing
            edited = truthx_editor.edit(hidden_states)

            if isinstance(output, tuple):
                return (edited,) + output[1:]
            else:
                return edited

        return hook_fn

    # Registra hooks per attention e FFN modules
    for layer_idx in target_layers:
        if layer_idx >= len(layers):
            print(
                f"Warning: Layer {layer_idx} not found in model with {len(layers)} layers. Skipping."
            )
            continue

        layer = layers[layer_idx]

        # Hook per attention output
        try:
            attn_module = LLMArchitectureDetector.get_attention_module(
                layer, arch_config
            )
            hook = attn_module.register_forward_hook(create_hook(f"{layer_idx}.attn"))
            hooks.append(hook)
        except AttributeError as e:
            print(
                f"Warning: Could not register attention hook for layer {layer_idx}: {e}"
            )

        # Hook per MLP/FFN output
        try:
            mlp_module = LLMArchitectureDetector.get_mlp_module(layer, arch_config)
            hook = mlp_module.register_forward_hook(create_hook(f"{layer_idx}.ffn"))
            hooks.append(hook)
        except AttributeError as e:
            print(f"Warning: Could not register MLP hook for layer {layer_idx}: {e}")

    return hooks


def remove_hooks(hooks: List):
    """Rimuovi tutti gli hooks registrati."""
    for hook in hooks:
        hook.remove()


def generate_with_truthx(
    model,
    tokenizer,
    prompts: List[str],
    truthx_editor: TruthX,
    target_layers: List[int],
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    do_sample: bool = True,
    arch_config: dict = None,
) -> List[str]:
    """
    Genera testo con TruthX editing attivo.

    Args:
        model: Modello LLM
        tokenizer: Tokenizer corrispondente
        prompts: Lista di prompt
        truthx_editor: Istanza TruthX
        target_layers: Layer da editare
        max_new_tokens: Numero massimo di token da generare
        temperature: Temperatura per sampling
        do_sample: Se usare sampling o greedy decoding
        arch_config: Configurazione architettura (opzionale)

    Returns:
        Lista di testi generati
    """
    # Applica hooks
    hooks = apply_truthx_hook(model, truthx_editor, target_layers, arch_config)

    try:
        # Tokenizza input
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Genera
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decodifica
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Rimuovi il prompt dall'output
        results = []
        for prompt, generated in zip(prompts, generated_texts):
            if generated.startswith(prompt):
                result = generated[len(prompt) :].strip()
            else:
                result = generated.strip()
            results.append(result)

        return results

    finally:
        # Rimuovi hooks
        remove_hooks(hooks)


def evaluate_truthfulness(
    model,
    tokenizer,
    truthx_editor: Optional[TruthX],
    dataset_path: str,
    target_layers: List[int],
    output_path: str,
    num_samples: int = None,
    arch_config: dict = None,
):
    """
    Valuta la truthfulness su un dataset.

    Args:
        model: Modello LLM
        tokenizer: Tokenizer
        truthx_editor: TruthX editor (None per baseline)
        dataset_path: Path al dataset JSON
        target_layers: Layer da editare
        output_path: Path per salvare i risultati
        num_samples: Numero di samples da valutare (None = tutti)
        arch_config: Configurazione architettura (opzionale)
    """
    # Carica dataset
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    if num_samples:
        dataset = dataset[:num_samples]

    results = []

    print(f"Evaluating on {len(dataset)} samples...")

    for item in tqdm(dataset):
        question = item["question"]
        correct_answer = item.get("correct_answer", "")

        # Genera con TruthX (se fornito)
        if truthx_editor:
            generated = generate_with_truthx(
                model,
                tokenizer,
                [question],
                truthx_editor,
                target_layers,
                max_new_tokens=100,
                do_sample=False,
                arch_config=arch_config,
            )[0]
        else:
            # Baseline senza TruthX
            inputs = tokenizer([question], return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if generated.startswith(question):
                generated = generated[len(question) :].strip()

        results.append(
            {
                "question": question,
                "generated": generated,
                "correct_answer": correct_answer,
            }
        )

    # Salva risultati
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")


def compare_baseline_vs_truthx(
    model_name: str,
    truthx_model_path: str,
    dataset_path: str,
    output_dir: str,
    target_layers: List[int],
    edit_strength: float = 1.0,
    num_samples: int = None,
):
    """
    Confronta baseline vs TruthX editing.
    Rileva automaticamente l'architettura del modello.
    """
    # Carica modello e tokenizer
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # Rileva architettura automaticamente
    layers, num_layers, arch_name, arch_config = get_model_layers_info(model)

    # Verifica che target_layers sia valido
    max_layer = max(target_layers) if target_layers else 0
    if max_layer >= num_layers:
        print(
            f"Warning: Max target layer {max_layer} exceeds model layers {num_layers}. Adjusting..."
        )
        target_layers = [l for l in target_layers if l < num_layers]
        if not target_layers:
            target_layers = list(range(max(0, num_layers - 10), num_layers))
            print(f"Using default layers: {target_layers}")

    # Ottieni hidden_size in modo agnostico
    hidden_size = LLMArchitectureDetector.get_hidden_size(model)
    print(f"Hidden size: {hidden_size}")

    # Baseline evaluation
    print("\n=== Baseline Evaluation ===")
    baseline_output = os.path.join(output_dir, "baseline_results.json")
    evaluate_truthfulness(
        model,
        tokenizer,
        None,
        dataset_path,
        target_layers,
        baseline_output,
        num_samples,
        arch_config,
    )

    # TruthX evaluation
    print("\n=== TruthX Evaluation ===")
    print(f"Loading TruthX model from {truthx_model_path}...")
    truthx_editor = TruthX(
        model_path=truthx_model_path,
        hidden_size=hidden_size,
        edit_strength=edit_strength,
        top_layers=len(target_layers),
    )

    truthx_output = os.path.join(output_dir, "truthx_results.json")
    evaluate_truthfulness(
        model,
        tokenizer,
        truthx_editor,
        dataset_path,
        target_layers,
        truthx_output,
        num_samples,
        arch_config,
    )

    print("\n=== Evaluation Complete ===")
    print(f"Baseline results: {baseline_output}")
    print(f"TruthX results: {truthx_output}")


def interactive_demo(
    model_name: str,
    truthx_model_path: str,
    target_layers: List[int],
    edit_strength: float = 1.0,
):
    """
    Demo interattiva per testare TruthX.
    Rileva automaticamente l'architettura del modello.
    """
    # Carica modello
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # Rileva architettura automaticamente
    layers, num_layers, arch_name, arch_config = get_model_layers_info(model)

    # Verifica target_layers
    max_layer = max(target_layers) if target_layers else 0
    if max_layer >= num_layers:
        print("Warning: Adjusting target layers to fit model architecture...")
        target_layers = [l for l in target_layers if l < num_layers]
        if not target_layers:
            target_layers = list(range(max(0, num_layers - 10), num_layers))

    # Ottieni hidden_size in modo agnostico
    hidden_size = LLMArchitectureDetector.get_hidden_size(model)

    # Carica TruthX
    print(f"Loading TruthX from {truthx_model_path}...")
    truthx_editor = TruthX(
        model_path=truthx_model_path,
        hidden_size=hidden_size,
        edit_strength=edit_strength,
        top_layers=len(target_layers),
    )

    print("\n=== TruthX Interactive Demo ===")
    print(f"Architecture: {arch_name} ({num_layers} layers)")
    print("Type your questions (or 'quit' to exit)")
    print(f"Edit strength: {edit_strength}")
    print(f"Target layers: {target_layers}\n")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            break

        if not question:
            continue

        # Genera baseline
        print("\n[Baseline]")
        inputs = tokenizer([question], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        baseline = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if baseline.startswith(question):
            baseline = baseline[len(question) :].strip()
        print(baseline)

        # Genera con TruthX
        print("\n[TruthX Edited]")
        truthx_result = generate_with_truthx(
            model,
            tokenizer,
            [question],
            truthx_editor,
            target_layers,
            max_new_tokens=100,
            do_sample=False,
            arch_config=arch_config,
        )[0]
        print(truthx_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Modalità
    parser.add_argument(
        "--mode",
        type=str,
        choices=["evaluate", "interactive", "compare"],
        default="compare",
        help="Modalità di esecuzione",
    )

    # Modello
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Nome o path del modello HuggingFace (es. meta-llama/Llama-2-7b-chat-hf, gpt2, ecc.)",
    )
    parser.add_argument(
        "--truthx_model",
        type=str,
        required=True,
        help="Path al checkpoint TruthX (.pt)",
    )

    # Dataset (per evaluate/compare)
    parser.add_argument("--dataset_path", type=str, help="Path al dataset JSON")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory per salvare i risultati",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Numero di samples da valutare (None=tutti)",
    )

    # TruthX parameters
    parser.add_argument(
        "--target_layers",
        type=str,
        default="10,11,12,13,14,15,16,17,18,19",
        help="Layer da editare (comma-separated). L'architettura viene rilevata automaticamente.",
    )
    parser.add_argument(
        "--edit_strength", type=float, default=1.0, help="Forza dell'editing (alpha)"
    )

    args = parser.parse_args()
    args.target_layers = [int(x) for x in args.target_layers.split(",")]

    if args.mode == "interactive":
        interactive_demo(
            args.model_name,
            args.truthx_model,
            args.target_layers,
            args.edit_strength,
        )

    elif args.mode == "evaluate":
        if not args.dataset_path:
            raise ValueError("--dataset_path required for evaluate mode")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Rileva architettura
        layers, num_layers, arch_name, arch_config = get_model_layers_info(model)
        hidden_size = LLMArchitectureDetector.get_hidden_size(model)

        truthx_editor = TruthX(
            model_path=args.truthx_model,
            hidden_size=hidden_size,
            edit_strength=args.edit_strength,
            top_layers=len(args.target_layers),
        )

        evaluate_truthfulness(
            model,
            tokenizer,
            truthx_editor,
            args.dataset_path,
            args.target_layers,
            os.path.join(args.output_dir, "results.json"),
            args.num_samples,
            arch_config,
        )

    elif args.mode == "compare":
        if not args.dataset_path:
            raise ValueError("--dataset_path required for compare mode")

        compare_baseline_vs_truthx(
            args.model_name,
            args.truthx_model,
            args.dataset_path,
            args.output_dir,
            args.target_layers,
            args.edit_strength,
            args.num_samples,
        )
