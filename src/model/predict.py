import os
import argparse
from src.model.HallucinationDetection import HallucinationDetection


PROJECT_ROOT = os.getcwd()


def main(args):
    model_name = args.model_name
    data_name = args.data_name
    use_local = args.use_local
    use_gemini = args.use_gemini
    gemini_model = args.gemini_model
    max_samples = args.max_samples
    quantization = args.quantization
    belief_bank_data_type = args.belief_bank_data_type
    
    hallucination_detector = HallucinationDetection(project_dir=PROJECT_ROOT)

    llm_name = model_name.split("/")[-1]

    print("=="*50)
    print(f"Saving activations for {data_name} dataset")
    if data_name == "belief_bank":
        print(f"BeliefBank data type: {belief_bank_data_type}")
    if use_gemini:
        print(f"Using Gemini autorater: {gemini_model}")
    else:
        print("Using substring matching for hallucination detection")
    if max_samples:
        print(f"Processing only first {max_samples} samples")
    if quantization:
        print("Using 4-bit quantization")
    else:
        print("Using full precision (bfloat16)")
    print("=="*50)
    
    hallucination_detector.save_model_activations(
        llm_name=model_name, 
        use_local=use_local, 
        data_name=data_name,
        use_gemini_autorater=use_gemini,
        gemini_model=gemini_model,
        max_samples=max_samples,
        quantization=quantization,
        belief_bank_data_type=belief_bank_data_type
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save LLM activations for dataset.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="Name of the LLM model to use.")
    parser.add_argument("--data_name", type=str, default=HallucinationDetection.DEFAULT_DATASET, help="Name of the dataset: simpleqa, halu_bench, belief_bank")
    parser.add_argument("--use_local", action="store_true", help="Use local model instead of remote.")
    parser.add_argument("--use_gemini", action="store_true", help="Use Gemini API as autorater for hallucination detection.")
    parser.add_argument("--gemini_model", type=str, default="gemini-2.0-flash-lite", help="Gemini model to use (default: gemini-2.0-flash-lite).")
    parser.add_argument("--max_samples", type=int, default=None, help="Number of samples to process (default: all).")
    parser.add_argument("--quantization", action="store_true", help="Use 4-bit quantization (default: full precision).")
    parser.add_argument("--belief_bank_data_type", type=str, default="facts", choices=["facts", "constraints"], help="BeliefBank data type: facts or constraints (default: facts).")

    args = parser.parse_args()
    main(args)

