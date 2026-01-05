import os
import sys
import argparse

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

PROJECT_ROOT = project_root
print(f"Project root: {PROJECT_ROOT}")
from src.model.HallucinationDetection import HallucinationDetection



def main(args):
    model_name = args.model_name
    data_name = args.data_name
    use_local = args.use_local
    max_samples = args.max_samples
    quantization = args.quantization
    belief_bank_data_type = args.belief_bank_data_type
    
    hallucination_detector = HallucinationDetection(project_dir=PROJECT_ROOT)

    print("=="*50)
    print(f"Saving activations for {data_name} dataset")
    if data_name == "belief_bank":
        print(f"BeliefBank data type: {belief_bank_data_type}")
    if max_samples:
        print(f"Processing only first {max_samples} samples")
    if quantization:
        print("Using 4-bit quantization")
    else:
        print("Using full precision (bfloat16)")
    print("=="*50)
    
    hallucination_detector.save_model_activations(
        llm_name=model_name,  
        max_samples=max_samples,
        quantization=quantization,
        use_local=use_local,
        data_name=data_name,
        belief_bank_data_type=belief_bank_data_type
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save LLM activations for dataset.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="Name of the LLM model to use.")
    parser.add_argument("--data_name", type=str, default=HallucinationDetection.DEFAULT_DATASET, help="Name of the dataset: simpleqa, halu_bench, belief_bank")
    parser.add_argument("--use_local", action="store_true", help="Use local model instead of remote.")
    parser.add_argument("--max_samples", type=int, default=None, help="Number of samples to process (default: all).")
    parser.add_argument("--quantization", action="store_true", help="Use 4-bit quantization (default: full precision).")
    parser.add_argument("--belief_bank_data_type", type=str, default="facts", choices=["facts", "constraints"], help="BeliefBank data type: facts or constraints (default: facts).")

    args = parser.parse_args()
    main(args)

