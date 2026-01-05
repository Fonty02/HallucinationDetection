# Hallucination Detection with LLM Activations

This project provides tools to extract and save internal activations from Large Language Models (LLMs) for hallucination detection research. It focuses on collecting hidden states, MLP outputs, and attention outputs across model layers using the SimpleQA dataset.

## 🎯 Project Goal

The goal is to save LLM activations when processing questions and answers from the SimpleQA-verified dataset. These activations can later be used to train hallucination detection classifiers or analyze model behavior.

## 🛠️ Setup

1. Clone the repository:
```bash
git clone https://github.com/Fonty02/HallucinationDetection.git
cd HallucinationDetection
```

2. Create and activate a virtual environment using uv:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
uv sync
```

## 📊 Dataset

This project uses **SimpleQA-verified**, a 1,000-prompt factuality benchmark from Google DeepMind and Google Research, available on [🤗 HuggingFace](https://huggingface.co/datasets/google/simpleqa-verified).

The dataset contains:
- **problem**: Question testing parametric knowledge
- **answer**: Gold answer for verification
- **topic**: Subject category (e.g., Politics, Art, Sports)
- **answer_type**: Type of answer (Person, Date, Number, Place, Other)
- **multi_step**: Whether question requires multiple sources
- **requires_reasoning**: Whether complex reasoning is needed
- **urls**: Supporting URLs for verification

## 🚀 Usage

### Save Model Activations

To save LLM activations for the SimpleQA dataset:

```bash
python -m src.model.predict --model_name "meta-llama/Meta-Llama-3-8B" --data_name "simpleqa" --use_local
```

Parameters:
- `--model_name`: HuggingFace model identifier (default: "meta-llama/Meta-Llama-3-8B")
- `--data_name`: Dataset name (default: "simpleqa")
- `--use_local`: Use locally cached model and dataset

### What Gets Saved

The script saves:
1. **Hidden states** from each transformer layer (32 layers for Llama-3-8B)
2. **MLP outputs** from each layer
3. **Attention outputs** from each layer
4. **Model generations** (LLM responses to questions)
5. **Logits** (output probabilities)

All activations are saved in `activation_cache/{model_name}/simpleqa/`:
- `activation_hidden/` - Hidden states
- `activation_mlp/` - MLP layer outputs
- `activation_attn/` - Attention layer outputs
- `generations/` - Text generations
- `logits/` - Output logits

## 📁 Project Structure

```
HallucinationDetection/
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 setup.py
├── 📁 src/
│   ├── 📁 data/
│   │   ├── SimpleQADataset.py          # SimpleQA dataset loader
│   │   └── __init__.py
│   ├── 📁 model/
│   │   ├── HallucinationDetection.py   # Main class for activation extraction
│   │   ├── InspectOutputContext.py     # Context manager for layer inspection
│   │   ├── predict.py                  # Script to run activation saving
│   │   ├── prompts.py                  # Prompt templates
│   │   ├── utils.py                    # Utility functions
│   │   └── __init__.py
│   └── __init__.py
├── 📁 activation_cache/                 # Saved activations (created at runtime)
└── 📁 notebooks/                        # Analysis notebooks
```

## 💻 Example Code

```python
from src.model.HallucinationDetection import HallucinationDetection

# Initialize
detector = HallucinationDetection(project_dir=".")

# Save activations for SimpleQA
detector.save_model_activations(
    llm_name="meta-llama/Meta-Llama-3-8B",
    data_name="simpleqa",
    use_local=True
)
```

## 🔧 Technical Details

- **Supported Models**: Any HuggingFace Transformers model (tested with Llama-3-8B)
- **Activation Extraction**: Uses custom context manager to hook into model layers
- **Storage Format**: PyTorch tensors (.pt files) + JSON metadata
- **Layers Analyzed**: All 32 transformer layers (configurable via `TARGET_LAYERS`)
- **Quantization**: Supports BitsAndBytes 4-bit/8-bit quantization

## 📝 Notes

- Activation extraction requires significant disk space (~GB per model run)
- GPU with sufficient VRAM is recommended (tested on CUDA-enabled GPUs)
- The script processes all 1,000 examples from SimpleQA-verified
- Activations are saved per-instance and then combined into layer-wise tensors

