# Hallucination Detection with LLM Activations

This project provides tools to extract and save internal activations from Large Language Models (LLMs) for hallucination detection research. It focuses on collecting hidden states, MLP outputs, and attention outputs across model layers using the SimpleQA dataset.

## 🎯 Project Goal

The goal is to save LLM activations when processing questions and answers from the SimpleQA-verified dataset. These activations can later be used to train hallucination detection classifiers or analyze model behavior.

## 🛠️ Setup

1. Clone the repository:
```bash
git clone [github.repo]
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
## 📚 Methodology

This project implements the **One4All** approach, a unified model-agnostic framework for detecting three types of hallucinations in LLMs:

### Types of Hallucinations Detected

1. **Factual Hallucinations**: Inconsistencies between model outputs and established world knowledge
   - Datasets: BeliefBank Facts
   
2. **Logical Hallucinations**: Violations of logical consistency in reasoning
   - Dataset: BeliefBank Constraints
   
3. **Contextual Hallucinations**: Discrepancies between generated text and provided context
   - Dataset: HaluEval

### Activation Extraction

The framework extracts internal representations from different model components:

- **Hidden States**: Final layer representations before the output layer
- **MLP Outputs**: Feed-forward network outputs at each layer
- **Attention Outputs**: Multi-head attention outputs at each layer

These activations are extracted at specific token positions to capture the model's internal state when processing information.


## 🗂️ Project Structure

### Data Directories
- `data/beliefbank/`: Facts, constraints, and templates for factual/logical hallucination detection

### Models
- `models_frozen_head/`: Trained probe models organized by:
  - Dataset: `belief_bank_constraints/`, `belief_bank_facts/`, `halu_eval/`
  - Component: `attn/`, `hidden/`, `mlp/`
  - Architecture: encoder-only, shared-head, adapter-based

### Source Code
- `src/model/HallucinationDetection.py`: Used to store activations
- `src/data/`: Dataset loaders for BeliefBank and HaluEval
- `src/model/predict.py`: Inference utilities

### Notebooks
- `notebooks/HybridApproach/`: MLP adapter experiments
- `notebooks/linearApproach/`: Linear probe experiments
- `notebooks/nonLinearApproach/`: Non-linear probe variants
- `notebooks/layersStudies/`: Layer-wise analysis

## 🚀 Usage

### Extracting Activations

```bash

```

### Training Probes

```bash
# Train encoder probe on BeliefBank facts
python scripts/train_probe.py \
    --dataset belief_bank_facts \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --component hidden \
    --probe_type encoder
```

### Cross-Model Evaluation

```bash
# Evaluate cross-domain transfer
python scripts/generate_cross_domain_table.py \
    --source_model meta-llama/Llama-3.1-8B-Instruct \
    --target_model google/gemma-2-9b-it \
    --dataset belief_bank_constraints
```

## 📊 Results

The project includes comprehensive evaluation across:
- Multiple LLM families (Llama, Gemma, Qwen, Falcon)
- Different activation components (hidden states, MLP, attention)
- Cross-model transfer scenarios
- Layer-wise analysis (see `notebooks/layersStudies/`)

Confusion matrices and performance metrics are stored in `confusion_matrices_frozen_head/`.

## 🔬 Key Findings

Based on the One4All approach:

1. **Model-Agnostic Transfer**: Probes trained on one model can effectively detect hallucinations in other models
2. **Component Importance**: Hidden states and MLP outputs generally provide better hallucination signals than attention
3. **Layer Analysis**: Middle-to-late layers contain the most informative representations for hallucination detection
4. **Unified Detection**: A single probe can detect multiple types of hallucinations across different domains

