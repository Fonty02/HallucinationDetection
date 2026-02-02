# TruthX Training on BeliefBank Dataset

Questo script implementa il training del modello TruthX sul dataset BeliefBankFacts, come descritto nel paper "TruthX: Alleviating Hallucinations by Editing Large Language Models in Truthful Space".

## Panoramica

Lo script crea un subset bilanciato di 1000 esempi dal dataset BeliefBankFacts:
- 500 esempi positivi (label "yes" = truthful)
- 500 esempi negativi (label "no" = hallucinated/incorrect)

Il training del modello TruthX utilizza:
- **Reconstruction Loss**: Per preservare le rappresentazioni semantiche
- **Contrastive Loss** (InfoNCE): Per separare lo spazio truthful da quello non truthful
- **Editing Loss**: Per correggere le rappresentazioni non truthful

## Struttura del Progetto

```
src/truthx/
├── train_truthx_beliefbank.py    # Script principale per training su BeliefBank
├── truthx_model.py                # Implementazione del modello MLPAE
├── train_truthx.py                # Script di training originale
└── inference_with_truthx.py       # Script per inference

scripts/
├── train_truthx_beliefbank.bat    # Script di lancio (Windows)
└── train_truthx_beliefbank.sh      # Script di lancio (Linux/Mac)
```

## Utilizzo

### Prerequisiti

- Python 3.11+
- PyTorch con supporto CUDA (raccomandato)
- Un modello LLM pre-addestrato (es. Llama-3-8B)

### Esempio di Esecuzione

**Windows (Batch):**
```batch
REM Prima esecuzione: estrai le attivazioni e addestra
scripts\train_truthx_beliefbank.bat --extract-activations

# Esecuzioni successive: usa le attivazioni già estratte
scripts\train_truthx_beliefbank.bat
```

**Linux/Mac (Shell):**
```bash
# Prima esecuzione: estrai le attivazioni e addestra
./scripts/train_truthx_beliefbank.sh --extract-activations

# Esecuzioni successive: usa le attivazioni già estratte
./scripts/train_truthx_beliefbank.sh
```

**Direttamente con Python:**
```bash
# Prima esecuzione
python -m src.truthx.train_truthx_beliefbank --extract_activations

# Esecuzioni successive
python -m src.truthx.train_truthx_beliefbank
```

### Parametri

**Dataset:**
- `--num_positive`: Numero di esempi positivi (default: 500)
- `--num_negative`: Numero di esempi negativi (default: 500)

**Attivazioni:**
- `--extract_activations`: Flag per estrarre le attivazioni (default: False)
- `--quantization`: Flag per usare quantizzazione 4-bit (default: False)
- `--activation_type`: Tipo di attivazione ("attn", "hidden", "mlp", default: "attn")
- `--target_layers`: Layer target (default: "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")

**Modello:**
- `--model_name`: Nome del modello LLM (default: "meta-llama/Meta-Llama-3-8B")

**Hyperparameters (come da paper):**
- `--semantic_latent_dim`: Dimensione latente spazio semantico (default: 1024)
- `--truthful_latent_dim`: Dimensione latente spazio truthful (default: 1024)
- `--semantic_hidden_dims`: Dimensioni hidden encoder semantico (default: "2048,1024")
- `--truthful_hidden_dims`: Dimensioni hidden encoder truthful (default: "2048,1024")
- `--decoder_hidden_dims`: Dimensioni hidden decoder (default: "1024,2048")

**Training:**
- `--batch_size`: Dimensione batch (default: 32)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--num_epochs`: Numero di epoche (default: 50)
- `--contrastive_weight`: Peso della contrastive loss (default: 1.0)
- `--editing_weight`: Peso della editing loss (default: 1.0)
- `--temperature`: Temperatura per InfoNCE loss (default: 0.1)

**Output:**
- `--output_dir`: Directory di output (default: "truthx_models")

## Output

Lo script genera i seguenti output:

1. **Subset BeliefBank**: `data/beliefbank/beliefbank_subset_1000.jsonl`
   - Contiene 1000 esempi bilanciati (500 yes, 500 no)

2. **Attivazioni**: `activation_cache/{model_name}/belief_bank_subset/`
   - `activation_hidden/`: Attivazioni hidden state per ogni layer
   - `activation_mlp/`: Attivazioni MLP per ogni layer
   - `activation_attn/`: Attivazioni attention per ogni layer
   - `generations/`: Generazioni e labels di halluncinazione

3. **Modello TruthX**: `truthx_models/truthx_beliefbank_model.pt`
   - Checkpoint del modello addestrato
   - Include: state_dict, pos_center, neg_center, rank (layer ranking)

## Pipeline Completa

1. **Creazione Subset**: Lo script carica BeliefBankFacts e crea un subset bilanciato di 1000 esempi

2. **Estrazione Attivazioni**: (opzionale, se `--extract_activations`) Il modello LLM processa ogni esempio e le attivazioni vengono salvate

3. **Caricamento Attivazioni**: Le attivazioni vengono caricate e raggruppate per tipo (hallucinated/not_hallucinated)

4. **Training**:
   - Inizializzazione modello MLPAE
   - Training loop con reconstruction, contrastive, e editing loss
   - Calcolo dei centri truthful per ogni layer
   - Ranking dei layer per separabilità

5. **Salvataggio**: Il modello addestrato e i centri vengono salvati come checkpoint

## Note

- La prima esecuzione con `--extract_activations` richiede molto tempo e spazio disco (~GB)
- Le esecuzioni successive senza `--extract_activations` sono molto più veloci
- L'uso di quantizzazione (`--quantization`) riduce l'uso della memoria ma può influenzare la qualità delle attivazioni
- Il ranking dei layer mostra quali layer hanno la migliore separabilità tra truthful e non truthful

## Riferimenti

- TruthX: Alleviating Hallucinations by Editing Large Language Models in Truthful Space
- BeliefBank: A Benchmark for Fact-Based Belief Revision
