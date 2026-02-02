#!/bin/bash
# Script to run TruthX training on BeliefBank dataset
# This script creates a balanced subset of 1000 examples (500 correct, 500 incorrect)
# and trains the TruthX model as described in the paper

echo "=================================================="
echo "Training TruthX on BeliefBank Dataset"
echo "=================================================="
echo

# Default parameters
PROJECT_ROOT="."
CACHE_DIR="activation_cache_truthx"
MODEL_NAME="meta-llama/Meta-Llama-3-8B"
ACTIVATION_TYPE="attn"
TARGET_LAYERS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
OUTPUT_DIR="truthx_models"

# BeliefBank specific parameters
NUM_POSITIVE=500
NUM_NEGATIVE=500

# Hyperparameters (as per paper)
SEMANTIC_LATENT_DIM=1024
TRUTHFUL_LATENT_DIM=1024
SEMANTIC_HIDDEN_DIMS="2048,1024"
TRUTHFUL_HIDDEN_DIMS="2048,1024"
DECODER_HIDDEN_DIMS="1024,2048"

BATCH_SIZE=32
LEARNING_RATE=1e-4
NUM_EPOCHS=50
CONTRASTIVE_WEIGHT=1.0
EDITING_WEIGHT=1.0
TEMPERATURE=0.1

# Parse optional flags
EXTRACT_ACTIVATIONS=false
QUANTIZATION=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --extract-activations) EXTRACT_ACTIVATIONS=true ;;
        --quantization) QUANTIZATION=true ;;
        --project_root) PROJECT_ROOT="$2"; shift ;;
        --model_name) MODEL_NAME="$2"; shift ;;
        --num_epochs) NUM_EPOCHS="$2"; shift ;;
        *) shift ;;
    esac
    shift
done

echo "Parameters:"
echo "  Project Root: $PROJECT_ROOT"
echo "  Model Name: $MODEL_NAME"
echo "  Activation Type: $ACTIVATION_TYPE"
echo "  Target Layers: $TARGET_LAYERS"
echo "  Output Dir: $OUTPUT_DIR"
echo
echo "  Num Positive (truthful): $NUM_POSITIVE"
echo "  Num Negative (hallucinated): $NUM_NEGATIVE"
echo
echo "  Extract Activations: $EXTRACT_ACTIVATIONS"
echo "  Use Quantization: $QUANTIZATION"
echo

# Build command
CMD="python -m src.truthx.train_truthx_beliefbank \
    --project_root $PROJECT_ROOT \
    --cache_dir $CACHE_DIR \
    --model_name $MODEL_NAME \
    --activation_type $ACTIVATION_TYPE \
    --target_layers $TARGET_LAYERS \
    --output_dir $OUTPUT_DIR \
    --num_positive $NUM_POSITIVE \
    --num_negative $NUM_NEGATIVE \
    --semantic_latent_dim $SEMANTIC_LATENT_DIM \
    --truthful_latent_dim $TRUTHFUL_LATENT_DIM \
    --semantic_hidden_dims $SEMANTIC_HIDDEN_DIMS \
    --truthful_hidden_dims $TRUTHFUL_HIDDEN_DIMS \
    --decoder_hidden_dims $DECODER_HIDDEN_DIMS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --contrastive_weight $CONTRASTIVE_WEIGHT \
    --editing_weight $EDITING_WEIGHT \
    --temperature $TEMPERATURE"

# Add optional flags
if [ "$EXTRACT_ACTIVATIONS" = true ]; then
    CMD="$CMD --extract_activations"
fi

if [ "$QUANTIZATION" = true ]; then
    CMD="$CMD --quantization"
fi

# Run the command
eval $CMD

echo
echo "=================================================="
echo "Training Complete!"
echo "=================================================="
