@echo off
REM Script to run TruthX training on BeliefBank dataset
REM This script creates a balanced subset of 1000 examples (500 correct, 500 incorrect)
REM and trains the TruthX model as described in the paper

echo ==================================================
echo Training TruthX on BeliefBank Dataset
echo ==================================================
echo.

REM Default parameters
set PROJECT_ROOT=.
set CACHE_DIR=activation_cache_truthx
set MODEL_NAME=meta-llama/Meta-Llama-3-8B
set ACTIVATION_TYPE=attn
set TARGET_LAYERS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
set OUTPUT_DIR=truthx_models

REM BeliefBank specific parameters
set NUM_POSITIVE=500
set NUM_NEGATIVE=500

REM Hyperparameters (as per paper)
set SEMANTIC_LATENT_DIM=1024
set TRUTHFUL_LATENT_DIM=1024
set SEMANTIC_HIDDEN_DIMS=2048,1024
set TRUTHFUL_HIDDEN_DIMS=2048,1024
set DECODER_HIDDEN_DIMS=1024,2048

set BATCH_SIZE=32
set LEARNING_RATE=1e-4
set NUM_EPOCHS=50
set CONTRASTIVE_WEIGHT=1.0
set EDITING_WEIGHT=1.0
set TEMPERATURE=0.1

REM Check if --extract-activations flag is provided
set EXTRACT_ACTIVATIONS=0
for %%i in (%*) do (
    if "%%i"=="--extract-activations" set EXTRACT_ACTIVATIONS=1
)

REM Check if --quantization flag is provided
set QUANTIZATION=0
for %%i in (%*) do (
    if "%%i"=="--quantization" set QUANTIZATION=1
)

echo Parameters:
echo   Project Root: %PROJECT_ROOT%
echo   Model Name: %MODEL_NAME%
echo   Activation Type: %ACTIVATION_TYPE%
echo   Target Layers: %TARGET_LAYERS%
echo   Output Dir: %OUTPUT_DIR%
echo.
echo   Num Positive (truthful): %NUM_POSITIVE%
echo   Num Negative (hallucinated): %NUM_NEGATIVE%
echo.
echo   Extract Activations: %EXTRACT_ACTIVATIONS%
echo   Use Quantization: %QUANTIZATION%
echo.

python -m src.truthx.train_truthx_beliefbank ^
    --project_root %PROJECT_ROOT% ^
    --cache_dir %CACHE_DIR% ^
    --model_name %MODEL_NAME% ^
    --activation_type %ACTIVATION_TYPE% ^
    --target_layers %TARGET_LAYERS% ^
    --output_dir %OUTPUT_DIR% ^
    --num_positive %NUM_POSITIVE% ^
    --num_negative %NUM_NEGATIVE% ^
    --semantic_latent_dim %SEMANTIC_LATENT_DIM% ^
    --truthful_latent_dim %TRUTHFUL_LATENT_DIM% ^
    --semantic_hidden_dims %SEMANTIC_HIDDEN_DIMS% ^
    --truthful_hidden_dims %TRUTHFUL_HIDDEN_DIMS% ^
    --decoder_hidden_dims %DECODER_HIDDEN_DIMS% ^
    --batch_size %BATCH_SIZE% ^
    --learning_rate %LEARNING_RATE% ^
    --num_epochs %NUM_EPOCHS% ^
    --contrastive_weight %CONTRASTIVE_WEIGHT% ^
    --editing_weight %EDITING_WEIGHT% ^
    --temperature %TEMPERATURE% ^
    %*

echo.
echo ==================================================
echo Training Complete!
echo ==================================================
