#!/bin/bash

cd /lustrehome/fonty/HallucinationDetection

uv sync

source .venv/bin/activate

# rm -rf ~/.cache/pip
# rm -rf ~/.cache/uv
# rm -rf ~/.cache/python
# 

export HF_TOKEN="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"
hf download google/gemma-2-9b-it
hf download meta-llama/Llama-3.1-8B-Instruct

