#!/bin/bash

cd /lustrehome/fonty/HallucinationDetection

uv sync

source .venv/bin/activate

# rm -rf ~/.cache/pip
# rm -rf ~/.cache/uv
# rm -rf ~/.cache/python
# 


hf download google/gemma-2-9b-it
hf download Qwen/Qwen2.5-7B
hf download meta-llama/Llama-3.1-8B-Instruct
hf download tiiuae/Falcon3-7B-Base
