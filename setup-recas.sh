#!/bin/bash

cd /lustrehome/fonty/HallucinationDetection

uv venv --python 3.11.5

source .venv/bin/activate

# rm -rf ~/.cache/pip
# rm -rf ~/.cache/uv
# rm -rf ~/.cache/python
# 
uv pip sync requirements.lock

huggingface-cli login
huggingface-cli download google/gemma-2-9b-it
huggingface-cli download Qwen/Qwen2.5-7B

huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
huggingface-cli download tiiuae/Falcon3-7B-Base
