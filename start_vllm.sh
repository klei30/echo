#!/bin/bash
# Start vLLM with LoRA support in WSL Ubuntu-24.04
# Usage: wsl -d Ubuntu-24.04 bash /mnt/c/Users/ASUS/Desktop/echo/start_vllm.sh

set -e

export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
export VLLM_ATTENTION_BACKEND=FLASHINFER

# Activate venv if it exists
if [ -f ~/vllm-env/bin/activate ]; then
    source ~/vllm-env/bin/activate
fi

# Start vLLM
exec /home/klei/vllm-env/bin/vllm serve Qwen/Qwen2.5-7B-Instruct \
  --enable-lora \
  --max-lora-rank 64 \
  --port 8001 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0
