#!/bin/bash
# Start the experimental Gemma 4 E2B vLLM lane in WSL Ubuntu-24.04.
# Usage from Windows:
#   wsl -d Ubuntu-24.04 bash /mnt/c/Users/ASUS/Desktop/echo/start_gemma4_e2b_vllm.sh
#
# This runs on port 8003 so the stable Qwen lane on 8001 is untouched.
# If Qwen was started with high GPU memory utilization, stop Qwen first or
# restart both lanes with smaller utilization values.

set -e

export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-1}"
export HF_HOME=/mnt/c/Users/ASUS/.cache/huggingface
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
unset VLLM_NUM_GPU_BLOCKS_OVERRIDE

# Gemma 4 support currently needs newer Transformers than the stable Qwen env.
# Keep that override isolated so ~/vllm-env remains usable for Qwen.
if [ -d /home/klei/gemma4-transformers ]; then
  export PYTHONPATH="/home/klei/gemma4-transformers${PYTHONPATH:+:$PYTHONPATH}"
fi

MODEL_ID="${GEMMA4_MODEL_ID:-/mnt/c/Users/ASUS/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it/snapshots/f0c5915f17ad6c66dbeb577fb06ff8925bf8d7ae}"
SERVED_MODEL_NAME="${GEMMA4_SERVED_MODEL_NAME:-gemma4_e2b}"
PORT="${GEMMA4_PORT:-8003}"
MAX_MODEL_LEN="${GEMMA4_MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GEMMA4_GPU_MEMORY_UTILIZATION:-0.55}"
ENABLE_LORA="${GEMMA4_ENABLE_LORA:-1}"

LORA_ARGS=()
if [ "$ENABLE_LORA" = "1" ]; then
  LORA_ARGS=(--enable-lora --max-lora-rank 64)
fi

if [ -f ~/vllm-env/bin/activate ]; then
    source ~/vllm-env/bin/activate
fi

exec /home/klei/vllm-env/bin/vllm serve "$MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  "${LORA_ARGS[@]}" \
  --trust-remote-code \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --limit-mm-per-prompt '{"image":0,"audio":0}' \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --host 0.0.0.0
