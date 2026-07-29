#!/bin/bash
# Start the experimental Gemma 4 E2B vLLM lane in WSL Ubuntu-24.04.
# Set GEMMA4_MODEL_ID when the model is not in the standard Hugging Face cache.
#
# This runs on port 8003 so the stable Qwen lane on 8001 is untouched.
# If Qwen was started with high GPU memory utilization, stop Qwen first or
# restart both lanes with smaller utilization values.

set -e

export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-1}"
if [ -z "${HF_HOME:-}" ]; then
  WINDOWS_PROFILE="$(cmd.exe /c echo %USERPROFILE% 2>/dev/null | tr -d '\r' || true)"
  WINDOWS_HF_HOME=""
  if [ -n "$WINDOWS_PROFILE" ]; then
    WINDOWS_HF_HOME="$(wslpath -u "$WINDOWS_PROFILE" 2>/dev/null || true)/.cache/huggingface"
  fi
  if [ -n "$WINDOWS_HF_HOME" ] && [ -d "$WINDOWS_HF_HOME" ]; then
    export HF_HOME="$WINDOWS_HF_HOME"
  else
    export HF_HOME="$HOME/.cache/huggingface"
  fi
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
unset VLLM_NUM_GPU_BLOCKS_OVERRIDE

# Gemma 4 support currently needs newer Transformers than the stable Qwen env.
# Keep that override isolated so ~/vllm-env remains usable for Qwen.
TRANSFORMERS_OVERRIDE="${GEMMA4_TRANSFORMERS_PATH:-$HOME/gemma4-transformers}"
if [ -d "$TRANSFORMERS_OVERRIDE" ]; then
  export PYTHONPATH="$TRANSFORMERS_OVERRIDE${PYTHONPATH:+:$PYTHONPATH}"
fi

if [ -n "${GEMMA4_MODEL_ID:-}" ]; then
  MODEL_ID="$GEMMA4_MODEL_ID"
else
  MODEL_ID="$(find "$HF_HOME/hub/models--unsloth--gemma-4-E2B-it/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1)"
  if [ -z "$MODEL_ID" ]; then
    echo "Gemma 4 model not found. Set GEMMA4_MODEL_ID or populate $HF_HOME." >&2
    exit 1
  fi
fi
SERVED_MODEL_NAME="${GEMMA4_SERVED_MODEL_NAME:-gemma4_e2b}"
PORT="${GEMMA4_PORT:-8003}"
MAX_MODEL_LEN="${GEMMA4_MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GEMMA4_GPU_MEMORY_UTILIZATION:-0.55}"
ENABLE_LORA="${GEMMA4_ENABLE_LORA:-1}"

LORA_ARGS=()
if [ "$ENABLE_LORA" = "1" ]; then
  LORA_ARGS=(--enable-lora --max-lora-rank 64)
fi

VLLM_ENV="${GEMMA4_VLLM_ENV:-$HOME/vllm-env}"
if [ -f "$VLLM_ENV/bin/activate" ]; then
    source "$VLLM_ENV/bin/activate"
fi

VLLM_BIN="${GEMMA4_VLLM_BIN:-$(command -v vllm || true)}"
if [ -z "$VLLM_BIN" ]; then
  echo "vLLM executable not found. Activate its environment or set GEMMA4_VLLM_BIN." >&2
  exit 1
fi

exec "$VLLM_BIN" serve "$MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  "${LORA_ARGS[@]}" \
  --trust-remote-code \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --limit-mm-per-prompt '{"image":0,"audio":0}' \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --host 0.0.0.0
