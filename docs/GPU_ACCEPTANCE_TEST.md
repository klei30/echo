# GPU Acceptance Test

The CPU test suite verifies dataset isolation, deterministic evaluation,
single-flight coordination, promotion decisions, checkpoint ordering, and
rollback. This acceptance test covers only the hardware boundary that a CPU
laptop cannot prove: Gemma 4 + Unsloth + vLLM LoRA compatibility.

## Preconditions

- A Linux or Kaggle runtime with a supported NVIDIA GPU
- Gemma 4 model files available locally
- Unsloth, PyTorch, Transformers, TRL, and vLLM installed
- `ECHO_TRAINING_RUNTIME=linux_local`
- `GEMMA4_TRAINING_MODEL_PATH` points to the local model
- vLLM exposes the base model on `http://127.0.0.1:8003/v1`

## Bounded smoke test

Start Echo and seed at least seven high-quality examples for the test user
(four training examples plus three frozen holdout examples).
Then call:

```bash
curl -X POST http://127.0.0.1:8002/v1/training/demo-loop \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ECHO_SECRET" \
  -H "X-Echo-User-Id: gpu-acceptance" \
  -d '{
    "lane": "gemma4_e2b",
    "max_pairs": 8,
    "max_steps": 8,
    "min_pairs": 4
  }'
```

The test passes only when all of the following are true:

1. Unsloth produces an adapter containing `adapter_config.json` and adapter weights.
2. vLLM restarts and lists `gemma4_e2b`.
3. The candidate loads through vLLM's LoRA endpoint.
4. At least three held-out generations complete.
5. A failed or skipped evaluation does not create a new checkpoint.
6. A passing evaluation creates exactly one new checkpoint.
7. `/v1/training/runs` records the adapter path, evaluation, and final status.

Keep the returned JSON and the Echo/vLLM logs as the acceptance artifact. The
test is intentionally bounded; it verifies compatibility and lifecycle, not
model quality.
