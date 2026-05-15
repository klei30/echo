"""
providers/gemma.py — vLLM Gemma-4 2B client with streaming support.
Supports per-user LoRA adapter loading via vLLM's lora_request param.
"""

import json
import time
import uuid
from typing import AsyncGenerator

from openai import AsyncOpenAI
from config import settings

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key="not-needed",
            base_url=settings.gemma4_vllm_base_url,
        )
    return _client


async def is_gemma_available() -> bool:
    try:
        client = get_client()
        models = await client.models.list()
        return any(settings.gemma4_base_model in m.id for m in models.data)
    except Exception:
        return False


async def chat_with_gemma(
    messages: list[dict],
    lora_adapter: str | None = None,
) -> tuple[str, str]:
    """
    Non-streaming Gemma response.
    lora_adapter: user_id string for per-user LoRA via vLLM dynamic loading.
    Returns (response_text, model_used).
    """
    client = get_client()
    extra = {}
    model = settings.gemma4_base_model

    # vLLM supports named LoRA adapters — pass user_id as adapter name
    if lora_adapter:
        extra["extra_body"] = {"lora_request": {"lora_name": lora_adapter}}

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        **extra,
    )

    content = response.choices[0].message.content or ""
    return content, model


async def stream_gemma_response(
    messages: list[dict],
    lora_adapter: str | None = None,
) -> AsyncGenerator[tuple[str, str], None]:
    """
    Real SSE streaming from vLLM Gemma.
    Yields (sse_line, accumulated_text) — same interface as stream_teacher_response.
    """
    client = get_client()
    extra = {}

    if lora_adapter:
        extra["extra_body"] = {"lora_request": {"lora_name": lora_adapter}}

    accumulated = ""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    model = settings.gemma4_base_model

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            stream=True,
            **extra,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            content = (delta.content or "") if delta else ""
            finish = chunk.choices[0].finish_reason if chunk.choices else None

            accumulated += content

            sse_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": content} if content else {},
                    "finish_reason": finish,
                }],
            }
            yield f"data: {json.dumps(sse_chunk)}\n\n", accumulated

    except Exception:
        # vLLM not available / streaming fails — fall back to single chunk
        text, _ = await chat_with_gemma(messages, lora_adapter)
        accumulated = text
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n", accumulated
        done_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done_chunk)}\n\n", accumulated

    yield "data: [DONE]\n\n", accumulated
