import json
import time
import uuid
from typing import AsyncIterator

from openai import AsyncOpenAI
from config import settings
from teacher_policy import record_teacher_usage, should_use_teacher

_client: AsyncOpenAI | None = None

# Models that support logprobs (needed for GKD training)
LOGPROBS_SUPPORTED = {"gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini", "grok-3", "grok-3-mini"}

# Models that accept custom temperature
TEMPERATURE_SUPPORTED = {"gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "grok-3", "grok-3-mini"}


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.teacher_base_url,
        )
    return _client


async def chat_with_teacher(
    messages: list[dict],
    model: str | None = None,
    user_id: str | None = None,
    purpose: str = "judge_answer",
    confidence: float | None = None,
    importance: str = "normal",
    explicit_user_request: bool = False,
    recent_failure: bool = False,
    require_policy: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[str, list | None, str]:
    """Non-streaming. Returns (response_text, logprobs_or_None, model_used)"""
    if user_id:
        prompt = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        decision = await should_use_teacher(
            user_id,
            purpose,
            confidence=confidence,
            importance=importance,
            prompt=str(prompt),
            recent_failure=recent_failure,
            explicit_user_request=explicit_user_request,
        )
        if not decision.allowed:
            if require_policy:
                return "", None, f"teacher_skipped:{decision.reason}"
        else:
            await record_teacher_usage(
                user_id,
                purpose,
                decision.reason,
                {"model": model or settings.teacher_model, "decision": decision.to_dict()},
            )

    client = get_client()
    used_model = model or settings.teacher_model
    supports_logprobs = used_model in LOGPROBS_SUPPORTED

    kwargs: dict = {"model": used_model, "messages": messages}
    if used_model in TEMPERATURE_SUPPORTED:
        kwargs["temperature"] = 0.7 if temperature is None else temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if supports_logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 5

    response = await client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""
    raw_logprobs = response.choices[0].logprobs if supports_logprobs else None

    serialized = None
    if raw_logprobs and raw_logprobs.content:
        serialized = [
            {
                "token": t.token,
                "logprob": t.logprob,
                "top": [{"token": tl.token, "logprob": tl.logprob} for tl in (t.top_logprobs or [])],
            }
            for t in raw_logprobs.content[:100]
        ]

    return content, serialized, used_model


async def stream_teacher_response(
    messages: list[dict],
    model: str | None = None,
    user_id: str | None = None,
    purpose: str = "tool_chat",
    confidence: float | None = None,
    importance: str = "normal",
    explicit_user_request: bool = False,
    recent_failure: bool = False,
) -> AsyncIterator[tuple[str, str]]:
    """
    Streaming generator. Yields (sse_line, accumulated_text) pairs.
    The final yield has the complete response text.
    """
    client = get_client()
    used_model = model or settings.teacher_model
    if user_id:
        prompt = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        decision = await should_use_teacher(
            user_id,
            purpose,
            confidence=confidence,
            importance=importance,
            prompt=str(prompt),
            recent_failure=recent_failure,
            explicit_user_request=explicit_user_request,
        )
        if not decision.allowed:
            yield "data: [DONE]\n\n", ""
            return
        await record_teacher_usage(
            user_id,
            purpose,
            decision.reason,
            {"model": used_model, "stream": True, "decision": decision.to_dict()},
        )

    kwargs: dict = {"model": used_model, "messages": messages, "stream": True}
    if used_model in TEMPERATURE_SUPPORTED:
        kwargs["temperature"] = 0.7

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    full_text = ""
    first_chunk = True

    stream = await client.chat.completions.create(**kwargs)
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        finish_reason = chunk.choices[0].finish_reason
        content = delta.content or ""
        full_text += content

        delta_payload: dict = {}
        if first_chunk:
            delta_payload["role"] = "assistant"
            first_chunk = False
        if content:
            delta_payload["content"] = content

        sse_data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": used_model,
            "choices": [{"index": 0, "delta": delta_payload, "finish_reason": finish_reason}],
        }
        yield f"data: {json.dumps(sse_data)}\n\n", full_text

    yield "data: [DONE]\n\n", full_text
