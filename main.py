import asyncio
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime as _dt, timezone as _timezone
from typing import Optional

import httpx
import uvicorn
import bcrypt
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from auth.jwt_utils import create_token, verify_token
from db.database import init_tables, get_conn
from memory.mem0_client import warmup as mem0_warmup, add_memories, add_raw_memory
from memory.context import get_active_rules, get_active_skills, get_context
from router.confidence import get_confidence, update_confidence, detect_topic
from router.compass import record_topic
from training.collector import save_pair, count_untrained_pairs, mark_chat_feedback_pair
from training.adapter import adapter_exists, adapter_status, ensure_adapter_loaded, lora_name_for_user
from training.runtime import start_gemma_vllm_after_training, stop_gemma_vllm_for_training, tcp_ok, vllm_models_health
from loop_events import loop_snapshot, record_event, record_life_event, record_outcome
from loop_priority import get_today_priority
from proactive_engine import (
    acknowledge_intervention,
    get_daily_mission,
    get_growth_timeline,
    get_intervention_settings,
    get_latest_clone_mission,
    get_or_create_next_intervention,
    get_reality_check,
    get_revelation_status,
    update_intervention_settings,
)
from thesis import get_current_thesis, refresh_current_thesis
from training.summary import get_training_summary
from training.state import (
    finish_training_run,
    latest_training_run,
    mark_interrupted_training_runs,
    try_create_training_run,
)
from models.schemas import ContextRequest, ContextResponse, SaveRequest
from teacher_policy import infer_importance, record_teacher_usage, should_use_teacher

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("echo")

ECHO_PRODUCT_SYSTEM_PROMPT = (
    "You are Echo, a local-first opportunity engine built on one belief: talent is common, "
    "but discovery is uneven. Many people only find what they are good at by accident; Echo "
    "turns scattered moments into direction, practice, proof, outcomes, and eval-gated adapter updates.\n"
    "Echo's product loop is: Signal -> Pattern Map -> Next Proof Step -> "
    "Outcome -> Proof Card -> Direction.\n"
    "Behave less like a generic chatbot and more like a precise private operating layer. Be "
    "direct, calm, evidence-based, and specific. Help the user leave with one concrete next "
    "step, one better decision, or one piece of proof they can build.\n"
    "Use memory and loop context as evidence. Never invent personal facts. If context is missing, "
    "say so simply and ask at most one useful question.\n"
    "When advice matters, end with a small action and what outcome the user should log later.\n"
    "Avoid public-facing clone, shadow, battle, tournament, lab, or anime-style language. "
    "Use product language: Pattern Map, Today, Proof Card, Decision Room, Practice, Outcome, "
    "Direction, Runtime, Home Brain, Home Brain Adapter, This Device.\n"
    "Stage awareness: if the user has fewer than 8 stored moments (early stage), listen more and "
    "advise less — build the picture before claiming to know them, ask at most one clarifying "
    "question per turn. If the user has an active thesis and practice rep, reference both when "
    "the moment fits. If training is ready, briefly mention it before closing the conversation."
)

# Shared voice anchor used by all feature prompts — keeps identity coherent across endpoints.
ECHO_FEATURE_HEADER = (
    "You are Echo - a local-first AI system building a longitudinal picture of one person "
    "so hidden talent becomes direction, proof, and opportunity.\n"
)


def _meaningful_chat_message(value: str | None) -> bool:
    text = " ".join((value or "").strip().lower().split())
    if len(text) < 28:
        return False
    blocked = (
        "generate a concise title",
        "generate a title",
        "conversation content:",
        "please return the result",
        "continue",
    )
    return not any(marker in text for marker in blocked)


def _loop_context_injection(
    thesis: dict,
    priority: dict,
    practice: dict | None = None,
    training: dict | None = None,
) -> str:
    thesis_title = thesis.get("title") or "Still Forming"
    thesis_statement = thesis.get("statement") or "Echo is still forming a read."
    confidence = thesis.get("confidence_label") or "early"
    priority_title = priority.get("title") or "No priority yet"
    priority_body = priority.get("body") or ""
    action = priority.get("action") if isinstance(priority.get("action"), dict) else {}
    action_label = action.get("label") or "Keep talking"
    practice = practice or {}
    training = training or {}
    practice_title = practice.get("rep_title") or "No practice rep set yet"
    practice_instruction = practice.get("rep_instruction") or ""
    ready_for_training = bool(training.get("ready_for_training"))
    untrained = int(training.get("untrained_pairs") or 0)
    required = int(training.get("required_pairs") or settings.min_pairs_for_training)
    dpo_ready = int(training.get("dpo_ready_pairs") or 0)
    dpo_required = int(training.get("dpo_required_pairs") or 4)
    return (
        ECHO_PRODUCT_SYSTEM_PROMPT
        + "\n\n## Current Echo Loop\n"
        f"- Pattern map / direction: {thesis_title} ({confidence}). {thesis_statement}\n"
        f"- Next proof step: {priority_title}. {priority_body}\n"
        f"- Best next action: {action_label}.\n"
        f"- Today's practice / proof rep: {practice_title}. {practice_instruction}\n"
        f"- Home Brain Adapter: {'ready' if ready_for_training else 'not ready'} "
        f"({untrained}/{required} new moments, {dpo_ready}/{dpo_required} preference pairs).\n"
        "\n## How to use this context\n"
        f"- If the user's question touches anything related to '{thesis_title}': connect it to their current direction.\n"
        f"- If the practice rep is set and the user describes a relevant action: mention logging the outcome.\n"
        f"- If training is ready and the conversation is winding down: briefly note that a Home Brain Adapter update is available.\n"
        f"- If an action label is set ('{action_label}'): surface it once as a concrete next step if the conversation warrants it.\n"
        "- Do not force the loop — let the user's question lead. One nudge per turn is enough."
    )


def _has_tool_prompt(system_prompt: str | None) -> bool:
    return bool(
        system_prompt
        and (
            "<tool_definitions>" in system_prompt
            or "<tool_usage_instructions>" in system_prompt
            or "<function name=" in system_prompt
        )
    )


def _looks_like_tool_request(text: str | None) -> bool:
    lower = " ".join((text or "").lower().split())
    if not lower:
        return False
    service_markers = (
        "gmail",
        "email",
        "mail",
        "slack",
        "calendar",
        "google drive",
        "drive",
        "notion",
        "github",
        "jira",
        "discord",
        "telegram",
        "whatsapp",
        "sheets",
        "docs",
    )
    action_markers = (
        "connect",
        "authenticate",
        "auth",
        "authorize",
        "read",
        "check",
        "send",
        "reply",
        "create",
        "schedule",
        "search",
        "find",
        "list",
        "open",
        "update",
        "delete",
        "upload",
        "download",
    )
    return any(service in lower for service in service_markers) and any(action in lower for action in action_markers)


def _tool_safe_context(system_injection: str) -> str:
    return system_injection.replace(
        "You have NO tools and NO external functions. Do NOT output <function>, <tool_call>, or any XML/function syntax — ever.",
        "You may use the tools explicitly provided by the client system prompt. Do not invent tools.",
    )


def _requested_model_lane(request: Request, body: dict) -> str:
    lane = (
        request.headers.get("x-echo-model-lane")
        or body.get("echo_model_lane")
        or body.get("model_lane")
        or ""
    )
    lane = str(lane).strip().lower()
    if lane in {"gemma", "gemma4", "gemma4_e2b", "gemma-4-e2b"}:
        return "gemma4_e2b"
    return "auto"


def _training_lane(value: str | None) -> str:
    return "gemma4_e2b"


def _training_status_key(user_id: str, lane: str = "gemma4_e2b") -> str:
    return f"{user_id}:{lane}"


def _local_target_for_lane(lane: str, user_id: str, use_adapter: bool) -> tuple[str, str, str]:
    """Return (base_url, model, model_used) for the selected local lane."""
    target = lora_name_for_user(user_id, lane="gemma4_e2b") if use_adapter else settings.gemma4_base_model
    return settings.gemma4_vllm_base_url, target, "gemma4_e2b"


async def _call_gemma_feature(
    user_id: str,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.65,
    max_tokens: int = 400,
) -> str:
    use_adapter = adapter_exists(user_id, lane="gemma4_e2b")
    if use_adapter:
        use_adapter = await ensure_adapter_loaded(user_id, lane="gemma4_e2b")
    base_url, model, _ = _local_target_for_lane("gemma4_e2b", user_id, use_adapter=use_adapter)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": ["<function", "<tool_call>", "<|tool_call|>", "✿FUNCTION✿"],
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"].get("content", "").strip()


async def _call_feature_model(
    user_id: str,
    purpose: str,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.65,
    max_tokens: int = 400,
    importance: str = "normal",
) -> tuple[str, str]:
    try:
        text = await _call_gemma_feature(
            user_id,
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if text:
            return text, "gemma4_e2b"
    except Exception as e:
        log.warning("Gemma feature call failed user=%s purpose=%s: %s", user_id, purpose, e)

    decision = await should_use_teacher(
        user_id,
        purpose,
        importance=importance,
        prompt=prompt,
        recent_failure=True,
    )
    if not decision.allowed:
        return "", f"teacher_skipped:{decision.reason}"
    from providers.teacher import chat_with_teacher
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    text, _, model = await chat_with_teacher(
        messages,
        user_id=user_id,
        purpose=purpose,
        importance=importance,
        recent_failure=True,
        explicit_user_request=False,
        require_policy=True,
    )
    return text, model


async def _context_with_loop(user_id: str, message: str) -> dict:
    ctx = await get_context(user_id, message)
    try:
        thesis, priority, training, practice = await asyncio.gather(
            get_current_thesis(user_id),
            get_today_priority(user_id),
            get_training_summary(user_id, lane="gemma4_e2b"),
            _cached_practice_today(user_id),
        )
        ctx["system_injection"] = (
            ctx["system_injection"]
            + "\n\n"
            + _loop_context_injection(thesis, priority, practice, training)
        )
        ctx["loop_state"] = {
            "thesis": thesis,
            "today_priority": priority,
            "training_summary": training,
            "practice": practice,
        }
    except Exception as e:
        log.warning("loop context failed for user=%s: %s", user_id, e)
        ctx["loop_state"] = {}
    return ctx


async def _cached_practice_today(user_id: str) -> dict | None:
    today = _dt.utcnow().strftime("%Y-%m-%d")
    async with get_conn() as db:
        async with db.execute(
            "SELECT * FROM practice_reps WHERE user_id=? AND date=?",
            (user_id, today),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        rep = dict(row)
        async with db.execute(
            "SELECT done FROM practice_log WHERE user_id=? AND rep_id=?",
            (user_id, rep["id"]),
        ) as cur:
            log_row = await cur.fetchone()
        return {
            "rep_id": rep["id"],
            "observation": rep["observation"],
            "rep_title": rep["rep_title"],
            "rep_instruction": rep["rep_instruction"],
            "arc_label": rep.get("arc_label"),
            "logged": log_row is not None,
            "done": bool(log_row["done"]) if log_row else None,
        }


async def _loop_delta_for_turn(user_id: str, event_id: str, topic: str, model_used: str) -> dict:
    thesis, priority, snapshot = await asyncio.gather(
        get_current_thesis(user_id),
        get_today_priority(user_id),
        loop_snapshot(user_id),
    )
    return {
        "event_id": event_id,
        "topic": topic,
        "model_used": model_used,
        "thesis": thesis,
        "today_priority": priority,
        "snapshot": snapshot,
        "suggested_actions": [
            {
                "type": "run_tournament",
                "label": "Choose best path",
                "payload": (priority.get("action") or {}).get("payload", {}),
            },
            {"type": "outcome", "label": "Log outcome", "outcome": "helped", "score": 1.0},
            {"type": "outcome", "label": "Not true", "outcome": "not_true", "score": -0.7},
            {"type": "outcome", "label": "Add to proof", "outcome": "saved_signal", "score": 1.2},
        ],
    }


async def _loop_delta_for_save(
    user_id: str,
    event_id: str,
    *,
    topic: str,
    model_used: str,
    thesis_updated: bool = False,
    proof_created: bool = False,
    opportunity_unlocked: bool = False,
    training_signal_saved: bool = False,
    next_action: str = "Add evidence when you have it.",
    receipt_title: str = "Echo updated",
    receipt_detail: str = "This moment was saved to your loop.",
) -> dict:
    base = await _loop_delta_for_turn(user_id, event_id, topic, model_used)
    base.update(
        {
            "thesis_updated": thesis_updated,
            "proof_created": proof_created,
            "opportunity_unlocked": opportunity_unlocked,
            "training_signal_saved": training_signal_saved,
            "next_action": next_action,
            "receipt_title": receipt_title,
            "receipt_detail": receipt_detail,
        }
    )
    return base


async def _auto_load_adapters():
    """On startup, reload all trained adapters into vLLM (lost on every vLLM restart)."""
    await asyncio.sleep(15)  # wait for vLLM to finish booting
    from pathlib import Path
    from training.adapter import adapter_user_from_dirname, hot_swap_adapter
    adapters_dir = Path(settings.adapters_dir)
    candidates: list[tuple[str, str]] = []
    try:
        async with get_conn() as db:
            async with db.execute(
                """
                SELECT c.user_id, c.path
                FROM checkpoints c
                JOIN (
                    SELECT user_id, lane, MAX(created_at) AS created_at
                    FROM checkpoints
                    WHERE lane='gemma4_e2b'
                    GROUP BY user_id, lane
                ) latest
                  ON c.user_id=latest.user_id
                 AND c.lane=latest.lane
                 AND c.created_at=latest.created_at
                """
            ) as cur:
                rows = await cur.fetchall()
        candidates.extend((r["user_id"], r["path"]) for r in rows)
    except Exception as e:
        log.warning("Could not read adapter checkpoints on startup: %s", e)

    if adapters_dir.exists():
        for p in sorted(adapters_dir.iterdir()):
            if not p.is_dir():
                continue
            user_id = adapter_user_from_dirname(p.name)
            if user_id:
                candidates.append((user_id, str(p)))

    if not candidates:
        return

    seen: set[tuple[str, str]] = set()
    loaded = 0
    for user_id, path in candidates:
        key = (user_id, path)
        if key in seen:
            continue
        seen.add(key)
        if not Path(path).exists():
            continue
        ok = await hot_swap_adapter(user_id, path, record_checkpoint=False, lane="gemma4_e2b")
        if ok:
            loaded += 1
            log.info("Auto-loaded adapter for user=%s path=%s", user_id, path)
    log.info("Auto-loaded %d adapter(s) on startup", loaded)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.jwt_secret == "echo-jwt-secret-change-in-production":
        log.warning("JWT secret is the default dev value — change it in .env before exposing to a network")
    await init_tables()
    await mark_interrupted_training_runs()
    try:
        await asyncio.wait_for(mem0_warmup(), timeout=8)
    except asyncio.TimeoutError:
        log.warning("mem0 warmup timed out; continuing startup")
    from scheduler import start_scheduler, _extract_skills, _seed_proof_from_history
    start_scheduler()
    asyncio.create_task(_auto_load_adapters())
    asyncio.create_task(_startup_skill_and_proof_seed(_extract_skills, _seed_proof_from_history))
    log.info("Echo sidecar ready on port %d", settings.port)
    yield


async def _startup_skill_and_proof_seed(extract_skills_fn, seed_proof_fn) -> None:
    """On startup: if any user has training pairs but no skills or proof, seed them."""
    await asyncio.sleep(10)  # let mem0 and vLLM settle first
    try:
        async with get_conn() as db:
            async with db.execute(
                "SELECT DISTINCT user_id FROM training_pairs"
            ) as cur:
                rows = await cur.fetchall()
        for row in rows:
            uid = row["user_id"]
            try:
                async with get_conn() as db:
                    async with db.execute(
                        "SELECT COUNT(*) as cnt FROM user_skills WHERE user_id=? AND active=1", (uid,)
                    ) as cur:
                        skill_cnt = (await cur.fetchone())["cnt"]
                if skill_cnt == 0:
                    log.info("Startup: no skills for user=%s — running extraction", uid)
                    await extract_skills_fn(uid)
            except Exception as e:
                log.warning("Startup skill extraction failed user=%s: %s", uid, e)
            try:
                await seed_proof_fn(uid)
            except Exception as e:
                log.warning("Startup proof seed failed user=%s: %s", uid, e)
    except Exception as e:
        log.warning("Startup skill/proof seed error: %s", e)


app = FastAPI(title="Echo Sidecar", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Public paths — no token required
_PUBLIC = {
    "/health",
    "/auth/register",
    "/auth/login",
    "/v1/models",
    "/v1/demo/seed",
    "/docs",
    "/openapi.json",
    "/redoc",
}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Let CORS preflight through — browser sends OPTIONS with no auth header
    if request.method == "OPTIONS":
        return await call_next(request)
    if request.url.path in _PUBLIC:
        return await call_next(request)
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        # Valid JWT — extract user_id from it
        user_id = verify_token(token)
        if user_id:
            request.state.user_id = user_id
            return await call_next(request)
        # Static app API key — identity comes from x-echo-user-id header or "default"
        if token == settings.echo_secret:
            uid = request.headers.get("x-echo-user-id") or "default"
            request.state.user_id = uid
            return await call_next(request)
    # Unauthenticated local dev via x-echo-user-id header only
    legacy_uid = request.headers.get("x-echo-user-id")
    if legacy_uid and settings.echo_secret == "echo-local-secret":
        request.state.user_id = legacy_uid
        return await call_next(request)
    return JSONResponse({"error": "Unauthorized"}, status_code=401)


# Register auth routes
from auth.router import router as auth_router
app.include_router(auth_router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "shadow", "object": "model", "owned_by": "echo"}],
    }


def _create_livekit_token(room_name: str, identity: str) -> str:
    """Generate a LiveKit JWT token using the configured API key/secret."""
    from jose import jwt as jose_jwt
    now = int(time.time())
    claims = {
        "iss": settings.livekit_api_key,
        "nbf": now,
        "exp": now + 3600,
        "sub": identity,
        "video": {"roomJoin": True, "room": room_name, "canPublish": True, "canSubscribe": True},
    }
    return jose_jwt.encode(claims, settings.livekit_api_secret, algorithm="HS256")


@app.post("/v1/voice/token")
async def voice_token(request: Request):
    """Return a LiveKit JWT token + room name for the authenticated user."""
    user_id = getattr(request.state, "user_id", None) or request.headers.get("x-echo-user-id") or "default"
    room_name = f"voice-{user_id}"
    token = _create_livekit_token(room_name, user_id)
    return {"token": token, "room": room_name, "url": settings.livekit_url}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, background_tasks: BackgroundTasks):
    """
    OpenAI-compatible endpoint. ChatMCP points here as a custom provider.
    1. Enriches messages with memory context
    2. Routes to vLLM (personal model) or OpenAI (teacher)
    3. Saves the pair for training
    User-id comes from X-Echo-User-Id header, defaults to 'default'.
    """
    body = await request.json()
    user_id = getattr(request.state, "user_id", None) or request.headers.get("x-echo-user-id") or body.get("user") or "default"
    messages: list[dict] = body.get("messages", [])
    stream: bool = body.get("stream", False)
    model_lane = _requested_model_lane(request, body)

    # Extract last user message — content can be a string or an OpenAI array
    raw = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    if isinstance(raw, list):
        user_msg = " ".join(p.get("text", "") for p in raw if isinstance(p, dict) and p.get("type") == "text")
    else:
        user_msg = raw or ""

    existing_system = next((m["content"] for m in messages if m.get("role") == "system"), None)
    has_tool_prompt = _has_tool_prompt(existing_system)
    use_tool_route = has_tool_prompt and _looks_like_tool_request(user_msg)

    confidence = await get_confidence(user_id, user_msg)
    # Gemma is the default local lane. Use the personal LoRA when it is valid and loaded;
    # otherwise use base Gemma so the user does not have to select it manually.
    explicit_gemma = model_lane == "gemma4_e2b"
    auto_gemma = model_lane == "auto"
    wants_gemma = settings.gemma4_enabled and not use_tool_route and (explicit_gemma or auto_gemma)
    teacher_decision = None
    use_gemma_adapter = False
    if wants_gemma:
        vllm_health = await vllm_models_health(timeout=2.0)
        if not vllm_health.get("ok"):
            log.warning(
                "Gemma lane requested but vLLM is unavailable; falling back to teacher user=%s lane=%s reason=%s",
                user_id,
                model_lane,
                vllm_health.get("error") or vllm_health.get("status_code") or "unknown",
            )
            wants_gemma = False
    if wants_gemma and adapter_exists(user_id, lane="gemma4_e2b"):
        use_gemma_adapter = await ensure_adapter_loaded(user_id, lane="gemma4_e2b")
        if not use_gemma_adapter:
            log.warning("Gemma adapter exists but is not loadable; user=%s using base Gemma", user_id)
    use_local = wants_gemma
    if not use_local and use_tool_route:
        teacher_decision = await should_use_teacher(
            user_id,
            "tool_chat",
            confidence=confidence,
            importance=infer_importance(user_msg),
            prompt=user_msg,
            explicit_user_request=True,
        )
        if not teacher_decision.allowed:
            log.info("Teacher tool route skipped user=%s reason=%s", user_id, teacher_decision.reason)
            return JSONResponse(
                {
                    "error": {
                        "message": "Echo is Gemma-first and skipped the teacher tool route for this turn.",
                        "type": "teacher_policy_skipped",
                        "teacher_policy": teacher_decision.to_dict(),
                    }
                },
                status_code=429,
            )
    model_used = "gemma4_e2b" if use_local else "openai"

    ctx = await _context_with_loop(user_id, user_msg)
    context_injection = _tool_safe_context(ctx["system_injection"]) if use_tool_route else ctx["system_injection"]
    if existing_system and (not has_tool_prompt or use_tool_route):
        # Put caller's persona first, Echo's memory injection last — model follows the final instruction
        if use_tool_route:
            combined = context_injection + "\n\n" + existing_system
        else:
            combined = existing_system + "\n\n" + context_injection
        enriched = _inject_system(messages, combined)
    else:
        enriched = _inject_system(messages, context_injection)

    log.info(
        "/v1/chat/completions user=%s confidence=%.2f route=%s lane=%s",
        user_id, confidence, model_used, model_lane,
    )

    if use_local:
        base_url, target_model, local_model_used = _local_target_for_lane(
            "gemma4_e2b",
            user_id,
            use_adapter=use_gemma_adapter,
        )
        model_used = local_model_used
        target_url = f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
    else:
        target_url = f"{settings.teacher_base_url}/chat/completions"
        target_model = settings.teacher_model
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.llm_api_key}",
        }
        if teacher_decision:
            await record_teacher_usage(
                user_id,
                "tool_chat",
                teacher_decision.reason,
                {"model": target_model, "confidence": confidence, "decision": teacher_decision.to_dict()},
            )

    payload = {**body, "messages": enriched, "model": target_model}
    # Strip MCP tool definitions — Echo handles memory/context itself; tool calls break streaming
    if not use_tool_route:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)
    # Stop tokens only for local models (OpenAI limits to 4, and doesn't need these Qwen tokens)
    if use_local and "stop" not in payload and not use_tool_route:
        payload["stop"] = ["<function", "<tool_call>", "<|tool_call|>", "✿FUNCTION✿"]
    # Strip unknown fields so OpenAI doesn't 400 on custom Echo fields
    _OPENAI_FIELDS = {
        "model", "messages", "stream", "temperature", "top_p", "max_tokens",
        "frequency_penalty", "presence_penalty", "stop", "n", "user",
        "tools", "tool_choice", "seed", "logprobs", "top_logprobs", "stream_options",
    }
    if not use_local:
        payload = {k: v for k, v in payload.items() if k in _OPENAI_FIELDS}

    if stream:
        return StreamingResponse(
            _stream_and_save(
                target_url,
                headers,
                payload,
                user_id,
                user_msg,
                model_used,
                allow_tools=use_tool_route,
                fallback_messages=enriched if use_local else None,
            ),
            media_type="text/event-stream",
        )

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(target_url, headers=headers, json=payload)
            if resp.status_code >= 400:
                log.error(
                    "chat_completions upstream error status=%s url=%s model=%s body=%s",
                    resp.status_code, target_url, target_model, resp.text[:1000],
                )
                return JSONResponse(
                    {
                        "error": {
                            "message": f"upstream error {resp.status_code}: {resp.text[:500]}",
                            "type": "upstream_error",
                        }
                    },
                    status_code=502,
                )
            data = resp.json()
    except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as e:
        log.error("chat_completions connection error url=%s: %s", target_url, e)
        if use_local and settings.llm_api_key:
            try:
                from providers.teacher import chat_with_teacher
                fallback_text, _, fallback_model = await chat_with_teacher(
                    enriched,
                    model=settings.teacher_model,
                    user_id=None,
                    purpose="chat_fallback",
                    recent_failure=True,
                    explicit_user_request=True,
                )
                if fallback_text:
                    background_tasks.add_task(_do_save_raw, user_id, user_msg, fallback_text, f"{fallback_model}:fallback")
                    return {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": fallback_model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": fallback_text},
                                "finish_reason": "stop",
                            }
                        ],
                    }
            except Exception as fallback_e:
                log.error("teacher fallback failed after local connection error: %s", fallback_e)
        return JSONResponse(
            {"error": {"message": "Echo model unreachable — is vLLM running?", "type": "connection_error"}},
            status_code=503,
        )
    except (httpx.TimeoutException, httpx.ReadTimeout) as e:
        log.error("chat_completions timeout url=%s: %s", target_url, e)
        if use_local and settings.llm_api_key:
            try:
                from providers.teacher import chat_with_teacher
                fallback_text, _, fallback_model = await chat_with_teacher(
                    enriched,
                    model=settings.teacher_model,
                    user_id=None,
                    purpose="chat_fallback",
                    recent_failure=True,
                    explicit_user_request=True,
                )
                if fallback_text:
                    background_tasks.add_task(_do_save_raw, user_id, user_msg, fallback_text, f"{fallback_model}:fallback")
                    return {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": fallback_model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": fallback_text},
                                "finish_reason": "stop",
                            }
                        ],
                    }
            except Exception as fallback_e:
                log.error("teacher fallback failed after local timeout: %s", fallback_e)
        return JSONResponse(
            {"error": {"message": "Echo model timed out — vLLM may be overloaded.", "type": "timeout_error"}},
            status_code=504,
        )

    assistant_msg = data["choices"][0]["message"]["content"] or ""
    background_tasks.add_task(_do_save_raw, user_id, user_msg, assistant_msg, model_used)
    return data


def _strip_tool_tags(text: str) -> str:
    """Remove any Qwen/Claude-style tool call blocks from a streamed chunk."""
    import re
    # Qwen format: <function name="...">...</function> and <function_response>...</function_response>
    text = re.sub(r"<function[^>]*>.*?</function[^>]*>", "", text, flags=re.DOTALL)
    # Generic tool_call blocks
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    text = re.sub(r"<functioncall>.*?</functioncall>", "", text, flags=re.DOTALL)
    # Truncate at any opening function/tool tag that wasn't closed yet
    for tag in ("<function", "<tool_call>", "<|tool_call|>", "✿FUNCTION✿"):
        idx = text.find(tag)
        if idx != -1:
            text = text[:idx]
    return text


async def _stream_teacher_fallback(
    messages: list[dict],
    user_id: str,
    user_msg: str,
    reason: str,
):
    if not settings.llm_api_key:
        msg = f"\n\n[Echo: local model failed and no cloud fallback key is configured - {reason}]"
        yield f"data: {json.dumps({'choices': [{'delta': {'content': msg}, 'finish_reason': None}]})}\n\n"
        yield "data: [DONE]\n\n"
        return

    from providers.teacher import stream_teacher_response

    full_text = ""
    try:
        async for sse_line, accumulated in stream_teacher_response(
            messages,
            model=settings.teacher_model,
            user_id=None,
            purpose="chat_fallback",
            recent_failure=True,
            explicit_user_request=True,
        ):
            if accumulated:
                full_text = accumulated
            yield sse_line
    except Exception as e:
        log.error("teacher stream fallback failed: %s", e)
        msg = f"\n\n[Echo: local model failed and cloud fallback also failed - {type(e).__name__}: {e or 'no details'}]"
        yield f"data: {json.dumps({'choices': [{'delta': {'content': msg}, 'finish_reason': None}]})}\n\n"
        yield "data: [DONE]\n\n"
        return

    if full_text:
        asyncio.create_task(_do_save_raw(user_id, user_msg, full_text, f"{settings.teacher_model}:fallback"))


async def _stream_and_save(
    url: str,
    headers: dict,
    payload: dict,
    user_id: str,
    user_msg: str,
    model_used: str,
    allow_tools: bool = False,
    fallback_messages: list[dict] | None = None,
):
    collected = []
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    try:
                        err = json.loads(body).get("error", {}).get("message", body.decode())
                    except Exception:
                        err = body.decode()
                    log.error("_stream_and_save upstream error %s url=%s: %s", resp.status_code, url, err)
                    err_chunk = {"choices": [{"delta": {"content": f"\n\n[Echo: upstream error {resp.status_code} — {err}]"}, "finish_reason": None}]}
                    yield f"data: {json.dumps(err_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content") or ""
                            if content:
                                clean = content if allow_tools else _strip_tool_tags(content)
                                delta["content"] = clean
                                chunk["choices"][0]["delta"] = delta
                                collected.append(clean)
                                yield f"data: {json.dumps(chunk)}\n\n"
                            else:
                                yield f"{line}\n\n"
                        except Exception:
                            yield f"{line}\n\n"
                    else:
                        yield f"{line}\n\n"
    except httpx.ConnectError:
        log.error("_stream_and_save connect error url=%s", url)
        if fallback_messages is not None:
            async for item in _stream_teacher_fallback(fallback_messages, user_id, user_msg, "local model connection failed"):
                yield item
            return
        msg = f"\n\n[Echo: cannot reach model at {url} — is vLLM running?]"
        yield f"data: {json.dumps({'choices': [{'delta': {'content': msg}, 'finish_reason': None}]})}\n\n"
        yield "data: [DONE]\n\n"
        return
    except (httpx.TimeoutException, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as e:
        log.error("_stream_and_save timeout url=%s: %s", url, type(e).__name__)
        if fallback_messages is not None:
            async for item in _stream_teacher_fallback(fallback_messages, user_id, user_msg, "local model timed out"):
                yield item
            return
        msg = f"\n\n[Echo: model timed out at {url} — vLLM may be overloaded or not running]"
        yield f"data: {json.dumps({'choices': [{'delta': {'content': msg}, 'finish_reason': None}]})}\n\n"
        yield "data: [DONE]\n\n"
        return
    except httpx.RemoteProtocolError as e:
        log.error("_stream_and_save protocol error url=%s: %s", url, e)
        if fallback_messages is not None:
            async for item in _stream_teacher_fallback(fallback_messages, user_id, user_msg, "local model disconnected"):
                yield item
            return
        msg = f"\n\n[Echo: model disconnected unexpectedly — {e or 'server closed the connection'}]"
        yield f"data: {json.dumps({'choices': [{'delta': {'content': msg}, 'finish_reason': None}]})}\n\n"
        yield "data: [DONE]\n\n"
        return
    except httpx.ReadError as e:
        log.error("_stream_and_save read error url=%s: %s", url, e)
        if fallback_messages is not None:
            async for item in _stream_teacher_fallback(fallback_messages, user_id, user_msg, "local model read failed"):
                yield item
            return
        msg = "\n\n[Echo: connection dropped while reading the response — vLLM may have restarted or run out of memory. Try again in a moment.]"
        yield f"data: {json.dumps({'choices': [{'delta': {'content': msg}, 'finish_reason': None}]})}\n\n"
        yield "data: [DONE]\n\n"
        return
    except Exception as e:
        log.error("_stream_and_save error url=%s: %s", url, e)
        if fallback_messages is not None:
            async for item in _stream_teacher_fallback(fallback_messages, user_id, user_msg, f"local model failed with {type(e).__name__}"):
                yield item
            return
        msg = f"\n\n[Echo: unexpected error — {type(e).__name__}: {e or 'no details'}]"
        yield f"data: {json.dumps({'choices': [{'delta': {'content': msg}, 'finish_reason': None}]})}\n\n"
        yield "data: [DONE]\n\n"
        return

    assistant_msg = "".join(collected)
    if assistant_msg:
        asyncio.create_task(_do_save_raw(user_id, user_msg, assistant_msg, model_used))


def _inject_system(messages: list[dict], system_injection: str) -> list[dict]:
    enriched = [m for m in messages if m.get("role") != "system"]
    return [{"role": "system", "content": system_injection}] + enriched


@app.post("/swap-adapter")
async def swap_adapter(request: Request):
    """Directly hot-swap an existing adapter into vLLM (for testing without retraining)."""
    from training.adapter import hot_swap_adapter, adapter_path_for_user
    body = await request.json()
    user_id = body.get("user_id", "default")
    lane = "gemma4_e2b"
    path = body.get("path") or adapter_path_for_user(user_id, lane=lane)
    if not path:
        return {"status": "no_adapter_found", "user_id": user_id, "lane": lane}
    ok = await hot_swap_adapter(user_id, path, lane=lane)
    return {"status": "swapped" if ok else "failed", "path": path, "lane": lane}


# In-memory training status per user: "idle" | "running" | "complete"
_training_status: dict[str, str] = {}


@app.get("/v1/training/status")
async def training_status(request: Request):
    user_id = request.state.user_id
    lane = _training_lane(request.query_params.get("lane") or request.headers.get("x-echo-model-lane"))
    memory_status = _training_status.get(_training_status_key(user_id, lane))
    latest = await latest_training_run(user_id, lane)
    status = memory_status or (latest["status"] if latest else "idle")
    summary = await get_training_summary(user_id, lane=lane)
    return {
        "status": status,
        "lane": lane,
        "latest_run": latest,
        "summary": summary,
    }


@app.get("/v1/user/history")
async def user_history(request: Request):
    """Return the last 50 conversation pairs for the authenticated user."""
    user_id = request.state.user_id
    async with get_conn() as db:
        async with db.execute(
            """SELECT user_msg, assistant_msg, topic, created_at
               FROM training_pairs
               WHERE user_id=?
               ORDER BY created_at DESC LIMIT 50""",
            (user_id,)
        ) as cur:
            rows = await cur.fetchall()
    return {"pairs": [dict(r) for r in rows]}


@app.get("/v1/training/history")
async def training_history(request: Request):
    user_id = request.state.user_id
    lane = _training_lane(request.query_params.get("lane"))
    async with get_conn() as db:
        async with db.execute(
            "SELECT created_at, lane FROM checkpoints WHERE user_id=? AND lane=? ORDER BY created_at ASC",
            (user_id, lane),
        ) as cur:
            rows = await cur.fetchall()
    return {"checkpoints": [{"created_at": r["created_at"], "lane": r["lane"]} for r in rows]}


async def _http_ok(url: str, timeout: float = 1.2) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
        return resp.status_code < 500
    except Exception:
        return False


async def _tcp_ok(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return bool(reader)
    except Exception:
        return False


async def _user_data_counts(user_id: str) -> dict:
    async with get_conn() as db:
        async with db.execute("SELECT COUNT(*) as cnt FROM life_events WHERE user_id=?", (user_id,)) as cur:
            life_events = await cur.fetchone()
        async with db.execute("SELECT COUNT(*) as cnt FROM shadow_outcomes WHERE user_id=?", (user_id,)) as cur:
            outcomes = await cur.fetchone()
        async with db.execute("SELECT COUNT(*) as cnt FROM proof_items WHERE user_id=? AND status='active'", (user_id,)) as cur:
            proof = await cur.fetchone()
        async with db.execute("SELECT COUNT(*) as cnt FROM training_pairs WHERE user_id=?", (user_id,)) as cur:
            pairs = await cur.fetchone()
    return {
        "life_events": life_events["cnt"] if life_events else 0,
        "outcomes": outcomes["cnt"] if outcomes else 0,
        "proof_items": proof["cnt"] if proof else 0,
        "training_pairs": pairs["cnt"] if pairs else 0,
    }


@app.get("/v1/runtime/capabilities")
async def runtime_capabilities(request: Request):
    """Capability negotiation for mobile, Home Brain, cloud, and offline modes."""
    user_id = request.state.user_id
    lane = _training_lane(request.query_params.get("lane") or request.headers.get("x-echo-model-lane") or "gemma4_e2b")
    vllm_health, livekit_ok, training, counts = await asyncio.gather(
        vllm_models_health(timeout=3.0),
        tcp_ok(settings.livekit_host, settings.livekit_port),
        get_training_summary(user_id, lane=lane),
        _user_data_counts(user_id),
    )
    teacher_ready = bool(settings.llm_api_key and settings.teacher_base_url)
    memory_ready = counts["life_events"] > 0 or counts["training_pairs"] > 0
    training_ready = bool(training.get("can_train_now"))
    home_brain_ready = settings.gemma4_enabled and bool(vllm_health.get("ready"))
    cloud_ready = teacher_ready
    return {
        "user_id": user_id,
        "generated_at": _dt.now(_timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode_recommendation": "home_brain" if home_brain_ready else ("cloud" if cloud_ready else "this_device"),
        "runtimes": {
            "home_brain": {
                "label": "Home Brain",
                "available": home_brain_ready,
                "model": settings.gemma4_base_model,
                "private_memory": memory_ready,
                "training": training_ready,
                "voice": livekit_ok,
                "connected_apps": True,
                "status": "ready" if home_brain_ready else vllm_health.get("status", "model_unreachable"),
                "latency_ms": vllm_health.get("latency_ms"),
                "models": vllm_health.get("models", []),
            },
            "cloud": {
                "label": "Cloud Echo",
                "available": cloud_ready,
                "model": settings.teacher_model,
                "private_memory": memory_ready,
                "training": False,
                "voice": False,
                "connected_apps": True,
                "status": "ready" if cloud_ready else "missing_api_key",
            },
            "this_device": {
                "label": "This Device",
                "available": True,
                "model": "client_negotiated_litert_lm",
                "private_memory": memory_ready,
                "memory_source": "synced_memory_pack",
                "training": False,
                "voice": False,
                "connected_apps": False,
                "status": "requires_client_model",
            },
        },
        "features": {
            "talk": {"home_brain": home_brain_ready, "cloud": cloud_ready, "this_device": True},
            "today": {"home_brain": True, "cloud": True, "this_device": memory_ready},
            "proof": {"home_brain": True, "cloud": True, "this_device": True},
            "opportunities": {"home_brain": True, "cloud": True, "this_device": True, "scoring": "proof_rules_v1"},
            "decision_room": {"home_brain": home_brain_ready, "cloud": cloud_ready, "this_device": False},
            "training_studio": {"home_brain": home_brain_ready and training_ready, "cloud": False, "this_device": False},
            "connected_apps": {"home_brain": True, "cloud": True, "this_device": False},
            "voice": {"home_brain": livekit_ok, "cloud": False, "this_device": False},
        },
        "counts": counts,
        "training": training,
        "limits": {
            "this_device": [
                "Offline Gemma uses compressed memory context.",
                "Personal training and connected-app actions sync when Home Brain or Cloud is reachable.",
            ],
            "cloud": ["Cloud can help online but should not be treated as the private training runtime."],
        },
    }


def _event_from_row(row) -> dict:
    item = dict(row)
    try:
        item["payload"] = json.loads(item.pop("payload_json") or "{}")
    except Exception:
        item["payload"] = {}
    event_type = item.get("event_type") or ""
    item["action"] = item["payload"].get("action") if isinstance(item.get("payload"), dict) else None
    item["is_product_event"] = event_type in {e["type"] for e in _EVENT_TAXONOMY} or event_type.startswith(("proof_", "opportunity_", "training_"))
    return item


@app.get("/v1/events/taxonomy")
async def event_taxonomy():
    return {"taxonomy": _EVENT_TAXONOMY}


@app.get("/v1/events/recent")
async def recent_events(request: Request):
    user_id = request.state.user_id
    limit = min(int(request.query_params.get("limit", "40")), 100)
    async with get_conn() as db:
        async with db.execute(
            """
            SELECT id, event_type, source, summary, payload_json, weight, created_at
            FROM echo_events
            WHERE user_id=?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ) as cur:
            rows = await cur.fetchall()
    return {"events": [_event_from_row(r) for r in rows], "taxonomy": _EVENT_TAXONOMY}


@app.get("/v1/events/stream")
async def stream_events(request: Request):
    """Small SSE stream for proactive mobile surfaces. Use recent events for replay."""
    user_id = request.state.user_id
    since = request.query_params.get("since") or _dt.now(_timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    async def event_generator():
        nonlocal since
        while True:
            if await request.is_disconnected():
                break
            async with get_conn() as db:
                async with db.execute(
                    """
                    SELECT id, event_type, source, summary, payload_json, weight, created_at
                    FROM echo_events
                    WHERE user_id=? AND created_at > ?
                    ORDER BY created_at ASC
                    LIMIT 20
                    """,
                    (user_id, since),
                ) as cur:
                    rows = await cur.fetchall()
            if rows:
                for row in rows:
                    event = _event_from_row(row)
                    since = event["created_at"]
                    yield f"event: {event['event_type']}\ndata: {json.dumps(event, ensure_ascii=True)}\n\n"
            else:
                yield ": keepalive\n\n"
            await asyncio.sleep(3)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/v1/loop/snapshot")
async def get_loop_snapshot(request: Request):
    """Single product loop summary for Today/You: events, outcomes, latest tournament."""
    user_id = request.state.user_id
    lane = _training_lane(request.query_params.get("lane"))
    data = await loop_snapshot(user_id)
    latest = await latest_training_run(user_id, lane)
    data["training_status"] = _training_status.get(_training_status_key(user_id, lane)) or (latest["status"] if latest else "idle")
    data["latest_training_run"] = latest
    return data


@app.get("/v1/today/priority")
async def today_priority(request: Request):
    """The product brain for Today: return the single most useful next card."""
    user_id = request.state.user_id
    return await get_today_priority(user_id)


@app.get("/v1/today/mission")
async def today_mission(request: Request):
    """Daily mission: one priority plus practice, clone, reality, and proof context."""
    user_id = request.state.user_id
    return await get_daily_mission(user_id)


@app.get("/v1/reality/check")
async def reality_check(request: Request):
    """Compare stated intent with available behavioral evidence."""
    user_id = request.state.user_id
    return await get_reality_check(user_id)


@app.get("/v1/growth/timeline")
async def growth_timeline(request: Request):
    """Longitudinal proof that Echo's loop is creating change."""
    user_id = request.state.user_id
    return await get_growth_timeline(user_id)


@app.get("/v1/revelation/status")
async def revelation_status(request: Request):
    """Readiness gate for the earned talent revelation moment."""
    user_id = request.state.user_id
    data = await get_revelation_status(user_id)
    if data.get("ready") or data.get("state") == "revealed":
        try:
            await _record_product_event_once(
                user_id,
                "revelation_available",
                "revelation",
                data.get("headline") or "A deeper Echo read is ready.",
                {"state": data.get("state"), "score": data.get("score"), "action": {"type": "open_you"}},
                weight=1.6,
            )
        except Exception as e:
            log.warning("revelation event check failed user=%s: %s", user_id, e)
    return data


@app.get("/v1/clone-mission/latest")
async def clone_mission_latest(request: Request):
    """Latest action returned by a winning shadow clone."""
    user_id = request.state.user_id
    mission = await get_latest_clone_mission(user_id)
    return {"mission": mission}


@app.get("/v1/interventions/next")
async def intervention_next(request: Request):
    """Return the next trusted proactive nudge Echo is allowed to send."""
    user_id = request.state.user_id
    intervention = await get_or_create_next_intervention(user_id)
    settings_data = await get_intervention_settings(user_id)
    return {
        "intervention": intervention,
        "settings": settings_data,
        "trust_rules": [
            "Echo only nudges when it can name the reason.",
            "Quiet hours and daily limits are enforced before scheduling.",
            "Every nudge deep-links to the exact loop action.",
            "Categories can be disabled independently.",
        ],
    }


@app.post("/v1/interventions/ack")
async def intervention_ack(request: Request):
    user_id = request.state.user_id
    body = await request.json()
    intervention_id = body.get("id") or body.get("intervention_id")
    if not intervention_id:
        return JSONResponse({"error": "id required"}, status_code=400)
    return await acknowledge_intervention(user_id, intervention_id, body.get("status", "acknowledged"))


@app.get("/v1/interventions/settings")
async def intervention_settings_get(request: Request):
    user_id = request.state.user_id
    return await get_intervention_settings(user_id)


@app.post("/v1/interventions/settings")
async def intervention_settings_post(request: Request):
    user_id = request.state.user_id
    body = await request.json()
    return await update_intervention_settings(user_id, body)


@app.post("/v1/life/events")
async def life_event_ingest(request: Request):
    """Opt-in local ingestion path for future real-world signals."""
    user_id = request.state.user_id
    body = await request.json()
    event_type = (body.get("event_type") or "").strip()
    if not event_type:
        return JSONResponse({"error": "event_type required"}, status_code=400)
    event_id = await record_life_event(
        user_id=user_id,
        event_domain=(body.get("event_domain") or "manual").strip()[:80],
        event_type=event_type[:120],
        source=(body.get("source") or "user").strip()[:80],
        title=(body.get("title") or "").strip(),
        summary=(body.get("summary") or "").strip(),
        payload=body.get("payload") if isinstance(body.get("payload"), dict) else {},
        confidence=float(body.get("confidence", 0.5)),
        privacy_level=(body.get("privacy_level") or "local").strip()[:40],
        subject_type=body.get("subject_type"),
        subject_id=body.get("subject_id"),
    )
    return {"saved": True, "event_id": event_id}


@app.get("/v1/thesis/current")
async def current_thesis(request: Request):
    """The durable center of Echo: current belief, evidence, and next test."""
    user_id = request.state.user_id
    return await get_current_thesis(user_id)


@app.get("/v1/training/summary")
async def training_summary(request: Request):
    """Clone-learning summary built from battles, outcomes, pairs, and checkpoints."""
    user_id = request.state.user_id
    lane = _training_lane(request.query_params.get("lane"))
    data = await get_training_summary(user_id, lane=lane)
    latest = await latest_training_run(user_id, lane)
    data["status"] = _training_status.get(_training_status_key(user_id, lane)) or (latest["status"] if latest else "idle")
    data["lane"] = lane
    data["latest_run"] = latest
    if data.get("ready_for_training") or data.get("dpo_ready_for_training"):
        try:
            await _record_product_event_once(
                user_id,
                "training_ready",
                "training",
                "Echo has enough saved moments for a personal update.",
                {
                    "lane": lane,
                    "untrained_pairs": data.get("untrained_pairs"),
                    "dpo_ready_pairs": data.get("dpo_ready_pairs"),
                    "action": {"type": "open_training_studio"},
                },
                weight=1.5,
            )
        except Exception as e:
            log.warning("training ready event check failed user=%s: %s", user_id, e)
    return data


@app.get("/v1/training/runs")
async def training_runs(request: Request):
    """All training runs for the user — used to show improvement over time."""
    user_id = request.state.user_id
    lane = _training_lane(request.query_params.get("lane"))
    async with get_conn() as db:
        async with db.execute(
            """SELECT id, status, untrained_pairs, adapter_path, error, summary_json, started_at, finished_at
               FROM training_runs WHERE user_id=? AND lane=?
               ORDER BY started_at DESC LIMIT 20""",
            (user_id, lane),
        ) as cur:
            rows = await cur.fetchall()
    runs = []
    for r in rows:
        summary = {}
        try:
            import json as _json
            summary = _json.loads(r["summary_json"] or "{}") if r["summary_json"] else {}
        except Exception:
            pass
        eval_data = summary.get("eval") or {}
        runs.append({
            "id": r["id"],
            "status": r["status"],
            "pairs": r["untrained_pairs"],
            "started_at": r["started_at"],
            "finished_at": r["finished_at"],
            "eval_score": eval_data.get("score"),
            "eval_passed": eval_data.get("passed"),
            "error": r["error"],
        })
    return {"runs": runs}


@app.get("/v1/training/eval")
async def training_eval(request: Request):
    """Last eval result for the user's adapter: score, passed, n_eval, details."""
    user_id = request.state.user_id
    lane = _training_lane(request.query_params.get("lane"))
    latest = await latest_training_run(user_id, lane)
    if not latest:
        return {"eval": None, "status": "no_runs", "run_id": None}
    return {
        "eval": latest["summary"].get("eval"),
        "status": latest["status"],
        "run_id": latest["id"],
        "finished_at": latest.get("finished_at"),
    }


@app.get("/v1/training/pipeline-trace")
async def training_pipeline_trace(request: Request):
    """Full self-improvement pipeline trace for Kaggle/demo audits.

    Safe by default: plain GET only reports existing pipeline state. Passing
    prepare=1/write=1 can prepare and write trace datasets for demos, but it
    does not stop vLLM or start a LoRA training job.
    """
    user_id = request.state.user_id
    lane = _training_lane(request.query_params.get("lane"))
    prepare = str(request.query_params.get("prepare", "0")).lower() not in {"0", "false", "no"}
    write = str(request.query_params.get("write", "0")).lower() not in {"0", "false", "no"}
    from training.adapter import adapter_status, adapter_path_for_user
    from training.orchestrator import build_pipeline_trace

    trace, latest, adapter = await asyncio.gather(
        build_pipeline_trace(user_id, lane=lane, prepare_augmented=prepare, write_datasets=write),
        latest_training_run(user_id, lane),
        adapter_status(user_id, lane=lane),
    )
    winner_path = adapter_path_for_user(user_id, lane=lane)
    trace["latest_run"] = latest
    trace["winner"] = {
        "adapter_path": winner_path or (latest or {}).get("adapter_path"),
        "source": "checkpoint_or_adapter_dir" if winner_path else ("latest_training_run" if latest else None),
        "eval": (latest or {}).get("summary", {}).get("eval") if latest else None,
    }
    trace["hot_swap"] = {
        "loaded": bool(adapter.get("loaded")),
        "vllm": adapter.get("vllm"),
        "model": adapter.get("serving_model"),
        "path": adapter.get("path"),
    }
    return trace


def _compute_rank(total_pairs: int, battles: int, checkpoints: int, practice_done: int) -> dict:
    """Compute a neutral growth stage from existing training data."""
    xp = (total_pairs * 3) + (battles * 15) + (checkpoints * 100) + (practice_done * 20)

    RANKS = [
        {"name": "Starting",   "title": "First Signals",  "min": 0,     "max": 300},
        {"name": "Practicing", "title": "Practice Loop",  "min": 300,   "max": 1000},
        {"name": "Proving",    "title": "Proof Builder",  "min": 1000,  "max": 3000},
        {"name": "Aligning",   "title": "Strong Signal",  "min": 3000,  "max": 7000},
        {"name": "Mastery",    "title": "Clear Direction", "min": 7000, "max": None},
    ]

    current = RANKS[0]
    for r in RANKS:
        if r["max"] is None or xp < r["max"]:
            current = r
            break

    xp_to_next = (current["max"] - xp) if current["max"] else 0
    progress = 0.0
    if current["max"]:
        span = current["max"] - current["min"]
        progress = min(1.0, (xp - current["min"]) / span) if span > 0 else 0.0

    return {
        "rank": current["name"],
        "title": current["title"],
        "xp": xp,
        "xp_to_next": xp_to_next,
        "progress": round(progress, 3),
    }


@app.get("/v1/user/rank")
async def user_rank(request: Request):
    """Growth stage derived from training activity. No new data needed."""
    user_id = request.state.user_id

    async with get_conn() as db:
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM training_pairs WHERE user_id=?", (user_id,)
        ) as cur:
            pairs_row = await cur.fetchone()

        async with db.execute(
            "SELECT COUNT(*) as cnt FROM tournament_runs WHERE user_id=? AND status='complete'",
            (user_id,),
        ) as cur:
            battles_row = await cur.fetchone()

        async with db.execute(
            "SELECT COUNT(*) as cnt FROM checkpoints WHERE user_id=?", (user_id,)
        ) as cur:
            ckpt_row = await cur.fetchone()

        async with db.execute(
            "SELECT COUNT(*) as cnt FROM practice_log WHERE user_id=? AND done=1",
            (user_id,),
        ) as cur:
            practice_row = await cur.fetchone()

    return _compute_rank(
        total_pairs=pairs_row["cnt"] if pairs_row else 0,
        battles=battles_row["cnt"] if battles_row else 0,
        checkpoints=ckpt_row["cnt"] if ckpt_row else 0,
        practice_done=practice_row["cnt"] if practice_row else 0,
    )


@app.get("/v1/system/health")
async def system_health(request: Request):
    """Operational health for the local Echo loop: DB, vLLM, adapter, LiveKit."""
    user_id = request.state.user_id
    lane = _training_lane(request.query_params.get("lane"))
    adapter, latest, vllm_health = await asyncio.gather(
        adapter_status(user_id, lane=lane),
        latest_training_run(user_id, lane),
        vllm_models_health(timeout=5.0),
    )
    health = {
        "echo": "ok",
        "database": "unknown",
        "vllm": "ok" if vllm_health.get("ready") else vllm_health.get("status", adapter["vllm"]),
        "vllm_health": vllm_health,
        "adapter": adapter,
        "adapter_loaded": adapter["loaded"],
        "livekit": "unknown",
        "training_status": _training_status.get(_training_status_key(user_id, lane)) or (latest["status"] if latest else "idle"),
        "latest_training_run": latest,
    }

    try:
        async with get_conn() as db:
            async with db.execute("SELECT 1") as cur:
                await cur.fetchone()
        health["database"] = "ok"
    except Exception as e:
        health["database"] = f"error: {e}"

    try:
        url = settings.livekit_url.replace("ws://", "http://").replace("wss://", "https://")
        async with httpx.AsyncClient(timeout=2) as client:
            resp = await client.get(url)
        health["livekit"] = "ok" if resp.status_code < 500 else f"http_{resp.status_code}"
    except Exception as e:
        health["livekit"] = f"error: {e}"

    return health


@app.get("/v1/experimental/gemma4/health")
async def gemma4_health(request: Request):
    """Health check for the experimental Gemma 4 E2B lane on its own vLLM port."""
    del request
    data = {
        "enabled": settings.gemma4_enabled,
        "base_url": settings.gemma4_vllm_base_url,
        "model": settings.gemma4_base_model,
        "status": "unknown",
        "models": [],
    }
    health = await vllm_models_health(timeout=5.0)
    data["status"] = "ok" if health.get("ready") else health.get("status", "error")
    data["models"] = health.get("models", [])
    data["latency_ms"] = health.get("latency_ms")
    return data


@app.get("/v1/teacher/policy")
async def teacher_policy_status(request: Request):
    """Show current teacher budget and the last sparse teacher uses."""
    user_id = request.state.user_id
    decision = await should_use_teacher(user_id, "tournament_challenger")
    async with get_conn() as db:
        async with db.execute(
            """
            SELECT purpose, reason, metadata_json, created_at
            FROM teacher_usage
            WHERE user_id=?
            ORDER BY created_at DESC
            LIMIT 20
            """,
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()
    return {
        "policy_enabled": settings.teacher_policy_enabled,
        "budget": decision.to_dict(),
        "recent": [dict(r) for r in rows],
    }


@app.post("/v1/experimental/gemma4/chat")
async def gemma4_chat(request: Request):
    """Small manual smoke test for Gemma 4 E2B. Does not affect normal Qwen routing."""
    body = await request.json()
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return JSONResponse({"error": "prompt required"}, status_code=400)
    messages = body.get("messages") or [{"role": "user", "content": prompt}]
    payload = {
        "model": settings.gemma4_base_model,
        "messages": messages,
        "temperature": float(body.get("temperature", 0.4)),
        "max_tokens": int(body.get("max_tokens", 320)),
        "stop": ["<function", "<tool_call>", "<|tool_call|>", "<functioncall>"],
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{settings.gemma4_vllm_base_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
    return {
        "model": settings.gemma4_base_model,
        "content": data["choices"][0]["message"].get("content", ""),
        "raw": data,
    }


@app.post("/v1/outcome")
async def record_user_outcome(request: Request):
    """Generic feedback endpoint: lets any feature teach Echo what helped."""
    user_id = request.state.user_id
    body = await request.json()
    subject_type = body.get("subject_type", "feature")
    subject_id = body.get("subject_id")
    outcome = body.get("outcome", "acknowledged")
    score = float(body.get("score", 0.5))
    note = body.get("note", "")
    training_feedback = None
    if subject_type == "chat_response":
        feedback_signal = {
            "thumbs_up": "thumbs_up",
            "helped": "thumbs_up",
            "saved_signal": "thumbs_up",
            "thumbs_down": "thumbs_down",
            "not_true": "thumbs_down",
        }.get(outcome)
        user_msg = (body.get("user_message") or "").strip()
        assistant_msg = (body.get("assistant_message") or "").strip()
        if feedback_signal and user_msg and assistant_msg:
            training_feedback = await mark_chat_feedback_pair(
                user_id=user_id,
                user_msg=user_msg,
                assistant_msg=assistant_msg,
                model_used=body.get("model_used") or "chat_feedback",
                engagement_signal=feedback_signal,
                topic=detect_topic(user_msg),
            )
    event_id = await record_event(
        user_id,
        "user_outcome",
        "user",
        f"User outcome recorded: {outcome}",
        {"subject_type": subject_type, "subject_id": subject_id, "score": score, "note": note},
        weight=max(0.2, min(score, 2.0)),
    )
    outcome_id = await record_outcome(
        user_id,
        subject_type=subject_type,
        subject_id=subject_id,
        outcome=outcome,
        score=score,
        event_id=event_id,
        note=note,
    )
    await refresh_current_thesis(user_id)
    opportunity_unlocked = False
    if subject_type in {"opportunity", "practice_rep", "chat_practice", "today_priority"} and score > 0:
        try:
            await _record_opportunity_unlock_if_ready(user_id)
            opportunity_unlocked = True
        except Exception as e:
            log.warning("opportunity unlock check failed user=%s: %s", user_id, e)
    loop_delta = await _loop_delta_for_save(
        user_id,
        event_id,
        topic=detect_topic(f"{subject_type} {outcome} {note}"),
        model_used="outcome",
        thesis_updated=True,
        proof_created=False,
        opportunity_unlocked=opportunity_unlocked,
        training_signal_saved=score > 0,
        next_action="Add proof if this moment matters for a goal.",
        receipt_title="Outcome saved",
        receipt_detail="Echo updated your read and training signal.",
    )
    return {
        "saved": True,
        "event_id": event_id,
        "outcome_id": outcome_id,
        "training_feedback": training_feedback,
        "loop_delta": loop_delta,
    }


def _list_from_body(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            pass
        return [part.strip() for part in text.split(",") if part.strip()]
    return []


def _json_list(value) -> str:
    return json.dumps(_list_from_body(value), ensure_ascii=True)


def _decode_json_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    except Exception:
        pass
    return []


_OPPORTUNITY_CATALOG = [
    {
        "title": "Scholarship story",
        "type": "scholarship",
        "description": "Turn lived effort, proof, feedback, and a future plan into a scholarship-ready narrative.",
        "required_proof": ["lived challenge", "two proof items", "feedback quote", "future plan", "safe share version"],
    },
    {
        "title": "Starter portfolio",
        "type": "portfolio",
        "description": "Show one shipped artifact with a clear before/after result and plain-language tradeoff.",
        "required_proof": ["shipped artifact", "measured result", "tradeoff note", "public link"],
    },
    {
        "title": "First job application",
        "type": "job",
        "description": "Map proof of skill, communication, and reliability to one realistic role.",
        "required_proof": ["skill proof", "communication proof", "reliability proof", "role narrative"],
    },
    {
        "title": "Community project",
        "type": "project",
        "description": "Use proof to make a credible ask for a local, school, or online project.",
        "required_proof": ["problem statement", "contribution proof", "collaborator feedback", "next ask"],
    },
    {
        "title": "Open-source contribution",
        "type": "project",
        "description": "Turn a small technical fix into public proof that other people can verify.",
        "required_proof": ["issue context", "pull request or patch", "review response", "learning note"],
    },
    {
        "title": "Personal goal",
        "type": "personal_goal",
        "description": "Convert repeated practice into evidence that behavior is changing.",
        "required_proof": ["baseline", "repeated practice", "outcome trend", "reflection"],
    },
]


_EVENT_TAXONOMY = [
    {"type": "pattern_detected", "default_action": "open_today", "description": "Echo has enough repeated evidence to show a new read."},
    {"type": "training_ready", "default_action": "open_training_studio", "description": "Enough user-approved moments exist to update Echo."},
    {"type": "revelation_available", "default_action": "open_you", "description": "A deeper strength or readiness moment is ready to show."},
    {"type": "opportunity_unlocked", "default_action": "open_opportunities", "description": "Proof readiness crossed a meaningful threshold."},
    {"type": "memory_consent_requested", "default_action": "open_memory", "description": "Echo captured a user-approved memory proposal."},
    {"type": "sync_required", "default_action": "open_runtime", "description": "Local device work should sync to Home Brain or Cloud."},
]


def _proof_from_row(row) -> dict:
    item = dict(row)
    item["skill_tags"] = _decode_json_list(item.pop("skill_tags_json", "[]"))
    return item


def _opportunity_from_row(row, *, proof_items: list[dict] | None = None, outcomes: list[dict] | None = None) -> dict:
    item = dict(row)
    item["required_proof"] = _decode_json_list(item.pop("required_proof_json", "[]"))
    item["missing_proof"] = _decode_json_list(item.pop("missing_proof_json", "[]"))
    if proof_items is not None:
        item = _score_opportunity(item, proof_items, outcomes or [])
    return item


async def _proof_summary(user_id: str) -> dict:
    async with get_conn() as db:
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM proof_items WHERE user_id=? AND status='active'",
            (user_id,),
        ) as cur:
            count_row = await cur.fetchone()
        async with db.execute(
            """
            SELECT category, COUNT(*) as cnt
            FROM proof_items
            WHERE user_id=? AND status='active'
            GROUP BY category
            ORDER BY cnt DESC
            """,
            (user_id,),
        ) as cur:
            by_category = [dict(r) for r in await cur.fetchall()]
        async with db.execute(
            """
            SELECT opportunity_type, COUNT(*) as cnt
            FROM proof_items
            WHERE user_id=? AND status='active'
            GROUP BY opportunity_type
            ORDER BY cnt DESC
            """,
            (user_id,),
        ) as cur:
            by_opportunity_type = [dict(r) for r in await cur.fetchall()]
        async with db.execute(
            """
            SELECT subject_type, COUNT(*) as cnt
            FROM shadow_outcomes
            WHERE user_id=?
            GROUP BY subject_type
            ORDER BY cnt DESC
            """,
            (user_id,),
        ) as cur:
            outcomes_by_subject = [dict(r) for r in await cur.fetchall()]
    return {
        "count": count_row["cnt"] if count_row else 0,
        "by_category": by_category,
        "by_opportunity_type": by_opportunity_type,
        "outcomes_by_subject": outcomes_by_subject,
    }


async def _proof_items_for_scoring(user_id: str) -> list[dict]:
    async with get_conn() as db:
        async with db.execute(
            """
            SELECT *
            FROM proof_items
            WHERE user_id=? AND status='active'
            ORDER BY created_at DESC
            LIMIT 100
            """,
            (user_id,),
        ) as cur:
            return [_proof_from_row(r) for r in await cur.fetchall()]


async def _outcomes_for_scoring(user_id: str) -> list[dict]:
    async with get_conn() as db:
        async with db.execute(
            """
            SELECT subject_type, subject_id, outcome, score, note, created_at
            FROM shadow_outcomes
            WHERE user_id=?
            ORDER BY created_at DESC
            LIMIT 100
            """,
            (user_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


def _proof_haystack(item: dict) -> str:
    parts = [
        item.get("title"),
        item.get("description"),
        item.get("evidence"),
        item.get("category"),
        item.get("opportunity_type"),
        " ".join(item.get("skill_tags") or []),
    ]
    return " ".join(str(p or "").lower() for p in parts)


def _gap_requirements(gap: str) -> tuple[set[str], tuple[str, ...]]:
    text = gap.lower()
    categories: set[str] = set()
    keywords: list[str] = []
    if any(k in text for k in ("feedback", "quote", "review", "collaborator")):
        categories.add("feedback")
        keywords += ["feedback", "quote", "review", "mentor", "teacher", "client", "collaborator"]
    if any(k in text for k in ("artifact", "link", "public", "patch", "pull request", "shipped")):
        categories.add("artifact")
        keywords += ["artifact", "link", "public", "patch", "pull request", "shipped", "demo", "file"]
    if any(k in text for k in ("result", "trend", "baseline", "changed", "measured")):
        categories.add("outcome")
        keywords += ["result", "trend", "baseline", "metric", "measured", "before", "after", "changed"]
    if any(k in text for k in ("practice", "rep", "skill", "learning", "reflection")):
        categories.add("practice")
        keywords += ["practice", "rep", "skill", "learned", "lesson"]
    if any(k in text for k in ("story", "narrative", "plan", "ask", "statement", "challenge")):
        categories.add("story")
        keywords += ["story", "narrative", "plan", "ask", "statement", "challenge", "why"]
    if not keywords:
        keywords = [part for part in re.split(r"[^a-z0-9]+", text) if len(part) > 3]
    return categories, tuple(dict.fromkeys(keywords))


def _find_matching_proof(gap: str, proof_items: list[dict], outcomes: list[dict], used_proof_ids: set[str]) -> dict | None:
    categories, keywords = _gap_requirements(gap)
    lower = gap.lower()
    for item in proof_items:
        proof_id = str(item.get("id") or item.get("title") or "")
        if proof_id and proof_id in used_proof_ids:
            continue
        category = str(item.get("category") or "").lower()
        haystack = _proof_haystack(item)
        if "repeated practice" in lower and category != "practice" and not any(k in haystack for k in ("practice", "rep", "streak")):
            continue
        if "outcome trend" in lower and not any(k in haystack for k in ("trend", "streak", "over time", "three", "3 days")):
            continue
        if "baseline" in lower and not any(k in haystack for k in ("baseline", "before", "starting point")):
            continue
        if category in categories:
            return {"kind": "proof", "id": item.get("id"), "title": item.get("title"), "category": category}
        if keywords and any(keyword in haystack for keyword in keywords):
            return {"kind": "proof", "id": item.get("id"), "title": item.get("title"), "category": category}

    positive_outcomes = [o for o in outcomes if float(o.get("score") or 0.0) > 0]
    practice_outcomes = [o for o in positive_outcomes if "practice" in str(o.get("subject_type") or "")]
    outcome_notes = " ".join(str(o.get("note") or "").lower() for o in positive_outcomes)
    if "outcome trend" in lower and len(positive_outcomes) >= 3:
        return {"kind": "outcome", "title": "three positive outcomes", "category": "outcome"}
    if "baseline" in lower and any(k in outcome_notes for k in ("baseline", "before", "starting point")):
        return {"kind": "outcome", "title": "baseline outcome", "category": "outcome"}
    if any(k in lower for k in ("result", "measured")) and positive_outcomes:
        return {"kind": "outcome", "title": "recent outcome", "category": "outcome"}
    if any(k in lower for k in ("practice", "rep", "repeated")) and len(practice_outcomes) >= 2:
        return {"kind": "outcome", "title": "repeated practice outcomes", "category": "practice"}
    if "two proof" in lower and len(proof_items) >= 2:
        return {"kind": "proof_count", "title": "two proof items", "category": "proof"}
    return None


def _next_step_for_missing(missing: list[str], opportunity_type: str) -> str:
    if not missing:
        return "Choose one real application, project, or ask and prepare a public-safe proof story."
    gap = missing[0].lower()
    if "feedback" in gap or "quote" in gap:
        return "Ask one person who saw your work for a specific sentence of feedback."
    if "public" in gap or "link" in gap or "artifact" in gap or "patch" in gap:
        return "Create a public-safe artifact or link that someone else can inspect."
    if "narrative" in gap or "story" in gap or "plan" in gap:
        return "Write the short story: what changed, what proof exists, and what you are ready for next."
    if "practice" in gap or "rep" in gap:
        return "Complete one focused practice rep and log the outcome."
    if opportunity_type == "scholarship":
        return "Add one proof item that connects effort, constraint, and future plan."
    return f"Build proof for: {missing[0]}."


def _score_opportunity(item: dict, proof_items: list[dict], outcomes: list[dict]) -> dict:
    required = list(item.get("required_proof") or [])
    if not required:
        required = ["practice proof", "outcome proof", "shareable artifact"]
    matched: list[dict] = []
    missing: list[str] = []
    used_proof_ids: set[str] = set()
    for gap in required:
        match = _find_matching_proof(gap, proof_items, outcomes, used_proof_ids)
        if match:
            matched.append({"gap": gap, **match})
            if match.get("kind") == "proof":
                used_proof_ids.add(str(match.get("id") or match.get("title") or ""))
        else:
            missing.append(gap)
    done = len(required) - len(missing)
    base = round((done / max(len(required), 1)) * 100)
    proof_boost = min(12, max(0, len(proof_items) - done) * 2)
    outcome_boost = min(8, len([o for o in outcomes if float(o.get("score") or 0.0) > 0]) * 2)
    readiness = max(0, min(100, base + proof_boost + outcome_boost))
    item = {**item}
    item["required_proof"] = required
    item["missing_proof"] = missing
    item["matched_proof"] = matched
    item["readiness"] = readiness
    item["readiness_label"] = "ready" if readiness >= 75 else ("building" if readiness >= 40 else "early")
    item["next_step"] = item.get("next_step") or _next_step_for_missing(missing, item.get("type") or "personal_goal")
    item["scoring_version"] = "proof_rules_v1"
    return item


async def _opportunity_suggestions(user_id: str, limit: int = 6) -> list[dict]:
    thesis, proof_items, outcomes = await asyncio.gather(
        get_current_thesis(user_id),
        _proof_items_for_scoring(user_id),
        _outcomes_for_scoring(user_id),
    )
    direction = thesis.get("title") or "your next direction"
    suggestions = []
    for seed in _OPPORTUNITY_CATALOG:
        item = {
            "id": None,
            "user_id": user_id,
            "title": seed["title"],
            "type": seed["type"],
            "description": seed["description"],
            "required_proof": seed["required_proof"],
            "missing_proof": [],
            "next_step": "",
            "status": "suggested",
            "generated": True,
            "seeded": True,
            "direction": direction,
        }
        suggestions.append(_score_opportunity(item, proof_items, outcomes))
    suggestions.sort(key=lambda x: (x.get("readiness", 0), -len(x.get("missing_proof") or [])), reverse=True)
    return suggestions[:limit]


async def _generated_opportunity(user_id: str) -> dict:
    suggestions = await _opportunity_suggestions(user_id, limit=1)
    if suggestions:
        return suggestions[0]
    thesis = await get_current_thesis(user_id)
    direction = thesis.get("title") or "your next direction"
    return _score_opportunity(
        {
            "id": None,
            "user_id": user_id,
            "title": f"Build proof for {direction}",
            "type": "personal_goal",
            "description": thesis.get("statement") or "Echo is still forming your direction. Start by creating one small proof item.",
            "required_proof": ["practice proof", "outcome proof", "shareable artifact"],
            "missing_proof": [],
            "next_step": "",
            "status": "suggested",
            "generated": True,
        },
        [],
        [],
    )


async def _record_opportunity_unlock_if_ready(user_id: str) -> None:
    suggestions = await _opportunity_suggestions(user_id, limit=1)
    if not suggestions:
        return
    top = suggestions[0]
    if int(top.get("readiness") or 0) < 70:
        return
    async with get_conn() as db:
        async with db.execute(
            """
            SELECT 1
            FROM echo_events
            WHERE user_id=? AND event_type='opportunity_unlocked'
              AND json_extract(payload_json, '$.title')=?
              AND created_at >= datetime('now', '-1 day')
            LIMIT 1
            """,
            (user_id, top.get("title")),
        ) as cur:
            if await cur.fetchone():
                return
    await record_event(
        user_id,
        "opportunity_unlocked",
        "opportunity",
        f"{top.get('title')} is {top.get('readiness')}% ready.",
        {
            "title": top.get("title"),
            "type": top.get("type"),
            "readiness": top.get("readiness"),
            "missing_proof": top.get("missing_proof") or [],
            "action": {"type": "open_opportunities"},
        },
        weight=1.7,
    )


async def _record_product_event_once(
    user_id: str,
    event_type: str,
    source: str,
    summary: str,
    payload: dict,
    weight: float = 1.0,
    window: str = "-1 day",
) -> None:
    async with get_conn() as db:
        async with db.execute(
            """
            SELECT 1
            FROM echo_events
            WHERE user_id=? AND event_type=? AND created_at >= datetime('now', ?)
            LIMIT 1
            """,
            (user_id, event_type, window),
        ) as cur:
            if await cur.fetchone():
                return
    await record_event(user_id, event_type, source, summary, payload, weight=weight)


def _compact_text(value, limit: int = 1200) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _coerce_text_list(value, limit: int = 12) -> list[str]:
    if isinstance(value, dict):
        items = [f"{k}: {v}" for k, v in value.items()]
    else:
        items = _list_from_body(value)
    return [_compact_text(item, 220) for item in items if _compact_text(item, 220)][:limit]


def _extract_json_object(text: str | None) -> dict | None:
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _public_safe_text(text: str, limit: int = 1000) -> str:
    safe = re.sub(r"[\w.\-+]+@[\w.\-]+\.\w+", "[email removed]", text or "")
    safe = re.sub(r"\+?\d[\d\s().-]{7,}\d", "[phone removed]", safe)
    return _compact_text(safe, limit)


def _artifact_lines(body: dict) -> list[str]:
    lines: list[str] = []
    for key in ("visible_text", "ocr_text", "labels", "detected_objects", "evidence"):
        lines.extend(_coerce_text_list(body.get(key)))
    for key in ("scene", "caption", "user_caption", "description", "note", "goal"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            lines.append(_compact_text(value, 260))
    return list(dict.fromkeys([line for line in lines if line]))[:20]


def _artifact_image_present(body: dict) -> bool:
    return any(body.get(key) for key in ("image_url", "image_base64", "image_data", "file_name", "mime_type"))


def _artifact_image_url(body: dict) -> str | None:
    if body.get("image_url"):
        return str(body.get("image_url"))
    raw = body.get("image_base64") or body.get("image_data")
    if not raw:
        return None
    raw_text = str(raw)
    if raw_text.startswith("data:image/"):
        return raw_text
    mime_type = str(body.get("mime_type") or "image/jpeg")
    return f"data:{mime_type};base64,{raw_text}"


async def _call_gemma_vision_feature(
    user_id: str,
    prompt: str,
    image_url: str,
    *,
    system: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 650,
) -> str:
    use_adapter = adapter_exists(user_id, lane="gemma4_e2b")
    if use_adapter:
        use_adapter = await ensure_adapter_loaded(user_id, lane="gemma4_e2b")
    base_url, model, _ = _local_target_for_lane("gemma4_e2b", user_id, use_adapter=use_adapter)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    )
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"].get("content", "").strip()


def _infer_artifact_skills(text: str) -> list[str]:
    lower = text.lower()
    skills: list[str] = []
    if any(k in lower for k in ("sensor", "hardware", "robot", "circuit", "electronics", "motor", "prototype")):
        skills.append("hardware prototyping")
    if any(k in lower for k in ("test", "stable", "minutes", "measured", "baseline", "outdoor")):
        skills.append("testing")
    if any(k in lower for k in ("cost", "price", "$", "cheap", "cheaper", "budget", "reduced")):
        skills.append("cost tradeoff")
    if any(k in lower for k in ("offline", "internet", "low-connectivity", "network")):
        skills.append("offline-first engineering")
    if any(k in lower for k in ("teacher", "feedback", "explains", "debug", "helps", "mentor", "review")):
        skills.append("communication")
    if any(k in lower for k in ("write", "description", "story", "public", "explain")):
        skills.append("public proof packaging")
    return skills or ["evidence gathering"]


def _heuristic_artifact_analysis(body: dict) -> dict:
    lines = _artifact_lines(body)
    text = "\n".join(lines)
    lower = text.lower()
    title = _compact_text(body.get("title") or "", 120)
    if not title:
        if "garden" in lower and "sensor" in lower:
            title = "Offline garden sensor field test"
        elif lines:
            title = lines[0][:120]
        else:
            title = "Proof artifact"

    evidence = [line for line in lines if line.lower() != title.lower()][:6]
    if not evidence and text:
        evidence = [_compact_text(text, 180)]

    if "stable" in lower and "40" in lower:
        summary = "A working prototype was tested and produced measurable stability evidence."
    elif evidence:
        summary = f"Artifact evidence captured: {evidence[0]}"
    else:
        summary = "A user-captured artifact is ready to be turned into proof."

    missing_context: list[str] = []
    if not _artifact_image_present(body):
        missing_context.append("one photo or short video of the artifact")
    if not any(k in lower for k in ("benefit", "helps", "community", "school", "user", "customer")):
        missing_context.append("one sentence about who benefits from this work")
    if not any(k in lower for k in ("feedback", "teacher", "review", "quote", "mentor")):
        missing_context.append("one reviewer quote or feedback sentence")

    privacy_risk = "medium" if re.search(r"[\w.\-+]+@[\w.\-]+\.\w+|\+?\d[\d\s().-]{7,}\d", text) else "low"
    public_safe = _public_safe_text("; ".join(evidence or [summary]), 900)

    return {
        "model_family": "heuristic_rules_v1",
        "proof_title": title,
        "artifact_summary": summary,
        "evidence": evidence,
        "skills_proved": _infer_artifact_skills(text),
        "privacy_risk": privacy_risk,
        "public_safe_version": public_safe or summary,
        "missing_context": missing_context[:4],
        "recommended_echo_action": "create_proof_item",
        "confidence": 0.68 if evidence else 0.42,
    }


def _normalize_artifact_analysis(parsed: dict | None, fallback: dict, model_used: str) -> dict:
    if not isinstance(parsed, dict):
        return fallback
    analysis = dict(fallback)
    title = parsed.get("proof_title") or parsed.get("title")
    summary = parsed.get("artifact_summary") or parsed.get("summary") or parsed.get("description")
    public_safe = parsed.get("public_safe_version") or parsed.get("public_safe_summary")
    evidence = _coerce_text_list(parsed.get("evidence"), limit=8)
    skills = _coerce_text_list(parsed.get("skills_proved") or parsed.get("skill_tags"), limit=8)
    missing = _coerce_text_list(parsed.get("missing_context") or parsed.get("missing_proof"), limit=6)
    if title:
        analysis["proof_title"] = _compact_text(title, 120)
    if summary:
        analysis["artifact_summary"] = _compact_text(summary, 700)
    if evidence:
        analysis["evidence"] = evidence
    if skills:
        analysis["skills_proved"] = skills
    if missing:
        analysis["missing_context"] = missing
    if public_safe:
        analysis["public_safe_version"] = _public_safe_text(str(public_safe), 900)
    if parsed.get("privacy_risk"):
        analysis["privacy_risk"] = str(parsed.get("privacy_risk"))[:40]
    if parsed.get("recommended_echo_action"):
        analysis["recommended_echo_action"] = str(parsed.get("recommended_echo_action"))[:80]
    analysis["model_family"] = model_used
    analysis["confidence"] = float(parsed.get("confidence") or analysis.get("confidence") or 0.7)
    return analysis


async def _analyze_artifact_for_proof(user_id: str, body: dict) -> dict:
    fallback = _heuristic_artifact_analysis(body)
    use_model = body.get("use_model", True) is not False
    if not use_model or not settings.gemma4_enabled:
        return fallback

    health = await vllm_models_health(timeout=1.5)
    if not health.get("ready"):
        fallback["model_note"] = f"Gemma runtime unavailable; used {fallback['model_family']}."
        return fallback

    lines = _artifact_lines(body)
    image_note = (
        "Image bytes or URL were provided. Use any textual hints below and return conservative claims."
        if _artifact_image_present(body)
        else "No image bytes were provided; analyze only the visible text and caption."
    )
    prompt = (
        ECHO_FEATURE_HEADER
        + "You are analyzing a user-captured artifact for Echo Proof Camera.\n"
        + "Return ONLY valid JSON with these keys: proof_title, artifact_summary, evidence, "
        + "skills_proved, privacy_risk, public_safe_version, missing_context, recommended_echo_action, confidence.\n"
        + "Rules: do not invent facts; separate private raw evidence from the public-safe version; "
        + "the recommended action should usually be create_proof_item when evidence is concrete.\n\n"
        + f"{image_note}\n"
        + f"User goal: {_compact_text(body.get('goal') or body.get('opportunity_type') or '', 400)}\n"
        + "Artifact notes:\n"
        + "\n".join(f"- {line}" for line in lines[:18])
    )
    try:
        image_url = _artifact_image_url(body)
        if image_url:
            text = await _call_gemma_vision_feature(
                user_id,
                prompt,
                image_url,
                system=ECHO_PRODUCT_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=650,
            )
        else:
            text = await _call_gemma_feature(
                user_id,
                prompt,
                system=ECHO_PRODUCT_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=650,
            )
        parsed = _extract_json_object(text)
        analysis = _normalize_artifact_analysis(parsed, fallback, "gemma4_e2b")
        analysis["raw_model_used"] = "gemma4_e2b"
        return analysis
    except Exception as e:
        log.warning("artifact analysis model failed user=%s: %s", user_id, e)
        fallback["model_note"] = f"Gemma analysis failed; used {fallback['model_family']}."
        return fallback


ECHO_TOOL_SCHEMAS = [
    {
        "name": "create_proof_item",
        "description": "Save a public-safe proof item from an artifact, outcome, or feedback quote.",
        "parameters": {
            "type": "object",
            "required": ["title", "description", "evidence", "category", "skill_tags", "opportunity_type"],
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "evidence": {"type": "string"},
                "category": {"type": "string", "enum": ["artifact", "feedback", "outcome", "practice", "story"]},
                "skill_tags": {"type": "array", "items": {"type": "string"}},
                "opportunity_type": {"type": "string"},
            },
        },
    },
    {
        "name": "log_outcome",
        "description": "Record what happened after a practice rep or real-world action.",
        "parameters": {
            "type": "object",
            "required": ["subject_type", "outcome", "score", "note"],
            "properties": {
                "subject_type": {"type": "string"},
                "subject_id": {"type": "string"},
                "outcome": {"type": "string"},
                "score": {"type": "number"},
                "note": {"type": "string"},
            },
        },
    },
    {
        "name": "generate_opportunity",
        "description": "Generate a scored opportunity path from current proof and missing evidence.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "ask_for_feedback",
        "description": "Create a public-safe feedback request for a teacher, peer, client, or collaborator.",
        "parameters": {
            "type": "object",
            "required": ["audience", "artifact", "question"],
            "properties": {
                "audience": {"type": "string"},
                "artifact": {"type": "string"},
                "question": {"type": "string"},
            },
        },
    },
]


def _tool_schema_summary() -> list[dict]:
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "required": tool.get("parameters", {}).get("required", []),
        }
        for tool in ECHO_TOOL_SCHEMAS
    ]


def _heuristic_tool_decision(body: dict) -> dict:
    data = body.get("artifact_analysis") or body.get("analysis") or body.get("input") or {}
    if not isinstance(data, dict):
        data = {}
    recommended = str(data.get("recommended_echo_action") or "").strip()
    if recommended in {tool["name"] for tool in ECHO_TOOL_SCHEMAS}:
        tool_name = recommended
    else:
        objective = " ".join(str(v or "") for v in (body.get("goal"), body.get("objective"), body.get("instruction"))).lower()
        tool_name = "generate_opportunity" if "opportun" in objective else "create_proof_item"

    if tool_name == "create_proof_item":
        args = {
            "title": data.get("proof_title") or body.get("title") or "Proof artifact",
            "description": data.get("public_safe_version") or data.get("artifact_summary") or body.get("description") or "",
            "evidence": "; ".join(_coerce_text_list(data.get("evidence"), limit=8)),
            "category": body.get("category") or "artifact",
            "skill_tags": data.get("skills_proved") or body.get("skill_tags") or [],
            "opportunity_type": body.get("opportunity_type") or "scholarship",
        }
    elif tool_name == "log_outcome":
        args = {
            "subject_type": body.get("subject_type") or "artifact",
            "outcome": body.get("outcome") or "evidence_captured",
            "score": float(body.get("score") or 1.0),
            "note": data.get("artifact_summary") or body.get("note") or "Artifact evidence captured.",
        }
    elif tool_name == "ask_for_feedback":
        args = {
            "audience": body.get("audience") or "teacher",
            "artifact": data.get("proof_title") or body.get("artifact") or "the artifact",
            "question": body.get("question") or "What is clear, and what proof is missing?",
        }
    else:
        args = {}

    return {
        "thought": "Selected the safest Echo action from the available tool schema.",
        "tool_name": tool_name,
        "arguments": args,
        "next_missing_proof": data.get("missing_context") or [],
        "model_used": "heuristic_rules_v1",
    }


def _normalize_tool_decision(parsed: dict | None, fallback: dict) -> dict:
    if not isinstance(parsed, dict):
        return fallback
    if isinstance(parsed.get("tool_calls"), list) and parsed["tool_calls"]:
        call = parsed["tool_calls"][0]
        fn = call.get("function", {}) if isinstance(call, dict) else {}
        name = fn.get("name")
        raw_args = fn.get("arguments") or {}
        if isinstance(raw_args, str):
            raw_args = _extract_json_object(raw_args) or {}
        parsed = {"tool_name": name, "arguments": raw_args, "thought": parsed.get("thought") or parsed.get("reasoning")}

    allowed = {tool["name"] for tool in ECHO_TOOL_SCHEMAS}
    tool_name = str(parsed.get("tool_name") or parsed.get("name") or fallback["tool_name"])
    if tool_name not in allowed:
        return fallback
    args = parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else fallback.get("arguments", {})
    return {
        "thought": _compact_text(parsed.get("thought") or parsed.get("reasoning") or fallback.get("thought"), 700),
        "tool_name": tool_name,
        "arguments": args,
        "next_missing_proof": _coerce_text_list(parsed.get("next_missing_proof") or parsed.get("missing_context"), limit=6)
        or fallback.get("next_missing_proof", []),
        "model_used": parsed.get("model_used") or "gemma4_e2b",
    }


async def _gemma_tool_decision(user_id: str, body: dict) -> dict:
    fallback = _heuristic_tool_decision(body)
    if body.get("use_model", True) is False or not settings.gemma4_enabled:
        return fallback
    health = await vllm_models_health(timeout=1.5)
    if not health.get("ready"):
        fallback["model_note"] = "Gemma runtime unavailable; used heuristic tool selection."
        return fallback
    prompt = (
        ECHO_FEATURE_HEADER
        + "Choose exactly one Echo tool for the user's situation.\n"
        + "Return ONLY valid JSON: {\"thought\": string, \"tool_name\": string, \"arguments\": object, \"next_missing_proof\": array}.\n"
        + "Do not invent tools. Prefer create_proof_item when the input has concrete evidence.\n\n"
        + f"Allowed tools:\n{json.dumps(ECHO_TOOL_SCHEMAS, ensure_ascii=True, indent=2)}\n\n"
        + f"User situation:\n{json.dumps(body, ensure_ascii=True, indent=2)[:6000]}"
    )
    try:
        text = await _call_gemma_feature(
            user_id,
            prompt,
            system=ECHO_PRODUCT_SYSTEM_PROMPT,
            temperature=0.05,
            max_tokens=650,
        )
        decision = _normalize_tool_decision(_extract_json_object(text), fallback)
        decision["model_used"] = "gemma4_e2b"
        return decision
    except Exception as e:
        log.warning("Gemma tool decision failed user=%s: %s", user_id, e)
        fallback["model_note"] = "Gemma tool decision failed; used heuristic tool selection."
        return fallback


async def _execute_echo_tool(user_id: str, decision: dict) -> dict:
    tool_name = decision.get("tool_name")
    args = decision.get("arguments") if isinstance(decision.get("arguments"), dict) else {}
    if tool_name == "create_proof_item":
        return await _create_proof_item_for_user(user_id, args)
    if tool_name == "log_outcome":
        subject_type = _compact_text(args.get("subject_type") or "artifact", 80)
        outcome = _compact_text(args.get("outcome") or "evidence_captured", 120)
        note = _compact_text(args.get("note") or "", 500)
        try:
            score = float(args.get("score", 1.0))
        except Exception:
            score = 1.0
        event_id = await record_event(
            user_id,
            "tool_outcome_logged",
            "gemma_tool",
            note or f"Outcome logged: {outcome}",
            {"tool": tool_name, "arguments": args},
            weight=1.1,
        )
        outcome_id = await record_outcome(
            user_id,
            subject_type=subject_type,
            subject_id=args.get("subject_id"),
            event_id=event_id,
            outcome=outcome,
            score=score,
            note=note,
        )
        await refresh_current_thesis(user_id)
        return {"saved": True, "event_id": event_id, "outcome_id": outcome_id}
    if tool_name == "generate_opportunity":
        generated = await _generated_opportunity(user_id)
        item_id = str(uuid.uuid4())
        async with get_conn() as db:
            await db.execute(
                """
                INSERT INTO opportunity_goals
                    (id, user_id, title, type, description, required_proof_json,
                     missing_proof_json, next_step, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')
                """,
                (
                    item_id,
                    user_id,
                    generated["title"][:160],
                    generated["type"][:60],
                    generated["description"][:1200],
                    _json_list(generated.get("required_proof")),
                    _json_list(generated.get("missing_proof")),
                    generated.get("next_step", "")[:500],
                ),
            )
            await db.commit()
            async with db.execute("SELECT * FROM opportunity_goals WHERE id=? AND user_id=?", (item_id, user_id)) as cur:
                row = await cur.fetchone()
        proof_items, outcomes = await asyncio.gather(_proof_items_for_scoring(user_id), _outcomes_for_scoring(user_id))
        return {"saved": True, "item": _opportunity_from_row(row, proof_items=proof_items, outcomes=outcomes)}
    if tool_name == "ask_for_feedback":
        audience = _compact_text(args.get("audience") or "reviewer", 80)
        artifact = _compact_text(args.get("artifact") or "this artifact", 160)
        question = _compact_text(args.get("question") or "What is clear, and what proof is missing?", 260)
        draft = f"Could you review {artifact}? {question}"
        event_id = await record_event(
            user_id,
            "feedback_request_drafted",
            "gemma_tool",
            f"Feedback request drafted for {audience}.",
            {"audience": audience, "artifact": artifact, "question": question, "draft": draft},
            weight=0.9,
        )
        return {"created": True, "event_id": event_id, "draft": draft}
    return {"executed": False, "error": f"Unknown tool: {tool_name}"}


@app.get("/v1/tools/schema")
async def echo_tools_schema(request: Request):
    return {"tools": ECHO_TOOL_SCHEMAS, "summary": _tool_schema_summary()}


@app.post("/v1/gemma/tool-call")
async def gemma_tool_call(request: Request):
    user_id = request.state.user_id
    body = await request.json()
    decision = await _gemma_tool_decision(user_id, body)
    execute = body.get("execute", True) is not False
    result = await _execute_echo_tool(user_id, decision) if execute else None
    return {
        "tools": _tool_schema_summary(),
        "decision": decision,
        "executed": execute,
        "result": result,
    }


@app.post("/v1/tools/decide")
async def tools_decide(request: Request):
    return await gemma_tool_call(request)


@app.post("/v1/vision/analyze")
async def vision_analyze(request: Request):
    user_id = request.state.user_id
    body = await request.json()
    analysis = await _analyze_artifact_for_proof(user_id, body)
    response = {"analyzed": True, "analysis": analysis, "input_mode": "image_or_text" if _artifact_image_present(body) else "text_hints"}
    if body.get("create_proof") is True:
        proof_body = {
            "title": analysis["proof_title"],
            "description": analysis["public_safe_version"],
            "evidence": "; ".join(analysis.get("evidence") or []),
            "category": body.get("category") or "artifact",
            "source_type": body.get("source_type") or "proof_camera",
            "source_id": body.get("source_id"),
            "skill_tags": analysis.get("skills_proved") or [],
            "opportunity_type": body.get("opportunity_type") or "scholarship",
        }
        response["proof"] = await _create_proof_item_for_user(user_id, proof_body)
    return response


@app.post("/v1/proof/from-artifact")
async def create_proof_from_artifact(request: Request):
    user_id = request.state.user_id
    body = await request.json()
    analysis = await _analyze_artifact_for_proof(user_id, body)
    proof_body = {
        "title": body.get("title") or analysis["proof_title"],
        "description": body.get("description") or analysis["public_safe_version"],
        "evidence": body.get("evidence") or "; ".join(analysis.get("evidence") or []),
        "category": body.get("category") or "artifact",
        "source_type": body.get("source_type") or "proof_camera",
        "source_id": body.get("source_id"),
        "skill_tags": body.get("skill_tags") or analysis.get("skills_proved") or [],
        "opportunity_type": body.get("opportunity_type") or "scholarship",
    }
    proof = await _create_proof_item_for_user(user_id, proof_body)
    return {
        "saved": True,
        "analysis": analysis,
        "proof": proof,
        "next_missing_proof": analysis.get("missing_context") or [],
    }


@app.get("/v1/proof/items")
async def list_proof_items(request: Request):
    user_id = request.state.user_id
    limit = min(int(request.query_params.get("limit", "50")), 100)
    async with get_conn() as db:
        async with db.execute(
            """
            SELECT *
            FROM proof_items
            WHERE user_id=? AND status='active'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ) as cur:
            rows = await cur.fetchall()
    return {
        "items": [_proof_from_row(r) for r in rows],
        "summary": await _proof_summary(user_id),
    }


async def _create_proof_item_for_user(user_id: str, body: dict) -> dict:
    title = " ".join((body.get("title") or "").strip().split())
    description = " ".join((body.get("description") or "").strip().split())
    evidence = " ".join((body.get("evidence") or "").strip().split())
    if not title:
        title = (description or evidence or "Proof item")[:80]
    item_id = str(uuid.uuid4())
    category = (body.get("category") or "practice").strip()[:40]
    source_type = (body.get("source_type") or "").strip()[:60] or None
    source_id = (body.get("source_id") or "").strip()[:120] or None
    opportunity_type = (body.get("opportunity_type") or "personal_goal").strip()[:60]
    async with get_conn() as db:
        await db.execute(
            """
            INSERT INTO proof_items
                (id, user_id, title, description, category, source_type, source_id,
                 evidence, skill_tags_json, opportunity_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                user_id,
                title[:160],
                description[:1200],
                category,
                source_type,
                source_id,
                evidence[:1200],
                _json_list(body.get("skill_tags")),
                opportunity_type,
            ),
        )
        await db.commit()
    event_id = await record_event(
        user_id,
        "proof_item_created",
        "proof",
        f"Proof created: {title}",
        {"proof_item_id": item_id, "category": category, "source_type": source_type, "opportunity_type": opportunity_type},
        weight=1.4,
    )
    await record_life_event(
        user_id,
        "growth",
        "proof_created",
        "proof",
        title=title,
        summary=description or evidence,
        payload={"proof_item_id": item_id, "category": category, "event_id": event_id},
        confidence=0.85,
        subject_type="proof_item",
        subject_id=item_id,
    )
    await refresh_current_thesis(user_id)
    try:
        await _record_opportunity_unlock_if_ready(user_id)
    except Exception as e:
        log.warning("opportunity unlock check failed user=%s: %s", user_id, e)
    async with get_conn() as db:
        async with db.execute("SELECT * FROM proof_items WHERE id=? AND user_id=?", (item_id, user_id)) as cur:
            row = await cur.fetchone()
    loop_delta = await _loop_delta_for_save(
        user_id,
        event_id,
        topic=detect_topic(f"{title} {description} {evidence}"),
        model_used="proof",
        thesis_updated=True,
        proof_created=True,
        opportunity_unlocked=True,
        training_signal_saved=True,
        next_action="Open Place to see what this proof can unlock.",
        receipt_title="Proof saved",
        receipt_detail="Echo added this evidence to your Place plan.",
    )
    return {"saved": True, "item": _proof_from_row(row), "loop_delta": loop_delta}


@app.post("/v1/proof/items")
async def create_proof_item(request: Request):
    user_id = request.state.user_id
    body = await request.json()
    return await _create_proof_item_for_user(user_id, body)


@app.delete("/v1/proof/items/{item_id}")
async def delete_proof_item(item_id: str, request: Request):
    user_id = request.state.user_id
    async with get_conn() as db:
        await db.execute(
            "UPDATE proof_items SET status='deleted', updated_at=datetime('now') WHERE id=? AND user_id=?",
            (item_id, user_id),
        )
        await db.commit()
    return {"deleted": True, "id": item_id}


@app.post("/v1/proof/from-outcome")
async def create_proof_from_outcome(request: Request):
    user_id = request.state.user_id
    body = await request.json()
    title = body.get("title") or "Proof from outcome"
    description = body.get("description") or body.get("note") or body.get("context") or ""
    body = {
        **body,
        "title": title,
        "description": description,
        "category": body.get("category") or "outcome",
        "source_type": body.get("source_type") or "outcome",
    }
    return await _create_proof_item_for_user(user_id, body)


@app.get("/v1/opportunities")
async def list_opportunities(request: Request):
    user_id = request.state.user_id
    proof_items, outcomes = await asyncio.gather(_proof_items_for_scoring(user_id), _outcomes_for_scoring(user_id))
    async with get_conn() as db:
        async with db.execute(
            """
            SELECT *
            FROM opportunity_goals
            WHERE user_id=? AND status!='deleted'
            ORDER BY updated_at DESC, created_at DESC
            LIMIT 30
            """,
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()
    items = [_opportunity_from_row(r, proof_items=proof_items, outcomes=outcomes) for r in rows]
    saved_titles = {str(item.get("title") or "").lower() for item in items}
    suggestions = [
        item for item in await _opportunity_suggestions(user_id, limit=6)
        if str(item.get("title") or "").lower() not in saved_titles
    ]
    if not items:
        items = suggestions
    else:
        items = (items + suggestions[: max(0, 6 - len(items))])[:8]
    return {
        "items": items,
        "proof_summary": await _proof_summary(user_id),
        "catalog_version": "seed_rules_v1",
    }


@app.post("/v1/opportunities")
async def create_opportunity(request: Request):
    user_id = request.state.user_id
    body = await request.json()
    title = " ".join((body.get("title") or "").strip().split())
    if not title:
        return JSONResponse({"error": "title required"}, status_code=400)
    item_id = str(uuid.uuid4())
    async with get_conn() as db:
        await db.execute(
            """
            INSERT INTO opportunity_goals
                (id, user_id, title, type, description, required_proof_json,
                 missing_proof_json, next_step, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                user_id,
                title[:160],
                (body.get("type") or "personal_goal")[:60],
                (body.get("description") or "")[:1200],
                _json_list(body.get("required_proof")),
                _json_list(body.get("missing_proof")),
                (body.get("next_step") or "")[:500],
                body.get("status") or "active",
            ),
        )
        await db.commit()
    await record_event(
        user_id,
        "opportunity_goal_created",
        "opportunity",
        f"Opportunity goal created: {title}",
        {"opportunity_id": item_id, "type": body.get("type") or "personal_goal"},
        weight=1.2,
    )
    async with get_conn() as db:
        async with db.execute("SELECT * FROM opportunity_goals WHERE id=? AND user_id=?", (item_id, user_id)) as cur:
            row = await cur.fetchone()
    proof_items, outcomes = await asyncio.gather(_proof_items_for_scoring(user_id), _outcomes_for_scoring(user_id))
    return {"saved": True, "item": _opportunity_from_row(row, proof_items=proof_items, outcomes=outcomes)}


@app.post("/v1/opportunities/generate")
async def generate_opportunity(request: Request):
    user_id = request.state.user_id
    generated = await _generated_opportunity(user_id)
    body = {
        "title": generated["title"],
        "type": generated["type"],
        "description": generated["description"],
        "required_proof": generated["required_proof"],
        "missing_proof": generated["missing_proof"],
        "next_step": generated["next_step"],
        "status": "active",
    }
    class _BodyRequest:
        state = request.state
        async def json(self_inner):
            return body
    return await create_opportunity(_BodyRequest())


def _demo_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _is_demo_identity(email: str | None, username: str | None) -> bool:
    email = (email or "").lower()
    username = username or ""
    return email.endswith("@echo.local") or username.startswith(("kaggle_demo", "echo_demo"))


async def _clear_demo_user_data(user_id: str) -> None:
    async with get_conn() as db:
        async with db.execute("SELECT id FROM echo_threads WHERE user_id=?", (user_id,)) as cur:
            thread_ids = [r["id"] for r in await cur.fetchall()]
        for thread_id in thread_ids:
            await db.execute("DELETE FROM thread_evidence WHERE thread_id=?", (thread_id,))
            await db.execute("DELETE FROM thread_escalations WHERE thread_id=?", (thread_id,))

        async with db.execute("SELECT id FROM tournament_runs WHERE user_id=?", (user_id,)) as cur:
            run_ids = [r["id"] for r in await cur.fetchall()]
        for run_id in run_ids:
            await db.execute("DELETE FROM tournament_candidates WHERE run_id=?", (run_id,))

        for table in (
            "confidence",
            "training_pairs",
            "topic_history",
            "checkpoints",
            "user_skills",
            "user_rules",
            "daily_checkins",
            "practice_reps",
            "practice_log",
            "twin_sessions",
            "fcm_tokens",
            "echo_interruptions",
            "echo_revelations",
            "echo_threads",
            "echo_events",
            "life_events",
            "echo_interventions",
            "intervention_settings",
            "clone_missions",
            "shadow_outcomes",
            "proof_items",
            "opportunity_goals",
            "tournament_runs",
            "user_theses",
            "thesis_evidence",
            "teacher_usage",
            "training_runs",
        ):
            await db.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
        await db.commit()


async def _get_or_create_demo_user(body: dict) -> dict:
    stable = body.get("stable", False) is True or body.get("reset", False) is True
    suffix = uuid.uuid4().hex[:8]
    email = (body.get("email") or ("kaggle-demo@echo.local" if stable else f"kaggle-demo-{suffix}@echo.local")).lower()
    username = body.get("username") or ("kaggle_demo" if stable else f"kaggle_demo_{suffix}")
    password = body.get("password") or "echo-demo-password"

    async with get_conn() as db:
        async with db.execute(
            "SELECT id, email, username FROM users WHERE email=? OR username=?",
            (email, username),
        ) as cur:
            row = await cur.fetchone()
    existing = dict(row) if row else None

    if existing:
        if not _is_demo_identity(existing.get("email"), existing.get("username")):
            return {"error": "Refusing to seed over a non-demo user.", "status_code": 409}
        user_id = existing["id"]
        if body.get("reset", True):
            await _clear_demo_user_data(user_id)
        async with get_conn() as db:
            await db.execute(
                "UPDATE users SET email=?, username=?, password=? WHERE id=?",
                (email, username, _demo_password_hash(password), user_id),
            )
            await db.commit()
    else:
        user_id = str(uuid.uuid4())
        async with get_conn() as db:
            await db.execute(
                "INSERT INTO users (id, email, username, password) VALUES (?, ?, ?, ?)",
                (user_id, email, username, _demo_password_hash(password)),
            )
            await db.commit()

    return {
        "id": user_id,
        "email": email,
        "username": username,
        "password": password,
        "token": create_token(user_id),
    }


async def _seed_kaggle_proof_camera_scenario(user_id: str) -> dict:
    turns = [
        ("I fixed a broken water sensor for our school garden, but I only showed it to two friends.", "That is already proof. Write down the before state, what you changed, and one photo or reading that verifies it."),
        ("I avoid posting my work because I feel it is too small.", "Small is not the issue. Invisible is the issue. Make one public-safe artifact this week."),
        ("The internet drops often, so I need things to work offline.", "Then your proof should include offline constraints. That is part of the engineering story, not an excuse."),
        ("I helped my cousin debug a motor driver and he said I explain electronics clearly.", "Save that as feedback proof. A specific quote from someone helped by your work is valuable evidence."),
        ("I want to build a low-cost attendance device, but I keep redesigning it instead of testing.", "Run one test with a rough version. Echo should judge progress by evidence, not elegance."),
        ("I made a spreadsheet of parts prices and found a cheaper sensor option.", "That is decision proof: you reduced cost with a tradeoff. Add the comparison as an artifact."),
        ("My teacher says I should apply for a grant, but I do not know what to show.", "Show shipped artifacts, feedback, and a measured outcome. The story is already forming."),
        ("Today I tested the sensor outside and it stayed stable for 40 minutes.", "Log that as an outcome. Stability over time is stronger than a claim that it works."),
        ("I rewrote my project description so a non-technical person can understand it.", "That is communication proof. Keep the technical diagram and the plain-language version together."),
        ("I keep waiting for a perfect demo before asking anyone to review it.", "Ask for review now. Feedback is the missing proof, not the final polish."),
        ("I found a cheaper moisture sensor but I am not sure that counts.", "It counts if you name the tradeoff: lower cost, same usable signal, and what risk you accepted."),
        ("I tested the prototype without Wi-Fi because our connection was down.", "That is offline-first evidence. Put the constraint in the proof instead of hiding it."),
        ("My classmates ask me to explain circuits when they get stuck.", "That is peer teaching evidence. Ask one person for a short quote while the memory is fresh."),
        ("I do not know whether this is scholarship material.", "Scholarship material is not polish. It is constraint, effort, evidence, and a credible next plan."),
        ("The project looks rough on camera.", "Rough is acceptable if the evidence is clear. Capture the test, the result, and the cost comparison."),
        ("I made a one-page explanation for a non-technical teacher.", "That is translation skill. Save the before/after: technical diagram plus plain-language version."),
        ("I missed the first grant deadline because I did not know what to submit.", "Now build the proof pack before the next deadline: artifact, outcome, feedback, future plan."),
        ("My family cannot afford expensive kit parts.", "Resource constraints can be part of the engineering story when you show the tradeoffs clearly."),
        ("I want someone to tell me if this is real enough.", "Ask for review with a specific question: what is clear, what is missing, and what would make this credible?"),
        ("I can make a short video tomorrow.", "Good. The video only needs to verify the artifact and the outcome. Keep it public-safe."),
        ("I reduced the project cost from 18 dollars to 11 dollars.", "That is a measurable result. Save it as proof with the original and alternate part names."),
        ("I want to apply for the low-connectivity engineering grant.", "You are close. Add a reviewer quote and one future-plan sentence, then apply."),
    ]
    dpo_prompts = [
        (
            "Should I wait until the sensor demo is polished?",
            "Apply after packaging the current evidence and one reviewer quote.",
            "Wait until it looks perfect and avoid showing the rough version.",
        ),
        (
            "What matters most for the grant application?",
            "Show the artifact, measured result, constraint, feedback, and future plan.",
            "Write a generic essay about being passionate.",
        ),
        (
            "How should I explain the low-cost sensor decision?",
            "Name the cost drop, the tradeoff, and the test that proved it still worked.",
            "Do not mention the cheaper part because it may look less professional.",
        ),
        (
            "What feedback should I ask my teacher for?",
            "Ask what is clear, what proof is missing, and whether the impact is understandable.",
            "Ask only if they think you are talented.",
        ),
    ]

    for idx, (user_msg, assistant_msg) in enumerate(turns):
        topic = detect_topic(user_msg)
        await save_pair(
            user_id=user_id,
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            model_used="gemma4_e2b:demo_seed",
            engagement_signal="thumbs_up" if idx % 4 == 0 else "continue",
            topic=topic,
        )
        await update_confidence(user_id, topic, "gemma4_e2b:demo_seed")
        await record_topic(user_id, topic)

    for prompt, chosen, rejected in dpo_prompts:
        topic = detect_topic(prompt)
        await save_pair(user_id, prompt, chosen, "gemma4_e2b:demo_seed", "thumbs_up", topic=topic)
        await save_pair(user_id, prompt, rejected, "gemma4_e2b:demo_seed", "thumbs_down", topic=topic)

    async with get_conn() as db:
        source_month = _dt.utcnow().strftime("%Y-%m")
        for rule in (
            "Push Noor toward public-safe proof before polish.",
            "Treat offline constraints as engineering evidence.",
            "When an opportunity appears, ask what proof is missing.",
        ):
            await db.execute(
                "INSERT INTO user_rules (user_id, rule_text, applies_to, confidence, source_month, active) VALUES (?, ?, 'all', '0.95', ?, 1)",
                (user_id, rule, source_month),
            )
        today = _dt.utcnow().strftime("%Y-%m-%d")
        practice_id = f"demo-practice-{uuid.uuid4().hex[:8]}"
        await db.execute(
            """
            INSERT OR REPLACE INTO practice_reps
                (id, user_id, date, observation, rep_title, rep_instruction, arc_label)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                practice_id,
                user_id,
                today,
                "Noor waits for polish before asking for review.",
                "Ask before polish",
                "Send one rough artifact to one reviewer today. Ask what is clear and what is missing.",
                "Building visible proof",
            ),
        )
        await db.execute(
            "INSERT OR REPLACE INTO practice_log (user_id, rep_id, date, done, note) VALUES (?, ?, ?, 1, ?)",
            (user_id, practice_id, today, "Sent the rough proof page to a peer for review."),
        )
        await db.execute(
            "INSERT OR REPLACE INTO daily_checkins (user_id, date, questions, answers) VALUES (?, ?, ?, ?)",
            (
                user_id,
                today,
                json.dumps(["What did you make visible today?", "What proof is still missing?", "Who can verify this?"], ensure_ascii=True),
                json.dumps(["The sensor stability test.", "Future plan and reviewer quote.", "A peer who learned from the repair."], ensure_ascii=True),
            ),
        )
        await db.commit()

    await record_life_event(
        user_id,
        "learning",
        "prototype_tested",
        "demo_seed",
        title="Garden sensor outdoor stability test",
        summary="The prototype stayed stable outdoors for 40 minutes in a low-connectivity environment.",
        payload={"duration_minutes": 40, "constraint": "offline"},
        confidence=0.9,
        privacy_level="local",
    )
    await record_life_event(
        user_id,
        "feedback",
        "peer_feedback",
        "demo_seed",
        title="Peer observed clear electronics explanation",
        summary="Peer note: Noor helped two younger students understand the pump switch.",
        payload={"audience": "peer", "public_safe": True},
        confidence=0.85,
        privacy_level="proof",
    )
    try:
        await add_raw_memory(
            "Noor builds practical offline-first hardware but delays turning it into visible proof.",
            user_id=user_id,
            source="demo_seed:training",
        )
    except Exception as e:
        log.info("Demo seed memory write skipped user=%s: %s", user_id, e)

    proof_results = []
    proof_results.append(await _create_proof_item_for_user(user_id, {
        "title": "Offline garden sensor field test",
        "description": "Prototype tested outdoors for 40 minutes without network access.",
        "evidence": "40 minute outdoor stability note; rough prototype photo placeholder.",
        "category": "artifact",
        "source_type": "demo_seed",
        "skill_tags": ["hardware prototyping", "testing", "offline-first engineering"],
        "opportunity_type": "scholarship",
    }))
    proof_results.append(await _create_proof_item_for_user(user_id, {
        "title": "Cost reduced from $18 to $11",
        "description": "A part substitution lowered prototype cost while preserving the useful sensor behavior.",
        "evidence": "Parts comparison spreadsheet; alternate moisture sensor; cost drop of 39%.",
        "category": "outcome",
        "source_type": "demo_seed",
        "skill_tags": ["cost tradeoff", "resourcefulness", "testing"],
        "opportunity_type": "scholarship",
    }))
    proof_results.append(await _create_proof_item_for_user(user_id, {
        "title": "Peer feedback on electronics explanation",
        "description": "A peer observed that Noor helps younger students understand electronics clearly.",
        "evidence": "Peer note from field-test review.",
        "category": "feedback",
        "source_type": "demo_seed",
        "skill_tags": ["communication", "peer teaching", "electronics"],
        "opportunity_type": "scholarship",
    }))

    opportunity_id = str(uuid.uuid4())
    async with get_conn() as db:
        await db.execute(
            """
            INSERT INTO opportunity_goals
                (id, user_id, title, type, description, required_proof_json,
                 missing_proof_json, next_step, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')
            """,
            (
                opportunity_id,
                user_id,
                "Low-connectivity engineering grant",
                "scholarship",
                "Turn practical offline hardware proof into a credible scholarship or grant application.",
                _json_list(["shipped artifact", "measured result", "feedback quote", "future plan", "safe share version"]),
                _json_list(["future plan", "safe share version"]),
                "Attach the sensor artifact, cost result, and reviewer quote; then write the future-plan sentence.",
            ),
        )
        run_id = str(uuid.uuid4())
        await db.execute(
            """
            INSERT INTO training_runs
                (id, user_id, lane, status, untrained_pairs, required_pairs, adapter_path, summary_json, finished_at)
            VALUES (?, ?, 'gemma4_e2b', 'complete', ?, ?, ?, ?, datetime('now'))
            """,
            (
                run_id,
                user_id,
                len(turns),
                settings.min_pairs_for_training,
                "demo://gemma4-proof-camera-adapter",
                json.dumps({"eval_score": 0.82, "eval_passed": True, "demo_only": True}, ensure_ascii=True),
            ),
        )
        tournament_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO tournament_runs (id, user_id, prompt, topic, status, winning_style) VALUES (?, ?, ?, ?, 'complete', ?)",
            (tournament_id, user_id, "Should Noor share the rough proof now or wait for polish?", "opportunity", "Builder"),
        )
        for style, response, score in (
            ("Builder", "Create the proof page first.", 0.92),
            ("Strategist", "Apply after one feedback quote.", 0.86),
            ("Examiner", "Stop hiding behind polish.", 0.74),
        ):
            await db.execute(
                "INSERT INTO tournament_candidates (id, run_id, style, response, score, signals_json) VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), tournament_id, style, response, score, "{}"),
            )
        await db.commit()

    await record_event(
        user_id,
        "demo_seeded",
        "demo",
        "Kaggle Proof Camera demo scenario seeded.",
        {"scenario": "proof_camera_noor", "proof_count": len(proof_results), "opportunity_id": opportunity_id},
        weight=1.8,
    )
    await refresh_current_thesis(user_id)
    counts = await _user_data_counts(user_id)
    return {
        "scenario": "proof_camera_noor",
        "training_pairs_seeded": len(turns) + len(dpo_prompts) * 2,
        "proof_items_seeded": len(proof_results),
        "opportunity_id": opportunity_id,
        "counts": counts,
        "next_step": "Run the Kaggle notebook live with this token.",
    }


@app.post("/v1/demo/seed")
async def demo_seed(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    configured = (settings.demo_seed_token or os.getenv("ECHO_DEMO_SEED_TOKEN", "")).strip()
    supplied = (
        request.headers.get("x-echo-demo-token")
        or body.get("demo_seed_token")
        or body.get("token")
        or ""
    ).strip()
    if not configured:
        return JSONResponse(
            {"error": "Demo seed is disabled. Set ECHO_DEMO_SEED_TOKEN to enable it."},
            status_code=404,
        )
    if supplied != configured:
        return JSONResponse({"error": "Invalid demo seed token."}, status_code=403)

    scenario = body.get("scenario") or "proof_camera_maya"
    if scenario != "proof_camera_maya":
        return JSONResponse({"error": "Unknown demo scenario.", "supported": ["proof_camera_maya"]}, status_code=400)

    user = await _get_or_create_demo_user(body)
    if user.get("error"):
        return JSONResponse({"error": user["error"]}, status_code=user.get("status_code", 400))
    seeded = await _seed_kaggle_proof_camera_scenario(user["id"])
    return {
        "seeded": True,
        "scenario": scenario,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "username": user["username"],
            "password": user["password"],
        },
        "token": user["token"],
        "auth_header": f"Bearer {user['token']}",
        "result": seeded,
    }


@app.post("/v1/onboarding/first-read")
async def onboarding_first_read(request: Request):
    """Save the first onboarding answer and return an immediate sparse first read."""
    user_id = request.state.user_id
    body = await request.json()
    answer = " ".join((body.get("answer") or "").strip().split())
    if len(answer) < 8:
        return JSONResponse({"error": "answer too short"}, status_code=400)

    lower = answer.lower()
    if any(k in lower for k in ["avoid", "putting off", "procrast", "stuck", "afraid", "fear"]):
        title = "You notice the thing you avoid."
        read = (
            "You did not start with a random fact. You started with friction. "
            "That usually means there is a part of you already tracking the real problem, even before you act on it."
        )
        next_move = "Name one small action that would make this less vague."
    elif any(k in lower for k in ["build", "create", "idea", "project", "app", "company"]):
        title = "You think in possible futures."
        read = (
            "Your first signal points toward making something real. Echo will watch whether your energy rises "
            "when an idea becomes a concrete next move."
        )
        next_move = "Bring one project decision to the clones."
    elif any(k in lower for k in ["people", "friend", "family", "relationship", "social", "connect"]):
        title = "Connection is part of the signal."
        read = (
            "Your first answer points toward people, not only tasks. Echo will watch where connection gives you energy "
            "and where it drains it."
        )
        next_move = "Tell Echo one recent interaction that stayed in your mind."
    else:
        title = "A first signal is visible."
        read = (
            "Your first answer gives Echo a starting thread. It is too early to call it truth, but it is enough "
            "to begin testing what pulls your attention back."
        )
        next_move = "Keep talking until Echo can test this with clones."

    assistant_msg = f"{title}\n\n{read}\n\nNext move: {next_move}"
    await save_pair(
        user_id=user_id,
        user_msg=f"Onboarding first signal: {answer}",
        assistant_msg=assistant_msg,
        model_used="onboarding:first_read",
        engagement_signal="thumbs_up",
        topic=detect_topic(answer),
    )
    event_id = await record_event(
        user_id,
        "onboarding_first_signal",
        "onboarding",
        title,
        {"answer": answer[:1000], "read": read, "next_move": next_move},
        weight=1.6,
    )
    await record_outcome(
        user_id,
        subject_type="onboarding_first_signal",
        subject_id=event_id,
        event_id=event_id,
        outcome="first_read_created",
        score=1.0,
        note=answer[:500],
    )
    await refresh_current_thesis(user_id)
    return {
        "saved": True,
        "event_id": event_id,
        "title": title,
        "read": read,
        "next_move": next_move,
        "loop_delta": await _loop_delta_for_turn(user_id, event_id, detect_topic(answer), "onboarding:first_read"),
    }


@app.get("/v1/user/onboarding-state")
async def user_onboarding_state(request: Request):
    """Cold-start stage for mobile TODAY variants: day0 / early / building / active."""
    user_id = request.state.user_id
    thesis, counts = await asyncio.gather(
        get_current_thesis(user_id),
        _user_data_counts(user_id),
    )
    today_str = _dt.utcnow().strftime("%Y-%m-%d")
    async with get_conn() as db:
        async with db.execute(
            "SELECT 1 FROM daily_checkins WHERE user_id=? AND date=?",
            (user_id, today_str),
        ) as cur:
            checkin_row = await cur.fetchone()
        async with db.execute(
            "SELECT MIN(created_at) as first_at FROM training_pairs WHERE user_id=?",
            (user_id,),
        ) as cur:
            first_row = await cur.fetchone()
        async with db.execute(
            "SELECT 1 FROM opportunity_goals WHERE user_id=? AND status='active' LIMIT 1",
            (user_id,),
        ) as cur:
            opp_row = await cur.fetchone()

    days_active = 0
    if first_row and first_row["first_at"]:
        try:
            dt = _dt.fromisoformat(first_row["first_at"])
            days_active = max(0, (_dt.utcnow() - dt).days)
        except Exception:
            pass

    has_thesis = bool(thesis.get("title") and thesis.get("statement"))
    has_proof = counts["proof_items"] > 0
    has_practice = counts["outcomes"] > 0

    if days_active == 0 and counts["training_pairs"] == 0:
        stage = "day0"
    elif counts["training_pairs"] < 5 or not has_thesis:
        stage = "early"
    elif not has_proof and counts["training_pairs"] < 20:
        stage = "building"
    else:
        stage = "active"

    return {
        "stage": stage,
        "days_active": days_active,
        "has_thesis": has_thesis,
        "has_proof": has_proof,
        "has_practice": has_practice,
        "has_checkin_today": checkin_row is not None,
        "has_opportunity": opp_row is not None,
        "counts": counts,
        "thesis_title": thesis.get("title") or None,
        "cold_start_complete": stage in ("building", "active"),
    }


@app.post("/v1/tournament/run")
async def tournament_run(request: Request):
    """MVP shadow clone tournament: four styles compete on one situation."""
    from training.tournament_mvp import create_tournament
    user_id = request.state.user_id
    body = await request.json()
    prompt = (body.get("prompt") or body.get("message") or "").strip()
    if not prompt:
        return JSONResponse({"error": "prompt required"}, status_code=400)
    try:
        return await create_tournament(user_id, prompt)
    except Exception as e:
        log.exception("tournament_run failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/v1/tournament/choose")
async def tournament_choose(request: Request):
    """Record the tournament winner and convert it into DPO-ready preference data."""
    from training.tournament_mvp import choose_candidate
    user_id = request.state.user_id
    body = await request.json()
    run_id = body.get("run_id", "")
    candidate_id = body.get("candidate_id", "")
    if not run_id or not candidate_id:
        return JSONResponse({"error": "run_id and candidate_id required"}, status_code=400)
    try:
        return await choose_candidate(user_id, run_id, candidate_id, body.get("outcome", "chosen"))
    except LookupError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        log.exception("tournament_choose failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/v1/training/demo-loop")
async def training_demo_loop(request: Request):
    """Synchronous bounded real Unsloth loop for Kaggle/demo notebooks."""
    user_id = request.state.user_id
    try:
        body = await request.json()
    except Exception:
        body = {}

    def _body_int(name: str, default: int) -> int:
        try:
            return int(body.get(name, default))
        except Exception:
            return default

    lane = _training_lane(body.get("lane") or body.get("model_lane") or request.headers.get("x-echo-model-lane"))
    key = _training_status_key(user_id, lane)
    current_status = _training_status.get(key)
    if current_status in {"running", "running_demo", "loading_adapter"}:
        return JSONResponse(
            {"status": "training_already_running", "lane": lane, "current_status": current_status},
            status_code=409,
        )

    n = await count_untrained_pairs(user_id)
    run_id = await try_create_training_run(user_id, lane, n)
    if not run_id:
        return JSONResponse(
            {"status": "training_already_running", "lane": lane},
            status_code=409,
        )
    _training_status[key] = "running_demo"

    try:
        await record_event(
            user_id,
            "training_demo_started",
            "training",
            "Bounded demo training started.",
            {
                "run_id": run_id,
                "lane": lane,
                "pairs": n,
                "profile": "bounded_demo",
                "max_pairs": _body_int("max_pairs", 8),
                "max_steps": _body_int("max_steps", 8),
            },
            weight=1.3,
        )
        from training.orchestrator import run_demo_training_loop

        evidence = await run_demo_training_loop(
            user_id,
            lane=lane,
            max_pairs=_body_int("max_pairs", 8),
            max_steps=_body_int("max_steps", 8),
            min_pairs=_body_int("min_pairs", 4),
            run_id=run_id,
        )
        evidence["run_id"] = run_id
        status = evidence.get("status") or "failed"
        final_status = status if status.startswith("complete") or status == "failed" else "skipped"
        summary = await get_training_summary(user_id, lane=lane)
        summary["demo_loop"] = {
            "profile": evidence.get("profile"),
            "status": status,
            "bounds": evidence.get("bounds"),
            "dataset": evidence.get("dataset"),
            "promotion": evidence.get("promotion"),
        }
        adapter_path = (evidence.get("unsloth") or {}).get("output_dir") if evidence.get("real_training") else None
        await finish_training_run(
            run_id,
            final_status,
            adapter_path=adapter_path,
            error=(evidence.get("unsloth") or {}).get("error") if status == "failed" else None,
            summary=summary,
        )
        _training_status[key] = status
        try:
            await record_event(
                user_id,
                "training_demo_completed" if status.startswith("complete") else "training_demo_skipped",
                "training",
                "Bounded demo training completed." if status.startswith("complete") else "Bounded demo training did not produce an adapter.",
                {"run_id": run_id, "lane": lane, "status": status, "evidence": evidence},
                weight=1.5 if status.startswith("complete") else 0.7,
            )
        except Exception:
            log.exception("Could not record demo training completion user=%s run=%s", user_id, run_id)
        return evidence
    except Exception as e:
        _training_status[key] = "idle"
        await finish_training_run(run_id, "failed", error=str(e))
        await record_event(
            user_id,
            "training_demo_failed",
            "training",
            "Bounded demo training failed.",
            {"run_id": run_id, "lane": lane, "error": str(e)},
            weight=1.0,
        )
        log.exception("Bounded demo training failed user=%s lane=%s", user_id, lane)
        return JSONResponse({"status": "failed", "lane": lane, "run_id": run_id, "error": str(e)}, status_code=500)


@app.post("/trigger-training")
async def trigger_training(request: Request, background_tasks: BackgroundTasks):
    user_id = request.state.user_id
    try:
        body = await request.json()
    except Exception:
        body = {}
    lane = _training_lane(body.get("lane") or body.get("model_lane") or request.headers.get("x-echo-model-lane"))
    n, summary = await asyncio.gather(count_untrained_pairs(user_id), get_training_summary(user_id, lane=lane))
    if not summary.get("can_train_now"):
        status = "runtime_unavailable" if summary.get("data_ready_for_training") else "not_enough_data"
        return {
            "status": status,
            "pairs": n,
            "required": settings.min_pairs_for_training,
            "dpo_pairs": summary.get("dpo_ready_pairs", 0),
            "dpo_required": summary.get("dpo_required_pairs", 4),
            "dpo_requires_existing_adapter": bool(summary.get("dpo_requires_existing_adapter")),
            "blocked_reason": summary.get("blocked_reason"),
            "runtime_ready_for_training": summary.get("runtime_ready_for_training"),
            "lane": lane,
        }
    key = _training_status_key(user_id, lane)
    run_id = await try_create_training_run(user_id, lane, n)
    if not run_id:
        return JSONResponse(
            {"status": "training_already_running", "lane": lane},
            status_code=409,
        )
    _training_status[key] = "running"
    try:
        await record_event(
            user_id,
            "training_started",
            "training",
            f"Shadow training started with {n} untrained moments.",
            {"run_id": run_id, "pairs": n, "required": settings.min_pairs_for_training, "lane": lane},
            weight=1.5,
        )
        background_tasks.add_task(_run_training_bg, user_id, lane, run_id)
    except Exception as e:
        _training_status[key] = "idle"
        await finish_training_run(run_id, "failed", error=str(e))
        raise
    return {"status": "training_started", "pairs": n, "lane": lane, "run_id": run_id}


async def _run_training_bg(user_id: str, lane: str = "qwen", run_id: str | None = None) -> None:
    from training.coordinator import run_training_cycle
    lane = _training_lane(lane)
    key = _training_status_key(user_id, lane)
    try:
        if not run_id:
            raise RuntimeError("Training run id is required")
        result = await run_training_cycle(user_id, lane, run_id)
        status = result.get("status") or "failed"
        promoted = bool(result.get("promoted"))
        restore_ok = bool(result.get("restore_ok", True))
        await finish_training_run(
            run_id,
            status,
            adapter_path=result.get("promoted_path"),
            error=(
                result.get("reason")
                if status == "skipped"
                else ("Previous adapter restore failed." if status == "complete_restore_failed" else None)
            ),
            summary=result.get("summary") or result,
        )
        try:
            await record_event(
                user_id,
                "training_completed" if promoted else "training_not_promoted",
                "training",
                "Your shadow trained and promoted a stronger Home Brain Adapter."
                if promoted
                else (
                    "Your shadow trained, but restoring the previous adapter needs attention."
                    if not restore_ok
                    else "Your shadow trained, but the previous Home Brain was stronger and was kept."
                ),
                {"run_id": run_id, "lane": lane, "result": result},
                weight=2.0 if promoted else 1.0,
            )
        except Exception:
            log.exception("Could not record training completion event user=%s run=%s", user_id, run_id)
        _training_status[key] = status
    except Exception as e:
        _training_status[key] = "idle"
        if run_id:
            await finish_training_run(run_id, "failed", error=str(e))
        await record_event(
            user_id,
            "training_failed",
            "training",
            "Shadow training failed.",
            {"run_id": run_id, "error": str(e), "lane": lane},
            weight=1.0,
        )
        log.exception("Training failed for user=%s lane=%s", user_id, lane)


@app.post("/context", response_model=ContextResponse)
async def context_endpoint(req: ContextRequest):
    confidence, ctx = await asyncio.gather(
        get_confidence(req.user_id, req.message),
        _context_with_loop(req.user_id, req.message),
    )
    use_local = settings.gemma4_enabled
    recommended_model = "local" if use_local else "openai"
    lora_id = f"gemma4_user_{req.user_id}" if adapter_exists(req.user_id, lane="gemma4_e2b") else settings.gemma4_base_model
    log.info("/context user=%s confidence=%.2f model=%s", req.user_id, confidence, recommended_model)
    return ContextResponse(
        system_injection=ctx["system_injection"],
        recommended_model=recommended_model,
        lora_id=lora_id,
        confidence=confidence,
        loop_state=ctx.get("loop_state", {}),
    )


def _safe_json_loads(value: str | None, fallback):
    if not value:
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback


async def _offline_memories_for_user(user_id: str, limit: int = 80) -> list[dict]:
    try:
        ctx = await get_context(user_id, "offline memory sync profile preferences current read")
        return [
            {
                "id": None,
                "memory": memory,
                "created_at": None,
            }
            for memory in ctx.get("memories", [])[:limit]
        ][:limit]
    except Exception as e:
        log.warning("offline memory export context failed for user=%s: %s", user_id, e)
        return []


@app.get("/v1/offline/export")
async def offline_export(request: Request):
    """Compact Echo state pack for on-device Gemma when the phone goes offline."""
    user_id = request.state.user_id
    lane = _training_lane(request.query_params.get("lane"))

    snapshot, priority, thesis, mission, training, memories = await asyncio.gather(
        loop_snapshot(user_id),
        get_today_priority(user_id),
        get_current_thesis(user_id),
        get_daily_mission(user_id),
        get_training_summary(user_id, lane=lane),
        _offline_memories_for_user(user_id),
    )

    async with get_conn() as db:
        async with db.execute(
            "SELECT username, created_at FROM users WHERE id=?",
            (user_id,),
        ) as cur:
            user_row = await cur.fetchone()

        async with db.execute(
            "SELECT rule_text, applies_to, confidence FROM user_rules "
            "WHERE user_id=? AND active=1 ORDER BY confidence DESC LIMIT 20",
            (user_id,),
        ) as cur:
            rules = [dict(r) for r in await cur.fetchall()]

        async with db.execute(
            "SELECT skill_name, trigger, procedure, user_prefs FROM user_skills "
            "WHERE user_id=? AND active=1 ORDER BY rowid DESC LIMIT 12",
            (user_id,),
        ) as cur:
            skills = [dict(r) for r in await cur.fetchall()]

        async with db.execute(
            "SELECT id, date, observation, rep_title, rep_instruction, arc_label, created_at "
            "FROM practice_reps WHERE user_id=? ORDER BY date DESC LIMIT 1",
            (user_id,),
        ) as cur:
            practice_row = await cur.fetchone()

        async with db.execute(
            "SELECT name, topic, evidence_count, escalation_level, status, last_seen "
            "FROM echo_threads WHERE user_id=? ORDER BY last_seen DESC LIMIT 12",
            (user_id,),
        ) as cur:
            threads = [dict(r) for r in await cur.fetchall()]

        async with db.execute(
            "SELECT title, summary, event_domain, event_type, source, created_at "
            "FROM life_events WHERE user_id=? ORDER BY created_at DESC LIMIT 30",
            (user_id,),
        ) as cur:
            life_events = [dict(r) for r in await cur.fetchall()]

        async with db.execute(
            "SELECT user_msg, assistant_msg, topic, engagement_signal, created_at "
            "FROM training_pairs WHERE user_id=? ORDER BY created_at DESC LIMIT 35",
            (user_id,),
        ) as cur:
            pairs = [dict(r) for r in await cur.fetchall()]

        async with db.execute(
            "SELECT date, answers, created_at FROM daily_checkins "
            "WHERE user_id=? ORDER BY created_at DESC LIMIT 5",
            (user_id,),
        ) as cur:
            checkins = [
                {
                    "date": r["date"],
                    "answers": _safe_json_loads(r["answers"], []),
                    "created_at": r["created_at"],
                }
                for r in await cur.fetchall()
            ]

        counts = {}
        for table in (
            "training_pairs",
            "life_events",
            "echo_events",
            "echo_threads",
            "shadow_outcomes",
            "training_runs",
            "user_rules",
            "daily_checkins",
        ):
            async with db.execute(f"SELECT COUNT(*) as cnt FROM {table} WHERE user_id=?", (user_id,)) as cur:
                row = await cur.fetchone()
                counts[table] = row["cnt"] if row else 0

    rules, skills = await asyncio.gather(
        get_active_rules(user_id),
        get_active_skills(user_id),
    )

    training["latest_run"] = await latest_training_run(user_id, lane)
    training["status"] = _training_status.get(_training_status_key(user_id, lane)) or (
        training["latest_run"]["status"] if training.get("latest_run") else "idle"
    )

    practice = dict(practice_row) if practice_row else None
    public_rules = [
        {k: v for k, v in rule.items() if not k.startswith("_")}
        for rule in rules
    ]
    return {
        "schema_version": 1,
        "exported_at": _dt.utcnow().isoformat(timespec="seconds") + "Z",
        "user": {
            "id": user_id,
            "username": user_row["username"] if user_row else None,
            "created_at": user_row["created_at"] if user_row else None,
        },
        "counts": counts,
        "memories": memories,
        "rules": public_rules,
        "skills": skills,
        "loop_state": {
            "snapshot": snapshot,
            "today_priority": priority,
            "thesis": thesis,
            "mission": mission,
            "practice": practice,
            "training_summary": training,
        },
        "recent": {
            "threads": threads,
            "life_events": life_events,
            "pairs": pairs,
            "checkins": checkins,
        },
        "device_prompt": {
            "model_family": "LiteRT-LM Gemma 4 E2B",
            "instruction": "Use this pack as cached evidence. Do not claim live sync while offline.",
        },
    }


@app.get("/v1/user/rules")
async def get_rules(request: Request):
    user_id = request.state.user_id
    include_inactive = str(request.query_params.get("include_inactive", "")).lower() in {"1", "true", "yes"}
    active_clause = "" if include_inactive else " AND active=1"
    async with get_conn() as db:
        async with db.execute(
            "SELECT id, rule_text, applies_to, confidence, active FROM user_rules "
            f"WHERE user_id=?{active_clause} ORDER BY confidence DESC",
            (user_id,)
        ) as cur:
            rows = await cur.fetchall()
    rules: list[dict] = []
    for row in rows:
        item = dict(row)
        if item.get("active"):
            cleaned = _clean_extracted_rule(item.get("rule_text", ""))
            real_name = re.search(r"\breal name is\s+([a-z][a-z .'-]{1,60})", cleaned or "", flags=re.IGNORECASE)
            if real_name and "forget" in (cleaned or "").lower():
                name = real_name.group(1).split(",")[0].strip(" .'-")
                cleaned = f"Use the user's real name: {name.title()}." if name else cleaned
            if not cleaned:
                continue
            item["rule_text"] = cleaned
        rules.append(item)
    return {"rules": rules}


@app.post("/v1/user/rules")
async def add_rule(request: Request):
    user_id = request.state.user_id
    body = await request.json()
    rule_text = body.get("rule_text", "").strip()
    if not rule_text:
        return JSONResponse({"error": "rule_text required"}, status_code=400)
    applies_to = body.get("applies_to", "all")
    from datetime import datetime as _dt2
    source_month = _dt2.utcnow().strftime("%Y-%m")
    async with get_conn() as db:
        await db.execute(
            "INSERT INTO user_rules (user_id, rule_text, applies_to, confidence, source_month, active) VALUES (?,?,?,?,?,1)",
            (user_id, rule_text, applies_to, "0.99", source_month),
        )
        await db.commit()
    log.info("Rule added user=%s: %s", user_id, rule_text)
    return {"added": True, "rule_text": rule_text}


@app.delete("/v1/user/rules/{rule_id}")
async def delete_rule(rule_id: int, request: Request):
    user_id = request.state.user_id
    async with get_conn() as db:
        await db.execute(
            "DELETE FROM user_rules WHERE id=? AND user_id=?", (rule_id, user_id)
        )
        await db.commit()
    return {"deleted": rule_id}


@app.post("/save")
async def save_endpoint(req: SaveRequest):
    loop_delta = await _do_save(req)
    return {"saved": True, "loop_delta": loop_delta}


async def _do_save(req: SaveRequest) -> dict | None:
    return await _do_save_raw(req.user_id, req.user_message, req.assistant_message, req.model_used, req.engagement_signal)


# Meta-request prefixes that must never be saved to mem0 or training pairs
_SKIP_SAVE_PREFIXES = (
    "generate a concise title",
    "generate a title",
    "create a title for",
    "give this conversation a title",
    "what is a good title for",
)

# Keywords that suggest the user is expressing a behavioral preference
_PREFERENCE_KEYWORDS = (
    "don't", "dont", "stop ", "never ", "always ", "prefer",
    "hate ", "no more", "please don't", "i want you to",
    "i like when", "i don't like", "i dislike", "from now on",
)


def _clean_extracted_rule(text: str) -> str | None:
    cleaned = re.sub(r"<thought>.*?</thought>", "", text or "", flags=re.IGNORECASE | re.DOTALL)
    cleaned = cleaned.replace("```", "")
    cleaned = " ".join(cleaned.split()).strip().strip('"')
    if not cleaned:
        return None
    if cleaned.upper() == "NO_PREFERENCE" or "NO_PREFERENCE" in cleaned.upper():
        return None
    if "<" in cleaned or ">" in cleaned:
        return None
    if len(cleaned) > 180:
        return None
    return cleaned


async def _extract_preferences(user_id: str, user_msg: str) -> None:
    """If user_msg sounds like a behavioral preference, extract and save it as a rule."""
    msg_lower = user_msg.lower()
    if not any(kw in msg_lower for kw in _PREFERENCE_KEYWORDS):
        return
    decision = await should_use_teacher(
        user_id,
        "preference_extraction",
        importance=infer_importance(user_msg),
        prompt=user_msg,
        explicit_user_request=True,
    )
    if not decision.allowed:
        log.info("Preference extraction skipped by teacher policy user=%s reason=%s", user_id, decision.reason)
        return
    prompt = (
        "Does this message express a clear preference about how the AI should respond "
        "(e.g. 'stop using bullet points', 'always be brief', 'never use markdown')?\n\n"
        f"Message: \"{user_msg}\"\n\n"
        "If YES: reply with ONLY the rule as a short imperative sentence under 15 words.\n"
        "If NO: reply with exactly: NO_PREFERENCE"
    )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{settings.teacher_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {settings.llm_api_key}", "Content-Type": "application/json"},
                json={
                    "model": settings.teacher_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 40,
                },
            )
            resp.raise_for_status()
            raw_rule = resp.json()["choices"][0]["message"]["content"].strip()
            rule_text = _clean_extracted_rule(raw_rule)
        await record_teacher_usage(
            user_id,
            "preference_extraction",
            decision.reason,
            {"decision": decision.to_dict()},
        )
        if not rule_text:
            return
        async with get_conn() as db:
            async with db.execute(
                "SELECT id FROM user_rules WHERE user_id=? AND rule_text=? AND active=1",
                (user_id, rule_text),
            ) as cur:
                if await cur.fetchone():
                    return
            source_month = _dt.utcnow().strftime("%Y-%m")
            await db.execute(
                "INSERT INTO user_rules (user_id, rule_text, applies_to, confidence, source_month, active) "
                "VALUES (?,?,?,?,?,1)",
                (user_id, rule_text, "all", "0.95", source_month),
            )
            await db.commit()
        log.info("Auto-extracted rule for user=%s: %s", user_id, rule_text)
    except Exception as e:
        log.warning("_extract_preferences failed for user=%s: %s", user_id, e)


async def _do_save_raw(
    user_id: str,
    user_msg: str,
    assistant_msg: str,
    model_used: str,
    engagement_signal: str = "continue",
) -> dict | None:
    if any(user_msg.lower().startswith(p) for p in _SKIP_SAVE_PREFIXES):
        log.debug("Skipping meta-request (genTitle) for user=%s", user_id)
        return None

    asyncio.create_task(_extract_preferences(user_id, user_msg))

    topic = detect_topic(user_msg)
    tasks = [
        save_pair(
            user_id=user_id,
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            model_used=model_used,
            engagement_signal=engagement_signal,
            topic=topic,
        ),
        update_confidence(user_id, topic, model_used),
        record_topic(user_id, topic),
    ]
    memory_decision = await should_use_teacher(
        user_id,
        "memory_update",
        importance=infer_importance(user_msg),
        prompt=user_msg,
        explicit_user_request=engagement_signal in {"thumbs_up", "saved_signal", "deep"},
    )
    if memory_decision.allowed:
        tasks.append(add_memories(
            [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
            user_id=user_id,
        ))
        await record_teacher_usage(
            user_id,
            "memory_update",
            memory_decision.reason,
            {"topic": topic, "model_used": model_used, "decision": memory_decision.to_dict()},
        )
    else:
        log.debug("Memory update skipped by teacher policy user=%s reason=%s", user_id, memory_decision.reason)

    await asyncio.gather(*tasks)
    event_id = await record_event(
        user_id,
        "conversation_turn",
        "chat",
        f"Saved {topic} turn via {model_used}.",
        {
            "topic": topic,
            "model_used": model_used,
            "engagement_signal": engagement_signal,
            "user_preview": user_msg[:240],
            "assistant_preview": assistant_msg[:240],
        },
        weight=1.0 if engagement_signal == "continue" else 1.5,
    )
    signal_scores = {
        "thumbs_up": 1.0,
        "deep": 0.9,
        "continue": 0.6,
        "quiet": 0.2,
        "thumbs_down": -1.0,
    }
    if engagement_signal != "continue":
        await record_outcome(
            user_id,
            subject_type="conversation_turn",
            subject_id=event_id,
            event_id=event_id,
            outcome=engagement_signal,
            score=signal_scores.get(engagement_signal, 0.5),
            note=f"Feedback signal from {model_used} response.",
        )

    if _meaningful_chat_message(user_msg):
        await refresh_current_thesis(user_id)
        return await _loop_delta_for_turn(user_id, event_id, topic, model_used)

    return {
        "event_id": event_id,
        "topic": topic,
        "model_used": model_used,
        "meaningful": False,
    }


@app.delete("/v1/debug/memories/{memory_id}")
async def delete_memory(memory_id: str, request: Request):
    """Admin: delete a specific memory by ID."""
    from memory.mem0_client import _get_memory
    import asyncio
    m = _get_memory()
    try:
        await asyncio.to_thread(m.delete, memory_id)
        return {"deleted": memory_id}
    except Exception as e:
        return {"error": str(e), "id": memory_id}


@app.get("/v1/user/memories")
async def user_memories(request: Request):
    """Return all stored memories for the authenticated user with IDs for deletion."""
    user_id = request.state.user_id
    from memory.mem0_client import _get_memory
    m = _get_memory()
    try:
        all_r = await asyncio.to_thread(m.get_all, filters={"user_id": user_id}, top_k=1000)
        items = all_r if isinstance(all_r, list) else all_r.get("results", [])
    except Exception as e:
        return {"memories": [], "error": str(e)}
    return {
        "count": len(items),
        "memories": [
            {"id": i.get("id"), "memory": i.get("memory", ""), "created_at": i.get("created_at")}
            for i in items
        ]
    }


@app.post("/v1/user/memories")
async def add_user_memory(request: Request):
    """Directly add a memory for the authenticated user (used by MCP teacher tools)."""
    user_id = request.state.user_id
    body = await request.json()
    memory_text = body.get("memory", "").strip()
    if not memory_text:
        return JSONResponse({"error": "memory required"}, status_code=400)
    await add_raw_memory(memory_text, user_id=user_id, source="manual")
    return {"added": True, "memory": memory_text}


@app.post("/v1/memory/propose")
async def propose_memory(request: Request):
    """Explicit memory consent endpoint used by Talk, Shadow Training, and proof flows."""
    user_id = request.state.user_id
    body = await request.json()
    text = " ".join((body.get("text") or body.get("memory") or "").strip().split())
    if len(text) < 4:
        return JSONResponse({"error": "text required"}, status_code=400)
    source_type = (body.get("source_type") or "talk").strip()[:80]
    privacy = (body.get("privacy") or "private").strip()[:40]
    if privacy not in {"private", "training", "never_share", "proof", "shareable"}:
        privacy = "private"

    memory_saved = await add_raw_memory(text, user_id=user_id, source=f"consent:{source_type}:{privacy}")
    event_id = await record_event(
        user_id,
        "memory_consent_requested",
        "memory",
        "User approved an Echo memory.",
        {"source_type": source_type, "privacy": privacy, "memory_saved": memory_saved, "text_preview": text[:180]},
        weight=1.1,
    )
    await record_life_event(
        user_id=user_id,
        event_domain="memory",
        event_type="memory_approved",
        source=source_type,
        title="Echo memory approved",
        summary=text[:700],
        payload={"event_id": event_id, "privacy": privacy, "memory_saved": memory_saved},
        confidence=0.8,
        privacy_level=privacy,
        subject_type="memory",
        subject_id=event_id,
    )
    return {"saved": True, "memory_saved": memory_saved, "event_id": event_id, "privacy": privacy}


@app.delete("/v1/user/memories")
async def delete_all_user_memories(request: Request):
    """Delete ALL memories for the authenticated user."""
    user_id = request.state.user_id
    from memory.mem0_client import _get_memory
    m = _get_memory()
    try:
        await asyncio.to_thread(m.delete_all, user_id=user_id)
        return {"deleted_all": True, "user_id": user_id}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.delete("/v1/user/memories/{memory_id}")
async def delete_user_memory(memory_id: str, request: Request):
    """Delete a specific memory by ID."""
    from memory.mem0_client import _get_memory
    m = _get_memory()
    try:
        await asyncio.to_thread(m.delete, memory_id)
        return {"deleted": memory_id}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/v1/user/skills")
async def get_skills(request: Request):
    """Return all active skills for the authenticated user."""
    user_id = request.state.user_id
    async with get_conn() as db:
        async with db.execute(
            "SELECT id, skill_name, trigger, procedure, user_prefs, source_week FROM user_skills "
            "WHERE user_id=? AND active=1 ORDER BY rowid DESC",
            (user_id,)
        ) as cur:
            rows = await cur.fetchall()
    return {"skills": [dict(r) for r in rows]}


@app.post("/v1/skills/extract")
async def trigger_skill_extraction(request: Request):
    """Manually trigger skill extraction for the current user (runs immediately, not on schedule)."""
    user_id = request.state.user_id
    from scheduler import _extract_skills
    try:
        await _extract_skills(user_id)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    async with get_conn() as db:
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM user_skills WHERE user_id=? AND active=1", (user_id,)
        ) as cur:
            cnt = (await cur.fetchone())["cnt"]
    return {"ok": True, "skills_extracted": cnt}


@app.post("/v1/proof/seed")
async def trigger_proof_seed(request: Request):
    """Seed proof items from existing life events and thesis evidence (idempotent)."""
    user_id = request.state.user_id
    from scheduler import _seed_proof_from_history
    try:
        seeded = await _seed_proof_from_history(user_id)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "seeded": seeded}


@app.get("/v1/debug/memories")
async def debug_memories(request: Request):
    """Debug: return all stored memories + raw search result for the authenticated user."""
    user_id = request.state.user_id
    from memory.mem0_client import _get_memory
    import asyncio
    m = _get_memory()
    try:
        all_r = await asyncio.to_thread(m.get_all, filters={"user_id": user_id})
        items = all_r if isinstance(all_r, list) else all_r.get("results", [])
    except Exception as e:
        items = []
        return {"error_get_all": str(e)}
    try:
        search_r = await asyncio.to_thread(m.search, "vacation", filters={"user_id": user_id}, top_k=5)
        search_items = search_r if isinstance(search_r, list) else search_r.get("results", [])
    except Exception as e:
        search_items = [{"error": str(e)}]
    return {
        "user_id": user_id,
        "total_stored": len(items),
        "memories": [{"id": i.get("id"), "memory": i.get("memory", "")} for i in items],
        "search_vacation_raw": search_items,
    }


@app.get("/v1/user/stats")
async def user_stats(request: Request):
    """Return conversation count, weeks active, last trained date, and pattern count."""
    user_id = request.state.user_id
    async with get_conn() as db:
        async with db.execute(
            "SELECT COUNT(*) as cnt, MIN(created_at) as first_at FROM training_pairs WHERE user_id=?",
            (user_id,)
        ) as cur:
            tp = await cur.fetchone()
        async with db.execute(
            "SELECT MAX(created_at) as last_at FROM checkpoints WHERE user_id=?",
            (user_id,)
        ) as cur:
            ck = await cur.fetchone()
        async with db.execute(
            "SELECT COUNT(DISTINCT topic) as cnt FROM confidence WHERE user_id=? AND score > 0.05",
            (user_id,)
        ) as cur:
            pf = await cur.fetchone()

    total_pairs = tp["cnt"] if tp else 0
    first_at = tp["first_at"] if tp else None
    last_trained = ck["last_at"] if ck else None
    patterns_found = pf["cnt"] if pf else 0

    weeks_active = 0
    if first_at:
        try:
            dt = _dt.fromisoformat(first_at)
            weeks_active = max(1, int((_dt.utcnow() - dt).days / 7) + 1)
        except Exception:
            pass

    return {
        "total_pairs": total_pairs,
        "weeks_active": weeks_active,
        "last_trained": last_trained,
        "patterns_found": patterns_found,
    }


@app.get("/v1/passport/growth-card")
async def passport_growth_card(request: Request):
    """Privacy-safe shareable Growth Card — no private memory, no raw conversations."""
    user_id = request.state.user_id
    thesis, proof_items, counts = await asyncio.gather(
        get_current_thesis(user_id),
        _proof_items_for_scoring(user_id),
        _user_data_counts(user_id),
    )
    shareable_proof = [
        {
            "title": p["title"],
            "category": p.get("category"),
            "opportunity_type": p.get("opportunity_type"),
        }
        for p in proof_items
        if p.get("category") in ("artifact", "feedback", "outcome") or p.get("opportunity_type")
    ][:5]
    async with get_conn() as db:
        async with db.execute(
            "SELECT topic, score FROM confidence WHERE user_id=? AND score > 0.15 ORDER BY score DESC LIMIT 4",
            (user_id,),
        ) as cur:
            signals = [dict(r) for r in await cur.fetchall()]
    weeks_active = 0
    if counts["training_pairs"] > 0:
        async with get_conn() as db:
            async with db.execute(
                "SELECT MIN(created_at) as first_at FROM training_pairs WHERE user_id=?",
                (user_id,),
            ) as cur:
                tp_row = await cur.fetchone()
        if tp_row and tp_row["first_at"]:
            try:
                dt = _dt.fromisoformat(tp_row["first_at"])
                weeks_active = max(1, int((_dt.utcnow() - dt).days / 7) + 1)
            except Exception:
                pass
    return {
        "direction": thesis.get("title") or "Still forming",
        "confidence_label": thesis.get("confidence_label") or "early",
        "strong_signals": [s["topic"] for s in signals],
        "proof_count": counts["proof_items"],
        "shareable_proof": shareable_proof,
        "weeks_active": weeks_active,
        "card_version": "growth_card_v1",
        "privacy": "no_private_memory_no_conversations",
    }


@app.get("/v1/user/confidence")
async def user_confidence_endpoint(request: Request):
    """Return all confidence scores for the user, ordered highest first."""
    user_id = request.state.user_id
    async with get_conn() as db:
        async with db.execute(
            "SELECT topic, score, updated_at FROM confidence WHERE user_id=? ORDER BY score DESC",
            (user_id,)
        ) as cur:
            rows = await cur.fetchall()
    return {"topics": [dict(r) for r in rows]}


@app.post("/v1/mirror/weekly")
async def mirror_weekly_endpoint(request: Request):
    """Generate a weekly mirror reflection from the user's last 7 days of conversations."""
    user_id = request.state.user_id

    async with get_conn() as db:
        async with db.execute(
            """
            SELECT user_msg, topic, created_at FROM training_pairs
            WHERE user_id=? AND created_at >= datetime('now', '-7 days')
            ORDER BY created_at DESC LIMIT 25
            """,
            (user_id,)
        ) as cur:
            rows = await cur.fetchall()
        async with db.execute(
            "SELECT COUNT(*) as cnt, MIN(created_at) as first_at FROM training_pairs WHERE user_id=?",
            (user_id,)
        ) as cur:
            tp = await cur.fetchone()

    pairs = [dict(r) for r in rows]

    if not pairs:
        return {
            "headline": "No conversations this week yet.",
            "observations": ["Start talking to Echo — your mirror fills as you do."],
            "sit_with_this": "What's one thing you've been meaning to say out loud?",
            "experiment": "Tell Echo something that's been on your mind.",
            "week_number": 1,
        }

    # Compute week number
    week_number = 1
    if tp and tp["first_at"]:
        try:
            dt = _dt.fromisoformat(tp["first_at"])
            week_number = max(1, int((_dt.utcnow() - dt).days / 7) + 1)
        except Exception:
            pass

    messages_summary = "\n".join(
        f"[{r['topic']}] {r['user_msg'][:140]}" for r in pairs[:20]
    )

    prompt = (
        ECHO_FEATURE_HEADER
        + "Based on these messages the user sent this week, write a weekly reflection.\n\n"
        f"User messages:\n{messages_summary}\n\n"
        "Return ONLY valid JSON:\n"
        '{\n'
        '  "headline": "A 1-2 sentence insight about what dominated this week — specific and introspective, not cheesy",\n'
        '  "observations": [\n'
        '    "Sharp specific observation about a pattern (1-2 sentences)",\n'
        '    "Another observation from a different angle",\n'
        '    "Optional third observation if strongly warranted"\n'
        '  ],\n'
        '  "sit_with_this": "A single uncomfortable question for them to carry this week",\n'
        '  "experiment": "A small concrete action to try this coming week",\n'
        '  "proof_this_week": "One artifact, outcome, or action from this week they could save as proof right now — one sentence, concrete"\n'
        '}'
    )

    try:
        content, model_used = await _call_feature_model(
            user_id,
            "weekly_calibration",
            prompt,
            temperature=0.85,
            max_tokens=420,
            importance="normal",
        )
        if not content:
            raise RuntimeError(f"weekly mirror unavailable: {model_used}")

        # Extract JSON — strip any markdown fence if present
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        mirror = json.loads(content.strip())
        mirror["week_number"] = week_number
        mirror["model_used"] = model_used
        return mirror

    except Exception as e:
        log.warning("Mirror generation failed: %s", e)
        sample = pairs[0]["user_msg"][:100] if pairs else "Keep talking to Echo."
        return {
            "headline": "You've been building something this week.",
            "observations": [sample],
            "sit_with_this": "What's driving you right now?",
            "experiment": "Write down the one thing you want to finish next week.",
            "week_number": week_number,
        }


@app.get("/v1/user/insights")
async def user_insights_endpoint(request: Request):
    """Return nightly training summary: turns analyzed, new patterns, latest insight text."""
    user_id = request.state.user_id
    async with get_conn() as db:
        # Turns from last training session (since last checkpoint)
        async with db.execute(
            "SELECT MAX(created_at) as last_ck FROM checkpoints WHERE user_id=?",
            (user_id,)
        ) as cur:
            ck = await cur.fetchone()
        last_ck = ck["last_ck"] if ck and ck["last_ck"] else "1970-01-01"

        async with db.execute(
            "SELECT COUNT(*) as cnt FROM training_pairs WHERE user_id=? AND created_at >= ?",
            (user_id, last_ck)
        ) as cur:
            turns_row = await cur.fetchone()

        # New topics whose confidence was updated since last checkpoint
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM confidence WHERE user_id=? AND updated_at >= ?",
            (user_id, last_ck)
        ) as cur:
            new_patt_row = await cur.fetchone()

        # Recent high-engagement pairs for the insight text
        async with db.execute(
            """SELECT user_msg, assistant_msg, topic FROM training_pairs
               WHERE user_id=? ORDER BY created_at DESC LIMIT 20""",
            (user_id,)
        ) as cur:
            recent = await cur.fetchall()

    turns_analyzed = turns_row["cnt"] if turns_row else 0
    new_patterns = new_patt_row["cnt"] if new_patt_row else 0
    pairs = [dict(r) for r in recent]

    if not pairs:
        return {
            "turns_analyzed": 0,
            "new_patterns": 0,
            "accuracy_delta": "+0.0%",
            "latest_pattern": "Keep talking to Echo — patterns emerge as you do.",
        }

    summary = "\n".join(f"[{p['topic']}] {p['user_msg'][:120]}" for p in pairs[:12])
    prompt = (
        "You are Echo — an AI that watches patterns in how a person thinks and behaves.\n"
        "Based on these recent messages from the user, identify ONE specific behavioral or cognitive pattern "
        "you've noticed. Be precise and personal — like a therapist who's been watching closely.\n\n"
        f"Recent messages:\n{summary}\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "pattern": "One specific pattern in 1-2 sentences — concrete, not generic. Name what you actually see."\n'
        "}"
    )

    try:
        content, model_used = await _call_feature_model(
            user_id,
            "judge_answer",
            prompt,
            temperature=0.75,
            max_tokens=220,
            importance="normal",
        )
        if not content:
            raise RuntimeError(f"insights unavailable: {model_used}")
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content.strip())
        latest_pattern = result.get("pattern", "")
    except Exception as e:
        log.warning("insights pattern generation failed: %s", e)
        latest_pattern = pairs[0]["user_msg"][:120] if pairs else "Keep chatting with Echo."

    return {
        "turns_analyzed": turns_analyzed,
        "new_patterns": new_patterns,
        "accuracy_delta": f"+{min(new_patterns * 0.7, 9.9):.1f}%",
        "latest_pattern": latest_pattern,
    }


@app.post("/v1/emergence")
async def emergence_endpoint(request: Request):
    """Generate an emergence insight — a cross-topic pattern Echo has discovered about the user."""
    user_id = request.state.user_id
    async with get_conn() as db:
        async with db.execute(
            """SELECT user_msg, topic FROM training_pairs
               WHERE user_id=? ORDER BY created_at DESC LIMIT 40""",
            (user_id,)
        ) as cur:
            rows = await cur.fetchall()
        async with db.execute(
            "SELECT topic, score FROM confidence WHERE user_id=? ORDER BY score DESC LIMIT 5",
            (user_id,)
        ) as cur:
            conf_rows = await cur.fetchall()

    pairs = [dict(r) for r in rows]
    top_topics = [dict(r) for r in conf_rows]

    if len(pairs) < 5:
        return {
            "phase_label": "STILL LEARNING",
            "lines": [
                ("Keep talking to Echo.", 3),
                ("Patterns take time to form.", 4),
            ],
            "climax": "I need more of you\nbefore I can say",
            "climax_highlight": "what I see.",
        }

    summary = "\n".join(f"[{p['topic']}] {p['user_msg'][:100]}" for p in pairs[:25])
    topics_str = ", ".join(f"{t['topic']} ({t['score']:.0%})" for t in top_topics)

    prompt = (
        "You are Echo — an AI that finds the single deepest pattern in how a person thinks across all topics.\n"
        "Look at these messages from a user across different topics and find ONE overarching insight "
        "about their core nature — something they likely don't see themselves.\n\n"
        f"Top confidence topics: {topics_str}\n\n"
        f"Recent messages:\n{summary}\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "build_lines": [\n'
        '    "Short poetic line that hints at what you noticed (vague, building)",\n'
        '    "Another line going deeper",\n'
        '    "A third line — more specific",\n'
        '    "Now naming the domain or behavior you see",\n'
        '    "The specific evidence — what they actually do",\n'
        '    "The moment of almost-revelation"\n'
        '  ],\n'
        '  "climax": "The core poetic insight — 2 lines, italic-worthy. Do NOT include the highlight word here.",\n'
        '  "climax_highlight": "The final phrase that names what they are — surprising but true"\n'
        "}"
    )

    try:
        content, model_used = await _call_feature_model(
            user_id,
            "judge_answer",
            prompt,
            temperature=0.9,
            max_tokens=520,
            importance="normal",
        )
        if not content:
            raise RuntimeError(f"emergence unavailable: {model_used}")
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content.strip())

        raw_lines = result.get("build_lines", [])
        grades = [0, 1, 2, 3, 4, 5]
        lines = []
        for i, line in enumerate(raw_lines[:6]):
            grade = grades[min(i, len(grades) - 1)]
            lines.append((line, grade))
            if i == 1 or i == 3:
                lines.append(("", -1))

        weeks = max(1, len(pairs) // 10)
        return {
            "phase_label": f"WEEK {weeks} · EMERGENCE",
            "lines": lines,
            "climax": result.get("climax", ""),
            "climax_highlight": result.get("climax_highlight", ""),
            "model_used": model_used,
        }
    except Exception as e:
        log.warning("Emergence generation failed: %s", e)
        return {
            "phase_label": "EMERGENCE",
            "lines": [
                ("I've been watching", 0),
                ("something.", 1),
                ("", -1),
                ("Across everything you've shared", 2),
                ("a single thread keeps appearing.", 3),
                ("", -1),
                ("The way you think.", 4),
                ("The way you decide.", 5),
                ("The way you come back.", 6),
            ],
            "climax": "You already know what\nI'm going to say.",
            "climax_highlight": "You just haven't said it yet.",
        }


@app.post("/v1/user/talent")
async def user_talent_endpoint(request: Request):
    """Generate the hidden talent narrative — Echo's deepest read on who this person is."""
    user_id = request.state.user_id
    async with get_conn() as db:
        async with db.execute(
            "SELECT COUNT(*) as cnt, MIN(created_at) as first_at FROM training_pairs WHERE user_id=?",
            (user_id,)
        ) as cur:
            tp = await cur.fetchone()
        async with db.execute(
            "SELECT topic, score FROM confidence WHERE user_id=? ORDER BY score DESC LIMIT 8",
            (user_id,)
        ) as cur:
            conf_rows = await cur.fetchall()
        async with db.execute(
            """SELECT user_msg, topic FROM training_pairs
               WHERE user_id=? ORDER BY created_at DESC LIMIT 50""",
            (user_id,)
        ) as cur:
            pair_rows = await cur.fetchall()
        async with db.execute(
            "SELECT rule_text FROM user_rules WHERE user_id=? AND active=1 ORDER BY confidence DESC LIMIT 6",
            (user_id,)
        ) as cur:
            rule_rows = await cur.fetchall()

    total_pairs = tp["cnt"] if tp else 0
    first_at = tp["first_at"] if tp else None
    weeks_active = 1
    if first_at:
        try:
            dt = _dt.fromisoformat(first_at)
            weeks_active = max(1, int((_dt.utcnow() - dt).days / 7) + 1)
        except Exception:
            pass

    if total_pairs < 10:
        return {
            "weeks_active": weeks_active,
            "total_pairs": total_pairs,
            "trait_name": "Still Forming",
            "narrative": "Echo is still learning who you are. Keep talking — the picture takes time to form.",
            "evidence": [],
            "rarity_pct": None,
            "what_it_means": "Come back after a few more conversations.",
        }

    conf_list = [dict(r) for r in conf_rows]
    pairs = [dict(r) for r in pair_rows]
    rules = [r["rule_text"] for r in rule_rows]

    topics_str = ", ".join(f"{c['topic']} ({c['score']:.0%})" for c in conf_list)
    msgs_str = "\n".join(f"[{p['topic']}] {p['user_msg'][:120]}" for p in pairs[:30])
    rules_str = "\n".join(f"- {r}" for r in rules) if rules else "No rules extracted yet."

    prompt = (
        ECHO_FEATURE_HEADER
        + "You have been watching this person closely. You are about to reveal what you found: "
        "their hidden cognitive talent. This is NOT a generic personality type — it is the "
        "specific, unusual way THIS person thinks.\n\n"
        f"Weeks watching: {weeks_active}\n"
        f"Conversations analyzed: {total_pairs}\n"
        f"Top confidence topics: {topics_str}\n\n"
        f"Behavioral rules extracted:\n{rules_str}\n\n"
        f"Recent messages sample:\n{msgs_str}\n\n"
        "Return ONLY valid JSON with these fields:\n"
        "{\n"
        '  "trait_name": "2-4 word name for their core gift (e.g. Systems Architect, Pattern Synthesizer, Reluctant Visionary)",\n'
        '  "narrative": "3-4 paragraph flowing narrative — intimate, direct, specific. Start with the time watched. Name what you see. Give evidence. Tell them what it means for them specifically.",\n'
        '  "evidence": ["Specific observation 1", "Specific observation 2", "Specific observation 3"],\n'
        '  "rarity_pct": 4,\n'
        '  "what_it_means": "One paragraph: what this trait means for who they become and what they should pursue.",\n'
        '  "what_to_do_now": "The single most concrete thing they should do in the next 7 days based on this trait — something that produces a real proof item."\n'
        "}"
    )

    try:
        content, model_used = await _call_feature_model(
            user_id,
            "weekly_calibration",
            prompt,
            temperature=0.85,
            max_tokens=760,
            importance="high",
        )
        if not content:
            raise RuntimeError(f"talent read unavailable: {model_used}")
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content.strip())
        response = {
            "weeks_active": weeks_active,
            "total_pairs": total_pairs,
            "trait_name": result.get("trait_name", ""),
            "narrative": result.get("narrative", ""),
            "evidence": result.get("evidence", []),
            "rarity_pct": result.get("rarity_pct", 5),
            "what_it_means": result.get("what_it_means", ""),
            "model_used": model_used,
        }
        await record_event(
            user_id,
            "talent_read",
            "talent",
            f"Echo named a hidden talent: {response['trait_name']}.",
            {
                "trait_name": response["trait_name"],
                "evidence": response["evidence"],
                "rarity_pct": response["rarity_pct"],
            },
            weight=1.4,
        )
        await refresh_current_thesis(user_id)
        return response
    except Exception as e:
        log.warning("Talent generation failed: %s", e)
        return {
            "weeks_active": weeks_active,
            "total_pairs": total_pairs,
            "trait_name": "Deep Pattern Thinker",
            "narrative": f"I've been watching you for {weeks_active} weeks across {total_pairs} conversations. Something keeps appearing that most people miss in themselves.",
            "evidence": [],
            "rarity_pct": 5,
            "what_it_means": "Keep talking to Echo — the picture gets clearer every day.",
        }


@app.get("/v1/user/notable-quote")
async def notable_quote_endpoint(request: Request):
    """Return the most memorable/unusual thing the user ever said, extracted by the teacher LLM."""
    user_id = request.state.user_id
    async with get_conn() as db:
        async with db.execute(
            """SELECT user_msg, created_at FROM training_pairs
               WHERE user_id=? AND perplexity >= 0.6
               ORDER BY perplexity DESC LIMIT 80""",
            (user_id,)
        ) as cur:
            rows = await cur.fetchall()

    if not rows:
        return {"quote": None, "date": None}

    samples = "\n".join(f'- "{r["user_msg"][:180]}"' for r in rows[:60])
    prompt = (
        ECHO_FEATURE_HEADER
        + "Find the single most memorable, surprising, or self-revealing statement this person made — "
        "something that sounds like it came from real depth, not small talk. "
        "It should be the kind of thing that would stop someone mid-scroll.\n\n"
        f"Statements:\n{samples}\n\n"
        'Return JSON: {"quote": "exact quote or close paraphrase", "why": "one sentence on why this stands out"}\n'
        "Return only valid JSON."
    )
    try:
        resp, model_used = await _call_feature_model(
            user_id,
            "judge_answer",
            prompt,
            temperature=0.65,
            max_tokens=180,
            importance="normal",
        )
        if not resp:
            raise RuntimeError(f"notable quote unavailable: {model_used}")
        content = resp.strip().lstrip("```json").rstrip("```").strip()
        result = json.loads(content)
        # Find approximate date of the quote
        quote_text = result.get("quote", "")
        date_str = None
        for r in rows:
            if quote_text[:30].lower() in r["user_msg"].lower():
                date_str = r["created_at"]
                break
        if not date_str and rows:
            date_str = rows[0]["created_at"]
        return {"quote": result.get("quote"), "why": result.get("why"), "date": date_str, "model_used": model_used}
    except Exception as e:
        log.warning("Notable quote failed: %s", e)
        return {"quote": None, "date": None}


@app.post("/v1/user/experiment")
async def user_experiment_endpoint(request: Request):
    """Generate a personalized behavioral experiment based on user's patterns and rules."""
    user_id = request.state.user_id
    async with get_conn() as db:
        async with db.execute(
            "SELECT topic, score FROM confidence WHERE user_id=? ORDER BY score DESC LIMIT 6",
            (user_id,)
        ) as cur:
            conf_rows = await cur.fetchall()
        async with db.execute(
            "SELECT rule_text FROM user_rules WHERE user_id=? AND active=1 ORDER BY confidence DESC LIMIT 8",
            (user_id,)
        ) as cur:
            rule_rows = await cur.fetchall()
        async with db.execute(
            """SELECT user_msg FROM training_pairs WHERE user_id=? AND perplexity >= 0.6
               AND topic IN ('personal', 'work', 'general') ORDER BY rowid DESC LIMIT 40""",
            (user_id,)
        ) as cur:
            pair_rows = await cur.fetchall()

    topics = [f"{r['topic']} ({r['score']:.0%})" for r in conf_rows]
    rules = [r["rule_text"] for r in rule_rows]
    recent = [r["user_msg"][:120] for r in pair_rows[:20]]

    prompt = (
        ECHO_FEATURE_HEADER
        + "Design ONE concrete 7-day behavioral experiment for this person based on their patterns.\n\n"
        f"Their strongest topics: {', '.join(topics)}\n"
        f"Rules Echo learned about them:\n" + "\n".join(f"- {r}" for r in rules) + "\n\n"
        f"Recent things they said:\n" + "\n".join(f'- "{m}"' for m in recent) + "\n\n"
        "Return JSON:\n"
        "{\n"
        '  "trigger": "one sentence: what pattern Echo noticed that motivated this",\n'
        '  "hypothesis": "one sentence: what Echo predicts will happen if they do this",\n'
        '  "title": "experiment title (5-8 words)",\n'
        '  "body": "2-3 paragraph description with the exact daily action to take",\n'
        '  "followup": "how Echo will check in during the 7 days",\n'
        '  "proof_moment": "one sentence: the specific thing they could save as proof when the experiment works"\n'
        "}\n"
        "Return only valid JSON. Make it specific and actionable, not generic."
    )
    try:
        resp, model_used = await _call_feature_model(
            user_id,
            "judge_answer",
            prompt,
            temperature=0.75,
            max_tokens=420,
            importance="normal",
        )
        if not resp:
            raise RuntimeError(f"experiment unavailable: {model_used}")
        content = resp.strip().lstrip("```json").rstrip("```").strip()
        result = json.loads(content)
        return {
            "trigger": result.get("trigger", ""),
            "hypothesis": result.get("hypothesis", ""),
            "title": result.get("title", ""),
            "body": result.get("body", ""),
            "followup": result.get("followup", "I'll check in every 2 days."),
            "duration_days": 7,
            "model_used": model_used,
        }
    except Exception as e:
        log.warning("Experiment generation failed: %s", e)
        return {
            "trigger": "Echo noticed a pattern worth exploring.",
            "hypothesis": "A small shift in behavior will reveal something true about you.",
            "title": "Speak without hedging. Just once a day.",
            "body": "Once a day — in a meeting, a message, or a conversation — say your point without 'I think,' 'maybe,' or 'I could be wrong.'\n\nNot every time. Just once. See what happens to the room.",
            "followup": "I'll check in every 2 days.",
            "duration_days": 7,
        }


@app.get("/v1/daily/questions")
async def daily_questions(request: Request):
    """Return 3 personalized evening check-in questions for the user."""
    user_id = request.state.user_id

    async with get_conn() as db:
        async with db.execute(
            "SELECT user_msg, topic FROM training_pairs WHERE user_id=? ORDER BY created_at DESC LIMIT 20",
            (user_id,)
        ) as cur:
            pair_rows = await cur.fetchall()
        async with db.execute(
            "SELECT rule_text FROM user_rules WHERE user_id=? AND active=1 ORDER BY confidence DESC LIMIT 5",
            (user_id,)
        ) as cur:
            rule_rows = await cur.fetchall()
        async with db.execute(
            "SELECT questions FROM daily_checkins WHERE user_id=? ORDER BY created_at DESC LIMIT 3",
            (user_id,)
        ) as cur:
            checkin_rows = await cur.fetchall()

    pairs = [dict(r) for r in pair_rows]
    rules = [r["rule_text"] for r in rule_rows]

    _fallback = [
        "What's one moment from today you keep replaying in your head?",
        "Did you say what you actually meant, or hold something back?",
        "What does today tell you about yourself that yesterday didn't?",
    ]

    if not pairs:
        return {"questions": _fallback}

    recent_topics = ", ".join(sorted({p["topic"] for p in pairs[:10]}))
    msgs_summary = "\n".join(f"[{p['topic']}] {p['user_msg'][:100]}" for p in pairs[:12])
    rules_str = "\n".join(f"- {r}" for r in rules) if rules else "None yet."

    prev_qs: list[str] = []
    for row in checkin_rows:
        try:
            prev_qs.extend(json.loads(row["questions"]))
        except Exception:
            pass
    prev_str = "\n".join(f"- {q}" for q in prev_qs[:6]) if prev_qs else "None yet."

    prompt = (
        ECHO_FEATURE_HEADER
        + "Generate 3 evening check-in questions for this user.\n\n"
        f"Recent conversation topics: {recent_topics}\n"
        f"Recent messages:\n{msgs_summary}\n\n"
        f"Rules Echo knows about them:\n{rules_str}\n\n"
        f"Previous check-in questions to AVOID repeating:\n{prev_str}\n\n"
        "Rules for the 3 questions:\n"
        "Q1: About a specific moment or action TODAY — grounded in their actual recent topics\n"
        "Q2: About self-honesty or what they held back — more vulnerable\n"
        "Q3: Meta-cognitive — 'What from today could become evidence of something real about you?' "
        "(ask this version at least 2x per week; otherwise ask what today reveals about who they're becoming)\n\n"
        "Style: Under 20 words each, slightly uncomfortable. No fluff, no affirmations.\n\n"
        'Return ONLY valid JSON: {"questions": ["Q1", "Q2", "Q3"]}'
    )

    try:
        resp, model_used = await _call_feature_model(
            user_id,
            "judge_answer",
            prompt,
            temperature=0.7,
            max_tokens=220,
            importance="normal",
        )
        if not resp:
            raise RuntimeError(f"daily questions unavailable: {model_used}")
        content = resp.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content.strip())
        questions = result.get("questions", [])
        if len(questions) >= 3:
            return {"questions": questions[:3], "model_used": model_used}
    except Exception as e:
        log.warning("daily_questions generation failed: %s", e)

    return {"questions": _fallback}


@app.get("/v1/daily/checkin/status")
async def daily_checkin_status(request: Request):
    """Return whether today's check-in is done and how it feeds opportunity readiness."""
    user_id = request.state.user_id
    today_str = _dt.utcnow().strftime("%Y-%m-%d")
    async with get_conn() as db:
        async with db.execute(
            "SELECT 1 FROM daily_checkins WHERE user_id=? AND date=?",
            (user_id, today_str),
        ) as cur:
            row = await cur.fetchone()
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM daily_checkins WHERE user_id=?",
            (user_id,),
        ) as cur:
            total_row = await cur.fetchone()
    done = row is not None
    total = total_row["cnt"] if total_row else 0
    return {
        "done": done,
        "date": today_str,
        "total_checkins": total,
        "today_checkin_affects_readiness": done,
        "checkin_streak_value": min(total * 2, 20),
    }


@app.post("/v1/daily/checkin")
async def submit_daily_checkin(request: Request, background_tasks: BackgroundTasks):
    """Save evening check-in Q&A as training pairs and return synthesis bullets."""
    user_id = request.state.user_id
    body = await request.json()
    qas: list[dict] = body.get("qas", [])
    date_str = body.get("date", _dt.utcnow().strftime("%Y-%m-%d"))

    if not qas:
        return JSONResponse({"error": "qas required"}, status_code=400)

    questions = [qa.get("q", "") for qa in qas]
    answers = [qa.get("a", "") for qa in qas]

    # Persist check-in record so questions endpoint can avoid repetition tomorrow
    async with get_conn() as db:
        await db.execute(
            "INSERT OR REPLACE INTO daily_checkins (user_id, date, questions, answers) VALUES (?, ?, ?, ?)",
            (user_id, date_str, json.dumps(questions), json.dumps(answers)),
        )
        await db.commit()

    # Save pairs in background — don't block the response
    background_tasks.add_task(_save_checkin_pairs, user_id, qas)

    # Generate synthesis bullets from answers
    qa_text = "\n".join(
        f"Q: {qa.get('q', '')}\nA: {qa.get('a', '')}" for qa in qas
    )
    prompt = (
        ECHO_FEATURE_HEADER
        + f"A user just completed their evening check-in:\n\n{qa_text}\n\n"
        "Write 3 short insight bullets about what you learned about them tonight.\n"
        "Style: Direct, Echo's voice ('You showed...', 'I noticed...', 'Tonight revealed...')\n"
        "Each bullet: 1 sentence, concrete, personal. No generic affirmations.\n"
        "At least one bullet should connect to what this means for proof or opportunity — "
        "something they're building toward, not just observing about themselves.\n\n"
        'Return ONLY valid JSON: {"synthesis": ["bullet1", "bullet2", "bullet3"]}'
    )

    try:
        resp, model_used = await _call_feature_model(
            user_id,
            "judge_answer",
            prompt,
            temperature=0.65,
            max_tokens=240,
            importance="normal",
        )
        if not resp:
            raise RuntimeError(f"checkin synthesis unavailable: {model_used}")
        content = resp.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content.strip())
        synthesis = result.get("synthesis", [])
        if len(synthesis) >= 3:
            return {"synthesis": synthesis[:3], "training_pairs_added": len(qas) * 3, "model_used": model_used}
    except Exception as e:
        log.warning("checkin synthesis failed: %s", e)

    return {
        "synthesis": [a[:120] for a in answers[:3]],
        "training_pairs_added": len(qas) * 3,
    }


@app.get("/v1/user/report")
async def user_report(request: Request):
    """Single call returning everything the Reflect tab needs."""
    user_id = request.state.user_id
    from datetime import timedelta
    weekday = _dt.utcnow().weekday()
    week_start = (_dt.utcnow() - timedelta(days=weekday)).strftime("%Y-%m-%d")

    async with get_conn() as db:
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM training_pairs WHERE user_id=?", (user_id,)
        ) as cur:
            pairs_row = await cur.fetchone()
        total_pairs = (pairs_row["cnt"] if pairs_row else 0)

        async with db.execute(
            "SELECT MIN(created_at) as first FROM training_pairs WHERE user_id=?", (user_id,)
        ) as cur:
            first_row = await cur.fetchone()
        weeks = 0
        if first_row and first_row["first"]:
            try:
                first_dt = _dt.fromisoformat(first_row["first"])
                weeks = max(1, int((_dt.utcnow() - first_dt).days / 7) + 1)
            except Exception:
                pass

        async with db.execute(
            "SELECT rule_text FROM user_rules WHERE user_id=? AND active=1 ORDER BY confidence DESC LIMIT 3",
            (user_id,)
        ) as cur:
            rules = [r["rule_text"] for r in await cur.fetchall()]

        async with db.execute(
            "SELECT user_msg, created_at FROM training_pairs WHERE user_id=? ORDER BY created_at DESC LIMIT 3",
            (user_id,)
        ) as cur:
            mem_rows = [dict(r) for r in await cur.fetchall()]

        async with db.execute(
            "SELECT MAX(created_at) as last FROM checkpoints WHERE user_id=?", (user_id,)
        ) as cur:
            trained_row = await cur.fetchone()
        last_trained = (trained_row["last"] if trained_row and trained_row["last"] else None)

        async with db.execute(
            "SELECT COUNT(*) as cnt FROM practice_log pl "
            "JOIN practice_reps pr ON pl.rep_id=pr.id "
            "WHERE pl.user_id=? AND pr.date>=? AND pl.done=1",
            (user_id, week_start)
        ) as cur:
            wc_row = await cur.fetchone()
        week_completions = (wc_row["cnt"] if wc_row else 0)

        async with db.execute(
            "SELECT AVG(score) as avg FROM confidence WHERE user_id=?", (user_id,)
        ) as cur:
            conf_row = await cur.fetchone()
        avg_confidence = float((conf_row["avg"] or 0.0) if conf_row else 0.0)

        async with db.execute(
            "SELECT user_msg, topic FROM training_pairs WHERE user_id=? ORDER BY created_at DESC LIMIT 30",
            (user_id,)
        ) as cur:
            pair_rows = [dict(r) for r in await cur.fetchall()]

    headline, observations, sit_with_this = "", [], ""

    if pair_rows and rules:
        msgs = "\n".join(f"[{p['topic']}] {p['user_msg'][:120]}" for p in pair_rows[:20])
        rules_str = "\n".join(f"- {r}" for r in rules)
        prompt = (
            ECHO_FEATURE_HEADER
            + f"Rules:\n{rules_str}\n\nRecent messages:\n{msgs}\n\n"
            "Write a weekly reflection:\n"
            "- headline: One striking sentence about what you saw this week (under 12 words, start with 'You')\n"
            "- observations: 3 specific things you noticed. Each 1-2 sentences. Start with the fact.\n"
            "  At least one observation must connect to what this means for proof or opportunity.\n"
            "- sit_with_this: One uncomfortable question for them (under 20 words)\n\n"
            'Return ONLY valid JSON: {"headline":"...","observations":["...","...","..."],"sit_with_this":"..."}'
        )
        try:
            resp, model_used = await _call_feature_model(
                user_id,
                "weekly_calibration",
                prompt,
                temperature=0.7,
                max_tokens=420,
                importance="normal",
            )
            if not resp:
                raise RuntimeError(f"user report unavailable: {model_used}")
            content = resp.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            parsed = json.loads(content.strip())
            headline = parsed.get("headline", "")
            observations = parsed.get("observations", [])
            sit_with_this = parsed.get("sit_with_this", "")
        except Exception as e:
            log.warning("user_report generation failed: %s", e)

    return {
        "total_pairs": total_pairs,
        "weeks": weeks,
        "last_trained": last_trained,
        "week_completions": week_completions,
        "avg_confidence": round(avg_confidence, 2),
        "rules": rules,
        "recent_messages": [{"text": m["user_msg"][:100], "date": m["created_at"]} for m in mem_rows],
        "headline": headline,
        "observations": observations,
        "sit_with_this": sit_with_this,
    }


@app.post("/v1/user/fcm-token")
async def register_fcm_token(request: Request):
    """Store or update a device's FCM token for push notifications."""
    user_id = request.state.user_id
    body = await request.json()
    token = body.get("token", "").strip()
    platform = body.get("platform", "android")
    if not token:
        return JSONResponse({"error": "token required"}, status_code=400)
    async with get_conn() as db:
        await db.execute(
            "INSERT OR REPLACE INTO fcm_tokens (user_id, token, platform, updated_at) VALUES (?, ?, ?, datetime('now'))",
            (user_id, token, platform)
        )
        await db.commit()
    return {"registered": True}


@app.get("/v1/user/signal")
async def user_signal(request: Request):
    """One sentence that captures who the user is right now."""
    user_id = request.state.user_id

    async with get_conn() as db:
        async with db.execute(
            "SELECT rule_text FROM user_rules WHERE user_id=? AND active=1 ORDER BY confidence DESC LIMIT 5",
            (user_id,)
        ) as cur:
            rules = [r["rule_text"] for r in await cur.fetchall()]
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM training_pairs WHERE user_id=?",
            (user_id,)
        ) as cur:
            row = await cur.fetchone()
            total_pairs = row["cnt"] if row else 0
        async with db.execute(
            "SELECT MIN(created_at) as first FROM training_pairs WHERE user_id=?",
            (user_id,)
        ) as cur:
            first_row = await cur.fetchone()

    weeks = 0
    if first_row and first_row["first"]:
        try:
            first_dt = _dt.fromisoformat(first_row["first"])
            weeks = max(1, int((_dt.utcnow() - first_dt).days / 7) + 1)
        except Exception:
            pass

    if not rules:
        return {"signal": None, "total_pairs": total_pairs, "weeks": weeks}

    rules_str = "\n".join(f"- {r}" for r in rules)
    prompt = (
        f"Based on these behavioral patterns you observed in a user:\n{rules_str}\n\n"
        "Write ONE sentence capturing the most essential truth about how they think or operate.\n"
        "Start with 'You'. Be specific, not generic. Under 18 words.\n"
        "Example: 'You redesign the frame before solving the problem.'\n"
        "Return ONLY the sentence, no quotes, no explanation."
    )

    try:
        resp, model_used = await _call_feature_model(
            user_id,
            "judge_answer",
            prompt,
            temperature=0.55,
            max_tokens=80,
            importance="normal",
        )
        if not resp:
            raise RuntimeError(f"user signal unavailable: {model_used}")
        signal = resp.strip().strip('"\'')
        return {"signal": signal, "total_pairs": total_pairs, "weeks": weeks, "model_used": model_used}
    except Exception as e:
        log.warning("user_signal generation failed: %s", e)

    return {"signal": rules[0] if rules else None, "total_pairs": total_pairs, "weeks": weeks}


@app.get("/v1/practice/today")
async def practice_today(request: Request):
    """Return today's behavioral rep, generating if not already cached for this user+date."""
    user_id = request.state.user_id
    from datetime import timedelta
    today = _dt.utcnow().strftime("%Y-%m-%d")
    weekday = _dt.utcnow().weekday()
    week_start = (_dt.utcnow() - timedelta(days=weekday)).strftime("%Y-%m-%d")

    async def _week_completions(db) -> int:
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM practice_log pl "
            "JOIN practice_reps pr ON pl.rep_id=pr.id "
            "WHERE pl.user_id=? AND pr.date>=? AND pl.done=1",
            (user_id, week_start)
        ) as cur:
            row = await cur.fetchone()
        return row["cnt"] if row else 0

    # Check for cached rep
    async with get_conn() as db:
        async with db.execute(
            "SELECT * FROM practice_reps WHERE user_id=? AND date=?",
            (user_id, today)
        ) as cur:
            existing = await cur.fetchone()

        if existing:
            ex = dict(existing)
            async with db.execute(
                "SELECT done FROM practice_log WHERE user_id=? AND rep_id=?",
                (user_id, ex["id"])
            ) as cur:
                log_row = await cur.fetchone()
            wc = await _week_completions(db)
            return {
                "rep_id": ex["id"],
                "observation": ex["observation"],
                "rep_title": ex["rep_title"],
                "rep_instruction": ex["rep_instruction"],
                "arc_label": ex.get("arc_label"),
                "logged": log_row is not None,
                "done": bool(log_row["done"]) if log_row else None,
                "week_completions": wc,
            }

    # Pull context for generation
    async with get_conn() as db:
        async with db.execute(
            "SELECT rule_text FROM user_rules WHERE user_id=? AND active=1 ORDER BY confidence DESC LIMIT 5",
            (user_id,)
        ) as cur:
            rules = [r["rule_text"] for r in await cur.fetchall()]
        async with db.execute(
            "SELECT answers FROM daily_checkins WHERE user_id=? ORDER BY created_at DESC LIMIT 3",
            (user_id,)
        ) as cur:
            checkin_rows = await cur.fetchall()
        async with db.execute(
            "SELECT observation, rep_title FROM practice_reps WHERE user_id=? ORDER BY created_at DESC LIMIT 5",
            (user_id,)
        ) as cur:
            prev_reps = [dict(r) for r in await cur.fetchall()]

    recent_answers: list[str] = []
    for row in checkin_rows:
        try:
            recent_answers.extend(json.loads(row["answers"]))
        except Exception:
            pass

    rules_str = "\n".join(f"- {r}" for r in rules) if rules else "None yet."
    answers_str = "\n".join(f"- {a}" for a in recent_answers[:6]) if recent_answers else "None yet."
    prev_str = "\n".join(f"- {r['rep_title']}: {r['observation']}" for r in prev_reps) if prev_reps else "None yet."

    _fallback = {
        "observation": "You hold back your strongest thought until the end.",
        "rep_title": "Lead with the answer",
        "rep_instruction": "In one conversation today, say your conclusion first — then explain. Not 'I think maybe...' Just: 'Here's what I see.' Notice what changes.",
        "arc_label": "Building epistemic confidence",
    }

    prompt = (
        "You are Echo — a talent scout who has watched this user closely.\n\n"
        f"Rules you've learned about them:\n{rules_str}\n\n"
        f"Recent check-in answers:\n{answers_str}\n\n"
        f"Previous reps (don't repeat):\n{prev_str}\n\n"
        "Generate today's practice rep — one specific behavioral micro-action.\n\n"
        "Fields:\n"
        "- observation: 1 sentence (≤15 words) about something Echo noticed. Specific, not generic. Start with 'You'.\n"
        "- rep_title: 2-4 words, active verb phrase. (e.g. 'Claim your perspective', 'Lead with the answer')\n"
        "- rep_instruction: 2-3 sentences. Concrete enough to execute today. When, how, exactly what to say or do.\n"
        "- arc_label: 3-5 words, the long-term pattern this builds. (e.g. 'Building epistemic courage')\n\n"
        'Return ONLY valid JSON: {"observation":"...","rep_title":"...","rep_instruction":"...","arc_label":"..."}'
    )

    rep_data = _fallback.copy()
    try:
        resp, model_used = await _call_feature_model(
            user_id,
            "judge_answer",
            prompt,
            temperature=0.65,
            max_tokens=320,
            importance="normal",
        )
        if not resp:
            raise RuntimeError(f"practice rep unavailable: {model_used}")
        content = resp.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content.strip())
        rep_data = {k: parsed.get(k, _fallback[k]) for k in _fallback}
    except Exception as e:
        log.warning("practice_today generation failed: %s", e)

    rep_id = str(uuid.uuid4())
    async with get_conn() as db:
        await db.execute(
            "INSERT OR IGNORE INTO practice_reps "
            "(id, user_id, date, observation, rep_title, rep_instruction, arc_label) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (rep_id, user_id, today,
             rep_data["observation"], rep_data["rep_title"],
             rep_data["rep_instruction"], rep_data["arc_label"])
        )
        await db.commit()
        wc = await _week_completions(db)

    return {
        "rep_id": rep_id,
        **rep_data,
        "logged": False,
        "done": None,
        "week_completions": wc,
    }


@app.post("/v1/practice/log")
async def practice_log_endpoint(request: Request):
    """Mark today's rep done or skipped."""
    user_id = request.state.user_id
    body = await request.json()
    rep_id = body.get("rep_id", "")
    done = bool(body.get("done", True))
    from datetime import timedelta
    today = _dt.utcnow().strftime("%Y-%m-%d")
    weekday = _dt.utcnow().weekday()
    week_start = (_dt.utcnow() - timedelta(days=weekday)).strftime("%Y-%m-%d")

    if not rep_id:
        return JSONResponse({"error": "rep_id required"}, status_code=400)

    async with get_conn() as db:
        await db.execute(
            "INSERT OR REPLACE INTO practice_log (user_id, rep_id, date, done) VALUES (?, ?, ?, ?)",
            (user_id, rep_id, today, 1 if done else 0)
        )
        await db.commit()
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM practice_log pl "
            "JOIN practice_reps pr ON pl.rep_id=pr.id "
            "WHERE pl.user_id=? AND pr.date>=? AND pl.done=1",
            (user_id, week_start)
        ) as cur:
            wc_row = await cur.fetchone()

    event_id = await record_event(
        user_id,
        "practice_logged",
        "practice",
        "User completed today's rep." if done else "User skipped today's rep.",
        {"rep_id": rep_id, "done": done},
        weight=1.2 if done else 0.4,
    )
    await record_outcome(
        user_id,
        subject_type="practice_rep",
        subject_id=rep_id,
        event_id=event_id,
        outcome="did_it" if done else "skipped",
        score=1.0 if done else 0.2,
        note="Practice log from Today/You.",
    )
    await record_life_event(
        user_id=user_id,
        event_domain="practice",
        event_type="practice_completed" if done else "practice_skipped",
        source="today",
        title="Practice rep completed" if done else "Practice rep skipped",
        summary="The user moved the loop into behavior." if done else "The user chose not to run the rep today.",
        payload={"rep_id": rep_id, "done": done},
        confidence=0.9,
        subject_type="practice_rep",
        subject_id=rep_id,
    )
    await refresh_current_thesis(user_id)

    return {
        "logged": True,
        "done": done,
        "week_completions": wc_row["cnt"] if wc_row else 0,
    }


async def _call_vllm(user_id: str, messages: list[dict], lane: str = "gemma4_e2b") -> str:
    """Call local vLLM (Gemma4 primary). Uses trained adapter if exists."""
    question = messages[-1].get("content", "") if messages else ""
    ctx = await get_context(user_id, question)
    enriched = _inject_system(messages, ctx["system_injection"])

    # Route to Gemma4 vLLM
    if lane == "gemma4_e2b":
        vllm_url = settings.gemma4_vllm_base_url
        if adapter_exists(user_id, lane="gemma4_e2b"):
            model_name = f"gemma4_user_{user_id}"
        else:
            model_name = settings.gemma4_base_model
    else:
        vllm_url = settings.vllm_base_url
        model_name = f"user_{user_id}"

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{vllm_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model_name,
                "messages": enriched,
                "temperature": 0.7,
                "max_tokens": 500,
                "stop": ["<function", "<tool_call>"],
            }
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"] or ""


@app.post("/v1/twin/ask")
async def twin_ask(request: Request):
    """Ask the same question to both the shadow clone and the teacher, return both anonymized."""
    user_id = request.state.user_id
    body = await request.json()
    question = body.get("question", "").strip()
    if not question:
        return JSONResponse({"error": "question required"}, status_code=400)

    # Twin uses Gemma4 (primary lane now)
    if not adapter_exists(user_id, lane="gemma4_e2b"):
        return JSONResponse({"error": "shadow clone not ready yet — keep chatting to train Gemma4"}, status_code=503)

    messages = [{"role": "user", "content": question}]

    from providers.teacher import chat_with_teacher
    clone_result, teacher_result = await asyncio.gather(
        _call_vllm(user_id, messages),
        chat_with_teacher(
            messages,
            user_id=user_id,
            purpose="twin_compare",
            importance=infer_importance(question),
            explicit_user_request=True,
            require_policy=True,
        ),
        return_exceptions=True,
    )

    clone_text = clone_result if isinstance(clone_result, str) else ""
    teacher_text = teacher_result[0] if isinstance(teacher_result, tuple) else ""

    if not clone_text or not teacher_text:
        skipped = teacher_result[2] if isinstance(teacher_result, tuple) and len(teacher_result) > 2 else ""
        return JSONResponse({"error": "one or both models unavailable", "teacher_policy": skipped}, status_code=503)

    import random
    a_is_clone = random.random() > 0.5
    response_a = clone_text if a_is_clone else teacher_text
    response_b = teacher_text if a_is_clone else clone_text

    session_id = str(uuid.uuid4())
    async with get_conn() as db:
        await db.execute(
            "INSERT INTO twin_sessions (id, user_id, question, response_a, response_b, a_is_clone) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, user_id, question, response_a, response_b, 1 if a_is_clone else 0)
        )
        await db.commit()

    return {
        "session_id": session_id,
        "question": question,
        "response_a": response_a,
        "response_b": response_b,
    }


@app.post("/v1/twin/choose")
async def twin_choose(request: Request, background_tasks: BackgroundTasks):
    """Record which response felt 'more me' and save a DPO training pair."""
    user_id = request.state.user_id
    body = await request.json()
    session_id = body.get("session_id", "")
    chosen = body.get("chosen", "")
    if not session_id or chosen not in ("A", "B"):
        return JSONResponse({"error": "session_id and chosen (A or B) required"}, status_code=400)

    async with get_conn() as db:
        async with db.execute(
            "SELECT * FROM twin_sessions WHERE id=? AND user_id=?",
            (session_id, user_id)
        ) as cur:
            session = await cur.fetchone()
        if not session:
            return JSONResponse({"error": "session not found"}, status_code=404)
        s = dict(session)
        existing_choice = (s.get("chosen") or "").strip()
        if existing_choice in {"A", "B"}:
            a_is_clone = bool(s["a_is_clone"])
            chose_clone = (existing_choice == "A") == a_is_clone
            return {
                "saved": False,
                "duplicate": True,
                "chosen": existing_choice,
                "chose_clone": chose_clone,
                "message": "This twin choice was already recorded.",
            }

        update_cur = await db.execute(
            "UPDATE twin_sessions SET chosen=? WHERE id=? AND user_id=? AND chosen IS NULL",
            (chosen, session_id, user_id),
        )
        await db.commit()
        if update_cur.rowcount == 0:
            async with db.execute(
                "SELECT * FROM twin_sessions WHERE id=? AND user_id=?",
                (session_id, user_id),
            ) as retry_cur:
                latest = await retry_cur.fetchone()
            if latest and latest["chosen"] in {"A", "B"}:
                latest_dict = dict(latest)
                a_is_clone = bool(latest_dict["a_is_clone"])
                chose_clone = (latest_dict["chosen"] == "A") == a_is_clone
                return {
                    "saved": False,
                    "duplicate": True,
                    "chosen": latest_dict["chosen"],
                    "chose_clone": chose_clone,
                    "message": "This twin choice was already recorded.",
                }
            return JSONResponse({"error": "choice could not be recorded"}, status_code=409)

    a_is_clone = bool(s["a_is_clone"])
    chose_clone = (chosen == "A") == a_is_clone
    chosen_text = s["response_a"] if chosen == "A" else s["response_b"]
    rejected_text = s["response_b"] if chosen == "A" else s["response_a"]

    background_tasks.add_task(
        save_pair,
        user_id=user_id,
        user_msg=s["question"],
        assistant_msg=chosen_text,
        model_used="twin:chosen",
        engagement_signal="thumbs_up",
        topic="personal",
    )
    background_tasks.add_task(
        save_pair,
        user_id=user_id,
        user_msg=s["question"],
        assistant_msg=rejected_text,
        model_used="twin:rejected",
        engagement_signal="thumbs_down",
        topic="personal",
    )

    return {
        "saved": True,
        "duplicate": False,
        "chosen": chosen,
        "chose_clone": chose_clone,
        "message": "Your twin just learned something." if chose_clone else "Noted — your twin has more to learn here.",
    }


async def _save_checkin_pairs(user_id: str, qas: list[dict]) -> None:
    """Save each check-in Q&A as a high-quality personal training pair."""
    for qa in qas:
        q = qa.get("q", "").strip()
        a = qa.get("a", "").strip()
        if not q or not a:
            continue
        await asyncio.gather(
            save_pair(
                user_id=user_id,
                user_msg=q,
                assistant_msg=a,
                model_used="checkin",
                engagement_signal="deep",
                topic="personal",
            ),
            add_memories(
                [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                user_id=user_id,
            ),
        )


# ──────────────────────────────────────────────────────────────────────────────
# PRESENCE SYSTEM — Timing Engine + Council
# ──────────────────────────────────────────────────────────────────────────────

_COUNCIL_PERSONAS: dict[str, str] = {
    "Builder": (
        "You are The Builder — strategic, systems-oriented. You see everything as architecture. "
        "Your lens: what does this enable, what does this build toward? "
        "Respond in EXACTLY ONE sentence. Be direct and specific."
    ),
    "Creative": (
        "You are The Creative — intuitive, emotionally intelligent, chaotic in a good way. "
        "You feel into decisions rather than analyze them. "
        "Your lens: what energy does this carry, what does it feel like to take this path? "
        "Respond in EXACTLY ONE sentence. Be vivid and honest."
    ),
    "Strategist": (
        "You are The Strategist — cold, long-term, utterly unsentimental. "
        "You see 5 years ahead and dismiss short-term noise. "
        "Your lens: where does this trajectory lead, what does this choice lock in? "
        "Respond in EXACTLY ONE sentence. Be blunt."
    ),
    "Examiner": (
        "You are The Examiner — skeptical and penetrating. You see what people avoid seeing. "
        "Your lens: what is the person avoiding, what fear or resistance is underneath this question? "
        "Respond in EXACTLY ONE sentence. Don't be gentle."
    ),
    "Connector": (
        "You are The Connector — deeply relational and empathetic. You consider human impact above all. "
        "Your lens: who does this affect, how does this change the relationships around this person? "
        "Respond in EXACTLY ONE sentence. Be warm but clear."
    ),
}


def _is_decision_question(message: str) -> bool:
    """Heuristic: does this message ask for a decision/choice?"""
    msg_lower = message.lower()
    question_starters = ("should", "would", "which", "what should", "do i", "will i", "can i")
    decision_keywords = (
        "should i", "help me decide", "what do you think", "which option",
        "which is better", "or not", " vs ", "or should", "worth it",
        "take it", "quit", "leave", "accept", "reject", "pursue",
    )
    starts_like_question = any(msg_lower.startswith(w) for w in question_starters)
    has_decision_word = any(kw in msg_lower for kw in decision_keywords)
    has_question_mark = "?" in message
    return (starts_like_question or has_question_mark) and has_decision_word


def _infer_clone_lead(message: str) -> str:
    msg_lower = message.lower()
    if any(w in msg_lower for w in ("build", "system", "architecture", "structure", "engineer")):
        return "Builder"
    if any(w in msg_lower for w in ("feel", "creative", "art", "design", "intuition", "vibe")):
        return "Creative"
    if any(w in msg_lower for w in ("career", "future", "5 years", "long term", "growth", "trajectory")):
        return "Strategist"
    if any(w in msg_lower for w in ("doubt", "worry", "avoid", "fear", "resist", "hiding")):
        return "Examiner"
    if any(w in msg_lower for w in ("people", "relationship", "team", "affect", "communication", "family")):
        return "Connector"
    return "Strategist"


def _hours_since(ts: Optional[str]) -> float:
    if not ts:
        return 99999.0
    try:
        from datetime import timezone
        dt = _dt.fromisoformat(ts.replace("Z", ""))
        if dt.tzinfo is None:
            now = _dt.utcnow()
        else:
            now = _dt.now(timezone.utc)
        return (now - dt).total_seconds() / 3600
    except Exception:
        return 99999.0


async def _generate_interruption(user_id: str) -> Optional[str]:
    """Ask the LLM to find a cross-topic behavioral pattern as a single statement."""
    async with get_conn() as db:
        async with db.execute(
            "SELECT user_msg, topic FROM training_pairs WHERE user_id=? ORDER BY created_at DESC LIMIT 50",
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()
    if len(rows) < 10:
        return None
    summary = "\n".join(f"[{r['topic']}] {r['user_msg'][:80]}" for r in rows[:30])
    prompt = (
        "You are Echo — a presence that observes behavioral patterns across all domains of someone's life.\n\n"
        "Read these messages from a user across different topics and find ONE behavioral pattern "
        "that appears in 3+ topics. Something they do consistently that they probably don't notice. "
        "Could be avoidance, a framing habit, a recurring delay, or a tendency.\n\n"
        f"Messages:\n{summary}\n\n"
        "Return ONLY a single declarative statement spoken directly to them. "
        "No 'I noticed', no 'you tend to'. Just the truth. Under 15 words.\n"
        "Examples: 'You avoid finishing things once they go public.'\n"
        "         'You frame every decision as risk, never as opportunity.'\n"
        "If no clear pattern exists, return exactly: NONE"
    )
    try:
        text, _ = await _call_feature_model(
            user_id,
            "judge_answer",
            prompt,
            temperature=0.85,
            max_tokens=40,
            importance="normal",
        )
        text = text.strip().strip('"')
        return None if text.upper().startswith("NONE") or len(text) < 8 else text
    except Exception:
        return None


async def _generate_revelation(user_id: str) -> Optional[str]:
    """Ask the LLM to write a prose letter of fundamental recognition."""
    async with get_conn() as db:
        async with db.execute(
            "SELECT user_msg, topic FROM training_pairs WHERE user_id=? ORDER BY created_at DESC LIMIT 80",
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()
        async with db.execute(
            "SELECT topic, COUNT(*) as cnt FROM training_pairs WHERE user_id=? GROUP BY topic ORDER BY cnt DESC",
            (user_id,),
        ) as cur:
            topic_rows = await cur.fetchall()
    if len(rows) < 30:
        return None
    summary = "\n".join(f"[{r['topic']}] {r['user_msg'][:100]}" for r in rows[:50])
    topic_str = ", ".join(f"{r['topic']} ({r['cnt']} messages)" for r in topic_rows[:6])
    prompt = (
        "You are Echo — a presence that has studied someone carefully over a long time.\n\n"
        f"Domains they engage with: {topic_str}\n\n"
        f"A sample of their messages:\n{summary}\n\n"
        "Write them a short letter — 3 to 4 sentences — that reveals something fundamental about who they are. "
        "Not what they do. Not a list of skills. Who they ARE at their core.\n\n"
        "This is recognition, not encouragement. It should feel like: "
        "'This thing understood something about me that no one ever has.'\n\n"
        "Start with: 'I've been watching you.'\n"
        "End with the core insight — the one that changes how they see themselves.\n"
        "Return ONLY the letter text."
    )
    try:
        text, _ = await _call_feature_model(
            user_id,
            "weekly_calibration",
            prompt,
            temperature=0.92,
            max_tokens=240,
            importance="high",
        )
        return text.strip() if text else None
    except Exception:
        return None


# ─── Thread lifecycle helpers ────────────────────────────────────────────────

def _days_since(ts: Optional[str]) -> float:
    return _hours_since(ts) / 24.0


async def _load_active_threads(user_id: str) -> list[dict]:
    async with get_conn() as db:
        async with db.execute(
            "SELECT * FROM echo_threads WHERE user_id=? AND status='active' ORDER BY escalation_level DESC, evidence_count DESC",
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def _add_thread_evidence(thread_id: str, snippet: str) -> None:
    import uuid as _uuid
    async with get_conn() as db:
        await db.execute(
            "INSERT INTO thread_evidence (id, thread_id, message_snippet) VALUES (?,?,?)",
            (str(_uuid.uuid4()), thread_id, snippet[:200]),
        )
        await db.execute(
            "UPDATE echo_threads SET evidence_count=evidence_count+1, last_seen=datetime('now') WHERE id=?",
            (thread_id,),
        )
        await db.commit()


async def _escalate_thread(thread_id: str, new_level: int) -> None:
    async with get_conn() as db:
        await db.execute(
            "UPDATE echo_threads SET escalation_level=? WHERE id=?",
            (new_level, thread_id),
        )
        await db.commit()


async def _resolve_thread(thread_id: str, note: str = "") -> None:
    async with get_conn() as db:
        await db.execute(
            "UPDATE echo_threads SET status='resolved', resolution_note=? WHERE id=?",
            (note, thread_id),
        )
        await db.commit()


async def _log_thread_escalation(thread_id: str, level: int, content: str) -> None:
    import uuid as _uuid
    async with get_conn() as db:
        await db.execute(
            "INSERT INTO thread_escalations (id, thread_id, level, content) VALUES (?,?,?,?)",
            (str(_uuid.uuid4()), thread_id, level, content),
        )
        await db.commit()


async def _create_thread(user_id: str, name: str, topic: str, first_snippet: str) -> str:
    import uuid as _uuid
    thread_id = str(_uuid.uuid4())
    async with get_conn() as db:
        await db.execute(
            "INSERT INTO echo_threads (id, user_id, name, topic, evidence_count) VALUES (?,?,?,?,1)",
            (thread_id, user_id, name, topic),
        )
        await db.execute(
            "INSERT INTO thread_evidence (id, thread_id, message_snippet) VALUES (?,?,?)",
            (str(_uuid.uuid4()), thread_id, first_snippet[:200]),
        )
        await db.commit()
    return thread_id


async def _detect_new_threads(user_id: str, existing_names: list[str]) -> list[dict]:
    """Ask LLM to find new named patterns not already tracked."""
    async with get_conn() as db:
        async with db.execute(
            "SELECT user_msg, topic FROM training_pairs WHERE user_id=? ORDER BY created_at DESC LIMIT 40",
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()
    if len(rows) < 10:
        return []
    summary = "\n".join(f"[{r['topic']}] {r['user_msg'][:80]}" for r in rows[:30])
    existing_str = ", ".join(f'"{n}"' for n in existing_names) if existing_names else "none"
    prompt = (
        "You observe behavioral patterns in people's messages. "
        f"Already tracked patterns: {existing_str}\n\n"
        f"Messages:\n{summary}\n\n"
        "Find up to 2 NEW behavioral patterns not already tracked. "
        "Each pattern must appear in at least 3 different messages.\n"
        "Return JSON array: [{\"name\": \"short pattern name under 8 words\", \"topic\": \"main domain\", \"snippet\": \"best example message\"}]\n"
        "If no new patterns, return: []"
    )
    try:
        raw, _ = await _call_feature_model(
            user_id,
            "judge_answer",
            prompt,
            temperature=0.7,
            max_tokens=220,
            importance="normal",
        )
        import json as _json
        raw = raw.strip()
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start >= 0 and end > start:
            return _json.loads(raw[start:end])
    except Exception:
        pass
    return []


async def _check_thread_still_active(thread: dict, recent_messages: list[str]) -> bool:
    """Quick check: does this pattern still appear in last 10 messages?"""
    if len(recent_messages) < 5:
        return True  # not enough data to resolve
    sample = " ".join(recent_messages[-10:])[:500]
    prompt = (
        f"Pattern: \"{thread['name']}\"\n"
        f"Recent messages: {sample}\n\n"
        "Does this pattern still appear in these recent messages? Reply YES or NO only."
    )
    try:
        answer, _ = await _call_feature_model(
            thread["user_id"],
            "judge_answer",
            prompt,
            temperature=0.3,
            max_tokens=5,
            importance="normal",
        )
        return "YES" in answer.strip().upper()
    except Exception:
        return True


async def _generate_interruption_for_thread(thread: dict) -> Optional[str]:
    """Generate a statement specifically about a named thread."""
    async with get_conn() as db:
        async with db.execute(
            "SELECT message_snippet FROM thread_evidence WHERE thread_id=? ORDER BY added_at DESC LIMIT 5",
            (thread["id"],),
        ) as cur:
            evidence = await cur.fetchall()
    snippets = "\n".join(f"- {e['message_snippet']}" for e in evidence)
    days = _days_since(thread.get("first_seen"))
    prompt = (
        f"You are Echo. You've been observing a pattern in someone for {int(days)} days.\n"
        f"Pattern name: \"{thread['name']}\"\n"
        f"Evidence:\n{snippets}\n\n"
        "Write ONE declarative statement to this person about this pattern. "
        "Direct, specific, under 15 words. No 'I noticed'. Just truth.\n"
        "Return ONLY the statement."
    )
    try:
        text, _ = await _call_feature_model(
            thread["user_id"],
            "judge_answer",
            prompt,
            temperature=0.85,
            max_tokens=40,
            importance="normal",
        )
        text = text.strip().strip('"')
        return None if text.upper().startswith("NONE") or len(text) < 8 else text
    except Exception:
        return None


async def _generate_revelation_for_thread(user_id: str, thread: dict) -> Optional[str]:
    """Write a revelation letter anchored to a specific thread's arc."""
    async with get_conn() as db:
        async with db.execute(
            "SELECT message_snippet FROM thread_evidence WHERE thread_id=? ORDER BY added_at ASC",
            (thread["id"],),
        ) as cur:
            evidence = await cur.fetchall()
        async with db.execute(
            "SELECT topic, COUNT(*) as cnt FROM training_pairs WHERE user_id=? GROUP BY topic ORDER BY cnt DESC LIMIT 6",
            (user_id,),
        ) as cur:
            topic_rows = await cur.fetchall()
    days = int(_days_since(thread.get("first_seen")))
    snippets = "\n".join(f"- {e['message_snippet']}" for e in evidence[:8])
    topic_str = ", ".join(r["topic"] for r in topic_rows)
    prompt = (
        f"You are Echo. For {days} days, you have observed this pattern in someone: \"{thread['name']}\"\n"
        f"Their domains: {topic_str}\n"
        f"Evidence collected:\n{snippets}\n\n"
        f"Write a short letter — 3 to 4 sentences — that begins:\n"
        f"\"For {days} days, you've been circling something.\"\n\n"
        "Reveal something fundamental about who they are based on this pattern. "
        "Not what they do — who they ARE. Recognition, not encouragement.\n"
        "Return ONLY the letter."
    )
    try:
        text, _ = await _call_feature_model(
            user_id,
            "weekly_calibration",
            prompt,
            temperature=0.92,
            max_tokens=270,
            importance="high",
        )
        return text.strip() if text else None
    except Exception:
        return None


@app.post("/v1/echo/decide")
async def echo_decide(request: Request):
    """
    Thread-aware Timing Engine: decides which of the 4 Echo states applies.
    Returns: { state, speak_now, reason, thread_id?, thread_name?, thread_day?,
               escalation_level?, statement?, letter?, clone_lead? }
    States: silence | interruption | council | revelation
    """
    user_id = request.state.user_id
    body = await request.json()
    user_message: str = body.get("message", "")

    async with get_conn() as db:
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM training_pairs WHERE user_id=?", (user_id,)
        ) as cur:
            total_pairs = (await cur.fetchone())["cnt"]
        async with db.execute(
            "SELECT COUNT(DISTINCT topic) as cnt FROM training_pairs WHERE user_id=?", (user_id,)
        ) as cur:
            topic_count = (await cur.fetchone())["cnt"]
        async with db.execute(
            "SELECT user_msg FROM training_pairs WHERE user_id=? ORDER BY created_at DESC LIMIT 20",
            (user_id,),
        ) as cur:
            recent_rows = await cur.fetchall()

    recent_messages = [r["user_msg"] for r in recent_rows]

    # Rule 1: Not enough data → silence
    if total_pairs < 5:
        return {"state": "silence", "speak_now": False, "reason": "not_enough_data"}

    # Rule 2: Decision question → user-triggered council
    is_decision = bool(body.get("is_decision", False))
    if not is_decision and user_message:
        is_decision = _is_decision_question(user_message)
    if is_decision:
        return {
            "state": "council",
            "speak_now": True,
            "reason": "decision_question",
            "clone_lead": _infer_clone_lead(user_message),
        }

    # ── Thread lifecycle ──────────────────────────────────────────────────────
    threads = await _load_active_threads(user_id)
    existing_names = [t["name"] for t in threads]

    # Detect new patterns at most once per 6 hours to avoid LLM spam + duplicate threads
    should_detect = False
    if total_pairs >= 10:
        async with get_conn() as db:
            async with db.execute(
                "SELECT MAX(last_seen) as last FROM echo_threads WHERE user_id=?", (user_id,)
            ) as cur:
                row = await cur.fetchone()
        last_detection = row["last"] if row else None
        should_detect = last_detection is None or _hours_since(last_detection) >= 6

    if should_detect:
        new_patterns = await _detect_new_threads(user_id, existing_names)
        for p in new_patterns:
            name = p.get("name", "").strip()
            # Fuzzy dedup: skip if any existing name shares 2+ words
            name_words = set(name.lower().split())
            is_dup = any(
                len(name_words & set(ex.lower().split())) >= 2
                for ex in existing_names
            )
            if name and not is_dup:
                tid = await _create_thread(user_id, name, p.get("topic", "general"), p.get("snippet", ""))
                threads.append({
                    "id": tid, "user_id": user_id, "name": name,
                    "topic": p.get("topic", "general"), "first_seen": _dt.utcnow().isoformat(),
                    "last_seen": _dt.utcnow().isoformat(),
                    "evidence_count": 1, "escalation_level": 1, "status": "active",
                })
                existing_names.append(name)

    # Update evidence + escalation for existing threads
    _STOP_WORDS = {"a", "an", "the", "in", "on", "of", "to", "and", "or", "is", "are", "was", "my", "your"}
    for thread in threads:
        tid = thread["id"]
        count = thread["evidence_count"]
        days = _days_since(thread.get("first_seen"))
        level = thread["escalation_level"]

        # Evidence: any meaningful word from thread name appears in recent messages
        keywords = [w for w in thread["name"].lower().split() if len(w) > 3 and w not in _STOP_WORDS]
        for msg in recent_messages[:5]:
            msg_lower = msg.lower()
            if keywords and any(kw in msg_lower for kw in keywords):
                await _add_thread_evidence(tid, msg[:200])
                count += 1
                break

        # Escalation rules
        new_level = level
        if level < 2 and count >= 3:
            new_level = 2
        elif level < 3 and count >= 6 and days >= 2:
            new_level = 3
        elif level < 4 and count >= 12 and days >= 5:
            new_level = 4
        elif level < 5 and count >= 20 and days >= 10 and topic_count >= 3:
            new_level = 5

        if new_level > level:
            await _escalate_thread(tid, new_level)
            thread["escalation_level"] = new_level

        # Resolution check at level 3+
        if level >= 3 and len(recent_messages) >= 10:
            still_active = await _check_thread_still_active(thread, recent_messages)
            if not still_active:
                await _resolve_thread(tid, "pattern no longer evident")
                thread["status"] = "resolved"

    # Filter to only active threads, sorted by escalation level
    active = [t for t in threads if t.get("status") == "active"]
    active.sort(key=lambda t: (t["escalation_level"], t["evidence_count"]), reverse=True)

    if not active:
        return {"state": "silence", "speak_now": False, "reason": "no_active_threads"}

    top = active[0]
    level = top["escalation_level"]
    thread_day = max(1, int(_days_since(top.get("first_seen"))) + 1)

    thread_base = {
        "thread_id": top["id"],
        "thread_name": top["name"],
        "thread_day": thread_day,
        "escalation_level": level,
    }

    # Level 1 → seed, stay silent
    if level <= 1:
        return {"state": "silence", "speak_now": False, "reason": "seeding", **thread_base}

    # Level 2 → quiet surface
    if level == 2:
        # Check cooldown: only surface once per day per thread
        async with get_conn() as db:
            async with db.execute(
                "SELECT shown_at FROM thread_escalations WHERE thread_id=? AND level=2 ORDER BY shown_at DESC LIMIT 1",
                (top["id"],),
            ) as cur:
                last_surface = await cur.fetchone()
        if last_surface and _hours_since(last_surface["shown_at"]) < 24:
            return {"state": "silence", "speak_now": False, "reason": "surface_cooldown", **thread_base}
        statement = await _generate_interruption_for_thread(top)
        if statement:
            await _log_thread_escalation(top["id"], 2, statement)
            return {"state": "interruption", "speak_now": True, "reason": "thread_surface",
                    "statement": statement, **thread_base}

    # Level 3 → interruption (12h cooldown)
    if level == 3:
        async with get_conn() as db:
            async with db.execute(
                "SELECT shown_at FROM thread_escalations WHERE thread_id=? AND level=3 ORDER BY shown_at DESC LIMIT 1",
                (top["id"],),
            ) as cur:
                last_interrupt = await cur.fetchone()
        if last_interrupt and _hours_since(last_interrupt["shown_at"]) < 12:
            return {"state": "silence", "speak_now": False, "reason": "interrupt_cooldown", **thread_base}
        statement = await _generate_interruption_for_thread(top)
        if statement:
            await _log_thread_escalation(top["id"], 3, statement)
            return {"state": "interruption", "speak_now": True, "reason": "thread_interrupt",
                    "statement": statement, **thread_base}

    # Level 4 → proactive council (48h cooldown)
    if level == 4:
        async with get_conn() as db:
            async with db.execute(
                "SELECT shown_at FROM thread_escalations WHERE thread_id=? AND level=4 ORDER BY shown_at DESC LIMIT 1",
                (top["id"],),
            ) as cur:
                last_council = await cur.fetchone()
        if last_council and _hours_since(last_council["shown_at"]) < 48:
            return {"state": "silence", "speak_now": False, "reason": "council_cooldown", **thread_base}
        thread_context = (
            f"Pattern: \"{top['name']}\" — observed for {thread_day} days across {top['evidence_count']} messages."
        )
        await _log_thread_escalation(top["id"], 4, thread_context)
        return {"state": "council", "speak_now": True, "reason": "thread_council",
                "thread_context": thread_context, **thread_base}

    # Level 5 → revelation (14-day cooldown per thread)
    if level >= 5:
        async with get_conn() as db:
            async with db.execute(
                "SELECT shown_at FROM thread_escalations WHERE thread_id=? AND level=5 ORDER BY shown_at DESC LIMIT 1",
                (top["id"],),
            ) as cur:
                last_rev = await cur.fetchone()
        if last_rev and _hours_since(last_rev["shown_at"]) < 336:
            return {"state": "silence", "speak_now": False, "reason": "revelation_cooldown", **thread_base}
        letter = await _generate_revelation_for_thread(user_id, top)
        if letter:
            await _log_thread_escalation(top["id"], 5, letter)
            return {"state": "revelation", "speak_now": True, "reason": "thread_revelation",
                    "letter": letter, **thread_base}

    return {"state": "silence", "speak_now": False, "reason": "no_pattern_yet"}


async def _ask_one_clone(user_id: str, name: str, persona: str, question: str) -> tuple[str, str]:
    try:
        text = await _call_gemma_feature(
            user_id,
            question,
            system=persona,
            temperature=0.88,
            max_tokens=80,
        )
        # Keep only first sentence
        for end in (".", "?", "!"):
            idx = text.find(end)
            if idx != -1:
                return name, text[: idx + 1]
        return name, text.split("\n")[0] + "."
    except Exception:
        return name, "I couldn't find a clear answer here."


@app.get("/v1/threads")
async def get_threads(request: Request):
    """Return active and resolved threads for this user."""
    user_id = request.state.user_id
    async with get_conn() as db:
        async with db.execute(
            """SELECT id, name, topic, first_seen, last_seen, evidence_count, escalation_level, status, resolution_note
               FROM echo_threads WHERE user_id=? ORDER BY escalation_level DESC, evidence_count DESC""",
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()

    active = []
    resolved = []
    for r in rows:
        days = max(1, int(_days_since(r["first_seen"])) + 1)
        entry = {
            "id": r["id"],
            "name": r["name"],
            "topic": r["topic"],
            "first_seen": r["first_seen"],
            "last_seen": r["last_seen"],
            "evidence_count": r["evidence_count"],
            "escalation_level": r["escalation_level"],
            "days_running": days,
        }
        if r["status"] == "active":
            active.append(entry)
        else:
            entry["resolution_note"] = r["resolution_note"]
            entry["resolved_at"] = r["last_seen"]
            resolved.append(entry)

    return {"active": active, "resolved": resolved}


@app.post("/v1/threads/{thread_id}/resolve")
async def resolve_thread_endpoint(thread_id: str, request: Request):
    """Manually resolve a thread (called after council verdict or revelation read)."""
    user_id = request.state.user_id
    body = await request.json()
    note = body.get("note", "user acknowledged")
    async with get_conn() as db:
        async with db.execute(
            "SELECT id FROM echo_threads WHERE id=? AND user_id=?", (thread_id, user_id)
        ) as cur:
            if not await cur.fetchone():
                from fastapi import HTTPException
                raise HTTPException(404, "Thread not found")
        await db.execute(
            "UPDATE echo_threads SET status='resolved', resolution_note=? WHERE id=?",
            (note, thread_id),
        )
        await db.commit()
    return {"ok": True, "thread_id": thread_id, "status": "resolved"}


@app.post("/v1/threads/deduplicate")
async def deduplicate_threads(request: Request):
    """Resolve duplicate/stale level-1 threads, keeping highest-evidence per name cluster."""
    user_id = request.state.user_id
    async with get_conn() as db:
        async with db.execute(
            "SELECT * FROM echo_threads WHERE user_id=? AND status='active' ORDER BY evidence_count DESC",
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()
    threads = [dict(r) for r in rows]
    seen_clusters: list[set] = []
    keep_ids = []
    resolve_ids = []
    for t in threads:
        words = set(t["name"].lower().split())
        matched = next((i for i, c in enumerate(seen_clusters) if len(words & c) >= 2), None)
        if matched is None:
            seen_clusters.append(words)
            keep_ids.append(t["id"])
        else:
            resolve_ids.append(t["id"])
    if resolve_ids:
        async with get_conn() as db:
            for rid in resolve_ids:
                await db.execute(
                    "UPDATE echo_threads SET status='resolved', resolution_note='duplicate' WHERE id=?",
                    (rid,),
                )
            await db.commit()
    return {"kept": len(keep_ids), "resolved": len(resolve_ids)}


@app.post("/v1/council/ask")
async def council_ask(request: Request):
    """
    Council API: run a question through all 5 clone personalities in parallel.
    Returns: { question, voices: {Builder,Creative,...}, verdict }
    """
    user_id = request.state.user_id
    body = await request.json()
    question = body.get("question", "").strip()
    if not question:
        return JSONResponse({"error": "question required"}, status_code=400)

    # Run all 5 clones in parallel
    results = await asyncio.gather(
        *[_ask_one_clone(user_id, name, persona, question) for name, persona in _COUNCIL_PERSONAS.items()]
    )
    voices = {name: text for name, text in results}

    # Generate consensus verdict
    voices_text = "\n".join(f"- {n}: {t}" for n, t in voices.items())
    verdict_prompt = (
        f"Five advisors answered: \"{question}\"\n\n{voices_text}\n\n"
        "Determine the consensus verdict. "
        "If 3+ advisors lean the same direction: return 'N/5 agree: [one-sentence direction].'\n"
        "If divided: return 'The council is divided — [the core tension in one sentence].'\n"
        "Return ONLY the verdict. No preamble."
    )
    verdict = "The council is divided."
    try:
        verdict = await _call_gemma_feature(
            user_id,
            verdict_prompt,
            temperature=0.3,
            max_tokens=60,
        )
    except Exception:
        pass

    log.info("/v1/council/ask user=%s question=%.60s", user_id, question)
    return {"question": question, "voices": voices, "verdict": verdict}


@app.post("/v1/reply/action")
async def reply_action(request: Request, background_tasks: BackgroundTasks):
    """Unified after-reply action strip for Talk, Today, and Coach flows.
    action: make_practice | save_memory | log_outcome | add_to_passport
    """
    user_id = request.state.user_id
    body = await request.json()
    action = (body.get("action") or "").strip()
    reply_text = (body.get("reply_text") or "").strip()[:2000]
    context = (body.get("context") or "").strip()[:500]

    _VALID_ACTIONS = {"make_practice", "save_memory", "log_outcome", "add_to_passport"}
    if action not in _VALID_ACTIONS:
        return JSONResponse({"error": f"action must be one of {sorted(_VALID_ACTIONS)}"}, status_code=400)
    if not reply_text:
        return JSONResponse({"error": "reply_text required"}, status_code=400)

    result: dict = {"action": action, "saved": False}

    if action == "save_memory":
        memory_saved = await add_raw_memory(reply_text, user_id=user_id, source="reply_strip:save_memory")
        event_id = await record_event(
            user_id, "memory_consent_requested", "reply_strip",
            "User saved an Echo reply as memory.",
            {"source_type": "reply_strip", "privacy": "private", "memory_saved": memory_saved},
            weight=1.0,
        )
        result.update({"saved": True, "memory_saved": memory_saved, "event_id": event_id, "message": "Remembered"})

    elif action == "make_practice":
        event_id = await record_event(
            user_id, "practice_rep_set", "reply_strip",
            f"Practice rep from reply: {reply_text[:120]}",
            {"rep_instruction": reply_text[:500], "context": context, "source": "reply_strip"},
            weight=1.3,
        )
        result.update({"saved": True, "event_id": event_id, "rep_instruction": reply_text[:500], "message": "Practice saved"})

    elif action == "log_outcome":
        score = float(body.get("score") or 0.5)
        note = (body.get("note") or reply_text)[:700]
        event_id = await record_event(
            user_id, "outcome_logged", "reply_strip",
            f"Outcome from reply strip: {note[:120]}",
            {"note": note, "score": score, "source": "reply_strip"},
            weight=1.2,
        )
        await record_outcome(
            user_id,
            subject_type="reply_strip",
            subject_id=event_id,
            event_id=event_id,
            outcome="logged",
            score=score,
            note=note,
        )
        result.update({"saved": True, "event_id": event_id, "message": "Outcome saved"})

    elif action == "add_to_passport":
        title = (body.get("title") or reply_text[:80]).strip()
        item_id = str(uuid.uuid4())
        async with get_conn() as db:
            await db.execute(
                """
                INSERT INTO proof_items
                    (id, user_id, title, description, category, source_type, skill_tags_json)
                VALUES (?, ?, ?, ?, 'story', 'reply_strip', '[]')
                """,
                (item_id, user_id, title[:160], reply_text[:1200]),
            )
            await db.commit()
        event_id = await record_event(
            user_id, "proof_item_added", "reply_strip",
            f"Proof item from reply: {title[:80]}",
            {"proof_id": item_id, "source": "reply_strip"},
            weight=1.4,
        )
        result.update({"saved": True, "proof_id": item_id, "title": title, "event_id": event_id, "message": "Proof saved"})

    if result.get("saved") and result.get("event_id"):
        try:
            await refresh_current_thesis(user_id)
            result["loop_delta"] = await _loop_delta_for_save(
                user_id,
                result["event_id"],
                topic=detect_topic(f"{context} {reply_text}"),
                model_used=f"reply_strip:{action}",
                thesis_updated=True,
                proof_created=action == "add_to_passport",
                opportunity_unlocked=action == "add_to_passport",
                training_signal_saved=action in {"make_practice", "log_outcome", "add_to_passport"},
                next_action="Open Place to see what this proof can unlock." if action == "add_to_passport" else "Use this signal in Today, Proof, or your next practice rep.",
                receipt_title=result.get("message") or "Echo updated",
                receipt_detail="Echo connected this action to your loop.",
            )
        except Exception as e:
            log.warning("reply action loop delta failed user=%s action=%s: %s", user_id, action, e)

    log.info("/v1/reply/action user=%s action=%s", user_id, action)
    return result


@app.post("/v1/echo/simulate")
async def echo_simulate(request: Request):
    """
    Parallel Self Simulation: two diverging trajectories based on current patterns.
    Not prediction — pattern-based projection of where current habits lead vs. avoided path.
    Returns: { current_path: {label, projection, detail}, avoided_path: {label, projection, detail} }
    """
    user_id = request.state.user_id

    async with get_conn() as db:
        async with db.execute(
            "SELECT user_msg, topic FROM training_pairs WHERE user_id=? ORDER BY created_at DESC LIMIT 60",
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()
        async with db.execute(
            "SELECT topic, COUNT(*) as cnt FROM training_pairs WHERE user_id=? GROUP BY topic ORDER BY cnt DESC",
            (user_id,),
        ) as cur:
            topic_rows = await cur.fetchall()
        async with db.execute(
            "SELECT topic, score FROM confidence WHERE user_id=? ORDER BY score DESC LIMIT 5",
            (user_id,),
        ) as cur:
            conf_rows = await cur.fetchall()

    if len(rows) < 10:
        return {
            "current_path": {
                "label": "Still emerging",
                "projection": "Keep talking to Echo.",
                "detail": "Patterns need more data before two paths can be seen.",
            },
            "avoided_path": {
                "label": "Unknown",
                "projection": "The alternate path isn't visible yet.",
                "detail": "Come back when you've had more conversations.",
            },
            "ready": False,
        }

    summary = "\n".join(f"[{r['topic']}] {r['user_msg'][:80]}" for r in rows[:40])
    topic_str = ", ".join(f"{r['topic']} ({r['cnt']})" for r in topic_rows[:6])
    conf_str = ", ".join(f"{r['topic']} ({r['score']:.0%})" for r in conf_rows)

    prompt = (
        "You are Echo — a pattern recognition system that has studied someone's behavior closely.\n\n"
        f"Their dominant topics: {topic_str}\n"
        f"Their confidence areas: {conf_str}\n\n"
        f"Recent messages:\n{summary}\n\n"
        "Based purely on the patterns you see — not predictions, not advice — describe two diverging paths:\n\n"
        "1. CURRENT PATH: Where their current habits and patterns lead if they continue unchanged. "
        "Be specific to what you actually see in their messages. Not doom, not flattery — just honest trajectory.\n\n"
        "2. AVOIDED PATH: The path that is available but they consistently avoid or delay. "
        "This should be visible in the patterns — something they orbit but don't commit to.\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "current_path": {\n'
        '    "label": "short name for this path (3-4 words)",\n'
        '    "projection": "one sentence: where this leads",\n'
        '    "detail": "two sentences: what the pattern actually shows"\n'
        "  },\n"
        '  "avoided_path": {\n'
        '    "label": "short name for this path (3-4 words)",\n'
        '    "projection": "one sentence: what becomes available",\n'
        '    "detail": "two sentences: what they keep orbiting but not taking"\n'
        "  }\n"
        "}"
    )

    try:
        content, model_used = await _call_feature_model(
            user_id,
            "judge_answer",
            prompt,
            temperature=0.88,
            max_tokens=360,
            importance="normal",
        )
        if not content:
            raise RuntimeError(f"simulate unavailable: {model_used}")
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content.strip())
        result["ready"] = True
        result["model_used"] = model_used
        log.info("/v1/echo/simulate user=%s", user_id)
        return result
    except Exception as e:
        log.warning("/v1/echo/simulate failed user=%s: %s", user_id, e)
        return {
            "current_path": {"label": "Current trajectory", "projection": "Echo is still learning.", "detail": ""},
            "avoided_path": {"label": "Alternate path", "projection": "Patterns need more time.", "detail": ""},
            "ready": False,
        }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.port, reload=False)
