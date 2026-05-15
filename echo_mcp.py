"""
Echo MCP Server — wraps Echo API (localhost:8002) as an MCP server.

Transport: stdio (default) — works with Claude Code, Cursor, Cline, any MCP client.
Auth: uses echo_secret as Bearer token + x-echo-user-id header.

HOW THE TEACHER MODEL WORKS WITH MCP CLIENTS
─────────────────────────────────────────────
When you use Echo through Claude Code / GPT-5 / Sonnet / Gemini:
  • The LLM you're already talking to IS the teacher.
  • After calling chat(), that LLM evaluates quality and calls save_training_pair().
  • It extracts facts and calls save_memory().
  • It identifies rules and calls add_rule().
  • No separate teacher API key needed — the calling LLM does the work.
This means: GPT-5 users get GPT-5 as teacher. Sonnet users get Sonnet.

ADD TO CLAUDE CODE (.mcp.json in project root or ~/.claude/claude_code_config.json):
{
  "mcpServers": {
    "echo": {
      "command": "python",
      "args": ["C:/Users/ASUS/Desktop/echo/echo_mcp.py"],
      "env": {
        "ECHO_USER_ID": "YOUR_USER_ID_HERE",
        "ECHO_SECRET": "echo-local-secret"
      }
    }
  }
}

ENV VARS:
  ECHO_BASE_URL   — default http://127.0.0.1:8002
  ECHO_SECRET     — default echo-local-secret
  ECHO_USER_ID    — your user id (from /v1/user/stats or login)
"""

import os
import httpx
from typing import Any
from fastmcp import FastMCP

ECHO_BASE = os.environ.get("ECHO_BASE_URL", "http://127.0.0.1:8002")
ECHO_SECRET = os.environ.get("ECHO_SECRET", "echo-local-secret")
ECHO_USER_ID = os.environ.get("ECHO_USER_ID", "default")
ECHO_TIMEOUT = float(os.environ.get("ECHO_MCP_TIMEOUT", "12"))

mcp = FastMCP("Echo")


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {ECHO_SECRET}",
        "x-echo-user-id": ECHO_USER_ID,
        "Content-Type": "application/json",
    }


def _get(path: str, params: dict | None = None) -> dict:
    with httpx.Client(timeout=ECHO_TIMEOUT) as client:
        r = client.get(f"{ECHO_BASE}{path}", headers=_headers(), params=params or {})
        r.raise_for_status()
        return r.json()


def _post(path: str, body: dict) -> dict:
    with httpx.Client(timeout=60) as client:
        r = client.post(f"{ECHO_BASE}{path}", headers=_headers(), json=body)
        r.raise_for_status()
        return r.json()


def _delete(path: str, body: dict | None = None) -> dict:
    with httpx.Client(timeout=30) as client:
        r = client.request("DELETE", f"{ECHO_BASE}{path}", headers=_headers(), json=body)
        r.raise_for_status()
        return r.json()


def _try_get(path: str, params: dict | None = None) -> dict:
    try:
        return _get(path, params)
    except Exception as e:
        return {"error": str(e), "path": path}


def _try_post(path: str, body: dict) -> dict:
    try:
        return _post(path, body)
    except Exception as e:
        return {"error": str(e), "path": path}


def _try_delete(path: str, body: dict | None = None) -> dict:
    try:
        return _delete(path, body)
    except Exception as e:
        return {"error": str(e), "path": path}


def _trim(value: Any, limit: int = 900) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit].rstrip() + "..."
    if isinstance(value, list):
        return [_trim(v, limit) for v in value]
    if isinstance(value, dict):
        return {k: _trim(v, limit) for k, v in value.items()}
    return value


def _training_ready(summary: dict) -> bool:
    return bool(summary.get("can_train_now"))


def _compact_run(run: dict | None) -> dict:
    if not isinstance(run, dict):
        return {}
    return {
        "id": run.get("id"),
        "status": run.get("status"),
        "lane": run.get("lane"),
        "untrained_pairs": run.get("untrained_pairs"),
        "required_pairs": run.get("required_pairs"),
        "adapter_path": run.get("adapter_path"),
        "error": run.get("error"),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
    }


def _compact_eval(data: dict) -> dict:
    eval_data = data.get("eval", data) if isinstance(data, dict) else {}
    if not isinstance(eval_data, dict):
        return {}
    return {
        "passed": eval_data.get("passed"),
        "score": eval_data.get("score"),
        "n_eval": eval_data.get("n_eval"),
        "skipped_reason": eval_data.get("skipped_reason"),
    }


def _compact_adapter(adapter: dict) -> dict:
    if not isinstance(adapter, dict):
        return {}
    return {
        "lane": adapter.get("lane"),
        "base_model": adapter.get("base_model"),
        "lora_name": adapter.get("lora_name"),
        "exists": adapter.get("exists"),
        "compatible": adapter.get("compatible"),
        "loaded": adapter.get("loaded"),
        "serving_model": adapter.get("serving_model"),
        "vllm": adapter.get("vllm"),
    }


@mcp.tool
def echo_training_center(action: str = "status", confirm: bool = False, include_runtime: bool = False) -> dict:
    """
    Training control center for Echo.

    action: status | trigger
    trigger only starts training when Echo has enough SFT or DPO signal.
    Set confirm=True to actually start a ready training run.
    """
    summary = _try_get("/v1/training/summary")
    status = _try_get("/v1/training/status") if include_runtime or action == "trigger" else {}
    health = _try_get("/v1/system/health") if include_runtime else {}
    eval_data = _try_get("/v1/training/eval")
    ready = _training_ready(summary)
    result = {
        "action": action,
        "ready": ready,
        "readiness": {
            "untrained_pairs": summary.get("untrained_pairs"),
            "required_pairs": summary.get("required_pairs"),
            "can_train_now": summary.get("can_train_now"),
            "ready_for_training": summary.get("ready_for_training"),
            "dpo_ready_pairs": summary.get("dpo_ready_pairs"),
            "dpo_required_pairs": summary.get("dpo_required_pairs"),
            "dpo_ready_for_training": summary.get("dpo_ready_for_training"),
            "dpo_requires_existing_adapter": summary.get("dpo_requires_existing_adapter"),
        },
        "training": {
            "status": status.get("status"),
            "lane": status.get("lane") or summary.get("lane"),
            "latest_run": _compact_run(status.get("latest_run")),
            "last_checkpoint": summary.get("last_checkpoint"),
            "eval": _compact_eval(eval_data),
        },
        "adapter": _compact_adapter(health.get("adapter") or summary.get("adapter", {})),
    }
    if not include_runtime:
        result["runtime_note"] = "Call echo_training_center(include_runtime=True) to check live vLLM/adapter state."
    if action == "trigger":
        if not ready:
            result["started"] = False
            result["reason"] = "not_ready"
            return _trim(result)
        if not confirm:
            result["started"] = False
            result["reason"] = "confirm_required"
            result["next"] = "Call echo_training_center(action='trigger', confirm=True)."
            return _trim(result)
        result["trigger_result"] = _try_post("/trigger-training", {})
        result["started"] = "error" not in result["trigger_result"]
    return _trim(result)


@mcp.tool
def echo_daily_brief(include_questions: bool = False, include_next_intervention: bool = False) -> dict:
    """
    Daily Echo dashboard: priority, mission, practice, and check-in state.

    Generated questions and proactive interventions can call the model, so they
    are opt-in for MCP clients that need a fast status read.
    """
    result = {
        "priority": _try_get("/v1/today/priority"),
        "mission": _try_get("/v1/today/mission"),
        "reality_check": _try_get("/v1/reality/check"),
        "practice": _try_get("/v1/practice/today"),
        "checkin": {
            "status": _try_get("/v1/daily/checkin/status"),
        },
    }
    if include_questions:
        result["checkin"]["questions"] = _try_get("/v1/daily/questions")
    if include_next_intervention:
        result["next_intervention"] = _try_get("/v1/interventions/next")
    return _trim(result)


@mcp.tool
def echo_current_read(include_generated_signal: bool = False) -> dict:
    """
    Current identity read: thesis, evidence, revelation readiness, and loop state.

    include_generated_signal=True adds the one-line user signal, which may call
    the model and can be slower.
    """
    thesis = _try_get("/v1/thesis/current")
    result = {
        "thesis": thesis,
        "revelation": _try_get("/v1/revelation/status"),
        "loop": _try_get("/v1/loop/snapshot"),
        "rank": _try_get("/v1/user/rank"),
        "confidence": _try_get("/v1/user/confidence"),
        "evidence": thesis.get("evidence", []) if isinstance(thesis, dict) else [],
    }
    if include_generated_signal:
        result["signal"] = _try_get("/v1/user/signal")
    return _trim(result)


@mcp.tool
def echo_decision_room(
    mode: str = "decide",
    question: str = "",
    run_id: str = "",
    candidate_id: str = "",
    session_id: str = "",
    chosen: str = "",
) -> dict:
    """
    Decision workflow.

    mode: decide | council | tournament | choose_tournament | twin | choose_twin | simulate
    """
    mode = mode.strip().lower()
    if mode == "decide":
        return _trim(_try_post("/v1/echo/decide", {"message": question, "is_decision": True}))
    if mode == "council":
        return _trim(_try_post("/v1/council/ask", {"question": question}))
    if mode == "tournament":
        return _trim(_try_post("/v1/tournament/run", {"prompt": question}))
    if mode == "choose_tournament":
        if not run_id or not candidate_id:
            return {"error": "run_id and candidate_id are required"}
        return _trim(_try_post("/v1/tournament/choose", {"run_id": run_id, "candidate_id": candidate_id}))
    if mode == "twin":
        return _trim(_try_post("/v1/twin/ask", {"question": question}))
    if mode == "choose_twin":
        if chosen not in ("A", "B"):
            return {"error": "chosen must be A or B"}
        if not session_id:
            return {"error": "session_id is required"}
        return _trim(_try_post("/v1/twin/choose", {"session_id": session_id, "chosen": chosen}))
    if mode == "simulate":
        return _trim(_try_post("/v1/echo/simulate", {}))
    return {"error": "invalid mode", "valid_modes": ["decide", "council", "tournament", "choose_tournament", "twin", "choose_twin", "simulate"]}


@mcp.tool
def echo_memory_editor(action: str = "list", target: str = "memory", text: str = "", item_id: str = "", limit: int = 50) -> dict:
    """
    Edit Echo's memory and operating system.

    action: list | add | delete
    target: memory | rule
    """
    action = action.strip().lower()
    target = target.strip().lower()
    if target not in ("memory", "rule"):
        return {"error": "target must be memory or rule"}
    if action == "list":
        path = "/v1/user/memories" if target == "memory" else "/v1/user/rules"
        data = _try_get(path)
        key = "memories" if target == "memory" else "rules"
        if key in data and isinstance(data[key], list):
            data[key] = data[key][:limit]
        return _trim(data)
    if action == "add":
        if not text.strip():
            return {"error": "text is required"}
        if target == "memory":
            return _trim(_try_post("/v1/user/memories", {"memory": text.strip()}))
        return _trim(_try_post("/v1/user/rules", {"rule_text": text.strip()}))
    if action == "delete":
        if not item_id:
            return {"error": "item_id is required for delete"}
        path = f"/v1/user/memories/{item_id}" if target == "memory" else f"/v1/user/rules/{item_id}"
        return _trim(_try_delete(path))
    return {"error": "invalid action", "valid_actions": ["list", "add", "delete"]}


@mcp.tool
def echo_signal_capture(
    action: str,
    user_message: str = "",
    assistant_response: str = "",
    quality_score: float = 0.8,
    text: str = "",
    event_type: str = "",
    title: str = "",
    summary: str = "",
    domain: str = "manual",
    outcome: str = "",
    context: str = "",
    rep_id: str = "",
    done: bool = True,
    q1: str = "",
    a1: str = "",
    q2: str = "",
    a2: str = "",
    q3: str = "",
    a3: str = "",
) -> dict:
    """
    Capture training and life signals.

    action: save_pair | memory | rule | life_event | outcome | practice | checkin
    """
    action = action.strip().lower()
    if action == "save_pair":
        if not user_message or not assistant_response:
            return {"error": "user_message and assistant_response are required"}
        signal = "thumbs_up" if quality_score >= 0.7 else "continue"
        return _trim(_try_post("/save", {
            "user_id": ECHO_USER_ID,
            "user_message": user_message,
            "assistant_message": assistant_response,
            "model_used": "mcp_workflow",
            "engagement_signal": signal,
        }))
    if action == "memory":
        return echo_memory_editor(action="add", target="memory", text=text)
    if action == "rule":
        return echo_memory_editor(action="add", target="rule", text=text)
    if action == "life_event":
        if not event_type or not title:
            return {"error": "event_type and title are required"}
        return _trim(_try_post("/v1/life/events", {
            "event_type": event_type,
            "event_domain": domain,
            "title": title,
            "summary": summary,
            "source": "mcp",
            "confidence": 0.9,
        }))
    if action == "outcome":
        if not outcome:
            return {"error": "outcome is required"}
        body = {"signal": outcome}
        if context:
            body["context"] = context
        return _trim(_try_post("/v1/outcome", body))
    if action == "practice":
        if not rep_id:
            return {"error": "rep_id is required"}
        return _trim(_try_post("/v1/practice/log", {"rep_id": rep_id, "done": done}))
    if action == "checkin":
        qas = [{"q": q1, "a": a1}, {"q": q2, "a": a2}, {"q": q3, "a": a3}]
        if not all(item["q"] and item["a"] for item in qas):
            return {"error": "q1/a1, q2/a2, and q3/a3 are required"}
        return _trim(_try_post("/v1/daily/checkin", {"qas": qas}))
    return {"error": "invalid action", "valid_actions": ["save_pair", "memory", "rule", "life_event", "outcome", "practice", "checkin"]}


@mcp.tool
def echo_threads_inbox(action: str = "list", thread_id: str = "", note: str = "user acknowledged") -> dict:
    """
    Manage Echo's active pattern threads.

    action: list | resolve | deduplicate
    """
    action = action.strip().lower()
    if action == "list":
        return _trim(_try_get("/v1/threads"))
    if action == "resolve":
        if not thread_id:
            return {"error": "thread_id is required"}
        return _trim(_try_post(f"/v1/threads/{thread_id}/resolve", {"note": note}))
    if action == "deduplicate":
        return _trim(_try_post("/v1/threads/deduplicate", {}))
    return {"error": "invalid action", "valid_actions": ["list", "resolve", "deduplicate"]}


# ── Core chat ────────────────────────────────────────────────────────────────

@mcp.tool
def chat(message: str, thread_id: str = "") -> str:
    """Send a message to Echo and get a personalized response from the user's current Echo context."""
    body: dict = {
        "model": "gemma4_e2b",
        "messages": [{"role": "user", "content": message}],
        "stream": False,
    }
    if thread_id:
        body["thread_id"] = thread_id
    try:
        data = _post("/v1/chat/completions", body)
    except Exception as e:
        return f"Echo chat failed: {e}"
    return data["choices"][0]["message"]["content"]


# ── Memory ───────────────────────────────────────────────────────────────────

@mcp.tool
def get_memories(limit: int = 20) -> str:
    """Retrieve what Echo remembers about you (mem0 memories)."""
    data = _get("/v1/user/memories")
    memories = data if isinstance(data, list) else data.get("memories", [])
    if not memories:
        return "No memories stored yet."
    lines = [f"- {m.get('memory', m)}" for m in memories[:limit]]
    return "\n".join(lines)


@mcp.tool
def get_operating_system() -> str:
    """Get your personal operating system — rules and preferences Echo has learned."""
    data = _get("/v1/user/rules")
    rules = data if isinstance(data, list) else data.get("rules", [])
    if not rules:
        return "No rules defined yet."
    return "\n".join(f"- {r.get('rule_text', r.get('rule', str(r)))}" for r in rules)


# ── Growth & insights ────────────────────────────────────────────────────────

@mcp.tool
def get_growth_timeline() -> str:
    """Get your growth timeline — key life events and milestones Echo has tracked."""
    data = _get("/v1/growth/timeline")
    milestones = data.get("milestones", data if isinstance(data, list) else [])
    stats = data.get("stats", {})
    headline = data.get("headline", "")
    if not milestones:
        return "No growth events recorded yet."
    lines = []
    if headline:
        lines.append(headline)
    if stats:
        lines.append(f"Total: {stats.get('milestones',0)} milestones | {stats.get('model_updates',0)} model updates | {stats.get('clone_battles',0)} decision comparisons | {stats.get('practice_done',0)} practice reps")
    lines.append("")
    for e in milestones[:20]:
        date = (e.get("date") or e.get("created_at", ""))[:10]
        label = e.get("title") or e.get("type") or e.get("event_type", "event")
        note = e.get("summary") or e.get("note") or ""
        lines.append(f"[{date}] {label}: {note}" if note else f"[{date}] {label}")
    return "\n".join(lines)


@mcp.tool
def get_today() -> str:
    """Get today's priority, mission, and reality check from Echo."""
    priority = _get("/v1/today/priority")
    mission = _get("/v1/today/mission")
    reality = _get("/v1/reality/check")
    return (
        f"Priority: {priority.get('priority', priority)}\n"
        f"Mission: {mission.get('mission', mission)}\n"
        f"Reality check: {reality.get('check', reality)}"
    )


@mcp.tool
def get_insights() -> str:
    """Get Echo's latest insights and patterns observed about you."""
    data = _get("/v1/user/insights")
    # Response is a flat object: {turns_analyzed, new_patterns, accuracy_delta, latest_pattern}
    if isinstance(data, list):
        return "\n".join(f"- {i.get('insight', i)}" for i in data[:10])
    pattern = data.get("latest_pattern") or data.get("insight") or ""
    if not pattern and not data.get("turns_analyzed"):
        return "No insights yet — keep chatting to build patterns."
    lines = []
    if pattern:
        lines.append(f"Latest pattern: {pattern}")
    if data.get("turns_analyzed") is not None:
        lines.append(f"Turns analyzed: {data['turns_analyzed']}")
    if data.get("new_patterns") is not None:
        lines.append(f"New patterns found: {data['new_patterns']}")
    if data.get("accuracy_delta"):
        lines.append(f"Accuracy delta: {data['accuracy_delta']}")
    return "\n".join(lines) if lines else "No insights yet."


@mcp.tool
def get_skills() -> str:
    """Get the skills and strengths Echo has identified from your conversations."""
    data = _get("/v1/user/skills")
    skills = data if isinstance(data, list) else data.get("skills", [])
    if not skills:
        return "No skills extracted yet. Skills are generated weekly from your conversation patterns — keep chatting and they'll appear after the next weekly cycle."
    return "\n".join(
        f"- {s.get('skill_name', s.get('skill', s.get('name', str(s))))}: {s.get('trigger', '')}"
        for s in skills
    )


# ── Clone / training ─────────────────────────────────────────────────────────

@mcp.tool
def get_clone_status() -> str:
    """Check the status of your personal AI clone — adapter loaded, training state, eval score."""
    data = _get("/v1/system/health")
    adapter = data.get("adapter", {})
    return (
        f"Adapter loaded: {adapter.get('loaded', False)}\n"
        f"Serving model: {adapter.get('serving_model', 'base')}\n"
        f"Training status: {data.get('training_status', 'idle')}\n"
        f"Last eval score: {data.get('last_eval_score', 'N/A')}"
    )


@mcp.tool
def get_training_summary() -> str:
    """Get a summary of training runs — pairs used, eval scores, adapter history."""
    data = _get("/v1/training/summary")
    runs = _get("/v1/training/runs")
    run_list = runs if isinstance(runs, list) else runs.get("runs", [])
    lines = [
        f"Untrained pairs: {data.get('untrained_pairs', '?')}",
        f"Required for training: {data.get('required_pairs', 20)}",
        f"Total runs: {len(run_list)}",
    ]
    for r in run_list[:3]:
        score = r.get("eval_score") or r.get("eval", {}).get("score", "?")
        lines.append(f"  Run {r.get('id','')[:8]}: {r.get('status')} score={score}")
    return "\n".join(lines)


@mcp.tool
def trigger_training(confirm: bool = False) -> str:
    """Legacy training trigger. Prefer echo_training_center(action='trigger', confirm=True)."""
    if not confirm:
        return "Confirmation required. Call trigger_training(confirm=True), or use echo_training_center(action='trigger', confirm=True)."
    data = _post("/trigger-training", {})
    return data.get("message") or str(data)


# ── Signals ──────────────────────────────────────────────────────────────────

@mcp.tool
def log_outcome(signal: str, context: str = "") -> str:
    """
    Log an outcome signal for the last conversation.
    signal: one of thumbs_up, thumbs_down, saved_signal, helped, practice_request
    """
    valid = {"thumbs_up", "thumbs_down", "saved_signal", "helped", "practice_request"}
    if signal not in valid:
        return f"Invalid signal. Use one of: {', '.join(valid)}"
    body: dict = {"signal": signal}
    if context:
        body["context"] = context
    data = _post("/v1/outcome", body)
    return data.get("message") or "Signal logged."


# ── Teacher tools — YOU are the teacher when using Claude Code / GPT-5 / etc. ──
#
# How to use these from Claude Code:
#   1. Call chat(message) to get Echo's response.
#   2. Read the response. You (the LLM) evaluate its quality.
#   3. Call save_training_pair(user_msg, response, score) to feed the training pipeline.
#   4. If the user shared a personal fact, call save_memory(fact).
#   5. If the user stated a preference, call add_rule(rule).
#
# This replaces the Gemma 4 31B teacher API call entirely — you're smarter anyway.

@mcp.tool
def save_training_pair(user_message: str, assistant_response: str, quality_score: float) -> str:
    """
    Save a conversation turn as a training pair for Echo's next LoRA fine-tuning run.

    Call this after chat() when the response was good (quality_score >= 0.6).
    quality_score: 0.0–1.0. Use your own judgment:
      1.0 = perfect, highly personalized
      0.7 = good, relevant
      0.5 = acceptable but generic
      below 0.6 = don't save
    """
    if quality_score < 0.0 or quality_score > 1.0:
        return "quality_score must be between 0.0 and 1.0"
    signal = "thumbs_up" if quality_score >= 0.7 else "continue"
    data = _post("/save", {
        "user_id": ECHO_USER_ID,
        "user_message": user_message,
        "assistant_message": assistant_response,
        "model_used": "mcp_teacher",
        "engagement_signal": signal,
    })
    return data.get("message") or f"Pair saved (score={quality_score}, signal={signal})."


@mcp.tool
def save_memory(fact: str) -> str:
    """
    Save a personal fact about the user into Echo's memory (mem0).

    Call this when the user shares something worth remembering:
    personal details, goals, preferences, important life events.
    Write the fact as a single clear sentence, e.g.:
      'User is working as a freelance CTO for an Italian startup.'
      'User prefers direct answers without bullet points.'
      'User wants to add swimming to their daily routine.'
    """
    fact = fact.strip()
    if not fact:
        return "fact cannot be empty"
    data = _post("/v1/user/memories", {"memory": fact})
    return data.get("message") or f"Memory saved: {fact}"


@mcp.tool
def add_rule(rule: str) -> str:
    """
    Add a preference rule to Echo's operating system.

    Call this when the user states a clear communication preference or behavioral rule.
    Write it as an actionable instruction, e.g.:
      'Always give a direct answer first, then explain reasoning.'
      'Never use bullet points unless explicitly asked.'
      'When user asks about work situations, use the Challenger style.'
    """
    rule = rule.strip()
    if not rule:
        return "rule cannot be empty"
    data = _post("/v1/user/rules", {"rule_text": rule})
    return data.get("message") or f"Rule added: {rule}"


# ── Identity & stats ─────────────────────────────────────────────────────────

@mcp.tool
def get_user_stats() -> str:
    """Get your Echo stats — total pairs, signals, memories, training runs, and usage summary."""
    data = _get("/v1/user/stats")
    return "\n".join(f"{k}: {v}" for k, v in data.items() if not isinstance(v, dict))


@mcp.tool
def get_user_rank() -> str:
    """Get your Echo growth stage and readiness progress."""
    data = _get("/v1/user/rank")
    return (
        f"Stage: {data.get('rank')} - {data.get('title')}\n"
        f"Readiness: {data.get('xp')} (next stage in {data.get('xp_to_next')})\n"
        f"Progress: {round((data.get('progress', 0)) * 100)}%"
    )


@mcp.tool
def get_thesis() -> str:
    """Get your current thesis — Echo's durable belief about who you are and where you're headed."""
    data = _get("/v1/thesis/current")
    if not data or not data.get("statement"):
        return "No thesis formed yet — keep chatting."
    return f"{data.get('title', 'Thesis')}: {data.get('statement', '')}"


@mcp.tool
def get_loop_snapshot() -> str:
    """Get a snapshot of your current momentum — what Echo is tracking about your active patterns."""
    data = _get("/v1/loop/snapshot")
    if not data:
        return "No loop data yet."
    lines = []
    if data.get("headline"):
        lines.append(data["headline"])
    if data.get("event_count") is not None:
        lines.append(f"Events tracked: {data['event_count']}")
    if data.get("untrained_pairs") is not None:
        lines.append(f"Untrained pairs queued: {data['untrained_pairs']}")
    for e in data.get("latest_events", [])[:5]:
        lines.append(f"  [{e.get('created_at','')[:10]}] {e.get('event_type','')} — {e.get('summary','')[:100]}")
    return "\n".join(lines) if lines else str(data)


@mcp.tool
def get_notable_quote() -> str:
    """Get the most notable quote Echo has saved from your conversations. May take ~10s (calls teacher LLM)."""
    with httpx.Client(timeout=45) as client:
        r = client.get(f"{ECHO_BASE}/v1/user/notable-quote", headers=_headers())
        r.raise_for_status()
        data = r.json()
    quote = data.get("quote")
    if not quote:
        return "No notable quote found yet."
    why = data.get("why", "")
    date = (data.get("date") or "")[:10]
    return f'"{quote}"\n{why}' + (f"\n[{date}]" if date else "")


@mcp.tool
def get_full_report() -> str:
    """
    Get your complete Echo report — stats, rank, thesis, signals, training, and recent activity.
    This is the single most comprehensive view of everything Echo knows.
    """
    data = _get("/v1/user/report")
    import json as _json
    return _json.dumps(data, indent=2)


# ── Life events ───────────────────────────────────────────────────────────────

@mcp.tool
def log_life_event(event_type: str, title: str, summary: str = "", domain: str = "manual") -> str:
    """
    Log a significant life event into Echo's growth timeline.

    Use this when the user mentions something important that happened:
    promotions, relationships, decisions, achievements, setbacks.

    event_type: short label, e.g. 'job_change', 'moved_city', 'relationship_start'
    domain: 'work', 'health', 'relationships', 'finance', 'personal', 'manual'
    """
    data = _post("/v1/life/events", {
        "event_type": event_type,
        "event_domain": domain,
        "title": title,
        "summary": summary,
        "source": "mcp",
        "confidence": 0.9,
    })
    return f"Life event logged: {title}" if data.get("saved") else str(data)


# ── Daily check-in ────────────────────────────────────────────────────────────

@mcp.tool
def get_daily_questions() -> str:
    """Get today's 3 personalized evening check-in questions from Echo."""
    data = _get("/v1/daily/questions")
    questions = data.get("questions", [])
    if not questions:
        return "No questions generated yet."
    return "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))


@mcp.tool
def get_checkin_status() -> str:
    """Check whether today's evening check-in is already done."""
    data = _get("/v1/daily/checkin/status")
    done = data.get("done", False)
    date = data.get("date", "today")
    return f"Check-in for {date}: {'done' if done else 'not done yet'}"


@mcp.tool
def submit_daily_checkin(q1: str, a1: str, q2: str, a2: str, q3: str, a3: str) -> str:
    """
    Submit the evening check-in with 3 question/answer pairs.
    Get today's questions first with get_daily_questions(), then submit answers here.
    Returns Echo's synthesis bullets from your answers.
    """
    qas = [
        {"q": q1, "a": a1},
        {"q": q2, "a": a2},
        {"q": q3, "a": a3},
    ]
    data = _post("/v1/daily/checkin", {"qas": qas})
    synthesis = data.get("synthesis", [])
    if synthesis:
        return "Echo's synthesis:\n" + "\n".join(f"• {s}" for s in synthesis)
    return "Check-in saved."


# ── Practice reps ─────────────────────────────────────────────────────────────

@mcp.tool
def get_practice_today() -> str:
    """Get today's behavioral rep — a single concrete action Echo designed based on your patterns."""
    data = _get("/v1/practice/today")
    if "error" in data:
        return data["error"]
    return (
        f"Observation: {data.get('observation', '')}\n"
        f"Rep: {data.get('rep_title', '')}\n"
        f"Instruction: {data.get('rep_instruction', '')}\n"
        f"Arc: {data.get('arc_label', '')}\n"
        f"Done today: {data.get('done', False)}\n"
        f"Week completions: {data.get('week_completions', 0)}\n"
        f"rep_id: {data.get('rep_id', '')}"
    )


@mcp.tool
def log_practice(rep_id: str, done: bool = True) -> str:
    """
    Mark today's practice rep as done or skipped.
    Get rep_id from get_practice_today() first.
    """
    data = _post("/v1/practice/log", {"rep_id": rep_id, "done": done})
    return data.get("message") or f"Practice {'completed' if done else 'skipped'}."


# ── Tournament (style battles) ────────────────────────────────────────────────

@mcp.tool
def run_tournament(prompt: str) -> str:
    """
    Run a style tournament — four clone personalities (Builder, Challenger, Mirror, Strategist)
    each respond to your prompt. You pick the one that feels most like you.
    Returns run_id and all four responses. Use choose_tournament_winner() to record your pick.
    """
    data = _post("/v1/tournament/run", {"prompt": prompt})
    if "error" in data:
        return data["error"]
    candidates = data.get("candidates", [])
    run_id = data.get("run_id", "")
    lines = [f"run_id: {run_id}", ""]
    for c in candidates:
        lines.append(f"[{c.get('style', c.get('id', ''))}] id={c.get('id', '')}")
        lines.append(c.get("response", c.get("text", ""))[:400])
        lines.append("")
    return "\n".join(lines)


@mcp.tool
def choose_tournament_winner(run_id: str, candidate_id: str) -> str:
    """
    Record which tournament response felt most like you.
    Use run_id and candidate_id from run_tournament() output.
    This creates a DPO training pair — winner is 'chosen', others are 'rejected'.
    """
    data = _post("/v1/tournament/choose", {"run_id": run_id, "candidate_id": candidate_id})
    return data.get("message") or f"Winner recorded: {candidate_id}"


# ── Twin (clone vs teacher comparison) ───────────────────────────────────────

@mcp.tool
def ask_twin(question: str) -> str:
    """
    Ask the same question to two anonymous versions — one is your trained clone, one is the base model.
    Both answers are shown as A and B (randomly shuffled). Pick which feels more like you.
    Use choose_twin_winner() with the session_id to record your choice as DPO data.
    Requires a trained adapter to be loaded.
    """
    data = _post("/v1/twin/ask", {"question": question})
    if "error" in data:
        return data["error"]
    return (
        f"session_id: {data.get('session_id')}\n\n"
        f"Response A:\n{data.get('response_a', '')}\n\n"
        f"Response B:\n{data.get('response_b', '')}"
    )


@mcp.tool
def choose_twin_winner(session_id: str, chosen: str) -> str:
    """
    Record which twin response felt more like you ('A' or 'B').
    This saves a DPO training pair where the chosen response is the preferred output.
    """
    if chosen not in ("A", "B"):
        return "chosen must be 'A' or 'B'"
    data = _post("/v1/twin/choose", {"session_id": session_id, "chosen": chosen})
    return data.get("message") or f"Choice '{chosen}' recorded as DPO pair."


# ── Council (multi-persona parallel responses) ────────────────────────────────

@mcp.tool
def ask_council(question: str) -> str:
    """
    Ask all 5 clone personas in parallel — Builder, Challenger, Mirror, Strategist, Creative.
    Returns each voice's response and a consensus verdict.
    Great for decisions where you want multiple angles before choosing.
    """
    data = _post("/v1/council/ask", {"question": question})
    if "error" in data:
        return data["error"]
    voices = data.get("voices", {})
    verdict = data.get("verdict", "")
    lines = []
    for name, text in voices.items():
        lines.append(f"[{name}]\n{text}\n")
    lines.append(f"Verdict: {verdict}")
    return "\n".join(lines)


# ── Parallel self simulation ──────────────────────────────────────────────────

@mcp.tool
def simulate_parallel_self() -> str:
    """
    Run a parallel self simulation — two diverging paths based on your current patterns.
    Not prediction. Pattern-based projection: current path vs. the avoided path.
    Shows where your habits lead if continued vs. what changes if you course-correct.
    """
    data = _post("/v1/echo/simulate", {})
    if "error" in data:
        return data["error"]
    current = data.get("current_path", {})
    avoided = data.get("avoided_path", {})
    return (
        f"Current path — {current.get('label', '')}:\n{current.get('projection', '')}\n{current.get('detail', '')}\n\n"
        f"Avoided path — {avoided.get('label', '')}:\n{avoided.get('projection', '')}\n{avoided.get('detail', '')}"
    )


# ── Conversation history ──────────────────────────────────────────────────────

@mcp.tool
def get_conversation_history(limit: int = 20) -> str:
    """Get your last N conversation turns stored in Echo's training pipeline."""
    data = _get("/v1/user/history")
    pairs = data if isinstance(data, list) else data.get("pairs", data.get("history", []))
    if not pairs:
        return "No conversation history yet."
    lines = []
    for p in pairs[:limit]:
        msg = (p.get("user_msg") or p.get("user_message") or "")[:120]
        resp = (p.get("assistant_msg") or p.get("assistant_message") or "")[:120]
        lines.append(f"Q: {msg}\nA: {resp}\n")
    return "\n".join(lines)


# ── Proof portfolio ───────────────────────────────────────────────────────────

@mcp.tool
def echo_proof(
    action: str = "list",
    title: str = "",
    description: str = "",
    category: str = "practice",
    evidence: str = "",
    skill_tags: str = "",
    opportunity_type: str = "personal_goal",
    item_id: str = "",
    seed_from_history: bool = False,
) -> dict:
    """
    Manage the user's proof portfolio — concrete evidence of skills and growth.

    action: list | add | delete | seed
    category: practice | achievement | learning | project | pattern | challenge | decision
    opportunity_type: personal_goal | job | school | scholarship | project

    seed=True auto-populates proof from existing life events and thesis evidence.
    Call echo_opportunities() to turn proof items into an opportunity plan.
    """
    action = action.strip().lower()

    if action in {"list", "seed"} or seed_from_history:
        result: dict = {}
        if action == "seed" or seed_from_history:
            result["seed"] = _try_post("/v1/proof/seed", {})
        data = _try_get("/v1/proof/items")
        items = data.get("items", []) if isinstance(data, dict) else []
        summary = data.get("summary", {}) if isinstance(data, dict) else {}
        result["count"] = summary.get("count", len(items))
        result["by_category"] = summary.get("by_category", [])
        result["items"] = [
            {
                "id": i.get("id", ""),
                "title": i.get("title", ""),
                "category": i.get("category", ""),
                "evidence": (i.get("evidence") or "")[:200],
                "skill_tags": i.get("skill_tags", []),
            }
            for i in items[:20]
        ]
        return _trim(result)

    if action == "add":
        if not title.strip():
            return {"error": "title is required"}
        tags = [t.strip() for t in skill_tags.split(",") if t.strip()] if skill_tags else []
        return _trim(_try_post("/v1/proof/items", {
            "title": title.strip(),
            "description": description.strip(),
            "category": category.strip() or "practice",
            "evidence": evidence.strip(),
            "skill_tags": tags,
            "opportunity_type": opportunity_type.strip() or "personal_goal",
        }))

    if action == "delete":
        if not item_id.strip():
            return {"error": "item_id is required"}
        return _trim(_try_delete(f"/v1/proof/items/{item_id.strip()}"))

    return {"error": "invalid action", "valid_actions": ["list", "add", "delete", "seed"]}


# ── Opportunities ─────────────────────────────────────────────────────────────

@mcp.tool
def echo_opportunities(
    action: str = "list",
    title: str = "",
    opportunity_type: str = "personal_goal",
    description: str = "",
    required_proof: str = "",
    missing_proof: str = "",
    next_step: str = "",
) -> dict:
    """
    Manage and generate opportunity plans from the user's proof portfolio.

    action: list | generate | add
    opportunity_type: personal_goal | job | school | scholarship | project

    generate — asks Echo to analyze current proof and produce an AI opportunity plan.
    add — manually create an opportunity with required/missing proof checklist.
    list — show all saved opportunities with their proof requirements.

    Workflow: echo_proof(action='list') → echo_opportunities(action='generate') → review plan
    """
    action = action.strip().lower()

    if action == "list":
        data = _try_get("/v1/opportunities")
        items = data.get("items", []) if isinstance(data, dict) else []
        proof_summary = data.get("proof_summary", {}) if isinstance(data, dict) else {}
        return _trim({
            "proof_count": proof_summary.get("count", 0),
            "opportunities": [
                {
                    "id": i.get("id", ""),
                    "title": i.get("title", ""),
                    "type": i.get("type", ""),
                    "next_step": i.get("next_step", ""),
                    "required_proof": (i.get("required_proof") or [])[:4],
                    "missing_proof": (i.get("missing_proof") or [])[:4],
                    "generated": i.get("generated", False),
                }
                for i in items[:10]
            ],
        })

    if action == "generate":
        result = _try_post("/v1/opportunities/generate", {})
        return _trim(result)

    if action == "add":
        if not title.strip():
            return {"error": "title is required"}
        required = [p.strip() for p in required_proof.split(",") if p.strip()] if required_proof else []
        missing = [p.strip() for p in missing_proof.split(",") if p.strip()] if missing_proof else []
        return _trim(_try_post("/v1/opportunities", {
            "title": title.strip(),
            "type": opportunity_type.strip() or "personal_goal",
            "description": description.strip(),
            "required_proof": required,
            "missing_proof": missing,
            "next_step": next_step.strip(),
        }))

    return {"error": "invalid action", "valid_actions": ["list", "generate", "add"]}


if __name__ == "__main__":
    mcp.run()  # stdio by default
