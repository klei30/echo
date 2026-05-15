import json
import uuid
from dataclasses import dataclass, asdict
from typing import Any

from config import settings
from db.database import get_conn


@dataclass
class TeacherDecision:
    allowed: bool
    purpose: str
    reason: str
    meaningful_pairs: int = 0
    daily_used: int = 0
    weekly_used: int = 0
    daily_cap: int = 0
    weekly_cap: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_IMPORTANT_MARKERS = (
    "important",
    "serious",
    "decision",
    "career",
    "health",
    "medical",
    "legal",
    "financial",
    "money",
    "relationship",
    "risk",
)


def infer_importance(text: str | None, explicit: str | None = None) -> str:
    if explicit in {"low", "normal", "high", "critical"}:
        return explicit
    lower = (text or "").lower()
    if any(marker in lower for marker in _IMPORTANT_MARKERS):
        return "high"
    return "normal"


async def _teacher_stats(user_id: str) -> tuple[int, int, int, int]:
    async with get_conn() as db:
        async with db.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM training_pairs
            WHERE user_id=?
              AND engagement_signal != 'thumbs_down'
              AND perplexity >= ?
            """,
            (user_id, settings.quality_threshold),
        ) as cur:
            row = await cur.fetchone()
            meaningful_pairs = int(row["cnt"] if row else 0)

        async with db.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM teacher_usage
            WHERE user_id=? AND created_at >= datetime('now', '-1 day')
            """,
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
            daily_used = int(row["cnt"] if row else 0)

        async with db.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM teacher_usage
            WHERE user_id=? AND created_at >= datetime('now', '-7 days')
            """,
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
            weekly_used = int(row["cnt"] if row else 0)

        async with db.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM shadow_outcomes
            WHERE user_id=?
              AND outcome IN ('not_true', 'thumbs_down', 'failed')
              AND created_at >= datetime('now', '-7 days')
            """,
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
            recent_negative = int(row["cnt"] if row else 0)

    return meaningful_pairs, daily_used, weekly_used, recent_negative


def _budget_for(meaningful_pairs: int) -> tuple[int, int]:
    if meaningful_pairs < settings.teacher_new_user_pair_threshold:
        return settings.teacher_new_user_daily_cap, max(settings.teacher_weekly_cap, settings.teacher_new_user_daily_cap)
    return settings.teacher_trained_daily_cap, settings.teacher_weekly_cap


def _within_budget(daily_used: int, weekly_used: int, daily_cap: int, weekly_cap: int) -> bool:
    return daily_used < daily_cap and weekly_used < weekly_cap


async def should_use_teacher(
    user_id: str,
    purpose: str,
    *,
    confidence: float | None = None,
    importance: str = "normal",
    prompt: str | None = None,
    recent_failure: bool = False,
    explicit_user_request: bool = False,
) -> TeacherDecision:
    if not settings.teacher_policy_enabled:
        return TeacherDecision(False, purpose, "teacher_policy_disabled")
    if not settings.llm_api_key:
        return TeacherDecision(False, purpose, "missing_teacher_api_key")

    importance = infer_importance(prompt, importance)
    meaningful_pairs, daily_used, weekly_used, recent_negative = await _teacher_stats(user_id)
    daily_cap, weekly_cap = _budget_for(meaningful_pairs)
    in_budget = _within_budget(daily_used, weekly_used, daily_cap, weekly_cap)

    base = {
        "purpose": purpose,
        "meaningful_pairs": meaningful_pairs,
        "daily_used": daily_used,
        "weekly_used": weekly_used,
        "daily_cap": daily_cap,
        "weekly_cap": weekly_cap,
    }

    if recent_failure:
        return TeacherDecision(in_budget, reason="failure_recovery" if in_budget else "budget_exhausted", **base)

    if purpose == "cold_start":
        allowed = meaningful_pairs < settings.teacher_new_user_pair_threshold and in_budget
        return TeacherDecision(allowed, reason="new_user_seed" if allowed else "not_cold_start_or_budget", **base)

    if purpose == "tournament_challenger":
        allowed_reason = None
        if meaningful_pairs < settings.teacher_new_user_pair_threshold:
            allowed_reason = "cold_start_tournament"
        elif confidence is not None and confidence < 0.55:
            allowed_reason = "low_confidence_tournament"
        elif importance in {"high", "critical"}:
            allowed_reason = "important_tournament"
        elif recent_negative > 0:
            allowed_reason = "recent_negative_feedback"
        allowed = bool(allowed_reason and in_budget)
        return TeacherDecision(allowed, reason=allowed_reason or ("budget_exhausted" if not in_budget else "gemma_only_tournament"), **base)

    if purpose == "judge_answer":
        allowed = in_budget and (
            (confidence is not None and confidence < 0.55)
            or importance in {"high", "critical"}
            or recent_negative > 0
        )
        return TeacherDecision(allowed, reason="quality_calibration" if allowed else "not_needed", **base)

    if purpose == "memory_update":
        allowed = in_budget and (
            meaningful_pairs < settings.teacher_new_user_pair_threshold
            or importance in {"high", "critical"}
            or explicit_user_request
        )
        return TeacherDecision(allowed, reason="selective_memory_update" if allowed else "skip_routine_memory_update", **base)

    if purpose == "preference_extraction":
        allowed = in_budget
        return TeacherDecision(allowed, reason="user_preference_signal" if allowed else "budget_exhausted", **base)

    if purpose == "twin_compare":
        allowed = in_budget and explicit_user_request
        return TeacherDecision(allowed, reason="explicit_teacher_comparison" if allowed else "budget_exhausted", **base)

    if purpose == "tool_chat":
        allowed = in_budget and (importance in {"high", "critical"} or explicit_user_request)
        return TeacherDecision(allowed, reason="tool_requires_teacher" if allowed else "tool_teacher_not_essential", **base)

    if purpose in {"weekly_calibration", "monthly_distillation"}:
        allowed = in_budget
        return TeacherDecision(allowed, reason="scheduled_calibration" if allowed else "budget_exhausted", **base)

    return TeacherDecision(False, purpose, "purpose_not_allowed", meaningful_pairs, daily_used, weekly_used, daily_cap, weekly_cap)


async def record_teacher_usage(
    user_id: str,
    purpose: str,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    usage_id = str(uuid.uuid4())
    async with get_conn() as db:
        await db.execute(
            """
            INSERT INTO teacher_usage (id, user_id, purpose, reason, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (usage_id, user_id, purpose, reason, json.dumps(metadata or {}, ensure_ascii=True)),
        )
        await db.commit()
    return usage_id
