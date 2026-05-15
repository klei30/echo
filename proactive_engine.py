import json
import uuid
from datetime import datetime as dt, timedelta
from typing import Any

from db.database import get_conn
from loop_priority import get_today_priority
from thesis import get_current_thesis
from training.summary import get_training_summary


def _json(data: dict[str, Any] | None) -> str:
    return json.dumps(data or {}, ensure_ascii=True, sort_keys=True)


def _loads(value: str | None) -> dict[str, Any]:
    try:
        return json.loads(value or "{}")
    except Exception:
        return {}


def _today() -> str:
    return dt.utcnow().strftime("%Y-%m-%d")


def _iso(value: dt) -> str:
    return value.replace(microsecond=0).isoformat()


async def get_intervention_settings(user_id: str) -> dict[str, Any]:
    async with get_conn() as db:
        await db.execute(
            """
            INSERT OR IGNORE INTO intervention_settings (user_id)
            VALUES (?)
            """,
            (user_id,),
        )
        await db.commit()
        async with db.execute(
            "SELECT * FROM intervention_settings WHERE user_id=?",
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
    data = dict(row) if row else {}
    return {
        "enabled": bool(data.get("enabled", 1)),
        "categories": {
            "morning": bool(data.get("morning_enabled", 1)),
            "evening": bool(data.get("evening_enabled", 1)),
            "clone": bool(data.get("clone_enabled", 1)),
            "training": bool(data.get("training_enabled", 1)),
            "proof": bool(data.get("proof_enabled", 1)),
        },
        "quiet_hours": {
            "start": int(data.get("quiet_start", 22)),
            "end": int(data.get("quiet_end", 8)),
        },
        "max_per_day": int(data.get("max_per_day", 3)),
    }


async def update_intervention_settings(user_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    current = await get_intervention_settings(user_id)
    categories = dict(current["categories"])
    categories.update(patch.get("categories") or {})
    quiet = dict(current["quiet_hours"])
    quiet.update(patch.get("quiet_hours") or {})
    max_per_day = int(patch.get("max_per_day", current["max_per_day"]))
    async with get_conn() as db:
        await db.execute(
            """
            INSERT INTO intervention_settings
                (user_id, enabled, morning_enabled, evening_enabled, clone_enabled,
                 training_enabled, proof_enabled, quiet_start, quiet_end, max_per_day, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(user_id) DO UPDATE SET
                enabled=excluded.enabled,
                morning_enabled=excluded.morning_enabled,
                evening_enabled=excluded.evening_enabled,
                clone_enabled=excluded.clone_enabled,
                training_enabled=excluded.training_enabled,
                proof_enabled=excluded.proof_enabled,
                quiet_start=excluded.quiet_start,
                quiet_end=excluded.quiet_end,
                max_per_day=excluded.max_per_day,
                updated_at=datetime('now')
            """,
            (
                user_id,
                1 if patch.get("enabled", current["enabled"]) else 0,
                1 if categories.get("morning", True) else 0,
                1 if categories.get("evening", True) else 0,
                1 if categories.get("clone", True) else 0,
                1 if categories.get("training", True) else 0,
                1 if categories.get("proof", True) else 0,
                int(quiet.get("start", 22)),
                int(quiet.get("end", 8)),
                max(0, min(max_per_day, 10)),
            ),
        )
        await db.commit()
    return await get_intervention_settings(user_id)


def _inside_quiet_hour(hour: int, start: int, end: int) -> bool:
    if start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def _scheduled_time(settings: dict[str, Any]) -> str:
    now = dt.utcnow()
    quiet = settings["quiet_hours"]
    start = int(quiet["start"])
    end = int(quiet["end"])
    if not _inside_quiet_hour(now.hour, start, end):
        return _iso(now + timedelta(minutes=2))
    scheduled = now.replace(hour=end, minute=0, second=0, microsecond=0)
    if scheduled <= now:
        scheduled += timedelta(days=1)
    return _iso(scheduled)


def _intervention_from_row(row: Any) -> dict[str, Any]:
    data = dict(row)
    payload = _loads(data.pop("action_payload_json", "{}"))
    data["action"] = {
        "type": data.pop("action_type", "open_today"),
        "payload": payload,
    }
    return data


async def _daily_intervention_count(db, user_id: str) -> int:
    async with db.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM echo_interventions
        WHERE user_id=? AND date(created_at)=date('now')
          AND status IN ('pending', 'delivered', 'acknowledged')
        """,
        (user_id,),
    ) as cur:
        row = await cur.fetchone()
    return int(row["cnt"] if row else 0)


async def _insert_intervention(
    user_id: str,
    kind: str,
    title: str,
    body: str,
    reason: str,
    action_type: str,
    action_payload: dict[str, Any] | None,
    priority: int,
    settings: dict[str, Any],
    source_event_id: str | None = None,
) -> dict[str, Any] | None:
    if not settings["enabled"]:
        return None
    category = {
        "morning_mission": "morning",
        "evening_reflection": "evening",
        "clone_returned": "clone",
        "training_ready": "training",
        "growth_proof": "proof",
        "reality_check": "proof",
        "talent_revelation": "proof",
    }.get(kind, "morning")
    if not settings["categories"].get(category, True):
        return None

    async with get_conn() as db:
        if await _daily_intervention_count(db, user_id) >= settings["max_per_day"]:
            return None
        async with db.execute(
            """
            SELECT *
            FROM echo_interventions
            WHERE user_id=? AND kind=? AND status IN ('pending', 'delivered')
              AND date(created_at)=date('now')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id, kind),
        ) as cur:
            existing = await cur.fetchone()
        if existing:
            return _intervention_from_row(existing)

        intervention_id = str(uuid.uuid4())
        scheduled_for = _scheduled_time(settings)
        await db.execute(
            """
            INSERT INTO echo_interventions
                (id, user_id, kind, title, body, reason, source_event_id,
                 action_type, action_payload_json, priority, scheduled_for, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                intervention_id,
                user_id,
                kind,
                title[:180],
                body[:900],
                reason[:500],
                source_event_id,
                action_type,
                _json(action_payload),
                priority,
                scheduled_for,
                _iso(dt.utcnow() + timedelta(days=2)),
            ),
        )
        await db.commit()
        async with db.execute(
            "SELECT * FROM echo_interventions WHERE id=?",
            (intervention_id,),
        ) as cur:
            row = await cur.fetchone()
    return _intervention_from_row(row) if row else None


async def get_reality_check(user_id: str) -> dict[str, Any]:
    async with get_conn() as db:
        async with db.execute(
            """
            SELECT user_msg, topic, created_at
            FROM training_pairs
            WHERE user_id=?
              AND (
                lower(user_msg) LIKE '%want%' OR lower(user_msg) LIKE '%need%' OR
                lower(user_msg) LIKE '%trying%' OR lower(user_msg) LIKE '%build%' OR
                lower(user_msg) LIKE '%learn%' OR lower(user_msg) LIKE '%goal%'
              )
            ORDER BY created_at DESC
            LIMIT 6
            """,
            (user_id,),
        ) as cur:
            stated = [dict(r) for r in await cur.fetchall()]
        async with db.execute(
            """
            SELECT le.event_domain, le.event_type, le.source, le.title, le.summary, le.created_at
            FROM life_events le
            WHERE le.user_id=? AND le.event_domain NOT IN ('echo')
            ORDER BY le.created_at DESC
            LIMIT 8
            """,
            (user_id,),
        ) as cur:
            behavior = [dict(r) for r in await cur.fetchall()]
        async with db.execute(
            """
            SELECT pl.done, pl.date, pr.rep_title, pr.arc_label
            FROM practice_log pl
            JOIN practice_reps pr ON pr.id=pl.rep_id
            WHERE pl.user_id=?
            ORDER BY pl.logged_at DESC
            LIMIT 8
            """,
            (user_id,),
        ) as cur:
            practice = [dict(r) for r in await cur.fetchall()]

    completed = sum(1 for r in practice if int(r.get("done") or 0) == 1)
    skipped = sum(1 for r in practice if int(r.get("done") or 0) == 0)
    checks: list[dict[str, Any]] = []

    if stated and not behavior:
        checks.append({
            "kind": "data_gap",
            "title": "Echo can hear your goals, but cannot see enough behavior yet.",
            "claim": stated[0]["user_msg"][:180],
            "behavior": "Current proof is limited to chat, practice reps, clone choices, and outcomes.",
            "next_step": "Log today's rep or connect a real-world source later when the core loop is stable.",
            "confidence": "early",
        })
    if stated and skipped > completed:
        checks.append({
            "kind": "practice_gap",
            "title": "There is a gap between intent and reps.",
            "claim": stated[0]["user_msg"][:180],
            "behavior": f"Recent practice logs show {skipped} skipped and {completed} completed.",
            "next_step": "Make tomorrow's mission smaller enough that it can be completed once.",
            "confidence": "emerging",
        })
    if completed >= 3:
        checks.append({
            "kind": "proof",
            "title": "There is behavioral proof, not just self-description.",
            "claim": "You have been testing Echo's read through reps.",
            "behavior": f"{completed} recent practice reps were completed.",
            "next_step": "Use the next clone mission to make this pattern repeatable.",
            "confidence": "strong" if completed >= 6 else "emerging",
        })
    if behavior:
        checks.append({
            "kind": "external_signal",
            "title": "Echo has behavior beyond chat to compare.",
            "claim": stated[0]["user_msg"][:180] if stated else "No stated goal found yet.",
            "behavior": behavior[0].get("summary") or behavior[0].get("title") or behavior[0]["event_type"],
            "next_step": "Compare this signal against your current read before changing the thesis.",
            "confidence": "emerging",
        })

    if not checks:
        checks.append({
            "kind": "empty",
            "title": "No reality check is strong enough yet.",
            "claim": "Echo needs more stated goals and outcomes.",
            "behavior": "Talk, run clones, and log one rep to create the first comparison.",
            "next_step": "Complete one tiny rep today.",
            "confidence": "early",
        })

    return {
        "headline": checks[0]["title"],
        "checks": checks[:3],
        "inputs": {
            "stated_goals": len(stated),
            "external_behavior_events": len(behavior),
            "practice_completed": completed,
            "practice_skipped": skipped,
        },
    }


async def get_growth_timeline(user_id: str) -> dict[str, Any]:
    async with get_conn() as db:
        async with db.execute(
            """
            SELECT event_domain, event_type, source, title, summary, created_at
            FROM life_events
            WHERE user_id=?
            ORDER BY created_at ASC
            LIMIT 200
            """,
            (user_id,),
        ) as cur:
            events = [dict(r) for r in await cur.fetchall()]
        async with db.execute(
            "SELECT COUNT(*) AS cnt FROM practice_log WHERE user_id=? AND done=1",
            (user_id,),
        ) as cur:
            practice_done = (await cur.fetchone())["cnt"]
        async with db.execute(
            "SELECT COUNT(*) AS cnt FROM tournament_runs WHERE user_id=? AND status='complete'",
            (user_id,),
        ) as cur:
            battles = (await cur.fetchone())["cnt"]
        async with db.execute(
            "SELECT COUNT(*) AS cnt FROM checkpoints WHERE user_id=?",
            (user_id,),
        ) as cur:
            checkpoints = (await cur.fetchone())["cnt"]

    important_types = {
        "onboarding_first_signal",
        "tournament_winner",
        "practice_logged",
        "training_completed",
        "training_eval_failed",
        "training_completed_adapter_not_loaded",
        "user_outcome",
        "practice_created",
    }
    _type_labels = {
        "training_completed": "Clone updated",
        "training_eval_failed": "Clone trained (eval conservative — adapter active)",
        "training_completed_adapter_not_loaded": "Clone trained (adapter pending load)",
        "tournament_winner": "Tournament won",
        "practice_created": "Practice rep created",
        "practice_logged": "Rep completed",
        "user_outcome": "Outcome signal",
        "onboarding_first_signal": "First signal captured",
    }
    milestones = []
    for event in events:
        if event["event_type"] in important_types:
            milestones.append({
                "type": event["event_type"],
                "title": _type_labels.get(event["event_type"]) or event.get("title") or event["event_type"].replace("_", " ").title(),
                "summary": event.get("summary") or "",
                "date": event["created_at"],
            })
    if not milestones and events:
        first = events[0]
        milestones.append({
            "type": "first_signal",
            "title": "First signal captured",
            "summary": first.get("summary") or "Echo started watching for repeated patterns.",
            "date": first["created_at"],
        })

    headline = "Proof is still forming."
    if checkpoints:
        headline = "Your clone has already been updated from your signals."
    elif practice_done >= 3:
        headline = "Your proof is moving from words into reps."
    elif battles:
        headline = "Your clone choices are starting to show a pattern."

    return {
        "headline": headline,
        "stats": {
            "milestones": len(milestones),
            "practice_done": practice_done,
            "clone_battles": battles,
            "model_updates": checkpoints,
        },
        "milestones": milestones[-12:],
    }


async def get_revelation_status(user_id: str) -> dict[str, Any]:
    async with get_conn() as db:
        async with db.execute(
            "SELECT COUNT(*) AS cnt, MIN(created_at) AS first_at FROM training_pairs WHERE user_id=?",
            (user_id,),
        ) as cur:
            pairs = await cur.fetchone()
        async with db.execute(
            "SELECT COUNT(*) AS cnt FROM tournament_runs WHERE user_id=? AND status='complete'",
            (user_id,),
        ) as cur:
            battles = await cur.fetchone()
        async with db.execute(
            "SELECT COUNT(*) AS cnt FROM practice_log WHERE user_id=? AND done=1",
            (user_id,),
        ) as cur:
            reps = await cur.fetchone()
        async with db.execute(
            "SELECT COUNT(*) AS cnt FROM shadow_outcomes WHERE user_id=?",
            (user_id,),
        ) as cur:
            outcomes = await cur.fetchone()
        async with db.execute(
            "SELECT COUNT(*) AS cnt FROM checkpoints WHERE user_id=?",
            (user_id,),
        ) as cur:
            checkpoints = await cur.fetchone()
        async with db.execute(
            """
            SELECT id, summary, created_at
            FROM echo_events
            WHERE user_id=? AND event_type='talent_read'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id,),
        ) as cur:
            revealed = await cur.fetchone()

    first_at = pairs["first_at"] if pairs else None
    weeks = 0
    if first_at:
        try:
            weeks = max(1, int((dt.utcnow() - dt.fromisoformat(first_at)).days / 7) + 1)
        except Exception:
            weeks = 1

    signals = {
        "language": min(1.0, (pairs["cnt"] if pairs else 0) / 50),
        "preference": min(1.0, (battles["cnt"] if battles else 0) / 3),
        "behavior": min(1.0, (reps["cnt"] if reps else 0) / 5),
        "outcomes": min(1.0, (outcomes["cnt"] if outcomes else 0) / 4),
        "clone_training": min(1.0, (checkpoints["cnt"] if checkpoints else 0) / 1),
    }
    score = round(sum(signals.values()) / len(signals), 3)
    ready = score >= 0.68 or (
        (pairs["cnt"] if pairs else 0) >= 40
        and (battles["cnt"] if battles else 0) >= 1
        and (outcomes["cnt"] if outcomes else 0) >= 2
    )
    if revealed:
        state = "revealed"
        headline = revealed["summary"] or "Echo has already named a hidden pattern."
    elif ready:
        state = "ready"
        headline = "I found something worth naming."
    elif score >= 0.38:
        state = "forming"
        headline = "A deeper read is forming."
    else:
        state = "watching"
        headline = "Echo is still watching for the deeper pattern."

    requirements = [
        {"key": "language", "label": "Language signal", "current": pairs["cnt"] if pairs else 0, "target": 50, "complete": signals["language"] >= 1.0},
        {"key": "preference", "label": "Clone choices", "current": battles["cnt"] if battles else 0, "target": 3, "complete": signals["preference"] >= 1.0},
        {"key": "behavior", "label": "Practice proof", "current": reps["cnt"] if reps else 0, "target": 5, "complete": signals["behavior"] >= 1.0},
        {"key": "outcomes", "label": "Reality outcomes", "current": outcomes["cnt"] if outcomes else 0, "target": 4, "complete": signals["outcomes"] >= 1.0},
        {"key": "clone_training", "label": "Clone adapted", "current": checkpoints["cnt"] if checkpoints else 0, "target": 1, "complete": signals["clone_training"] >= 1.0},
    ]
    return {
        "state": state,
        "ready": ready,
        "score": score,
        "headline": headline,
        "weeks_watched": weeks,
        "revealed_at": revealed["created_at"] if revealed else None,
        "requirements": requirements,
        "cta": "Open Revelation" if ready or revealed else "Keep building signal",
        "explanation": (
            "Echo waits for language, clone choices, practice, outcomes, and at least one model update "
            "before delivering a deep talent read."
        ),
    }


async def get_latest_clone_mission(user_id: str) -> dict[str, Any] | None:
    async with get_conn() as db:
        async with db.execute(
            """
            SELECT *
            FROM clone_missions
            WHERE user_id=?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None


async def create_clone_mission(
    user_id: str,
    run_id: str,
    candidate_id: str,
    winning_style: str,
    response: str,
    practice_rep_id: str | None,
) -> dict[str, Any]:
    mission_id = str(uuid.uuid4())
    cleaned = " ".join(response.split())
    strategy = cleaned[:280] or f"Use the {winning_style} clone's answer as your strategy."
    suggested_action = {
        "Strategist": "Take the smallest next move from the winning plan.",
        "Challenger": "Name the protected assumption, then test the opposite once.",
        "Mirror": "Say the pattern out loud before choosing your next move.",
        "Builder": "Turn the answer into one repeatable rule and run it once.",
        "Mentor": "Use the mentor answer as a higher bar, then execute one concrete step.",
    }.get(winning_style, "Practice the winning answer once today.")
    risk = "The mission fails if it stays as insight and never becomes a visible action."
    outcome_question = "Did this clone's strategy change what you actually did?"
    async with get_conn() as db:
        await db.execute(
            """
            INSERT INTO clone_missions
                (id, user_id, run_id, candidate_id, winning_style, strategy,
                 suggested_action, risk, outcome_question, practice_rep_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mission_id,
                user_id,
                run_id,
                candidate_id,
                winning_style,
                strategy,
                suggested_action,
                risk,
                outcome_question,
                practice_rep_id,
            ),
        )
        await db.commit()
    return {
        "id": mission_id,
        "run_id": run_id,
        "candidate_id": candidate_id,
        "winning_style": winning_style,
        "strategy": strategy,
        "suggested_action": suggested_action,
        "risk": risk,
        "outcome_question": outcome_question,
        "practice_rep_id": practice_rep_id,
        "status": "active",
    }


async def get_daily_mission(user_id: str) -> dict[str, Any]:
    priority = await get_today_priority(user_id)
    thesis = await get_current_thesis(user_id)
    reality = await get_reality_check(user_id)
    timeline = await get_growth_timeline(user_id)
    clone_mission = await get_latest_clone_mission(user_id)

    async with get_conn() as db:
        async with db.execute(
            """
            SELECT id, rep_title, observation, rep_instruction
            FROM practice_reps
            WHERE user_id=? AND date=?
            LIMIT 1
            """,
            (user_id, _today()),
        ) as cur:
            practice = await cur.fetchone()

    return {
        "date": _today(),
        "headline": priority.get("title") or "Echo has one mission today.",
        "why": priority.get("body") or "This is the next useful move in the loop.",
        "priority": priority,
        "current_read": thesis,
        "practice": dict(practice) if practice else None,
        "clone_mission": clone_mission,
        "reality_check": reality["checks"][0] if reality.get("checks") else None,
        "growth": {
            "headline": timeline["headline"],
            "stats": timeline["stats"],
        },
    }


async def get_or_create_next_intervention(user_id: str) -> dict[str, Any] | None:
    settings = await get_intervention_settings(user_id)
    async with get_conn() as db:
        async with db.execute(
            """
            SELECT *
            FROM echo_interventions
            WHERE user_id=? AND status='pending'
              AND (expires_at IS NULL OR expires_at > datetime('now'))
            ORDER BY priority DESC, scheduled_for ASC
            LIMIT 1
            """,
            (user_id,),
        ) as cur:
            pending = await cur.fetchone()
    if pending:
        return _intervention_from_row(pending)

    summary = await get_training_summary(user_id)
    clone = await get_latest_clone_mission(user_id)
    reality = await get_reality_check(user_id)
    mission = await get_daily_mission(user_id)
    timeline = await get_growth_timeline(user_id)
    revelation = await get_revelation_status(user_id)

    if clone:
        return await _insert_intervention(
            user_id,
            "clone_returned",
            "Your clone returned with a mission.",
            clone["suggested_action"],
            "A tournament winner created a concrete action to test.",
            "open_clone_mission",
            {"mission_id": clone["id"], "run_id": clone["run_id"]},
            4,
            settings,
        )
    if summary.get("ready_for_training"):
        return await _insert_intervention(
            user_id,
            "training_ready",
            "Your clone has enough new signal.",
            f"{summary.get('untrained_pairs', 0)} moments are ready to train.",
            "Training readiness crossed the configured threshold.",
            "open_training",
            {},
            3,
            settings,
        )
    if revelation.get("ready") and revelation.get("state") != "revealed":
        return await _insert_intervention(
            user_id,
            "talent_revelation",
            "I found something important.",
            "Echo has enough signal to name a deeper pattern.",
            "Revelation readiness crossed the signal threshold.",
            "open_revelation",
            {},
            4,
            settings,
        )
    first_check = reality.get("checks", [{}])[0]
    if first_check.get("kind") in {"practice_gap", "proof", "external_signal"}:
        return await _insert_intervention(
            user_id,
            "reality_check",
            first_check["title"],
            first_check.get("next_step", ""),
            "Echo found a comparison between intention and behavior.",
            "open_reality_check",
            {},
            3,
            settings,
        )
    if timeline["stats"]["milestones"] >= 3:
        return await _insert_intervention(
            user_id,
            "growth_proof",
            timeline["headline"],
            "Open the timeline to see what changed.",
            "Echo found enough milestones to show growth over time.",
            "open_growth_timeline",
            {},
            2,
            settings,
        )
    return await _insert_intervention(
        user_id,
        "morning_mission",
        mission["headline"],
        mission["why"],
        "Daily mission generated from the current Echo loop.",
        "open_today",
        {},
        1,
        settings,
    )


async def acknowledge_intervention(user_id: str, intervention_id: str, status: str = "acknowledged") -> dict[str, Any]:
    if status not in {"acknowledged", "dismissed", "delivered"}:
        status = "acknowledged"
    column = "acknowledged_at"
    if status == "delivered":
        column = "delivered_at"
    async with get_conn() as db:
        await db.execute(
            f"UPDATE echo_interventions SET status=?, {column}=datetime('now') WHERE id=? AND user_id=?",
            (status, intervention_id, user_id),
        )
        await db.commit()
    return {"saved": True, "id": intervention_id, "status": status}
