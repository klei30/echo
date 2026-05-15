from datetime import datetime as dt

from config import settings
from db.database import get_conn
from thesis import get_current_thesis
from training.summary import get_training_summary


def _days_since(value: str | None) -> int:
    if not value:
        return 9999
    try:
        return max(0, (dt.utcnow() - dt.fromisoformat(value)).days)
    except Exception:
        return 9999


def _hours_since(value: str | None) -> float:
    if not value:
        return 999999.0
    try:
        return max(0.0, (dt.utcnow() - dt.fromisoformat(value)).total_seconds() / 3600)
    except Exception:
        return 999999.0


def _card(
    kind: str,
    title: str,
    body: str,
    action_type: str,
    action_label: str,
    *,
    evidence_count: int = 0,
    confidence: str = "emerging",
    subject_type: str | None = None,
    subject_id: str | None = None,
    payload: dict | None = None,
) -> dict:
    return {
        "kind": kind,
        "title": title,
        "body": body,
        "evidence_count": evidence_count,
        "confidence": confidence,
        "subject_type": subject_type or kind,
        "subject_id": subject_id,
        "action": {
            "type": action_type,
            "label": action_label,
            "payload": payload or {},
        },
    }


async def get_today_priority(user_id: str) -> dict:
    summary = await get_training_summary(user_id)
    thesis = await get_current_thesis(user_id)
    today = dt.utcnow().strftime("%Y-%m-%d")

    async with get_conn() as db:
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM training_pairs WHERE user_id=?",
            (user_id,),
        ) as cur:
            total_pairs = (await cur.fetchone())["cnt"]

        async with db.execute(
            """
            SELECT id, name, topic, first_seen, evidence_count, escalation_level
            FROM echo_threads
            WHERE user_id=? AND status='active'
            ORDER BY escalation_level DESC, evidence_count DESC, last_seen DESC
            LIMIT 1
            """,
            (user_id,),
        ) as cur:
            top_thread = await cur.fetchone()

        async with db.execute(
            """
            SELECT id, rep_title, observation, rep_instruction
            FROM practice_reps
            WHERE user_id=? AND date=?
            LIMIT 1
            """,
            (user_id, today),
        ) as cur:
            today_rep = await cur.fetchone()

        rep_logged = False
        if today_rep:
            async with db.execute(
                "SELECT done FROM practice_log WHERE user_id=? AND rep_id=?",
                (user_id, today_rep["id"]),
            ) as cur:
                rep_logged = await cur.fetchone() is not None

        async with db.execute(
            """
            SELECT id, prompt, topic, winning_style, created_at
            FROM tournament_runs
            WHERE user_id=? AND status='complete'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id,),
        ) as cur:
            latest_battle = await cur.fetchone()

        async with db.execute(
            """
            SELECT id, created_at
            FROM tournament_runs
            WHERE user_id=?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id,),
        ) as cur:
            any_battle = await cur.fetchone()

        async with db.execute(
            """
            SELECT id, summary, payload_json, created_at
            FROM echo_events
            WHERE user_id=? AND event_type='onboarding_first_signal'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id,),
        ) as cur:
            first_signal = await cur.fetchone()

    if total_pairs < 5:
        if first_signal:
            payload = {}
            try:
                import json
                payload = json.loads(first_signal["payload_json"] or "{}")
            except Exception:
                payload = {}
            return _card(
                "first_read",
                first_signal["summary"] or "Echo has a first read.",
                payload.get("read") or "Echo has enough of a first signal to start watching a real pattern.",
                "none",
                "Keep talking",
                evidence_count=max(1, total_pairs),
                confidence="early",
                subject_type="onboarding_first_signal",
                subject_id=first_signal["id"],
            )
        return _card(
            "listen",
            "Echo is still learning your signal.",
            "Talk normally for a little longer. The first useful pattern appears after Echo has enough real moments.",
            "none",
            "Keep talking",
            evidence_count=total_pairs,
            confidence="early",
        )

    if top_thread and top_thread["escalation_level"] >= 4:
        days = max(1, _days_since(top_thread["first_seen"]) + 1)
        context = (
            f"Pattern: \"{top_thread['name']}\" - observed for {days} days "
            f"across {top_thread['evidence_count']} messages."
        )
        return _card(
            "council",
            "This pattern is ready for the council.",
            context,
            "open_council",
            "Enter council",
            evidence_count=top_thread["evidence_count"],
            confidence="strong",
            subject_type="thread",
            subject_id=top_thread["id"],
            payload={"thread_id": top_thread["id"], "thread_context": context},
        )

    if latest_battle and _hours_since(latest_battle["created_at"]) <= 36:
        style = latest_battle["winning_style"] or "A shadow"
        return _card(
            "tournament_result",
            f"{style} won your latest battle.",
            "That choice is now training signal. Echo is learning which version of you actually helps.",
            "run_tournament",
            "Send clones again",
            evidence_count=max(1, summary["tournament_battles"]),
            confidence="emerging",
            subject_type="tournament",
            subject_id=latest_battle["id"],
        )

    if top_thread and top_thread["escalation_level"] >= 2:
        prompt = (
            f"I keep seeing this pattern in myself: {top_thread['name']}. "
            "What version of me would handle it best?"
        )
        return _card(
            "thread_tournament",
            "Send clones into this pattern.",
            f"Echo has seen \"{top_thread['name']}\" {top_thread['evidence_count']} times.",
            "run_tournament",
            "Send clones",
            evidence_count=top_thread["evidence_count"],
            confidence="emerging",
            subject_type="thread",
            subject_id=top_thread["id"],
            payload={"prompt": prompt},
        )

    if (
        thesis.get("stage") != "forming"
        and float(thesis.get("confidence") or 0.0) < 0.75
        and total_pairs >= 8
        and (not any_battle or _days_since(any_battle["created_at"]) >= 2)
    ):
        action = thesis.get("next_action") or {}
        payload = action.get("payload") if isinstance(action.get("payload"), dict) else {}
        return _card(
            "thesis_test",
            "Test Echo's read with clones.",
            thesis.get("statement") or "Echo has a belief worth testing against a real choice.",
            "run_tournament",
            "Send clones",
            evidence_count=int(thesis.get("evidence_count") or 0),
            confidence=thesis.get("confidence_label") or "emerging",
            subject_type="thesis",
            subject_id=thesis.get("id"),
            payload=payload,
        )

    if today_rep and not rep_logged:
        return _card(
            "practice",
            today_rep["rep_title"],
            today_rep["observation"] or today_rep["rep_instruction"],
            "log_practice",
            "Mark done",
            evidence_count=1,
            confidence="emerging",
            subject_type="practice_rep",
            subject_id=today_rep["id"],
            payload={"rep_id": today_rep["id"]},
        )

    if summary["ready_for_training"]:
        return _card(
            "training_ready",
            "Your clone has enough new material.",
            (
                f"{summary['untrained_pairs']} moments are waiting. "
                f"{summary['preference_signals']} are direct preference signals."
            ),
            "open_training",
            "Update clone",
            evidence_count=summary["untrained_pairs"],
            confidence="strong",
        )

    if total_pairs >= 5 and (not any_battle or _days_since(any_battle["created_at"]) >= 3):
        return _card(
            "tournament",
            "Send clones.",
            "Bring one real situation. Four versions of you will compete, and your choice teaches Echo.",
            "run_tournament",
            "Send clones",
            evidence_count=total_pairs,
            confidence="early",
        )

    if total_pairs >= settings.min_pairs_for_training:
        return _card(
            "mirror",
            "Your weekly mirror is ready.",
            "Echo has enough moments to reflect one pattern back with evidence.",
            "open_mirror",
            "Open mirror",
            evidence_count=total_pairs,
            confidence="emerging",
        )

    return _card(
        "signal",
        "Echo is collecting your pattern.",
        "Keep using Talk or voice. The system is watching for repeated strengths, friction, and response styles.",
        "none",
        "Keep talking",
        evidence_count=total_pairs,
        confidence="early",
    )
