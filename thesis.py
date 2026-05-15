import json
import uuid
from typing import Any

from db.database import get_conn


def _json(data: dict[str, Any] | None) -> str:
    return json.dumps(data or {}, ensure_ascii=True, sort_keys=True)


def _confidence_label(value: float) -> str:
    if value >= 0.75:
        return "strong"
    if value >= 0.45:
        return "emerging"
    return "early"


def _safe_snippet(value: str | None, limit: int = 180) -> str:
    text = " ".join((value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _useful_message(value: str | None) -> bool:
    text = " ".join((value or "").lower().split())
    if len(text) < 28:
        return False
    blocked = (
        "generate a concise title",
        "conversation content:",
        "please return the result",
        "what do you remember about me",
        "can you hear me",
    )
    return not any(marker in text for marker in blocked)


def _topic_label(topic: str | None) -> str:
    value = (topic or "").strip().lower()
    if not value or value == "general":
        return "your recurring decisions"
    return value


def _action_for(stage: str, statement: str, top_thread: dict[str, Any] | None) -> dict[str, Any]:
    if stage == "forming":
        return {
            "type": "none",
            "label": "Keep talking",
            "payload": {},
        }

    if top_thread:
        prompt = (
            f"Echo currently believes this about me: {statement}\n\n"
            f"The strongest evidence thread is: {top_thread.get('name', '')}.\n"
            "Open Practice Versions and run this belief against a real situation — "
            "show which version of me would handle it best."
        )
    else:
        prompt = (
            f"Echo currently believes this about me: {statement}\n\n"
            "Open Practice Versions and challenge this belief with a real situation."
        )
    return {
        "type": "run_tournament",
        "label": "Practice Versions",
        "payload": {"prompt": prompt},
    }


async def _latest_active_thesis_id(user_id: str) -> str | None:
    async with get_conn() as db:
        async with db.execute(
            """
            SELECT id FROM user_theses
            WHERE user_id=? AND status='active'
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
    return row["id"] if row else None


async def _write_evidence(thesis_id: str, user_id: str, rows: list[dict[str, Any]]) -> None:
    async with get_conn() as db:
        await db.execute("DELETE FROM thesis_evidence WHERE thesis_id=?", (thesis_id,))
        for row in rows[:12]:
            await db.execute(
                """
                INSERT OR IGNORE INTO thesis_evidence
                    (id, thesis_id, user_id, source, subject_type, subject_id, summary, weight)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    thesis_id,
                    user_id,
                    row["source"],
                    row["subject_type"],
                    row.get("subject_id"),
                    _safe_snippet(row["summary"], 500),
                    float(row.get("weight", 1.0)),
                ),
            )
        await db.commit()


async def refresh_current_thesis(user_id: str) -> dict[str, Any]:
    """Build the user's current thesis from existing Echo evidence and persist it."""
    active_thesis_id = await _latest_active_thesis_id(user_id)
    async with get_conn() as db:
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM training_pairs WHERE user_id=?",
            (user_id,),
        ) as cur:
            total_pairs = (await cur.fetchone())["cnt"]

        async with db.execute(
            """
            SELECT topic, score FROM confidence
            WHERE user_id=?
            ORDER BY score DESC
            LIMIT 5
            """,
            (user_id,),
        ) as cur:
            confidence_rows = [dict(r) for r in await cur.fetchall()]

        async with db.execute(
            """
            SELECT id, name, topic, evidence_count, escalation_level, first_seen, last_seen
            FROM echo_threads
            WHERE user_id=? AND status='active'
            ORDER BY escalation_level DESC, evidence_count DESC, last_seen DESC
            LIMIT 1
            """,
            (user_id,),
        ) as cur:
            top_thread_row = await cur.fetchone()
            top_thread = dict(top_thread_row) if top_thread_row else None

        async with db.execute(
            """
            SELECT rule_text, confidence, applies_to, id
            FROM user_rules
            WHERE user_id=? AND active=1
            ORDER BY confidence DESC, created_at DESC
            LIMIT 5
            """,
            (user_id,),
        ) as cur:
            rules = [dict(r) for r in await cur.fetchall()]

        async with db.execute(
            """
            SELECT user_msg, topic, created_at, id
            FROM training_pairs
            WHERE user_id=?
            ORDER BY created_at DESC
            LIMIT 24
            """,
            (user_id,),
        ) as cur:
            messages = [dict(r) for r in await cur.fetchall()]

        async with db.execute(
            """
            SELECT winning_style, COUNT(*) as wins
            FROM tournament_runs
            WHERE user_id=? AND status='complete' AND winning_style IS NOT NULL
            GROUP BY winning_style
            ORDER BY wins DESC
            LIMIT 1
            """,
            (user_id,),
        ) as cur:
            leading_style_row = await cur.fetchone()
            leading_style = dict(leading_style_row) if leading_style_row else None

        async with db.execute(
            """
            SELECT id, event_type, source, summary, weight, created_at
            FROM echo_events
            WHERE user_id=?
            ORDER BY created_at DESC
            LIMIT 8
            """,
            (user_id,),
        ) as cur:
            events = [dict(r) for r in await cur.fetchall()]

        thesis_outcomes = []
        if active_thesis_id:
            async with db.execute(
                """
                SELECT outcome, score, note, created_at
                FROM shadow_outcomes
                WHERE user_id=? AND subject_type='thesis' AND subject_id=?
                ORDER BY created_at DESC
                LIMIT 5
                """,
                (user_id, active_thesis_id),
            ) as cur:
                thesis_outcomes = [dict(r) for r in await cur.fetchall()]

    evidence: list[dict[str, Any]] = []
    for rule in rules[:3]:
        evidence.append({
            "source": "rule",
            "subject_type": "user_rule",
            "subject_id": str(rule["id"]),
            "summary": rule["rule_text"],
            "weight": 1.2,
        })
    if top_thread:
        evidence.append({
            "source": "thread",
            "subject_type": "thread",
            "subject_id": top_thread["id"],
            "summary": (
                f"{top_thread['name']} appeared {top_thread['evidence_count']} times "
                f"with escalation level {top_thread['escalation_level']}."
            ),
            "weight": 1.5,
        })
    for event in events[:4]:
        if event.get("summary"):
            evidence.append({
                "source": event["source"],
                "subject_type": "event",
                "subject_id": event["id"],
                "summary": event["summary"],
                "weight": event.get("weight") or 1.0,
            })
    useful_messages = [message for message in messages if _useful_message(message.get("user_msg"))]
    for message in useful_messages[:3]:
        evidence.append({
            "source": "conversation",
            "subject_type": "training_pair",
            "subject_id": str(message["id"]),
            "summary": _safe_snippet(message["user_msg"], 220),
            "weight": 0.7,
        })
    for outcome in thesis_outcomes[:2]:
        label = str(outcome["outcome"]).replace("_", " ")
        evidence.append({
            "source": "user",
            "subject_type": "thesis_feedback",
            "subject_id": active_thesis_id,
            "summary": f"User marked the current read as {label}.",
            "weight": 1.6,
        })

    if total_pairs < 8:
        title = "Still Forming"
        statement = (
            "Echo does not have enough real moments yet. The first useful thesis forms "
            "after repeated conversations, choices, and outcomes."
        )
        stage = "forming"
        confidence = min(0.25, total_pairs / 32)
    else:
        top_topic = _topic_label(confidence_rows[0]["topic"] if confidence_rows else None)
        if top_thread:
            title = top_thread["name"]
            statement = (
                f"You may be strongest when you turn recurring tension around "
                f"{top_thread['name']} into a concrete move instead of leaving it as noise."
            )
        elif rules:
            title = _safe_snippet(rules[0]["rule_text"], 48)
            statement = (
                f"A repeated rule is emerging: {rules[0]['rule_text']} "
                "This may point to a hidden operating style worth testing."
            )
        else:
            title = "Signal Under Test" if top_topic == "your recurring decisions" else f"{top_topic.title()} Pattern"
            statement = (
                f"Your strongest signal is clustering around {top_topic}. "
                "Echo needs more outcomes to know whether this is a talent, a friction point, or both."
            )
        if leading_style:
            statement += (
                f" In shadow battles, {leading_style['winning_style']} is currently winning most often."
            )
        stage = "testing" if total_pairs >= 8 and len(evidence) >= 3 else "forming"
        outcome_adjustment = sum(float(o.get("score") or 0.0) for o in thesis_outcomes[:3]) * 0.08
        confidence = min(
            0.92,
            max(
                0.12,
                0.20
                + min(total_pairs, 80) / 160
                + min(len(evidence), 10) / 30
                + (0.12 if top_thread and top_thread["evidence_count"] >= 3 else 0.0)
                + outcome_adjustment,
            ),
        )

    action = _action_for(stage, statement, top_thread)
    evidence_count = len(evidence)
    thesis_id = active_thesis_id or str(uuid.uuid4())

    async with get_conn() as db:
        await db.execute(
            """
            INSERT INTO user_theses
                (id, user_id, title, statement, stage, status, confidence, evidence_count,
                 next_action_type, next_action_label, next_action_payload_json)
            VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                statement=excluded.statement,
                stage=excluded.stage,
                confidence=excluded.confidence,
                evidence_count=excluded.evidence_count,
                next_action_type=excluded.next_action_type,
                next_action_label=excluded.next_action_label,
                next_action_payload_json=excluded.next_action_payload_json,
                updated_at=datetime('now')
            """,
            (
                thesis_id,
                user_id,
                title[:120],
                statement[:1000],
                stage,
                confidence,
                evidence_count,
                action["type"],
                action["label"],
                _json(action["payload"]),
            ),
        )
        await db.commit()

    await _write_evidence(thesis_id, user_id, evidence)
    return await get_current_thesis(user_id, refresh=False)


async def get_current_thesis(user_id: str, refresh: bool = True) -> dict[str, Any]:
    if refresh:
        return await refresh_current_thesis(user_id)

    async with get_conn() as db:
        async with db.execute(
            """
            SELECT *
            FROM user_theses
            WHERE user_id=? AND status='active'
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (user_id,),
        ) as cur:
            thesis = await cur.fetchone()

        if not thesis:
            return {
                "id": None,
                "title": "Still Forming",
                "statement": "Echo is waiting for enough real moments to form a thesis.",
                "stage": "forming",
                "confidence": 0.0,
                "confidence_label": "early",
                "evidence_count": 0,
                "evidence": [],
                "next_action": {"type": "none", "label": "Keep talking", "payload": {}},
            }

        async with db.execute(
            """
            SELECT source, subject_type, subject_id, summary, weight, created_at
            FROM thesis_evidence
            WHERE thesis_id=?
            ORDER BY weight DESC, created_at DESC
            LIMIT 8
            """,
            (thesis["id"],),
        ) as cur:
            evidence = [dict(r) for r in await cur.fetchall()]

    confidence = float(thesis["confidence"] or 0.0)
    try:
        payload = json.loads(thesis["next_action_payload_json"] or "{}")
    except json.JSONDecodeError:
        payload = {}
    return {
        "id": thesis["id"],
        "title": thesis["title"],
        "statement": thesis["statement"],
        "stage": thesis["stage"],
        "confidence": confidence,
        "confidence_label": _confidence_label(confidence),
        "evidence_count": int(thesis["evidence_count"] or 0),
        "evidence": evidence,
        "updated_at": thesis["updated_at"],
        "next_action": {
            "type": thesis["next_action_type"] or "none",
            "label": thesis["next_action_label"] or "Keep talking",
            "payload": payload,
        },
    }
