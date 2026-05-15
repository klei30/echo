import json
import uuid
from typing import Any

from db.database import get_conn
from training.collector import count_untrained_pairs


def _json(data: dict[str, Any] | None) -> str:
    return json.dumps(data or {}, ensure_ascii=True, sort_keys=True)


async def record_event(
    user_id: str,
    event_type: str,
    source: str,
    summary: str = "",
    payload: dict[str, Any] | None = None,
    weight: float = 1.0,
) -> str:
    event_id = str(uuid.uuid4())
    async with get_conn() as db:
        await db.execute(
            """
            INSERT INTO echo_events (id, user_id, event_type, source, summary, payload_json, weight)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (event_id, user_id, event_type, source, summary[:500], _json(payload), weight),
        )
        await db.execute(
            """
            INSERT OR IGNORE INTO life_events
                (id, user_id, event_domain, event_type, source, title, summary, payload_json, confidence, privacy_level)
            VALUES (?, ?, 'echo', ?, ?, ?, ?, ?, ?, 'local')
            """,
            (
                event_id,
                user_id,
                event_type,
                source,
                event_type.replace("_", " ").title(),
                summary[:700],
                _json(payload),
                max(0.0, min(float(weight) / 2.0, 1.0)),
            ),
        )
        await db.commit()
    return event_id


async def record_life_event(
    user_id: str,
    event_domain: str,
    event_type: str,
    source: str,
    title: str = "",
    summary: str = "",
    payload: dict[str, Any] | None = None,
    confidence: float = 0.5,
    privacy_level: str = "local",
    subject_type: str | None = None,
    subject_id: str | None = None,
) -> str:
    event_id = str(uuid.uuid4())
    async with get_conn() as db:
        await db.execute(
            """
            INSERT INTO life_events
                (id, user_id, event_domain, event_type, source, subject_type, subject_id,
                 title, summary, payload_json, confidence, privacy_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                user_id,
                event_domain,
                event_type,
                source,
                subject_type,
                subject_id,
                title[:160],
                summary[:1000],
                _json(payload),
                max(0.0, min(confidence, 1.0)),
                privacy_level,
            ),
        )
        await db.commit()
    return event_id


async def record_outcome(
    user_id: str,
    subject_type: str,
    outcome: str,
    score: float,
    subject_id: str | None = None,
    event_id: str | None = None,
    note: str = "",
) -> str:
    outcome_id = str(uuid.uuid4())
    async with get_conn() as db:
        await db.execute(
            """
            INSERT INTO shadow_outcomes
                (id, user_id, event_id, subject_type, subject_id, outcome, score, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (outcome_id, user_id, event_id, subject_type, subject_id, outcome, score, note[:500]),
        )
        await db.commit()
    return outcome_id


async def loop_snapshot(user_id: str) -> dict[str, Any]:
    async with get_conn() as db:
        async with db.execute(
            """
            SELECT event_type, source, summary, payload_json, weight, created_at
            FROM echo_events
            WHERE user_id=?
            ORDER BY created_at DESC
            LIMIT 12
            """,
            (user_id,),
        ) as cur:
            events = [dict(r) for r in await cur.fetchall()]

        async with db.execute(
            """
            SELECT outcome, subject_type, subject_id, score, note, created_at
            FROM shadow_outcomes
            WHERE user_id=?
            ORDER BY created_at DESC
            LIMIT 8
            """,
            (user_id,),
        ) as cur:
            outcomes = [dict(r) for r in await cur.fetchall()]

        async with db.execute(
            """
            SELECT id, prompt, topic, status, winning_style, created_at
            FROM tournament_runs
            WHERE user_id=?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id,),
        ) as cur:
            latest_tournament = await cur.fetchone()

        async with db.execute(
            "SELECT COUNT(*) as cnt FROM echo_events WHERE user_id=?",
            (user_id,),
        ) as cur:
            event_count = await cur.fetchone()

        async with db.execute(
            """
            SELECT event_domain, event_type, source, title, summary, confidence, created_at
            FROM life_events
            WHERE user_id=?
            ORDER BY created_at DESC
            LIMIT 10
            """,
            (user_id,),
        ) as cur:
            life_events = [dict(r) for r in await cur.fetchall()]

    for event in events:
        try:
            event["payload"] = json.loads(event.pop("payload_json") or "{}")
        except json.JSONDecodeError:
            event["payload"] = {}

    latest = dict(latest_tournament) if latest_tournament else None
    if latest and latest.get("id"):
        async with get_conn() as db:
            async with db.execute(
                """
                SELECT id, style, response, score, signals_json
                FROM tournament_candidates
                WHERE run_id=?
                ORDER BY score DESC, created_at ASC
                """,
                (latest["id"],),
            ) as cur:
                latest["candidates"] = [dict(r) for r in await cur.fetchall()]

    headline = "Echo is still collecting signal."
    if latest and latest.get("winning_style"):
        headline = f"Your {latest['winning_style']} clone won the latest round."
    elif events:
        headline = events[0].get("summary") or headline

    untrained_pairs = await count_untrained_pairs(user_id)
    return {
        "headline": headline,
        "event_count": event_count["cnt"] if event_count else 0,
        "untrained_pairs": untrained_pairs,
        "latest_events": events,
        "latest_outcomes": outcomes,
        "latest_tournament": latest,
        "latest_life_events": life_events,
    }
