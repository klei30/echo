import math
import logging
from datetime import date
from db.database import get_conn

log = logging.getLogger("echo.compass")

KL_DRIFT_THRESHOLD = 0.5
ALL_TOPICS = ["coding", "ml", "writing", "math", "research", "general"]


async def record_topic(user_id: str, topic: str) -> None:
    week = date.today().isocalendar()[1]
    async with get_conn() as db:
        await db.execute(
            """
            INSERT INTO topic_history (user_id, topic, week_number, count)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(user_id, topic, week_number) DO UPDATE
            SET count = count + 1
            """,
            (user_id, topic, week),
        )
        await db.commit()


async def _get_distribution(user_id: str, week_start: int, week_end: int) -> dict[str, float]:
    async with get_conn() as db:
        async with db.execute(
            "SELECT topic, SUM(count) as total FROM topic_history "
            "WHERE user_id = ? AND week_number > ? AND week_number <= ? "
            "GROUP BY topic",
            (user_id, week_start, week_end),
        ) as cur:
            rows = await cur.fetchall()
    counts = {row["topic"]: row["total"] for row in rows}
    total = sum(counts.values()) or 1
    return {t: counts.get(t, 0) / total for t in ALL_TOPICS}


def _kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    eps = 1e-10
    return sum(
        (p.get(t, eps)) * math.log((p.get(t, eps)) / (q.get(t, eps) + eps))
        for t in ALL_TOPICS
    )


async def detect_drift(user_id: str) -> tuple[bool, float]:
    """Returns (is_drifting, kl_score). Compares this week vs rolling 4-week baseline."""
    current_week = date.today().isocalendar()[1]
    # Clamp to 1 to avoid negative week numbers at year boundary
    baseline_start = max(1, current_week - 5)
    current = await _get_distribution(user_id, current_week - 1, current_week)
    baseline = await _get_distribution(user_id, baseline_start, current_week - 1)

    if all(v == 0 for v in baseline.values()):
        return False, 0.0

    kl = _kl_divergence(current, baseline)
    drifting = kl > KL_DRIFT_THRESHOLD
    if drifting:
        log.info("COMPASS drift detected user=%s kl=%.3f", user_id, kl)
    return drifting, kl


async def get_rebalance_weights(user_id: str) -> dict[str, float]:
    """Sampling weights to nudge training distribution back toward 4-week baseline."""
    current_week = date.today().isocalendar()[1]
    current = await _get_distribution(user_id, current_week - 1, current_week)
    baseline = await _get_distribution(user_id, current_week - 5, current_week - 1)
    eps = 1e-10
    weights = {t: baseline.get(t, eps) / (current.get(t, eps) + eps) for t in ALL_TOPICS}
    total = sum(weights.values())
    return {t: w / total for t, w in weights.items()}
