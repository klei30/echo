import json
import uuid
from typing import Any

from config import settings
from db.database import get_conn


async def create_training_run(user_id: str, lane: str, untrained_pairs: int) -> str:
    run_id = str(uuid.uuid4())
    async with get_conn() as db:
        await db.execute(
            """
            INSERT INTO training_runs
                (id, user_id, lane, status, untrained_pairs, required_pairs)
            VALUES (?, ?, ?, 'running', ?, ?)
            """,
            (run_id, user_id, lane, untrained_pairs, settings.min_pairs_for_training),
        )
        await db.commit()
    return run_id


async def finish_training_run(
    run_id: str,
    status: str,
    *,
    adapter_path: str | None = None,
    error: str | None = None,
    summary: dict[str, Any] | None = None,
) -> None:
    async with get_conn() as db:
        await db.execute(
            """
            UPDATE training_runs
            SET status=?,
                adapter_path=?,
                error=?,
                summary_json=?,
                finished_at=datetime('now')
            WHERE id=?
            """,
            (
                status,
                adapter_path,
                error,
                json.dumps(summary or {}, ensure_ascii=True),
                run_id,
            ),
        )
        await db.commit()


async def latest_training_run(user_id: str, lane: str) -> dict[str, Any] | None:
    async with get_conn() as db:
        async with db.execute(
            """
            SELECT *
            FROM training_runs
            WHERE user_id=? AND lane=?
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (user_id, lane),
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return None
    data = dict(row)
    try:
        data["summary"] = json.loads(data.pop("summary_json") or "{}")
    except Exception:
        data["summary"] = {}
    return data


async def mark_interrupted_training_runs() -> None:
    async with get_conn() as db:
        await db.execute(
            """
            UPDATE training_runs
            SET status='interrupted',
                error='Echo restarted before this run finished.',
                finished_at=datetime('now')
            WHERE status='running'
            """
        )
        await db.commit()
