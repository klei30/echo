import logging
from db.database import get_conn
from config import settings

log = logging.getLogger("echo.collector")


def _quality_score(assistant_msg: str, engagement_signal: str) -> float:
    if engagement_signal == "thumbs_up":
        return 1.0
    if engagement_signal == "thumbs_down":
        return 0.2  # saved as DPO rejected pair, excluded from SFT

    words = len(assistant_msg.split())
    if words < 5:
        return 0.1
    if words > 2000:
        return 0.5

    return {"continue": 0.7, "quiet": 0.3}.get(engagement_signal, 0.5)




_BAD_PATTERNS = (
    "<function",
    "<tool_call",
    "<functioncall",
    "<function_response",
    "call_function_result",
    "fetch_memory",
    "web_search_exa",
    "web_fetch_exa",
    "update_user_memory",
    "composio_",
    '"tool"',
    "'tool'",
)


def _is_poisoned(text: str) -> bool:
    lower = (text or "").lower()
    return any(p in lower for p in _BAD_PATTERNS)


def is_poisoned_training_text(text: str) -> bool:
    return _is_poisoned(text)


async def save_pair(
    user_id: str,
    user_msg: str,
    assistant_msg: str,
    model_used: str,
    engagement_signal: str,
    topic: str = "general",
) -> bool:
    # Never save responses containing tool call syntax — they poison the LoRA adapter
    if _is_poisoned(assistant_msg):
        log.warning("Rejected poisoned pair user=%s (contains tool call syntax)", user_id)
        return False

    quality = _quality_score(assistant_msg, engagement_signal)

    # Always save preference signals for DPO; gate everything else on quality
    if quality < settings.quality_threshold and engagement_signal not in ("thumbs_up", "thumbs_down"):
        log.debug("Skipping low-quality pair user=%s quality=%.2f", user_id, quality)
        return False

    async with get_conn() as db:
        await db.execute(
            """
            INSERT INTO training_pairs
                (user_id, topic, user_msg, assistant_msg, model_used, engagement_signal, perplexity)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, topic, user_msg, assistant_msg, model_used, engagement_signal, quality),
        )
        await db.commit()

    log.info("Saved pair user=%s topic=%s quality=%.2f signal=%s", user_id, topic, quality, engagement_signal)
    return True


async def mark_chat_feedback_pair(
    user_id: str,
    user_msg: str,
    assistant_msg: str,
    model_used: str,
    engagement_signal: str,
    topic: str = "general",
) -> dict:
    """Attach direct chat feedback to the saved pair, or create it if missing."""
    if engagement_signal not in {"thumbs_up", "thumbs_down"}:
        return {"updated": False, "created": False, "reason": "not_preference_signal"}
    if _is_poisoned(assistant_msg):
        return {"updated": False, "created": False, "reason": "poisoned_text"}

    quality = _quality_score(assistant_msg, engagement_signal)
    user_key = " ".join((user_msg or "").split())
    assistant_key = " ".join((assistant_msg or "").split())
    if not user_key or not assistant_key:
        return {"updated": False, "created": False, "reason": "missing_text"}

    async with get_conn() as db:
        async with db.execute(
            """
            SELECT id, engagement_signal
            FROM training_pairs
            WHERE user_id=?
              AND user_msg=?
              AND assistant_msg=?
            ORDER BY id DESC
            LIMIT 1
            """,
            (user_id, user_msg, assistant_msg),
        ) as cur:
            row = await cur.fetchone()
        if row:
            if row["engagement_signal"] == engagement_signal:
                return {"updated": False, "created": False, "duplicate": True, "pair_id": row["id"]}
            await db.execute(
                """
                UPDATE training_pairs
                SET engagement_signal=?, perplexity=?, used_in_training=0
                WHERE id=?
                """,
                (engagement_signal, quality, row["id"]),
            )
            await db.commit()
            return {"updated": True, "created": False, "pair_id": row["id"]}

    created = await save_pair(
        user_id=user_id,
        user_msg=user_msg,
        assistant_msg=assistant_msg,
        model_used=model_used,
        engagement_signal=engagement_signal,
        topic=topic,
    )
    return {"updated": False, "created": created}


async def get_sft_pairs(user_id: str, limit: int = 500) -> list[dict]:
    # Include ALL high-quality pairs regardless of model (local + teacher)
    # The shadow clone learns from both teacher and its own responses
    async with get_conn() as db:
        async with db.execute(
            """
            SELECT id, user_msg, assistant_msg, topic, perplexity, model_used, engagement_signal
            FROM training_pairs
            WHERE user_id = ?
              AND engagement_signal != 'thumbs_down'
              AND perplexity >= ?
              AND used_in_training = 0
            ORDER BY perplexity DESC
            LIMIT ?
            """,
            (user_id, settings.quality_threshold, limit * 2),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows if not _is_poisoned(r["assistant_msg"])][:limit]


async def get_dpo_pairs(user_id: str, limit: int = 200) -> list[dict]:
    # DPO needs BOTH thumbs_up AND thumbs_down for the SAME prompt
    async with get_conn() as db:
        async with db.execute(
            """
            SELECT id, user_msg, assistant_msg, engagement_signal
            FROM training_pairs
            WHERE user_id = ?
              AND engagement_signal IN ('thumbs_up', 'thumbs_down')
              AND used_in_training = 0
            ORDER BY rowid DESC
            LIMIT ?
            """,
            (user_id, limit),
        ) as cur:
            rows = await cur.fetchall()
    
    # Group by user_msg to find prompts with BOTH signals
    by_prompt: dict[str, dict] = {}
    for r in rows:
        if _is_poisoned(r["assistant_msg"]):
            continue
        key = r["user_msg"]
        entry = by_prompt.setdefault(key, {"chosen": None, "rejected": None, "chosen_id": None, "rejected_id": None})
        if r["engagement_signal"] == "thumbs_up" and entry["chosen"] is None:
            entry["chosen"] = r["assistant_msg"]
            entry["chosen_id"] = r["id"]
        elif r["engagement_signal"] == "thumbs_down" and entry["rejected"] is None:
            entry["rejected"] = r["assistant_msg"]
            entry["rejected_id"] = r["id"]
    
    # Only keep pairs with BOTH chosen and rejected
    dpo = [
        {
            "user_msg": prompt,
            "chosen": v["chosen"],
            "rejected": v["rejected"],
            "chosen_id": v["chosen_id"],
            "rejected_id": v["rejected_id"],
        }
        for prompt, v in by_prompt.items()
        if v["chosen"] and v["rejected"]
    ]
    return dpo[:limit]


async def mark_used(user_id: str, pair_ids: list[int] | set[int] | None = None) -> None:
    async with get_conn() as db:
        if pair_ids is None:
            await db.execute(
                "UPDATE training_pairs SET used_in_training = 1 "
                "WHERE user_id = ? AND used_in_training = 0",
                (user_id,),
            )
        else:
            ids = [int(i) for i in pair_ids]
            if not ids:
                return
            placeholders = ",".join("?" for _ in ids)
            await db.execute(
                f"UPDATE training_pairs SET used_in_training = 1 "
                f"WHERE user_id = ? AND id IN ({placeholders})",
                [user_id, *ids],
            )
        await db.commit()


async def count_untrained_pairs(user_id: str) -> int:
    async with get_conn() as db:
        async with db.execute(
            "SELECT assistant_msg FROM training_pairs "
            "WHERE user_id = ? AND used_in_training = 0 "
            "AND engagement_signal != 'thumbs_down' AND perplexity >= ?",
            (user_id, settings.quality_threshold),
        ) as cur:
            rows = await cur.fetchall()
    return sum(1 for r in rows if not _is_poisoned(r["assistant_msg"]))
