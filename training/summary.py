from db.database import get_conn
from config import settings
from training.collector import count_untrained_pairs, get_dpo_pairs
from training.adapter import adapter_status
from training.platform import training_runtime_info


async def get_training_summary(user_id: str, lane: str = "gemma4_e2b") -> dict:
    async with get_conn() as db:
        async with db.execute(
            """
            SELECT COUNT(*) as cnt
            FROM training_pairs
            WHERE user_id=? AND used_in_training=0
              AND engagement_signal IN ('thumbs_up', 'thumbs_down')
            """,
            (user_id,),
        ) as cur:
            pref_row = await cur.fetchone()

        async with db.execute(
            """
            SELECT COUNT(*) as cnt
            FROM training_pairs
            WHERE user_id=? AND engagement_signal IN ('thumbs_up', 'thumbs_down')
            """,
            (user_id,),
        ) as cur:
            total_pref_row = await cur.fetchone()

        async with db.execute(
            """
            SELECT COUNT(*) as cnt, MAX(created_at) as latest
            FROM tournament_runs
            WHERE user_id=? AND status='complete'
            """,
            (user_id,),
        ) as cur:
            battle_row = await cur.fetchone()

        async with db.execute(
            """
            SELECT winning_style, COUNT(*) as wins
            FROM tournament_runs
            WHERE user_id=? AND status='complete' AND winning_style IS NOT NULL
            GROUP BY winning_style
            ORDER BY wins DESC
            """,
            (user_id,),
        ) as cur:
            style_rows = await cur.fetchall()

        async with db.execute(
            """
            SELECT id, prompt, topic, winning_style, created_at
            FROM tournament_runs
            WHERE user_id=? AND status='complete'
            ORDER BY created_at DESC
            LIMIT 5
            """,
            (user_id,),
        ) as cur:
            recent_battles = [dict(r) for r in await cur.fetchall()]

        async with db.execute(
            """
            SELECT path, created_at
            FROM checkpoints
            WHERE user_id=? AND lane=?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id, lane),
        ) as cur:
            checkpoint = await cur.fetchone()

    untrained_pairs = await count_untrained_pairs(user_id)
    dpo_pairs = await get_dpo_pairs(user_id)
    adapter = await adapter_status(user_id, lane=lane)
    runtime = await training_runtime_info()
    preference_signals = pref_row["cnt"] if pref_row else 0
    total_preference_signals = total_pref_row["cnt"] if total_pref_row else 0
    battles = battle_row["cnt"] if battle_row else 0
    style_wins = {r["winning_style"]: r["wins"] for r in style_rows}
    leading_style = style_rows[0]["winning_style"] if style_rows else None

    has_checkpoint = bool(checkpoint)
    dpo_ready_for_training = len(dpo_pairs) >= 4
    ready_for_training = untrained_pairs >= settings.min_pairs_for_training
    data_ready_for_training = ready_for_training or (dpo_ready_for_training and has_checkpoint)
    model_runtime_ready = adapter.get("vllm") == "ok"
    runtime_ready_for_training = bool(runtime["available"] and model_runtime_ready)

    blocked_reason = None
    if data_ready_for_training and not runtime["available"]:
        mode = runtime["mode"]
        if mode == "windows_wsl":
            distro = runtime.get("wsl_distro", settings.echo_wsl_distro)
            blocked_reason = f"WSL distro {distro} is unavailable"
        elif mode == "linux_local":
            blocked_reason = f"training runtime unavailable ({runtime['status']})"
        elif mode == "disabled":
            blocked_reason = "training runtime disabled"
        else:
            blocked_reason = f"training runtime unavailable (mode={mode})"
    elif data_ready_for_training and not model_runtime_ready:
        blocked_reason = "Gemma 4 vLLM is unavailable"

    # Keep wsl field for backwards compat — derived from runtime_info on windows_wsl,
    # set to a neutral stub on other platforms so existing clients don't break.
    if runtime["mode"] == "windows_wsl":
        wsl_compat = {
            "available": runtime["available"],
            "distro": runtime.get("wsl_distro", settings.echo_wsl_distro),
            "status": runtime["status"],
        }
    else:
        wsl_compat = {"available": False, "status": "not_applicable", "distro": None}

    return {
        "untrained_pairs": untrained_pairs,
        "required_pairs": settings.min_pairs_for_training,
        "ready_for_training": ready_for_training,
        "data_ready_for_training": data_ready_for_training,
        "runtime_ready_for_training": runtime_ready_for_training,
        "can_train_now": data_ready_for_training and runtime_ready_for_training,
        "blocked_reason": blocked_reason,
        "training_runtime": runtime,
        "wsl": wsl_compat,
        "preference_signals": preference_signals,
        "total_preference_signals": total_preference_signals,
        "dpo_ready_pairs": len(dpo_pairs),
        "dpo_required_pairs": 4,
        "dpo_ready_for_training": dpo_ready_for_training,
        "dpo_requires_existing_adapter": dpo_ready_for_training and not has_checkpoint,
        "tournament_battles": battles,
        "latest_battle_at": battle_row["latest"] if battle_row else None,
        "style_wins": style_wins,
        "leading_style": leading_style,
        "recent_battles": recent_battles,
        "last_checkpoint": dict(checkpoint) if checkpoint else None,
        "adapter": adapter,
        "lane": lane,
    }
