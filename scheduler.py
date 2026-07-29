import asyncio
import json
import logging
import re
from datetime import date, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config import settings
from db.database import get_conn
from training.collector import count_untrained_pairs
from training.coordinator import run_training_cycle
from training.summary import get_training_summary
from training.state import finish_training_run, try_create_training_run
from loop_events import record_event
from providers.teacher import chat_with_teacher

log = logging.getLogger("echo.scheduler")
scheduler = AsyncIOScheduler(timezone="UTC")


def _strip_markdown_json(text: str) -> str:
    """Strip ```json ... ``` or ``` ... ``` fences so json.loads works."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        return m.group(1).strip()
    return text


async def _active_users() -> list[str]:
    async with get_conn() as db:
        async with db.execute(
            "SELECT DISTINCT user_id FROM training_pairs WHERE used_in_training = 0"
        ) as cur:
            rows = await cur.fetchall()
    return [r["user_id"] for r in rows]


# ── Nightly @ 2:00 UTC ────────────────────────────────────────
async def nightly_training_job() -> None:
    log.info("Shadow Clone nightly training started")
    for user_id in await _active_users():
        lane = "gemma4_e2b"
        n = await count_untrained_pairs(user_id)
        summary_before = await get_training_summary(user_id, lane=lane)
        prev_before = (summary_before.get("last_checkpoint") or {}).get("path")
        dpo_ready = bool(summary_before.get("dpo_ready_for_training") and prev_before)
        if n < settings.min_pairs_for_training and not dpo_ready:
            log.info("Skip user=%s only %d pairs and dpo_ready=%s", user_id, n, dpo_ready)
            continue
        run_id = await try_create_training_run(user_id, lane, n)
        if not run_id:
            log.info("Skip user=%s because the GPU training slot is busy", user_id)
            continue
        try:
            await record_event(
                user_id,
                "training_started",
                "scheduler",
                f"Nightly shadow training started with {n} untrained moments.",
                {"run_id": run_id, "pairs": n},
                weight=1.5,
            )
            result = await run_training_cycle(user_id, lane, run_id)
            status = result.get("status") or "failed"
            promoted = bool(result.get("promoted"))
            restore_ok = bool(result.get("restore_ok", True))
            await finish_training_run(
                run_id,
                status,
                adapter_path=result.get("promoted_path"),
                error=(
                    result.get("reason")
                    if status == "skipped"
                    else ("Previous adapter restore failed." if status == "complete_restore_failed" else None)
                ),
                summary=result.get("summary") or result,
            )
            try:
                await record_event(
                    user_id,
                    "training_completed" if promoted else "training_not_promoted",
                    "scheduler",
                    "Nightly shadow training completed and promoted a stronger adapter."
                    if promoted
                    else (
                        "Nightly training rejected the candidate, but restoring the previous adapter needs attention."
                        if not restore_ok
                        else "Nightly shadow training completed; the previous Home Brain was kept."
                    ),
                    {"run_id": run_id, "result": result},
                    weight=2.0 if promoted else 1.0,
                )
            except Exception:
                log.exception("Could not record nightly training completion user=%s run=%s", user_id, run_id)
        except Exception as e:
            await finish_training_run(run_id, "failed", error=str(e))
            await record_event(
                user_id,
                "training_failed",
                "scheduler",
                "Nightly shadow training failed.",
                {"error": str(e)},
                weight=1.0,
            )
            log.exception("Nightly training failed user=%s", user_id)


# ── Weekly Sunday @ 3:00 UTC ──────────────────────────────────
async def weekly_skill_extraction_job() -> None:
    log.info("Weekly skill extraction started")
    async with get_conn() as db:
        async with db.execute("SELECT DISTINCT user_id FROM training_pairs") as cur:
            rows = await cur.fetchall()
    for row in rows:
        try:
            await _extract_skills(row["user_id"])
        except Exception as e:
            log.error("Skill extraction failed user=%s: %s", row["user_id"], e)


async def _extract_skills(user_id: str) -> None:
    # On first run (no skills yet) use all-time pairs so the user isn't stuck
    # waiting until next Sunday. After that, use the rolling 7-day window.
    async with get_conn() as db:
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM user_skills WHERE user_id = ? AND active = 1",
            (user_id,),
        ) as cur:
            skill_count = (await cur.fetchone())["cnt"]

    week_ago = (date.today() - timedelta(days=7)).isoformat()
    async with get_conn() as db:
        if skill_count == 0:
            # First extraction: look at all-time pairs so the user isn't stuck
            # waiting until next Sunday with an empty Skills screen.
            async with db.execute(
                "SELECT user_msg, assistant_msg, topic, perplexity FROM training_pairs "
                "WHERE user_id = ? AND perplexity >= 0.7 "
                "ORDER BY perplexity DESC LIMIT 200",
                (user_id,),
            ) as cur:
                rows = await cur.fetchall()
        else:
            async with db.execute(
                "SELECT user_msg, assistant_msg, topic, perplexity FROM training_pairs "
                "WHERE user_id = ? AND created_at >= ? AND perplexity >= 0.7 "
                "ORDER BY perplexity DESC LIMIT 200",
                (user_id, week_ago),
            ) as cur:
                rows = await cur.fetchall()

    if len(rows) < 5:
        return

    convos = "\n\n".join(
        f"Q: {r['user_msg'][:200]}\nA: {r['assistant_msg'][:300]}" for r in rows[:50]
    )
    prompt = (
        'Analyze these conversations and extract 3-8 REUSABLE SKILLS as a JSON array:\n'
        '[{"skill_name":"NAME","trigger":"when to apply","procedure":"steps","user_prefs":"preferences"}]\n\n'
        'A skill is worth extracting only if:\n'
        '- It appears across at least 3 different conversations or time periods\n'
        '- It reveals how THIS specific person thinks or prefers to work, not generic advice\n'
        '- It would change how Echo responds to their next question in a meaningful way\n'
        'Skip generic life skills (time management, communication, focus). Extract the specific and personal.\n\n'
        f'Conversations:\n{convos}\n\nReturn only valid JSON.'
    )
    content, _, model_used = await chat_with_teacher(
        [{"role": "user", "content": prompt}],
        user_id=user_id,
        purpose="weekly_calibration",
        require_policy=False,  # scheduled jobs bypass the per-chat budget
        temperature=0.3,
        max_tokens=1000,
    )
    if not content:
        log.info("Weekly skill extraction skipped user=%s model=%s", user_id, model_used)
        return
    try:
        parsed = json.loads(_strip_markdown_json(content))
        skills = parsed if isinstance(parsed, list) else parsed.get("skills", [])
    except Exception:
        log.warning("Weekly skill extraction: JSON parse failed user=%s content=%r", user_id, content[:200])
        return

    today = date.today().isoformat()
    async with get_conn() as db:
        await db.execute("UPDATE user_skills SET active = 0 WHERE user_id = ?", (user_id,))
        for s in skills:
            await db.execute(
                "INSERT INTO user_skills (user_id, skill_name, trigger, procedure, user_prefs, source_week) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, s.get("skill_name", "Skill"), s.get("trigger", ""),
                 s.get("procedure", ""), s.get("user_prefs", ""), today),
            )
        await db.commit()
    log.info("Extracted %d skills for user=%s", len(skills), user_id)


# ── Monthly 1st @ 4:00 UTC ────────────────────────────────────
async def monthly_rule_distillation_job() -> None:
    log.info("Monthly rule distillation started")
    async with get_conn() as db:
        async with db.execute("SELECT DISTINCT user_id FROM training_pairs") as cur:
            rows = await cur.fetchall()
    for row in rows:
        try:
            await _distill_rules(row["user_id"])
        except Exception as e:
            log.error("Rule distillation failed user=%s: %s", row["user_id"], e)


async def _distill_rules(user_id: str) -> None:
    async with get_conn() as db:
        async with db.execute(
            "SELECT skill_name, trigger, procedure FROM user_skills WHERE user_id = ? AND active = 1",
            (user_id,),
        ) as cur:
            skill_rows = await cur.fetchall()
        async with db.execute(
            "SELECT user_msg FROM training_pairs WHERE user_id = ? AND perplexity >= 0.85 "
            "ORDER BY rowid DESC LIMIT 100",
            (user_id,),
        ) as cur:
            pair_rows = await cur.fetchall()

    if not skill_rows and len(pair_rows) < 10:
        return

    skills_text = "\n".join(
        f"- {r['skill_name']}: {r['trigger']} | {r['procedure']}" for r in skill_rows
    )
    sample_qs = "\n".join(f"Q: {r['user_msg'][:150]}" for r in pair_rows[:30])
    prompt = (
        'Extract 5-15 PERMANENT DECLARATIVE RULES as JSON array:\n'
        '[{"rule_text":"rule","applies_to":"all|coding|...","confidence":"high|medium"}]\n\n'
        'A rule is worth keeping only if:\n'
        '- It creates a clear constraint or strong preference that changes how Echo should behave\n'
        '- It is specific to this person — not general wisdom anyone might have\n'
        '- It could not be inferred just from knowing their topic interests\n'
        'Skip rules that say "be direct", "be concise", or "be encouraging" — '
        'extract the surprising, personal, and non-obvious.\n\n'
        f'Active skills:\n{skills_text}\n\nSample questions:\n{sample_qs}\n\nReturn only valid JSON.'
    )
    content, _, model_used = await chat_with_teacher(
        [{"role": "user", "content": prompt}],
        user_id=user_id,
        purpose="monthly_distillation",
        require_policy=False,  # scheduled jobs bypass the per-chat budget
        temperature=0.2,
        max_tokens=1500,
    )
    if not content:
        log.info("Monthly rule distillation skipped user=%s model=%s", user_id, model_used)
        return
    try:
        parsed = json.loads(_strip_markdown_json(content))
        rules = parsed if isinstance(parsed, list) else parsed.get("rules", [])
    except Exception:
        log.warning("Monthly rule distillation: JSON parse failed user=%s content=%r", user_id, content[:200])
        return

    source_month = date.today().strftime("%Y-%m")
    async with get_conn() as db:
        # Only deactivate previously-distilled rules (confidence = high/medium/low).
        # Never touch manually-added (0.99) or auto-extracted (0.95) rules.
        await db.execute(
            "UPDATE user_rules SET active = 0 WHERE user_id = ? AND confidence IN ('high', 'medium', 'low')",
            (user_id,),
        )
        for r in rules:
            await db.execute(
                "INSERT INTO user_rules (user_id, rule_text, applies_to, confidence, source_month, active) "
                "VALUES (?, ?, ?, ?, ?, 1)",
                (user_id, r.get("rule_text", ""), r.get("applies_to", "all"),
                 r.get("confidence", "medium"), source_month),
            )
        await db.commit()
    log.info("Distilled %d rules for user=%s", len(rules), user_id)


async def _send_fcm(token: str, title: str, body: str) -> None:
    import httpx
    from config import settings
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                "https://fcm.googleapis.com/fcm/send",
                headers={
                    "Authorization": f"key={settings.fcm_server_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "to": token,
                    "notification": {"title": title, "body": body, "sound": "default"},
                    "data": {"type": "evening_signal"},
                    "priority": "high",
                },
            )
    except Exception as e:
        log.warning("FCM send failed: %s", e)


async def evening_signal_job() -> None:
    """Send Evening Signal push notification to all active users at 19:00 UTC."""
    log.info("Evening Signal notification job started")
    async with get_conn() as db:
        async with db.execute("SELECT user_id, token FROM fcm_tokens") as cur:
            rows = await cur.fetchall()
    for row in rows:
        await _send_fcm(
            row["token"],
            "Evening Signal",
            "Echo has something to ask. 3 questions · 5 minutes",
        )
    log.info("Evening Signal sent to %d device(s)", len(rows))


async def _seed_proof_from_history(user_id: str) -> int:
    """Convert life_events and thesis_evidence into proof_items for users with none yet.
    Idempotent — skips if the user already has proof items."""
    async with get_conn() as db:
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM proof_items WHERE user_id=?", (user_id,)
        ) as cur:
            existing = (await cur.fetchone())["cnt"]
    if existing > 0:
        return 0

    import uuid as _uuid
    seeded = 0

    async with get_conn() as db:
        # Seed from life events (confidence >= 0.7)
        async with db.execute(
            "SELECT id, event_type, event_domain, title, summary, confidence "
            "FROM life_events WHERE user_id=? AND confidence >= 0.7 "
            "ORDER BY rowid DESC LIMIT 30",
            (user_id,),
        ) as cur:
            life_rows = await cur.fetchall()

        for r in life_rows:
            category = _event_type_to_proof_category(r["event_type"])
            await db.execute(
                "INSERT OR IGNORE INTO proof_items "
                "(id, user_id, title, description, category, evidence, "
                " skill_tags_json, source_type, source_id, opportunity_type, status) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(_uuid.uuid4()), user_id,
                    r["title"] or r["event_type"],
                    r["summary"] or "",
                    category,
                    r["summary"] or "",
                    "[]",
                    "life_event", str(r["id"]),
                    "personal_goal", "active",
                ),
            )
            seeded += 1

        # Seed from thesis evidence (weight >= 1.2 = strong signal)
        # thesis_evidence columns: id, thesis_id, user_id, source, subject_type, subject_id, summary, weight
        async with db.execute(
            "SELECT id, source, subject_type, summary, weight "
            "FROM thesis_evidence WHERE user_id=? AND weight >= 1.2 "
            "ORDER BY weight DESC LIMIT 15",
            (user_id,),
        ) as cur:
            ev_rows = await cur.fetchall()

        for r in ev_rows:
            label = r["subject_type"] or r["source"] or "insight"
            await db.execute(
                "INSERT OR IGNORE INTO proof_items "
                "(id, user_id, title, description, category, evidence, "
                " skill_tags_json, source_type, source_id, opportunity_type, status) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(_uuid.uuid4()), user_id,
                    f"Pattern: {label}",
                    r["summary"] or "",
                    "pattern",
                    r["summary"] or "",
                    "[]",
                    "thesis_evidence", str(r["id"]),
                    "personal_goal", "active",
                ),
            )
            seeded += 1

        await db.commit()

    log.info("Seeded %d proof items for user=%s from history", seeded, user_id)
    return seeded


def _event_type_to_proof_category(event_type: str) -> str:
    mapping = {
        "achievement": "achievement",
        "milestone": "achievement",
        "learning": "learning",
        "insight": "learning",
        "skill": "skill",
        "practice": "practice",
        "project": "project",
        "challenge": "challenge",
        "decision": "decision",
    }
    return mapping.get((event_type or "").lower(), "practice")


def start_scheduler() -> AsyncIOScheduler:
    scheduler.add_job(nightly_training_job, "cron", hour=2, minute=0, id="nightly")
    scheduler.add_job(weekly_skill_extraction_job, "cron", day_of_week="sun", hour=3, id="weekly_skills")
    scheduler.add_job(monthly_rule_distillation_job, "cron", day=1, hour=4, id="monthly_rules")
    scheduler.add_job(evening_signal_job, "cron", hour=19, minute=0, id="evening_signal")
    scheduler.start()
    log.info("APScheduler: nightly@2am | weekly@Sun 3am | monthly@1st 4am | evening@7pm UTC")
    return scheduler
