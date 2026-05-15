"""
training/pipeline.py — Nightly orchestrator for the shadow clone tournament.

Full flow:
  1. Collect today's training pairs (quality >= 0.8)
  2. Fetch priority topics (flagged by last night's consensus phase)
  3. Train 4 clones in parallel (SeqKD, GroupDPO, SelfCritique, OnPolicyGKD)
  4. Consensus phase (PST-style agreement detection)
  5. Tournament — evaluate + rank clones
  6. Winner selected, losers archived to skill_archive
  7. SDFT stabilization — prevent catastrophic forgetting
  8. Merge with global LoRA (mergekit TIES)
  9. Deploy to R2 + vLLM hot-swap
  10. Notify user
"""

import asyncio
import json
import logging
import os
import subprocess
from datetime import date
from pathlib import Path

import asyncpg
from openai import AsyncOpenAI

from config import settings
from training.clones import build_default_clones, CloneDNA
from training.consensus import run_consensus_phase, load_priority_topics, mark_priority_topics_used
from training.tournament import (
    evaluate_clones,
    select_winner_and_archive,
    load_personal_holdout,
)
from training.sdft import run_sdft_stabilization

logger = logging.getLogger("echo.training")


async def collect_training_pairs(user_id: str, pool: asyncpg.Pool, min_quality: float = 0.8) -> list[dict]:
    """Fetch today's training pairs from DB."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT prompt, teacher_response, teacher_logprobs, gemma_response,
                   quality_score, topic
            FROM training_pairs
            WHERE user_id = $1
              AND quality_score >= $2
              AND used_in_training = FALSE
              AND created_at >= CURRENT_DATE
            ORDER BY quality_score DESC
            LIMIT 500
            """,
            user_id, min_quality,
        )
    return [dict(r) for r in rows]


async def mark_pairs_used(user_id: str, pool: asyncpg.Pool):
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE training_pairs
            SET used_in_training = TRUE
            WHERE user_id = $1 AND used_in_training = FALSE AND created_at >= CURRENT_DATE
            """,
            user_id,
        )


async def create_training_job(user_id: str, pairs_count: int, pool: asyncpg.Pool) -> int:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO training_jobs (user_id, status, stage, pairs_count, started_at)
            VALUES ($1, 'running', 'starting', $2, NOW())
            RETURNING id
            """,
            user_id, pairs_count,
        )
    return row["id"]


async def update_job_stage(job_id: int, stage: str, pool: asyncpg.Pool, **kwargs):
    fields = ["stage = $2"]
    values = [job_id, stage]
    for k, v in kwargs.items():
        fields.append(f"{k} = ${len(values) + 1}")
        values.append(v)
    async with pool.acquire() as conn:
        await conn.execute(
            f"UPDATE training_jobs SET {', '.join(fields)} WHERE id = $1",
            *values,
        )


def _merge_with_global(winner_adapter_path: str, global_adapter_path: str | None) -> str:
    """
    Merge winner LoRA with global LoRA using mergekit TIES.
    Falls back to winner-only if mergekit not installed or no global adapter.
    """
    if not global_adapter_path or not Path(global_adapter_path).exists():
        return winner_adapter_path

    merged_path = winner_adapter_path + "_merged"
    try:
        merge_config = {
            "merge_method": "ties",
            "models": [
                {"model": winner_adapter_path, "parameters": {"weight": 0.7, "density": 0.5}},
                {"model": global_adapter_path, "parameters": {"weight": 0.3, "density": 0.3}},
            ],
            "base_model": os.environ.get("BASE_MODEL", "google/gemma-4-2b-it"),
            "parameters": {"normalize": True},
            "dtype": "float16",
            "out_path": merged_path,
        }
        config_path = Path(winner_adapter_path) / "merge_config.yaml"
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(merge_config, f)

        result = subprocess.run(
            ["mergekit-yaml", str(config_path), merged_path],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            return merged_path
    except Exception as e:
        logger.warning(f"mergekit merge failed: {e} — using winner-only adapter")

    return winner_adapter_path


async def deploy_adapter(user_id: str, adapter_path: str, pool: asyncpg.Pool) -> str:
    """
    Upload adapter to R2 and trigger vLLM hot-swap.
    Returns the R2 URL.
    """
    r2_url = await _upload_to_r2(user_id, adapter_path)

    # Notify vLLM to hot-swap adapter
    vllm_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{vllm_url}/v1/load_lora_adapter",
                json={"lora_name": user_id, "lora_path": adapter_path},
                timeout=30,
            )
    except Exception as e:
        logger.warning(f"vLLM hot-swap failed (adapter still saved to R2): {e}")

    # Update DB: mark as deployed
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE training_jobs SET status = 'deployed', adapter_path = $2, finished_at = NOW()
            WHERE user_id = $1 AND status = 'running'
            ORDER BY started_at DESC LIMIT 1
            """,
            user_id, r2_url,
        )

    return r2_url


async def _upload_to_r2(user_id: str, adapter_path: str) -> str:
    """Upload adapter directory to Cloudflare R2. Returns public URL."""
    r2_bucket = os.environ.get("R2_BUCKET", "echo-adapters")
    r2_account = os.environ.get("R2_ACCOUNT_ID", "")
    night = date.today().isoformat()
    r2_key = f"{user_id}/{night}/adapter"

    try:
        import boto3
        s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{r2_account}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ.get("R2_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("R2_SECRET_KEY"),
        )
        # Upload all adapter files
        for f in Path(adapter_path).rglob("*"):
            if f.is_file():
                rel = str(f.relative_to(adapter_path))
                s3.upload_file(str(f), r2_bucket, f"{r2_key}/{rel}")

        return f"r2://{r2_bucket}/{r2_key}"
    except Exception as e:
        logger.warning(f"R2 upload failed: {e} — using local path")
        return adapter_path


async def notify_user(user_id: str, message: str, pool: asyncpg.Pool):
    """Store notification for user (retrieved via /notifications endpoint)."""
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO training_jobs (user_id, status, stage) VALUES ($1, 'notify', $2)",
                user_id, message,
            )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# Main nightly pipeline
# ─────────────────────────────────────────────────────────────
async def nightly_training_run(user_id: str, pool: asyncpg.Pool):
    today = date.today().isoformat()
    logger.info(f"[{today}] Starting ECHO v2 nightly training for user {user_id}")

    if not settings.legacy_openai_pipeline_enabled:
        logger.info("Legacy OpenAI clone pipeline disabled; use the Gemma4 scheduler/orchestrator pipeline")
        return {"status": "skipped", "reason": "legacy_openai_pipeline_disabled"}

    openai = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # ── 1. COLLECT ──────────────────────────────────────────
    pairs = await collect_training_pairs(user_id, pool)
    logger.info(f"Collected {len(pairs)} training pairs")

    if len(pairs) < 10:
        logger.info("Not enough pairs — skipping tonight")
        return {"status": "skipped", "reason": "insufficient_data", "pairs": len(pairs)}

    job_id = await create_training_job(user_id, len(pairs), pool)

    # ── 2. PRIORITY TOPICS (auto-flagged last night) ────────
    priority = await load_priority_topics(user_id, pool, limit=20)
    if priority:
        logger.info(f"Including {len(priority)} priority prompts from last night's flags")
        priority_pairs = []
        for prompt in priority:
            try:
                resp = await openai.chat.completions.create(
                    model=os.environ.get("TEACHER_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                )
                priority_pairs.append({
                    "prompt": prompt,
                    "teacher_response": resp.choices[0].message.content,
                    "quality_score": 0.9,
                })
            except Exception:
                pass
        pairs = priority_pairs + pairs
        await mark_priority_topics_used(user_id, priority, pool)

    # ── 3. TRAIN 4 CLONES ───────────────────────────────────
    await update_job_stage(job_id, "training_clones", pool)
    logger.info("Training 4 clones...")

    clone_trainers = build_default_clones(user_id, openai)

    # Clones train sequentially (single GPU) — could parallelize on multi-GPU
    trained_clones = []
    for trainer in clone_trainers:
        try:
            logger.info(f"Training clone: {trainer.dna.method}")
            clone = trainer.train(pairs)
            trained_clones.append(clone)
        except Exception as e:
            logger.error(f"Clone {trainer.dna.method} failed: {e}")

    if not trained_clones:
        logger.error("All clones failed — aborting")
        return {"status": "failed", "reason": "all_clones_failed"}

    # ── 4. CONSENSUS PHASE ──────────────────────────────────
    await update_job_stage(job_id, "consensus", pool)
    logger.info("Running consensus phase...")

    holdout = await load_personal_holdout(user_id, pool, limit=20)
    consensus_result = await run_consensus_phase(
        user_id=user_id,
        clones=trained_clones,
        test_prompts=holdout,
        pool=pool,
    )
    logger.info(
        f"Consensus: {len(consensus_result['consensus_data'])} agree, "
        f"{len(consensus_result['flagged_prompts'])} flagged for tomorrow"
    )

    # ── 5. TOURNAMENT ────────────────────────────────────────
    await update_job_stage(job_id, "tournament", pool)
    logger.info("Running tournament...")

    scores = await evaluate_clones(trained_clones, holdout, user_id, pool, openai)
    winner_score, loser_scores = await select_winner_and_archive(scores, user_id, pool)

    logger.info(
        f"Winner: {winner_score.clone.clone_type} "
        f"(score={winner_score.overall_score:.4f})"
    )

    # ── 6. SDFT STABILIZATION ───────────────────────────────
    await update_job_stage(job_id, "sdft_stabilization", pool)
    logger.info("Running SDFT stabilization...")

    sdft_result = await run_sdft_stabilization(
        user_id=user_id,
        winner_adapter_path=winner_score.clone.adapter_path,
        pool=pool,
        alpha=0.3,
    )

    if not sdft_result.get("skipped"):
        logger.info(
            f"CKA alignment: {sdft_result['cka_before']:.4f} → "
            f"{sdft_result['cka_after']:.4f} "
            f"(Δ={sdft_result['cka_delta']:+.4f})"
        )

    stable_adapter = sdft_result["stable_adapter_path"]

    # ── 7. MERGE WITH GLOBAL ─────────────────────────────────
    await update_job_stage(job_id, "merging", pool)
    global_adapter = os.environ.get("GLOBAL_ADAPTER_PATH")
    merged_adapter = _merge_with_global(stable_adapter, global_adapter)

    # ── 8. DEPLOY ────────────────────────────────────────────
    await update_job_stage(job_id, "deploying", pool)
    logger.info("Deploying adapter...")

    adapter_url = await deploy_adapter(user_id, merged_adapter, pool)
    await mark_pairs_used(user_id, pool)

    # ── 9. NOTIFY ────────────────────────────────────────────
    winner_type = winner_score.clone.clone_type.replace("_", " ").title()
    cka_info = (
        f" CKA: {sdft_result['cka_after']:.3f}"
        if not sdft_result.get("skipped") else ""
    )
    await notify_user(
        user_id,
        f"✓ Your AI trained overnight ({winner_type} won, "
        f"score={winner_score.overall_score:.3f},{cka_info} "
        f"{len(consensus_result['flagged_prompts'])} gaps queued)",
        pool,
    )

    result = {
        "status": "success",
        "date": today,
        "pairs_used": len(pairs),
        "clones_trained": len(trained_clones),
        "winner_type": winner_score.clone.clone_type,
        "winner_score": winner_score.overall_score,
        "cka_before": sdft_result.get("cka_before"),
        "cka_after": sdft_result.get("cka_after"),
        "flagged_topics": len(consensus_result["flagged_prompts"]),
        "adapter_url": adapter_url,
    }

    logger.info(f"[{today}] Nightly training complete: {json.dumps(result)}")
    return result
