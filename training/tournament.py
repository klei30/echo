"""
training/tournament.py — Clone evaluation, winner selection, skill archive.

Paper: TeLAPA (2604.15414) — preserve behavioral diversity, don't delete losers.
Instead of discarding losing clones, we archive them with per-topic scores.
get_best_adapter_for_topic() can retrieve a specialist from archive later.
"""

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import asyncpg

from training.clones import TrainedClone

ARCHIVE_DIR = Path(os.environ.get("ADAPTERS_DIR", "/tmp/echo_adapters")) / "archive"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

TOPICS = ["coding", "ml", "writing", "math", "research", "general"]


@dataclass
class CloneScore:
    clone: TrainedClone
    overall_score: float
    topic_scores: dict[str, float]
    rank: int = 0


async def evaluate_clones(
    clones: list[TrainedClone],
    holdout_prompts: list[str],
    user_id: str,
    pool: asyncpg.Pool,
    openai_client=None,
    gemma_fn=None,
) -> list[CloneScore]:
    """
    Score each clone on personal holdout set.
    Scoring = semantic similarity to teacher response + topic breakdown.
    """
    # Fetch teacher responses for holdout (or generate if not cached)
    teacher_responses = await _get_teacher_responses(holdout_prompts, pool, user_id, openai_client)

    scored = []
    for clone in clones:
        topic_scores = {}
        all_scores = []

        for prompt, teacher_resp in zip(holdout_prompts, teacher_responses):
            if not teacher_resp:
                continue

            if gemma_fn:
                try:
                    clone_resp = gemma_fn(clone.adapter_path, prompt)
                except Exception:
                    clone_resp = ""
            else:
                clone_resp = ""

            sim = _semantic_similarity(prompt, clone_resp, teacher_resp)
            all_scores.append(sim)

            topic = _detect_topic(prompt)
            topic_scores.setdefault(topic, []).append(sim)

        # Average per topic
        topic_avgs = {t: float(sum(v) / len(v)) for t, v in topic_scores.items() if v}
        overall = float(sum(all_scores) / len(all_scores)) if all_scores else 0.0

        scored.append(CloneScore(
            clone=clone,
            overall_score=overall,
            topic_scores=topic_avgs,
        ))

    # Rank by overall score
    scored.sort(key=lambda x: x.overall_score, reverse=True)
    for i, cs in enumerate(scored):
        cs.rank = i + 1

    return scored


async def select_winner_and_archive(
    scores: list[CloneScore],
    user_id: str,
    pool: asyncpg.Pool,
) -> tuple[CloneScore, list[CloneScore]]:
    """
    Winner = rank 1. Losers are archived to DB + kept on disk.
    Returns (winner, losers).
    """
    if not scores:
        raise ValueError("No clones to select winner from")

    winner = scores[0]
    losers = scores[1:]

    today = date.today()

    # Save all to clone_lineage
    async with pool.acquire() as conn:
        for cs in scores:
            await conn.execute(
                """
                INSERT INTO clone_lineage
                    (user_id, night, clone_type, clone_dna, overall_score, topic_scores, is_winner, adapter_path)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                user_id,
                today,
                cs.clone.clone_type,
                json.dumps(cs.clone.dna.__dict__),
                cs.overall_score,
                json.dumps(cs.topic_scores),
                cs.rank == 1,
                cs.clone.adapter_path,
            )

        # Save losers to skill_archive (keep them retrievable)
        for cs in losers:
            await conn.execute(
                """
                INSERT INTO skill_archive
                    (user_id, night, clone_type, clone_rank, overall_score, topic_scores, adapter_path)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                user_id,
                today,
                cs.clone.clone_type,
                cs.rank,
                cs.overall_score,
                json.dumps(cs.topic_scores),
                cs.clone.adapter_path,
            )

    return winner, losers


async def get_best_adapter_for_topic(
    user_id: str,
    topic: str,
    pool: asyncpg.Pool,
    current_adapter_path: str,
    current_topic_score: float = 0.0,
    improvement_threshold: float = 0.10,
    lookback_days: int = 30,
) -> str:
    """
    Check skill archive: is there a past clone that's 10% better on this topic?
    If yes, return its adapter path. Otherwise return current_adapter_path.

    Called by the router when routing a topic-specific request.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT adapter_path, topic_scores, overall_score
            FROM skill_archive
            WHERE user_id = $1
              AND night >= CURRENT_DATE - INTERVAL '$2 days'
              AND adapter_path IS NOT NULL
            ORDER BY night DESC
            LIMIT 60
            """,
            user_id, lookback_days,
        )

    best_path = current_adapter_path
    best_score = current_topic_score

    for row in rows:
        topic_scores = json.loads(row["topic_scores"] or "{}")
        archive_score = topic_scores.get(topic, row["overall_score"])
        if archive_score > best_score * (1 + improvement_threshold):
            best_score = archive_score
            best_path = row["adapter_path"]

    return best_path


async def load_personal_holdout(user_id: str, pool: asyncpg.Pool, limit: int = 50) -> list[str]:
    """Load recent high-quality prompts as personal holdout benchmark."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT prompt FROM training_pairs
            WHERE user_id = $1
              AND quality_score >= 0.8
            ORDER BY created_at DESC
            LIMIT $2
            """,
            user_id, limit,
        )
    return [r["prompt"] for r in rows]


async def _get_teacher_responses(
    prompts: list[str],
    pool: asyncpg.Pool,
    user_id: str,
    openai_client=None,
) -> list[str]:
    """Fetch teacher responses from training_pairs cache. Generate missing ones if client available."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT prompt, teacher_response FROM training_pairs
            WHERE user_id = $1 AND prompt = ANY($2)
            """,
            user_id, prompts,
        )
    cached = {r["prompt"]: r["teacher_response"] for r in rows}

    result = []
    for prompt in prompts:
        if prompt in cached:
            result.append(cached[prompt])
        elif openai_client:
            try:
                resp = await openai_client.chat.completions.create(
                    model=os.environ.get("TEACHER_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                )
                result.append(resp.choices[0].message.content)
            except Exception:
                result.append("")
        else:
            result.append("")

    return result


def _semantic_similarity(prompt: str, candidate: str, reference: str) -> float:
    if not candidate or not reference:
        return 0.0
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embs = model.encode([candidate, reference], normalize_embeddings=True)
        return float(embs[0] @ embs[1])
    except Exception:
        # Token overlap fallback
        toks_c = set(candidate.lower().split())
        toks_r = set(reference.lower().split())
        if not toks_c or not toks_r:
            return 0.0
        return len(toks_c & toks_r) / len(toks_c | toks_r)


def _detect_topic(text: str) -> str:
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["code", "python", "function", "class", "bug", "error", "debug"]):
        return "coding"
    if any(kw in text_lower for kw in ["model", "train", "neural", "ml", "ai", "loss", "dataset"]):
        return "ml"
    if any(kw in text_lower for kw in ["write", "essay", "story", "email", "draft", "paragraph"]):
        return "writing"
    if any(kw in text_lower for kw in ["math", "calculate", "equation", "solve", "proof"]):
        return "math"
    if any(kw in text_lower for kw in ["research", "paper", "study", "arxiv", "survey"]):
        return "research"
    return "general"
