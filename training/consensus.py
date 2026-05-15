"""
training/consensus.py — PST-style consensus phase before tournament.

Paper: Peer-Predictive Self-Training (2604.13356)
All 4 clones answer the same test prompts.
  - High agreement (>0.85) → strong training signal, add to consensus_data
  - Low agreement        → uncertainty flag, queue prompt for GPT-4.1 tomorrow

This auto-identifies knowledge gaps without any manual auditing.
"""

import os
from datetime import date
from typing import Any

import asyncpg
import numpy as np


async def run_consensus_phase(
    user_id: str,
    clones: list[Any],            # list of TrainedClone
    test_prompts: list[str],
    pool: asyncpg.Pool,
    gemma_fn=None,                # callable(adapter_path, prompt) -> str
    agreement_threshold: float = 0.85,
) -> dict:
    """
    Run all clones on same prompts, extract consensus pairs + flag uncertain topics.

    Returns:
        {
            "consensus_data":    [{prompt, response}],
            "flagged_prompts":   [str],
            "agreement_scores":  [float],
        }
    """
    if not test_prompts:
        return {"consensus_data": [], "flagged_prompts": [], "agreement_scores": []}

    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        # Fallback: simple token overlap similarity
        embedder = None

    clone_responses: dict[str, list[str]] = {}

    for clone in clones:
        responses = []
        for prompt in test_prompts:
            if gemma_fn:
                try:
                    resp = gemma_fn(clone.adapter_path, prompt)
                    responses.append(resp)
                except Exception:
                    responses.append("")
            else:
                responses.append("")
        clone_responses[clone.clone_id] = responses

    consensus_data = []
    flagged_prompts = []
    agreement_scores = []

    for i, prompt in enumerate(test_prompts):
        responses = [clone_responses[c.clone_id][i] for c in clones if clone_responses.get(c.clone_id)]
        responses = [r for r in responses if r]

        if len(responses) < 2:
            flagged_prompts.append(prompt)
            agreement_scores.append(0.0)
            continue

        agreement = _compute_agreement(responses, embedder)
        agreement_scores.append(agreement)

        if agreement >= agreement_threshold:
            # Pick response most similar to all others (central tendency)
            best_idx = _find_most_central(responses, embedder)
            consensus_data.append({
                "prompt": prompt,
                "response": responses[best_idx],
                "agreement": agreement,
            })
        else:
            flagged_prompts.append(prompt)

    # Save flagged prompts to DB for next-day GPT-4.1 priority queue
    if flagged_prompts:
        await _save_priority_topics(user_id, flagged_prompts, pool)

    return {
        "consensus_data": consensus_data,
        "flagged_prompts": flagged_prompts,
        "agreement_scores": agreement_scores,
    }


def _compute_agreement(responses: list[str], embedder) -> float:
    if embedder is None:
        return _token_overlap_agreement(responses)

    try:
        embeddings = embedder.encode(responses, normalize_embeddings=True)
        sim_matrix = embeddings @ embeddings.T
        n = len(responses)
        # Average off-diagonal similarity
        off_diag = (sim_matrix.sum() - np.trace(sim_matrix)) / (n * (n - 1))
        return float(off_diag)
    except Exception:
        return _token_overlap_agreement(responses)


def _token_overlap_agreement(responses: list[str]) -> float:
    if len(responses) < 2:
        return 1.0
    scores = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            toks_a = set(responses[i].lower().split())
            toks_b = set(responses[j].lower().split())
            if not toks_a or not toks_b:
                continue
            overlap = len(toks_a & toks_b) / len(toks_a | toks_b)
            scores.append(overlap)
    return float(np.mean(scores)) if scores else 0.0


def _find_most_central(responses: list[str], embedder) -> int:
    if embedder is None or len(responses) == 1:
        return 0
    try:
        embeddings = embedder.encode(responses, normalize_embeddings=True)
        sim_matrix = embeddings @ embeddings.T
        centrality = sim_matrix.sum(axis=1)
        return int(centrality.argmax())
    except Exception:
        return 0


async def _save_priority_topics(user_id: str, prompts: list[str], pool: asyncpg.Pool):
    today = date.today()
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO priority_topics (user_id, prompt, flagged_night)
            VALUES ($1, $2, $3)
            ON CONFLICT DO NOTHING
            """,
            [(user_id, p, today) for p in prompts],
        )


async def load_priority_topics(user_id: str, pool: asyncpg.Pool, limit: int = 20) -> list[str]:
    """Load unprocessed priority topics for this user (for tomorrow's teacher calls)."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT prompt FROM priority_topics
            WHERE user_id = $1 AND used = FALSE
            ORDER BY created_at DESC
            LIMIT $2
            """,
            user_id, limit,
        )
    return [r["prompt"] for r in rows]


async def mark_priority_topics_used(user_id: str, prompts: list[str], pool: asyncpg.Pool):
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE priority_topics SET used = TRUE WHERE user_id = $1 AND prompt = ANY($2)",
            user_id, prompts,
        )
