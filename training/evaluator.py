import logging
from typing import Any, Iterable

import httpx

from config import settings
from db.database import get_conn
from training.adapter import lora_name_for_user

log = logging.getLogger("echo.evaluator")

def _word_overlap(pred: str, ref: str) -> float:
    pred_w = set(pred.lower().split())
    ref_w = set(ref.lower().split())
    if not ref_w:
        return 0.0
    return len(pred_w & ref_w) / len(ref_w)


def _length_score(pred: str, ref: str) -> float:
    """Penalize collapse (< 20% of reference length) and explosion (> 5x)."""
    ref_len = max(len(ref.split()), 1)
    ratio = len(pred.split()) / ref_len
    if ratio < 0.2:
        return 0.0
    if ratio > 5.0:
        return 0.5
    return 1.0


async def _fetch_held_out_pairs(
    user_id: str,
    n: int,
    exclude_ids: set[int] | list[int] | None = None,
) -> list[dict]:
    excluded = {int(i) for i in (exclude_ids or []) if i is not None}
    extra_sql = ""
    extra_args: list[int] = []
    if excluded:
        placeholders = ",".join("?" for _ in excluded)
        extra_sql = f" AND id NOT IN ({placeholders})"
        extra_args = list(excluded)

    async with get_conn() as db:
        async with db.execute(
            f"""
            SELECT id, user_msg, assistant_msg FROM training_pairs
            WHERE user_id = ?
              AND used_in_training = 0
              AND engagement_signal = 'thumbs_up'
              AND typeof(user_msg) = 'text'
              AND typeof(assistant_msg) = 'text'
              AND length(user_msg) > 10
              AND length(assistant_msg) > 10
              {extra_sql}
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, *extra_args, n),
        ) as cur:
            rows = await cur.fetchall()
        results = [dict(r) for r in rows]
        if len(results) >= min(settings.min_evaluation_pairs, n):
            return results

        async with db.execute(
            """
            SELECT id, user_msg, assistant_msg FROM training_pairs
            WHERE user_id = ?
              AND used_in_training = 1
              AND engagement_signal = 'thumbs_up'
              AND typeof(user_msg) = 'text'
              AND typeof(assistant_msg) = 'text'
              AND length(user_msg) > 10
              AND length(assistant_msg) > 10
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, max(0, n - len(results))),
        ) as cur:
            rows = await cur.fetchall()
        results.extend(dict(r) for r in rows)
    return results[:n]


async def reserve_evaluation_pairs(
    user_id: str,
    n: int | None = None,
) -> list[dict]:
    """Reserve one deterministic, current-batch holdout for every candidate.

    These rows remain ``used_in_training = 0`` and their ids/prompts must be
    excluded from all candidate datasets. Ordering by id makes a run
    reproducible and avoids giving each clone a different random exam.
    """
    limit = max(1, int(n or settings.evaluation_holdout_pairs))
    async with get_conn() as db:
        async with db.execute(
            """
            SELECT id, user_msg, assistant_msg
            FROM training_pairs
            WHERE user_id = ?
              AND used_in_training = 0
              AND engagement_signal = 'thumbs_up'
              AND perplexity >= ?
              AND typeof(user_msg) = 'text'
              AND typeof(assistant_msg) = 'text'
              AND length(user_msg) > 10
              AND length(assistant_msg) > 10
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, settings.quality_threshold, limit),
        ) as cur:
            return [dict(row) for row in await cur.fetchall()]


def score_response(prediction: str, reference: str) -> dict[str, float]:
    overlap = _word_overlap(prediction, reference)
    length = _length_score(prediction, reference)
    return {
        "overlap": overlap,
        "length_score": length,
        "pair_score": overlap * 0.7 + length * 0.3,
    }


def promotion_decision(
    *,
    baseline_score: float | None,
    candidate_score: float | None,
    n_eval: int,
    min_pairs: int | None = None,
    min_score: float | None = None,
    margin: float | None = None,
) -> dict[str, Any]:
    """Return a conservative, explainable adapter promotion decision."""
    required = int(min_pairs if min_pairs is not None else settings.min_evaluation_pairs)
    floor = float(min_score if min_score is not None else settings.adapter_promotion_min_score)
    required_margin = float(margin if margin is not None else settings.adapter_promotion_margin)

    if n_eval < required:
        return {
            "promote": False,
            "reason": "not_enough_held_out_pairs",
            "required_pairs": required,
            "n_eval": n_eval,
        }
    if baseline_score is None:
        return {"promote": False, "reason": "baseline_evaluation_failed", "n_eval": n_eval}
    if candidate_score is None:
        return {"promote": False, "reason": "candidate_evaluation_failed", "n_eval": n_eval}
    if candidate_score < floor:
        return {
            "promote": False,
            "reason": "candidate_below_absolute_floor",
            "candidate_score": candidate_score,
            "minimum_score": floor,
            "n_eval": n_eval,
        }
    delta = candidate_score - baseline_score
    if delta >= required_margin:
        reason = "candidate_improved"
    elif delta < 0:
        reason = "candidate_regressed"
    else:
        reason = "insufficient_improvement"
    return {
        "promote": delta >= required_margin,
        "reason": reason,
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "delta": delta,
        "required_margin": required_margin,
        "n_eval": n_eval,
    }


async def _generate_response(prompt: str, lora_name: str, vllm_base_url: str) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(
                f"{vllm_base_url.rstrip('/')}/chat/completions",
                json={
                    "model": lora_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                    "temperature": 0.0,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log.debug("Inference failed for eval lora=%s: %s", lora_name, e)
        return None


async def evaluate_model_on_pairs(
    model_name: str,
    pairs: Iterable[dict],
    *,
    vllm_base_url: str | None = None,
) -> dict[str, Any]:
    """Evaluate a named vLLM model on an already-frozen set of pairs."""
    base_url = vllm_base_url or settings.gemma4_vllm_base_url
    pair_list = list(pairs)
    scores: list[float] = []
    details: list[dict] = []
    pair_ids: list[int] = []

    for pair in pair_list:
        prompt = pair.get("user_msg", "")
        expected = pair.get("assistant_msg", "")
        if not isinstance(prompt, str) or not isinstance(expected, str):
            continue
        generated = await _generate_response(prompt[:1500], model_name, base_url)
        if generated is None:
            continue
        scored = score_response(generated, expected)
        scores.append(scored["pair_score"])
        if pair.get("id") is not None:
            pair_ids.append(int(pair["id"]))
        details.append({
            "pair_id": pair.get("id"),
            "overlap": round(scored["overlap"], 3),
            "length_score": round(scored["length_score"], 3),
            "pair_score": round(scored["pair_score"], 3),
        })

    return {
        "score": round(sum(scores) / len(scores), 4) if scores else None,
        "n_eval": len(scores),
        "pair_ids": pair_ids,
        "details": details,
        "inference_failures": max(0, len(pair_list) - len(scores)),
    }


async def evaluate_new_adapter(
    user_id: str,
    new_adapter_path: str,
    prev_adapter_path: str | None = None,
    lane: str = "gemma4_e2b",
    exclude_pair_ids: set[int] | list[int] | None = None,
    pairs: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Run held-out evaluation of the newly trained adapter (already loaded in vLLM).

    Uses a caller-supplied frozen holdout when available, otherwise selects a
    deterministic fallback set, then scores word-overlap + response length.

    Returns: {passed, score, n_eval, details, skipped_reason}
    Does NOT load or unload adapters — caller decides rollback based on 'passed'.
    """
    lora_name = lora_name_for_user(user_id, lane=lane)
    vllm_base_url = settings.gemma4_vllm_base_url

    if pairs is None:
        pairs = await _fetch_held_out_pairs(
            user_id,
            settings.evaluation_holdout_pairs,
            exclude_ids=exclude_pair_ids,
        )

    if len(pairs) < settings.min_evaluation_pairs:
        log.info(
            "Eval skipped user=%s lane=%s: only %d held-out pairs (need %d)",
            user_id, lane, len(pairs), settings.min_evaluation_pairs,
        )
        return {
            "passed": False,
            "score": None,
            "n_eval": 0,
            "details": [],
            "skipped_reason": f"not_enough_held_out_pairs ({len(pairs)} < {settings.min_evaluation_pairs})",
        }

    result = await evaluate_model_on_pairs(lora_name, pairs, vllm_base_url=vllm_base_url)
    if result["score"] is None:
        log.warning("Eval user=%s: all inference calls failed", user_id)
        return {
            "passed": False,
            "score": None,
            "n_eval": 0,
            "details": [],
            "skipped_reason": "all_inference_calls_failed",
        }

    avg = float(result["score"])
    passed = avg >= settings.adapter_promotion_min_score

    log.info(
        "Eval user=%s lane=%s n=%d score=%.3f passed=%s",
        user_id, lane, result["n_eval"], avg, passed,
    )
    return {
        "passed": passed,
        "score": round(avg, 4),
        "n_eval": result["n_eval"],
        "details": result["details"],
        "skipped_reason": None,
    }
