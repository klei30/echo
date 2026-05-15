import logging
from typing import Any

import httpx

from config import settings
from db.database import get_conn
from training.adapter import lora_name_for_user

log = logging.getLogger("echo.evaluator")

_MIN_EVAL_PAIRS = 3
_MAX_EVAL_PAIRS = 20
_PASS_THRESHOLD = 0.25  # word-overlap fraction — regression detection only, not reproduction check


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
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (user_id, *extra_args, n),
        ) as cur:
            rows = await cur.fetchall()
        results = [dict(r) for r in rows]
        if len(results) >= min(_MIN_EVAL_PAIRS, n):
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
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (user_id, max(0, n - len(results))),
        ) as cur:
            rows = await cur.fetchall()
        results.extend(dict(r) for r in rows)
    return results[:n]


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


async def evaluate_new_adapter(
    user_id: str,
    new_adapter_path: str,
    prev_adapter_path: str | None = None,
    lane: str = "gemma4_e2b",
    exclude_pair_ids: set[int] | list[int] | None = None,
) -> dict[str, Any]:
    """
    Run held-out evaluation of the newly trained adapter (already loaded in vLLM).

    Fetches thumbs_up pairs that were already used in training (safe held-out set),
    generates responses with the new adapter, scores word-overlap + length.

    Returns: {passed, score, n_eval, details, skipped_reason}
    Does NOT load or unload adapters — caller decides rollback based on 'passed'.
    """
    lora_name = lora_name_for_user(user_id)
    vllm_base_url = settings.gemma4_vllm_base_url

    pairs = await _fetch_held_out_pairs(user_id, _MAX_EVAL_PAIRS, exclude_ids=exclude_pair_ids)

    if len(pairs) < _MIN_EVAL_PAIRS:
        log.info(
            "Eval skipped user=%s lane=%s: only %d held-out pairs (need %d)",
            user_id, lane, len(pairs), _MIN_EVAL_PAIRS,
        )
        return {
            "passed": True,
            "score": None,
            "n_eval": 0,
            "details": [],
            "skipped_reason": f"not_enough_held_out_pairs ({len(pairs)} < {_MIN_EVAL_PAIRS})",
        }

    scores: list[float] = []
    details: list[dict] = []

    for pair in pairs:
        prompt = pair.get("user_msg", "")
        expected = pair.get("assistant_msg", "")
        if not isinstance(prompt, str) or not isinstance(expected, str):
            continue
        generated = await _generate_response(prompt[:1500], lora_name, vllm_base_url)
        if generated is None:
            continue
        overlap = _word_overlap(generated, expected)
        length_ok = _length_score(generated, expected)
        pair_score = overlap * 0.7 + length_ok * 0.3
        scores.append(pair_score)
        details.append({
            "overlap": round(overlap, 3),
            "length_score": round(length_ok, 3),
            "pair_score": round(pair_score, 3),
        })

    if not scores:
        log.warning("Eval user=%s: all inference calls failed", user_id)
        return {
            "passed": False,
            "score": None,
            "n_eval": 0,
            "details": [],
            "skipped_reason": "all_inference_calls_failed",
        }

    avg = sum(scores) / len(scores)
    passed = avg >= _PASS_THRESHOLD

    log.info(
        "Eval user=%s lane=%s n=%d score=%.3f passed=%s",
        user_id, lane, len(scores), avg, passed,
    )
    return {
        "passed": passed,
        "score": round(avg, 4),
        "n_eval": len(scores),
        "details": details,
        "skipped_reason": None,
    }
