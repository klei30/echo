"""Shared, hardware-independent lifecycle for one Shadow Clone training run."""

import logging
from typing import Any

from config import settings
from training.adapter import (
    ensure_adapter_loaded,
    get_last_checkpoint,
    hot_swap_adapter,
    lora_name_for_user,
    unload_adapter,
)
from training.collector import mark_used
from training.evaluator import (
    evaluate_model_on_pairs,
    promotion_decision,
    reserve_evaluation_pairs,
)
from training.orchestrator import (
    pick_clone_winner,
    prepare_clone_datasets,
    run_clone_training,
    run_training,
)
from training.runtime import start_gemma_vllm_after_training, stop_gemma_vllm_for_training
from training.summary import get_training_summary

log = logging.getLogger("echo.training.coordinator")


async def _restore_baseline(user_id: str, lane: str, previous_path: str | None) -> bool:
    if previous_path:
        restored = await hot_swap_adapter(
            user_id,
            previous_path,
            record_checkpoint=False,
            lane=lane,
        )
        if restored:
            return True
        # Never leave a rejected candidate live if the previous LoRA cannot load.
        await unload_adapter(user_id, lane=lane)
        return False
    return await unload_adapter(user_id, lane=lane)


async def run_training_cycle(user_id: str, lane: str, run_id: str) -> dict[str, Any]:
    """Train, compare, and conditionally promote one set of clone candidates.

    Promotion is deliberately the last operation: candidate adapters are loaded
    for evaluation without being checkpointed, and the checkpoint is written
    only after the winner passes the baseline comparison.
    """
    previous_path = await get_last_checkpoint(user_id, lane=lane)
    holdout = await reserve_evaluation_pairs(user_id)
    holdout_ids = {int(p["id"]) for p in holdout if p.get("id") is not None}

    baseline_model = settings.gemma4_base_model
    if previous_path and await ensure_adapter_loaded(user_id, lane=lane):
        baseline_model = lora_name_for_user(user_id, lane=lane)
    baseline_eval = await evaluate_model_on_pairs(baseline_model, holdout)

    clone_data: dict = {}
    use_clone_tournament = lane == "gemma4_e2b"
    if use_clone_tournament:
        try:
            clone_data = await prepare_clone_datasets(
                user_id,
                lane,
                holdout_pairs=holdout,
            )
            if not any(clone_data.get(k) for k in ("seqkd", "self_critique", "group_dpo", "on_policy")):
                use_clone_tournament = False
        except Exception:
            log.exception("Clone dataset preparation failed user=%s; using single-path fallback", user_id)
            use_clone_tournament = False

    stopped = False
    used_pair_ids: set[int] = set()
    adapter_paths: dict[str, str] = {}
    try:
        if lane == "gemma4_e2b":
            stopped = await stop_gemma_vllm_for_training()

        if use_clone_tournament:
            adapter_paths = await run_clone_training(
                user_id,
                clone_data,
                prev_checkpoint=previous_path,
                lane=lane,
                mark_pairs_used=False,
                used_ids_out=used_pair_ids,
                run_id=run_id,
            )
        else:
            single_path = await run_training(
                user_id,
                prev_checkpoint=previous_path,
                lane=lane,
                mark_pairs_used=False,
                used_ids_out=used_pair_ids,
                run_id=run_id,
                holdout_pairs=holdout,
            )
            if single_path:
                adapter_paths = {"base": single_path}
    finally:
        if stopped:
            await start_gemma_vllm_after_training()

    if not adapter_paths:
        return {
            "status": "skipped",
            "reason": "no_adapter_produced",
            "run_id": run_id,
            "baseline": baseline_eval,
            "holdout_pair_ids": sorted(holdout_ids),
            "candidates": {},
            "promoted": False,
            "promoted_path": None,
        }

    tournament = await pick_clone_winner(
        user_id,
        adapter_paths,
        lane=lane,
        evaluation_pairs=holdout,
    )
    winner_path = (tournament or {}).get("winner_path")
    winner_score = (tournament or {}).get("winner_score")
    n_eval = int((tournament or {}).get("n_eval") or 0)
    decision = promotion_decision(
        baseline_score=baseline_eval.get("score"),
        candidate_score=winner_score,
        n_eval=n_eval,
    )

    promoted = False
    restore_ok = True
    if winner_path and decision["promote"]:
        # This is the only call that commits a candidate to checkpoints.
        promoted = await hot_swap_adapter(
            user_id,
            winner_path,
            record_checkpoint=True,
            lane=lane,
        )
        if not promoted:
            decision = {**decision, "promote": False, "reason": "promotion_hot_swap_failed"}
            restore_ok = await _restore_baseline(user_id, lane, previous_path)
    else:
        restore_ok = await _restore_baseline(user_id, lane, previous_path)

    # The examples were consumed by this experiment even if the candidate was
    # staged rather than promoted. The frozen holdout remains unused.
    if used_pair_ids:
        await mark_used(user_id, used_pair_ids)

    summary = await get_training_summary(user_id, lane=lane)
    summary["evaluation"] = {
        "baseline": baseline_eval,
        "tournament": tournament,
        "promotion": decision,
        "holdout_pair_ids": sorted(holdout_ids),
    }
    status = (
        "complete"
        if promoted
        else ("complete_not_promoted" if restore_ok else "complete_restore_failed")
    )
    return {
        "status": status,
        "run_id": run_id,
        "baseline": baseline_eval,
        "tournament": tournament,
        "candidates": adapter_paths,
        "candidate_path": winner_path,
        "promotion": decision,
        "promoted": promoted,
        "promoted_path": winner_path if promoted else None,
        "previous_path": previous_path,
        "restore_ok": restore_ok,
        "holdout_pair_ids": sorted(holdout_ids),
        "used_pair_ids": sorted(used_pair_ids),
        "summary": summary,
    }
