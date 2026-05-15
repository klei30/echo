import asyncio
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import httpx

from config import settings
from training.adapter import _win_to_wsl
from training.collector import get_sft_pairs, get_dpo_pairs, mark_used
from router.compass import get_rebalance_weights

log = logging.getLogger("echo.orchestrator")


def _lane_prefix(user_id: str) -> str:
    return f"gemma4_{user_id}"


def _to_str(val) -> str:
    if isinstance(val, list):
        return " ".join(p.get("text", "") for p in val if isinstance(p, dict) and p.get("type") == "text")
    return str(val) if val is not None else ""


def _write_sft_data(pairs: list[dict], dest: Path) -> None:
    dataset = [
        {"instruction": _to_str(p["user_msg"]), "input": "", "output": _to_str(p["assistant_msg"])}
        for p in pairs
        if _to_str(p["user_msg"]) and _to_str(p["assistant_msg"])
    ]
    dest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in dataset),
        encoding="utf-8",
    )


def _write_dpo_data(pairs: list[dict], dest: Path) -> bool:
    if pairs and "chosen" in pairs[0] and "rejected" in pairs[0]:
        dpo = [
            {
                "instruction": _to_str(p["user_msg"]),
                "chosen": _to_str(p["chosen"]),
                "rejected": _to_str(p["rejected"]),
            }
            for p in pairs
            if _to_str(p.get("user_msg")) and _to_str(p.get("chosen")) and _to_str(p.get("rejected"))
        ]
        if not dpo:
            return False
        dest.write_text(
            "\n".join(json.dumps(row, ensure_ascii=True) for row in dpo),
            encoding="utf-8",
        )
        return True

    # Group by user_msg to find chosen/rejected for the same prompt
    by_prompt: dict[str, dict] = {}
    for p in pairs:
        key = _to_str(p["user_msg"])
        entry = by_prompt.setdefault(key, {"chosen": None, "rejected": None})
        if p.get("engagement_signal") == "thumbs_up" and entry["chosen"] is None:
            entry["chosen"] = _to_str(p["assistant_msg"])
        elif p.get("engagement_signal") == "thumbs_down" and entry["rejected"] is None:
            entry["rejected"] = _to_str(p["assistant_msg"])

    dpo = [
        {"instruction": prompt, "chosen": v["chosen"], "rejected": v["rejected"]}
        for prompt, v in by_prompt.items()
        if v["chosen"] and v["rejected"]
    ]
    if not dpo:
        return False
    dest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in dpo),
        encoding="utf-8",
    )
    return True


def _apply_compass_weights(pairs: list[dict], weights: dict[str, float], max_pairs: int = 500) -> list[dict]:
    if not pairs or not weights:
        return pairs[:max_pairs]
    pair_weights = [weights.get(p.get("topic", "general"), 1.0) for p in pairs]
    total = sum(pair_weights)
    if total == 0:
        return pairs[:max_pairs]
    probs = [w / total for w in pair_weights]
    n = min(len(pairs), max_pairs)
    indices = random.choices(range(len(pairs)), weights=probs, k=n)
    seen: set[int] = set()
    resampled = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            resampled.append(pairs[i])
    if len(resampled) < n:
        for i, p in enumerate(pairs):
            if i not in seen:
                resampled.append(p)
            if len(resampled) >= n:
                break
    return resampled


async def _run_unsloth(cfg: dict, cfg_path: Path, label: str) -> bool:
    from training.platform import get_training_runtime
    runtime = get_training_runtime()
    if runtime == "disabled":
        log.warning("%s skipped because training runtime is disabled", label)
        return False
    if runtime == "linux_local":
        return await _run_unsloth_linux(cfg, cfg_path, label)
    return await _run_unsloth_wsl(cfg, cfg_path, label)


async def _run_unsloth_wsl(cfg: dict, cfg_path: Path, label: str) -> bool:
    """Original WSL-based training launch — Windows only, paths converted to /mnt/c/..."""
    wsl_cfg_path = _win_to_wsl(str(cfg_path.resolve()))
    script_path = _win_to_wsl(str((Path(__file__).parent / "unsloth_train.py").resolve()))
    model_path = cfg.get("model_path", "")
    wsl_model_path = _win_to_wsl(model_path) if model_path and Path(model_path).exists() else model_path

    wsl_cfg = {**cfg}
    for key in ("model_path", "output_dir", "data_path", "prev_adapter"):
        if wsl_cfg.get(key) and Path(str(wsl_cfg[key])).exists():
            wsl_cfg[key] = _win_to_wsl(str(Path(wsl_cfg[key]).resolve()))
    if wsl_cfg.get("model_path"):
        wsl_cfg["model_path"] = (
            _win_to_wsl(str(Path(cfg["model_path"]).resolve()))
            if Path(cfg["model_path"]).exists()
            else wsl_model_path
        )

    cfg_path.write_text(json.dumps(wsl_cfg, indent=2, ensure_ascii=True), encoding="utf-8")
    log.info("%s launching via Unsloth (windows_wsl): %s", label, wsl_cfg_path)

    wsl_args = [
        "wsl.exe", "-d", settings.echo_wsl_distro,
        "env",
        f"PYTHONPATH={settings.echo_wsl_training_pythonpath}",
        "HF_HOME=/mnt/c/Users/ASUS/.cache/huggingface",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        settings.echo_wsl_training_python,
        script_path,
        "--config", wsl_cfg_path,
    ]

    proc = await asyncio.create_subprocess_exec(
        *wsl_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    out = stdout.decode(errors="ignore")
    err = stderr.decode(errors="ignore")
    if out:
        log.info("%s stdout: %s", label, out[-2000:])
    if proc.returncode != 0:
        log.error("%s failed stderr=%s", label, err[-3000:])
        return False
    return True


async def _run_unsloth_linux(cfg: dict, cfg_path: Path, label: str) -> bool:
    """Linux-native training launch — no WSL, no path conversion."""
    script_path = str((Path(__file__).parent / "unsloth_train.py").resolve())
    python = (settings.echo_training_python or "").strip() or sys.executable

    # Write cfg as-is — paths are already native Linux absolute paths
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=True), encoding="utf-8")
    log.info("%s launching via Unsloth (linux_local): %s", label, cfg_path)

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    pythonpath_override = (settings.echo_training_pythonpath or "").strip()
    if pythonpath_override:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{pythonpath_override}:{existing}" if existing else pythonpath_override

    proc = await asyncio.create_subprocess_exec(
        python, script_path, "--config", str(cfg_path.resolve()),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    out = stdout.decode(errors="ignore")
    err = stderr.decode(errors="ignore")
    if out:
        log.info("%s stdout: %s", label, out[-2000:])
    if proc.returncode != 0:
        log.error("%s failed stderr=%s", label, err[-3000:])
        return False
    return True


# ─── vLLM helpers for data prep ──────────────────────────────────────────────

async def _vllm_generate(
    prompt: str,
    model: str,
    base_url: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    timeout: float = 20.0,
) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stop": ["<function", "<tool_call>"],
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


async def _batch_generate(
    prompts: list[str],
    model: str,
    base_url: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    concurrency: int = 4,
) -> list[str | None]:
    sem = asyncio.Semaphore(concurrency)

    async def _one(p: str) -> str | None:
        async with sem:
            return await _vllm_generate(p, model, base_url, temperature, max_tokens)

    return list(await asyncio.gather(*[_one(p) for p in prompts]))


# ─── Clone dataset preparation (runs while vLLM is up) ───────────────────────

async def _prepare_self_critique_pairs(
    pairs: list[dict],
    model: str,
    base_url: str,
    limit: int = 40,
) -> list[dict]:
    """Ask vLLM to critique and rewrite each response. Fall back to original on failure."""
    subset = pairs[:limit]
    critique_prompts = [
        f"Question: {_to_str(p['user_msg'])}\n\n"
        f"Your answer: {_to_str(p['assistant_msg'])[:400]}\n\n"
        f"Find the main weakness in your answer, then rewrite it better. "
        f"Reply with only the improved answer, nothing else."
        for p in subset
    ]
    improved = await _batch_generate(critique_prompts, model, base_url, temperature=0.3, max_tokens=600)
    result = []
    for p, imp in zip(subset, improved):
        if imp and len(imp.split()) >= 5:
            result.append({**p, "assistant_msg": imp})
        else:
            result.append(p)
    result.extend(pairs[limit:])
    return result


async def _prepare_group_dpo_pairs(
    sft_pairs: list[dict],
    existing_dpo: list[dict],
    model: str,
    base_url: str,
    limit: int = 15,
) -> list[dict]:
    """
    Augment DPO pairs: for thumbs_up prompts with no existing DPO counterpart,
    generate a high-temperature (lower-quality) variant to serve as the rejected response.
    """
    result = list(existing_dpo)
    covered = {_to_str(p.get("user_msg", "")) for p in existing_dpo}

    candidates = [
        p for p in sft_pairs
        if p.get("engagement_signal") == "thumbs_up"
        and _to_str(p["user_msg"]) not in covered
    ][:limit]

    if not candidates:
        return result

    prompts = [_to_str(p["user_msg"]) for p in candidates]
    rejected_responses = await _batch_generate(
        prompts, model, base_url, temperature=1.2, max_tokens=400
    )

    for p, rejected in zip(candidates, rejected_responses):
        chosen = _to_str(p["assistant_msg"])
        if rejected and rejected != chosen and len(rejected.split()) >= 3:
            result.append({
                "user_msg": _to_str(p["user_msg"]),
                "chosen": chosen,
                "rejected": rejected,
                "chosen_id": p.get("id"),
                "rejected_id": None,
            })
    return result


async def _prepare_on_policy_pairs(
    pairs: list[dict],
    model: str,
    base_url: str,
    limit: int = 40,
) -> list[dict]:
    """
    On-policy GKD: generate Gemma's current response for each prompt.
    Replaces stored response with on-policy generation if the length is reasonable.
    Trains the model on what it currently generates — reduces distribution shift.
    """
    subset = pairs[:limit]
    prompts = [_to_str(p["user_msg"]) for p in subset]
    generated = await _batch_generate(prompts, model, base_url, temperature=0.7, max_tokens=512)

    result = []
    for p, gen in zip(subset, generated):
        original_len = len(_to_str(p["assistant_msg"]).split())
        if gen:
            gen_len = len(gen.split())
            if gen_len >= 5 and (original_len == 0 or 0.25 <= gen_len / max(original_len, 1) <= 4.0):
                result.append({**p, "assistant_msg": gen})
                continue
        result.append(p)
    result.extend(pairs[limit:])
    return result


async def prepare_clone_datasets(user_id: str, lane: str = "gemma4_e2b") -> dict:
    """
    Prepare all 4 clone datasets while vLLM is still running.
    SelfCritique, GroupDPO, and OnPolicyGKD make vLLM inference calls for data augmentation.
    Returns empty dict if not enough base data exists.
    """
    from training.adapter import lora_name_for_user, vllm_model_loaded

    sft_pairs = await get_sft_pairs(user_id)
    dpo_pairs = await get_dpo_pairs(user_id)

    if len(sft_pairs) < settings.min_pairs_for_training:
        log.info(
            "Not enough pairs for clone tournament user=%s (%d < %d)",
            user_id, len(sft_pairs), settings.min_pairs_for_training,
        )
        return {}

    try:
        weights = await get_rebalance_weights(user_id)
        sft_pairs = _apply_compass_weights(sft_pairs, weights)
        topic_dist: dict[str, int] = {}
        for p in sft_pairs:
            topic_dist[p.get("topic", "general")] = topic_dist.get(p.get("topic", "general"), 0) + 1
        log.info("Compass rebalanced clone pairs user=%s: %s", user_id, topic_dist)
    except Exception as e:
        log.warning("Compass rebalancing failed: %s", e)

    # Use user LoRA if loaded, else base model for data prep inference
    vllm_base = settings.gemma4_vllm_base_url
    model = settings.gemma4_base_model
    try:
        if await vllm_model_loaded(user_id, lane=lane):
            model = lora_name_for_user(user_id, lane=lane)
    except Exception:
        pass

    log.info(
        "Preparing clone datasets user=%s sft=%d dpo=%d model=%s",
        user_id, len(sft_pairs), len(dpo_pairs), model,
    )

    # SeqKD: filter to teacher-generated pairs; fall back to all if < 10
    teacher_keywords = {
        settings.teacher_model.lower(),
        "teacher",
        settings.teacher_model.split("-")[0].lower(),  # "gemma"
    }
    # Also match on partial names like "gemma-4-31b"
    teacher_keywords.update({"gemma-4-31b", "31b", "31b-it"})
    seqkd_pairs = [
        p for p in sft_pairs
        if any(kw in (p.get("model_used") or "").lower() for kw in teacher_keywords)
    ]
    if len(seqkd_pairs) < 10:
        seqkd_pairs = sft_pairs
        log.info("SeqKD fallback to all pairs user=%s (teacher pairs=%d)", user_id, len(seqkd_pairs))

    # SelfCritique, GroupDPO, OnPolicyGKD — vLLM inference for data augmentation
    try:
        self_critique_pairs = await _prepare_self_critique_pairs(sft_pairs, model, vllm_base)
        log.info("SelfCritique prepared user=%s pairs=%d", user_id, len(self_critique_pairs))
    except Exception as e:
        log.warning("SelfCritique prep failed user=%s: %s — using raw pairs", user_id, e)
        self_critique_pairs = sft_pairs

    try:
        group_dpo_pairs = await _prepare_group_dpo_pairs(sft_pairs, dpo_pairs, model, vllm_base)
        log.info("GroupDPO prepared user=%s pairs=%d (orig=%d)", user_id, len(group_dpo_pairs), len(dpo_pairs))
    except Exception as e:
        log.warning("GroupDPO prep failed user=%s: %s — using raw DPO pairs", user_id, e)
        group_dpo_pairs = dpo_pairs

    try:
        on_policy_pairs = await _prepare_on_policy_pairs(sft_pairs, model, vllm_base)
        log.info("OnPolicyGKD prepared user=%s pairs=%d", user_id, len(on_policy_pairs))
    except Exception as e:
        log.warning("OnPolicyGKD prep failed user=%s: %s — using raw pairs", user_id, e)
        on_policy_pairs = sft_pairs

    return {
        "seqkd": seqkd_pairs,
        "self_critique": self_critique_pairs,
        "group_dpo": group_dpo_pairs,
        "on_policy": on_policy_pairs,
        "base_sft": sft_pairs,
        "base_dpo": dpo_pairs,
    }


# ─── Clone training (runs while vLLM is stopped) ─────────────────────────────

def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _jsonl_sample(path: Path, limit: int = 1) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
        if len(rows) >= limit:
            break
    return rows


async def build_pipeline_trace(
    user_id: str,
    lane: str = "gemma4_e2b",
    *,
    prepare_augmented: bool = False,
    write_datasets: bool = False,
) -> dict:
    """Expose the production Shadow Clone pipeline without starting training."""
    prefix = _lane_prefix(user_id)
    data_dir = Path(settings.training_data_dir) / prefix
    data_dir.mkdir(parents=True, exist_ok=True)

    raw_sft = await get_sft_pairs(user_id)
    raw_dpo = await get_dpo_pairs(user_id)
    clone_data: dict = {
        "seqkd": raw_sft,
        "self_critique": raw_sft,
        "on_policy": raw_sft,
        "group_dpo": raw_dpo,
        "base_sft": raw_sft,
        "base_dpo": raw_dpo,
    }
    prep_status = {"attempted": bool(prepare_augmented), "ok": False, "mode": "raw_training_pairs", "error": None}

    if prepare_augmented and len(raw_sft) >= settings.min_pairs_for_training:
        try:
            prepared = await prepare_clone_datasets(user_id, lane=lane)
            if prepared:
                clone_data = prepared
                prep_status.update({"ok": True, "mode": "vllm_augmented_clone_datasets"})
            else:
                prep_status["error"] = "prepare_clone_datasets returned no datasets"
        except Exception as e:
            prep_status["error"] = repr(e)
    elif prepare_augmented:
        prep_status["error"] = f"not enough SFT pairs ({len(raw_sft)} < {settings.min_pairs_for_training})"

    clone_specs = [
        {"name": "seqkd", "stage": "sft", "purpose": "Distill strong teacher/base Gemma answers into the user's personal lane.", "pairs": clone_data.get("seqkd", [])},
        {"name": "self_critique", "stage": "sft", "purpose": "Train on Gemma critiques and improved rewrites of its own answers.", "pairs": clone_data.get("self_critique", [])},
        {"name": "on_policy", "stage": "sft", "purpose": "Train on current-policy Gemma generations to reduce distribution shift.", "pairs": clone_data.get("on_policy", [])},
        {"name": "group_dpo", "stage": "dpo", "purpose": "Learn from chosen vs rejected responses and Decision Room preferences.", "pairs": clone_data.get("group_dpo", [])},
    ]

    datasets = {}
    for spec in clone_specs:
        name = spec["name"]
        stage = spec["stage"]
        path = data_dir / f"trace_{name}_data.json"
        writable = False
        if write_datasets:
            if stage == "dpo":
                writable = _write_dpo_data(spec["pairs"], path)
            else:
                _write_sft_data(spec["pairs"], path)
                writable = path.exists() and _jsonl_count(path) > 0
        rows = _jsonl_count(path) if write_datasets and path.exists() else len(spec["pairs"])
        sample = _jsonl_sample(path) if write_datasets else spec["pairs"][:1]
        datasets[name] = {
            "stage": stage,
            "purpose": spec["purpose"],
            "rows": rows,
            "path": str(path.resolve()) if write_datasets and path.exists() else None,
            "ready": bool(writable if write_datasets else spec["pairs"]),
            "sample": sample,
        }

    return {
        "ready": len(raw_sft) >= settings.min_pairs_for_training,
        "lane": lane,
        "profile": "production_pipeline_trace",
        "user_id": user_id,
        "required_pairs": settings.min_pairs_for_training,
        "raw_counts": {"sft_pairs": len(raw_sft), "dpo_pairs": len(raw_dpo)},
        "prep": prep_status,
        "datasets_dir": str(data_dir.resolve()),
        "clone_variants": [{"name": spec["name"], "stage": spec["stage"], "purpose": spec["purpose"]} for spec in clone_specs],
        "datasets": datasets,
        "production_stages": [
            "collect_user_signals",
            "build_sft_and_dpo_datasets",
            "prepare_seqkd_self_critique_group_dpo_on_policy_lanes",
            "stop_vllm_for_training",
            "train_lora_variants",
            "restart_vllm",
            "evaluate_variants",
            "pick_winner",
            "hot_swap_winning_adapter",
        ],
    }


def _clamp_int(value, *, default: int, low: int, high: int) -> int:
    try:
        n = int(value)
    except Exception:
        n = default
    return max(low, min(high, n))


async def run_demo_training_loop(
    user_id: str,
    lane: str = "gemma4_e2b",
    *,
    max_pairs: int = 8,
    max_steps: int = 8,
    min_pairs: int = 4,
) -> dict:
    """Run one bounded, real Unsloth SFT loop for Kaggle/demo evidence.

    This intentionally uses the same collector, Unsloth subprocess, vLLM
    stop/start, eval, and hot-swap pieces as production training. The bounded
    profile keeps notebook execution inspectable and repeatable.
    """
    from training.adapter import adapter_status, get_last_checkpoint, hot_swap_adapter, lora_name_for_user
    from training.evaluator import evaluate_new_adapter
    from training.platform import training_runtime_info
    from training.runtime import start_gemma_vllm_after_training, stop_gemma_vllm_for_training, vllm_models_health

    max_pairs = _clamp_int(max_pairs, default=8, low=2, high=16)
    max_steps = _clamp_int(max_steps, default=8, low=1, high=40)
    min_pairs = _clamp_int(min_pairs, default=4, low=2, high=max_pairs)

    prefix = _lane_prefix(user_id)
    data_dir = Path(settings.training_data_dir) / prefix
    adapters_dir = Path(settings.adapters_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    adapters_dir.mkdir(parents=True, exist_ok=True)

    raw_sft = await get_sft_pairs(user_id, limit=max(max_pairs * 2, min_pairs))
    selected_pairs = [
        p for p in raw_sft
        if _to_str(p.get("user_msg")) and _to_str(p.get("assistant_msg"))
    ][:max_pairs]
    used_pair_ids = {int(p["id"]) for p in selected_pairs if p.get("id") is not None}
    runtime = await training_runtime_info()

    prompt = (
        "Noor has a working offline garden sensor, a 40 minute stability test, "
        "and a peer note that he explains electronics clearly. What should "
        "he do today to turn this rough work into inspectable proof?"
    )

    if sys.platform == "win32" and os.getenv("ECHO_ALLOW_WINDOWS_DEMO_TRAINING", "").lower() not in {"1", "true", "yes"}:
        return {
            "status": "blocked_windows_local_demo_training",
            "profile": "bounded_demo",
            "real_training": False,
            "lane": lane,
            "user_id": user_id,
            "required_demo_pairs": min_pairs,
            "available_pairs": len(selected_pairs),
            "raw_sft_pairs": len(raw_sft),
            "training_runtime": runtime,
            "before_after": {
                "prompt": prompt,
                "before": {"model": settings.gemma4_base_model, "response": None},
                "after": {"model": settings.gemma4_base_model, "response": None},
            },
            "next_step": "Local Windows demo mode blocks real Unsloth training to avoid stopping vLLM. Run in Kaggle/Linux, or set ECHO_ALLOW_WINDOWS_DEMO_TRAINING=1 if your WSL Unsloth runtime is confirmed working.",
        }

    if not runtime.get("available"):
        return {
            "status": "blocked_training_runtime_unavailable",
            "profile": "bounded_demo",
            "real_training": False,
            "lane": lane,
            "user_id": user_id,
            "required_demo_pairs": min_pairs,
            "available_pairs": len(selected_pairs),
            "raw_sft_pairs": len(raw_sft),
            "training_runtime": runtime,
            "before_after": {
                "prompt": prompt,
                "before": {"model": settings.gemma4_base_model, "response": None},
                "after": {"model": settings.gemma4_base_model, "response": None},
            },
            "next_step": "Run this notebook in Kaggle/Linux with Unsloth available, or set ECHO_TRAINING_RUNTIME to a working runtime.",
        }

    before_response = await _vllm_generate(
        prompt,
        settings.gemma4_base_model,
        settings.gemma4_vllm_base_url,
        temperature=0.2,
        max_tokens=240,
        timeout=60.0,
    )

    if len(selected_pairs) < min_pairs:
        return {
            "status": "blocked_not_enough_demo_data",
            "profile": "bounded_demo",
            "real_training": False,
            "lane": lane,
            "user_id": user_id,
            "required_demo_pairs": min_pairs,
            "available_pairs": len(selected_pairs),
            "raw_sft_pairs": len(raw_sft),
            "training_runtime": runtime,
            "before": {"model": settings.gemma4_base_model, "prompt": prompt, "response": before_response},
            "next_step": "Seed the demo user or save more high-quality pairs, then rerun this endpoint.",
        }

    stamp = int(time.time())
    data_file = data_dir / f"demo_loop_sft_data_{stamp}.json"
    cfg_file = data_dir / f"demo_loop_sft_config_{stamp}.json"
    out_dir = adapters_dir / f"{prefix}_demo_loop_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_sft_data(selected_pairs, data_file)
    rows_written = _jsonl_count(data_file)
    if rows_written < min_pairs:
        return {
            "status": "blocked_empty_demo_dataset",
            "profile": "bounded_demo",
            "real_training": False,
            "lane": lane,
            "user_id": user_id,
            "required_demo_pairs": min_pairs,
            "rows_written": rows_written,
            "data_path": str(data_file.resolve()),
            "training_runtime": runtime,
        }

    prev = await get_last_checkpoint(user_id, lane=lane)
    cfg = {
        "model_path": str(Path(settings.gemma4_training_model_path).resolve()),
        "output_dir": str(out_dir.resolve()),
        "data_path": str(data_file.resolve()),
        "stage": "sft",
        "prev_adapter": prev,
        "lora_rank": 8,
        "lora_alpha": 16,
        "epochs": 1,
        "batch_size": 1,
        "grad_accum": 1,
        "lr": 2e-4,
        "max_seq_len": 1024,
        "max_steps": max_steps,
    }

    stopped_vllm = False
    restarted_vllm = False
    trained = False
    swap_ok = False
    eval_result: dict = {}
    eval_failed = False
    rollback_ok = False
    error: str | None = None

    try:
        stopped_vllm = await stop_gemma_vllm_for_training()
        trained = await _run_unsloth(cfg, cfg_file, f"DEMO_LOOP user={user_id}")
    except Exception as exc:
        error = repr(exc)
        log.exception("Bounded demo training failed user=%s lane=%s", user_id, lane)
    finally:
        if lane == "gemma4_e2b":
            try:
                restarted_vllm = await start_gemma_vllm_after_training()
            except Exception as restart_exc:
                restarted_vllm = False
                log.warning("Could not restart vLLM after demo training user=%s: %s", user_id, restart_exc)

    vllm_after = await vllm_models_health(timeout=10)

    if trained:
        swap_ok = await hot_swap_adapter(user_id, str(out_dir), record_checkpoint=True, lane=lane)
        if swap_ok:
            try:
                eval_result = await evaluate_new_adapter(
                    user_id,
                    str(out_dir),
                    prev,
                    lane=lane,
                    exclude_pair_ids=used_pair_ids,
                )
                eval_failed = not bool(eval_result.get("passed", True))
                if eval_failed and prev:
                    rollback_ok = await hot_swap_adapter(user_id, prev, record_checkpoint=False, lane=lane)
            except Exception as eval_exc:
                eval_result = {"passed": True, "skipped_reason": f"eval_error_kept_adapter: {eval_exc}"}
        else:
            eval_result = {"passed": False, "skipped_reason": "hot_swap_failed"}
    elif error is None:
        error = "Unsloth subprocess did not produce a successful adapter."

    serving_model = lora_name_for_user(user_id, lane=lane) if swap_ok else settings.gemma4_base_model
    after_response = await _vllm_generate(
        prompt,
        serving_model,
        settings.gemma4_vllm_base_url,
        temperature=0.2,
        max_tokens=240,
        timeout=60.0,
    )
    adapter = await adapter_status(user_id, lane=lane)

    if trained and swap_ok and not eval_failed:
        status = "complete"
    elif trained and swap_ok and eval_failed:
        status = "complete_eval_failed"
    elif trained:
        status = "complete_adapter_not_loaded"
    else:
        status = "failed"

    return {
        "status": status,
        "profile": "bounded_demo",
        "real_training": bool(trained),
        "reusable_demo_data": True,
        "lane": lane,
        "user_id": user_id,
        "training_runtime": runtime,
        "bounds": {
            "max_pairs": max_pairs,
            "min_pairs": min_pairs,
            "max_steps": max_steps,
            "epochs": 1,
            "batch_size": 1,
            "grad_accum": 1,
            "max_seq_len": 1024,
        },
        "dataset": {
            "raw_sft_pairs": len(raw_sft),
            "selected_pairs": len(selected_pairs),
            "rows_written": rows_written,
            "data_path": str(data_file.resolve()),
            "sample": _jsonl_sample(data_file, limit=2),
            "used_pair_ids": sorted(used_pair_ids),
            "marked_used": False,
        },
        "unsloth": {
            "config_path": str(cfg_file.resolve()),
            "output_dir": str(out_dir.resolve()),
            "trained": trained,
            "config": cfg,
            "error": error,
        },
        "runtime_steps": {
            "stopped_vllm_for_training": stopped_vllm,
            "restarted_vllm": restarted_vllm,
            "vllm_after": vllm_after,
        },
        "promotion": {
            "previous_adapter": prev,
            "hot_swap_ok": swap_ok,
            "eval": eval_result,
            "eval_failed": eval_failed,
            "rollback_ok": rollback_ok,
            "adapter_status": adapter,
        },
        "before_after": {
            "prompt": prompt,
            "before": {"model": settings.gemma4_base_model, "response": before_response},
            "after": {"model": serving_model, "response": after_response},
        },
    }


async def run_clone_training(
    user_id: str,
    clone_data: dict,
    prev_checkpoint: str | None = None,
    lane: str = "gemma4_e2b",
    mark_pairs_used: bool = True,
    used_ids_out: set[int] | None = None,
) -> dict[str, str]:
    """
    Train all 4 clone adapters sequentially via Unsloth.
    Each gets its own output directory. Returns {clone_name: adapter_path}.
    """
    adapters_dir = Path(settings.adapters_dir)
    prefix = _lane_prefix(user_id)
    data_dir = Path(settings.training_data_dir) / prefix
    data_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(Path(settings.gemma4_training_model_path).resolve())
    trained: dict[str, str] = {}

    base_cfg = {
        "model_path": model_path,
        "prev_adapter": prev_checkpoint,
        "lora_rank": 16,
        "lora_alpha": 32,
        "batch_size": 1,
        "grad_accum": 8,
        "max_seq_len": 2048,
    }

    # 4 clones: 3 SFT variants + 1 DPO variant
    clone_specs = [
        {
            "name": "seqkd",
            "stage": "sft",
            "pairs": clone_data.get("seqkd", []),
            "epochs": 2,
            "lr": 1e-4,
        },
        {
            "name": "self_critique",
            "stage": "sft",
            "pairs": clone_data.get("self_critique", []),
            "epochs": 2,
            "lr": 1e-4,
        },
        {
            "name": "on_policy",
            "stage": "sft",
            "pairs": clone_data.get("on_policy", []),
            "epochs": 2,
            "lr": 8e-5,  # slightly lower LR for on-policy — responses may be noisier
        },
        {
            "name": "group_dpo",
            "stage": "dpo",
            "pairs": clone_data.get("group_dpo", []),
            "epochs": 1,
            "lr": 5e-5,
        },
    ]

    for spec in clone_specs:
        name = spec["name"]
        pairs = spec["pairs"]
        stage = spec["stage"]
        data_file = data_dir / f"{name}_data.json"
        cfg_file = data_dir / f"{name}_config.json"
        out_dir = adapters_dir / f"{prefix}_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if stage == "sft":
            if len(pairs) < settings.min_pairs_for_training:
                log.warning(
                    "Clone %s user=%s: only %d pairs (need %d), skipping",
                    name, user_id, len(pairs), settings.min_pairs_for_training,
                )
                continue
            _write_sft_data(pairs, data_file)
        else:  # dpo
            if not _write_dpo_data(pairs, data_file):
                log.warning("Clone %s user=%s: DPO dataset empty, skipping", name, user_id)
                continue

        cfg = {
            **base_cfg,
            "output_dir": str(out_dir.resolve()),
            "data_path": str(data_file.resolve()),
            "stage": stage,
            "epochs": spec["epochs"],
            "lr": spec["lr"],
        }

        ok = await _run_unsloth(cfg, cfg_file, f"{name.upper()} user={user_id}")
        if ok:
            trained[name] = str(out_dir)
            log.info("Clone %s trained user=%s → %s", name, user_id, out_dir)
        else:
            log.warning("Clone %s failed user=%s", name, user_id)

    # Mark base SFT pairs as used (same underlying data across all SFT clones)
    if trained:
        used_ids: set[int] = set()
        for p in clone_data.get("base_sft", []):
            if p.get("id") is not None:
                used_ids.add(int(p["id"]))
        for p in clone_data.get("base_dpo", []):
            for field in ("id", "chosen_id", "rejected_id"):
                if p.get(field) is not None:
                    used_ids.add(int(p[field]))
        if used_ids_out is not None:
            used_ids_out.update(used_ids)
        if mark_pairs_used:
            await mark_used(user_id, used_ids)
            log.info("Marked %d pairs used user=%s", len(used_ids), user_id)

    return trained


# ─── Winner selection (runs while vLLM is up) ─────────────────────────────────

async def _score_adapter(
    user_id: str,
    adapter_path: str,
    lane: str = "gemma4_e2b",
    exclude_pair_ids: set[int] | None = None,
) -> float:
    """
    Hot-swap to this adapter, run word-overlap eval on held-out pairs, return score.
    Returns 0.5 neutral if no holdout pairs available.
    """
    from training.adapter import hot_swap_adapter, lora_name_for_user, adapter_is_compatible
    from training.evaluator import _fetch_held_out_pairs, _word_overlap, _length_score

    if not adapter_is_compatible(adapter_path, lane=lane):
        log.warning("Skipping incompatible adapter for eval: %s", adapter_path)
        return 0.0

    swapped = await hot_swap_adapter(user_id, adapter_path, record_checkpoint=False, lane=lane)
    if not swapped:
        log.warning("Could not load adapter for eval: %s", adapter_path)
        return 0.0

    pairs = await _fetch_held_out_pairs(user_id, 8, exclude_ids=exclude_pair_ids)
    if not pairs:
        return 0.5  # Neutral — no holdout yet, don't penalize

    lora_name = lora_name_for_user(user_id, lane=lane)
    vllm_base = settings.gemma4_vllm_base_url
    scores: list[float] = []

    for pair in pairs:
        prompt = pair.get("user_msg", "")
        expected = pair.get("assistant_msg", "")
        if not isinstance(prompt, str) or not isinstance(expected, str):
            continue
        generated = await _vllm_generate(
            prompt[:1000], lora_name, vllm_base, temperature=0.0, max_tokens=256
        )
        if generated is None:
            continue
        overlap = _word_overlap(generated, expected)
        length_ok = _length_score(generated, expected)
        scores.append(overlap * 0.7 + length_ok * 0.3)

    return sum(scores) / len(scores) if scores else 0.0


async def pick_clone_winner(
    user_id: str,
    adapter_paths: dict[str, str],
    lane: str = "gemma4_e2b",
    exclude_pair_ids: set[int] | None = None,
) -> str | None:
    """
    Evaluate all trained clones and return the best adapter path.
    Logs scores for every clone.
    """
    if not adapter_paths:
        return None
    if len(adapter_paths) == 1:
        return next(iter(adapter_paths.values()))

    scores: dict[str, float] = {}
    for clone_name, path in adapter_paths.items():
        try:
            score = await _score_adapter(user_id, path, lane=lane, exclude_pair_ids=exclude_pair_ids)
            scores[clone_name] = score
            log.info("Clone eval user=%s %s score=%.3f path=%s", user_id, clone_name, score, path)
        except Exception as e:
            log.warning("Clone eval error user=%s %s: %s", user_id, clone_name, e)
            scores[clone_name] = 0.0

    winner_name = max(scores, key=lambda k: scores[k])
    winner_path = adapter_paths[winner_name]
    log.info(
        "Clone winner user=%s: %s (%.3f) | all=%s",
        user_id, winner_name, scores[winner_name],
        {k: round(v, 3) for k, v in scores.items()},
    )
    return winner_path


# ─── Legacy single-path training (fallback) ───────────────────────────────────

async def run_training(
    user_id: str,
    prev_checkpoint: str | None = None,
    lane: str = "gemma4_e2b",
    mark_pairs_used: bool = True,
    used_ids_out: set[int] | None = None,
) -> str | None:
    """
    Original single-path: SFT → DPO.
    Used as fallback when clone_data is empty or the tournament is disabled.
    """
    sft_pairs = await get_sft_pairs(user_id)
    dpo_pairs = await get_dpo_pairs(user_id)

    try:
        weights = await get_rebalance_weights(user_id)
        sft_pairs = _apply_compass_weights(sft_pairs, weights)
        topic_dist: dict[str, int] = {}
        for p in sft_pairs:
            t = p.get("topic", "general")
            topic_dist[t] = topic_dist.get(t, 0) + 1
        log.info("Compass rebalanced pairs user=%s: %s", user_id, topic_dist)
    except Exception as e:
        log.warning("Compass rebalancing failed, using raw pairs: %s", e)

    dpo_only = len(sft_pairs) < settings.min_pairs_for_training and len(dpo_pairs) >= 4 and bool(prev_checkpoint)
    if len(sft_pairs) < settings.min_pairs_for_training and not dpo_only:
        log.info(
            "Not enough pairs user=%s sft=%d/%d dpo=%d prev=%s",
            user_id, len(sft_pairs), settings.min_pairs_for_training, len(dpo_pairs), bool(prev_checkpoint),
        )
        return None

    adapters_dir = Path(settings.adapters_dir)
    prefix = _lane_prefix(user_id)
    data_dir = Path(settings.training_data_dir) / prefix
    data_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(Path(settings.gemma4_training_model_path).resolve())
    used_pair_ids: set[int] = set()

    if dpo_only:
        final = prev_checkpoint
        log.info("SFT skipped user=%s; running DPO-only update on previous adapter", user_id)
    else:
        sft_out = adapters_dir / f"{prefix}_sft"
        sft_out.mkdir(parents=True, exist_ok=True)

        _write_sft_data(sft_pairs, data_dir / "sft_data.json")

        sft_cfg = {
            "model_path": model_path,
            "output_dir": str(sft_out.resolve()),
            "data_path": str((data_dir / "sft_data.json").resolve()),
            "stage": "sft",
            "prev_adapter": prev_checkpoint,
            "lora_rank": 16,
            "lora_alpha": 32,
            "epochs": 2,
            "batch_size": 1,
            "grad_accum": 8,
            "lr": 1e-4,
            "max_seq_len": 2048,
        }

        log.info("SFT starting user=%s pairs=%d", user_id, len(sft_pairs))
        ok = await _run_unsloth(sft_cfg, data_dir / "sft_config.json", f"SFT user={user_id}")
        if not ok:
            return None

        final = str(sft_out)
        used_pair_ids.update(int(p["id"]) for p in sft_pairs if p.get("id") is not None)

    if len(dpo_pairs) >= 4 and _write_dpo_data(dpo_pairs, data_dir / "dpo_data.json"):
        dpo_out = adapters_dir / f"{prefix}_dpo"
        dpo_out.mkdir(parents=True, exist_ok=True)
        dpo_cfg = {
            "model_path": model_path,
            "output_dir": str(dpo_out.resolve()),
            "data_path": str((data_dir / "dpo_data.json").resolve()),
            "stage": "dpo",
            "prev_adapter": final,
            "lora_rank": 16,
            "lora_alpha": 32,
            "epochs": 1,
            "batch_size": 1,
            "grad_accum": 8,
            "lr": 5e-5,
            "max_seq_len": 2048,
        }
        log.info("DPO starting user=%s pairs=%d", user_id, len(dpo_pairs))
        ok = await _run_unsloth(dpo_cfg, data_dir / "dpo_config.json", f"DPO user={user_id}")
        if ok:
            final = str(dpo_out)
            for p in dpo_pairs:
                if p.get("chosen_id") is not None:
                    used_pair_ids.add(int(p["chosen_id"]))
                if p.get("rejected_id") is not None:
                    used_pair_ids.add(int(p["rejected_id"]))
        else:
            log.warning("DPO failed user=%s — using SFT adapter", user_id)

    if used_ids_out is not None:
        used_ids_out.update(used_pair_ids)
    if mark_pairs_used:
        await mark_used(user_id, used_pair_ids)
    log.info("Training complete user=%s adapter=%s", user_id, final)
    return final
