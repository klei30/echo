import logging
import json
from pathlib import Path

import httpx

from config import settings
from db.database import get_conn

log = logging.getLogger("echo.adapter")

_ADAPTER_SUFFIX_PRIORITY = (
    "_group_dpo",
    "_dpo",
    "_on_policy",
    "_self_critique",
    "_seqkd",
    "_sft",
)


def _adapter_prefix(user_id: str, lane: str = "gemma4_e2b") -> str:
    return f"gemma4_{user_id}"


def lora_name_for_user(user_id: str, lane: str = "gemma4_e2b") -> str:
    return f"gemma4_user_{user_id}"


def _adapter_config_path(path: Path) -> Path:
    return path / "adapter_config.json"


def _latest_checkpoint_dir(path: Path) -> Path | None:
    checkpoints = [p for p in path.glob("checkpoint-*") if p.is_dir() and _adapter_config_path(p).exists()]
    if not checkpoints:
        return None

    def checkpoint_step(p: Path) -> int:
        try:
            return int(p.name.rsplit("-", 1)[-1])
        except ValueError:
            return -1

    return max(checkpoints, key=checkpoint_step)


def adapter_is_compatible(adapter_path: str, lane: str = "gemma4_e2b") -> bool:
    path = Path(adapter_path)
    cfg_path = _adapter_config_path(path)
    if not cfg_path.exists():
        return False
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Invalid adapter_config.json path=%s: %s", adapter_path, e)
        return False
    base_model = str(cfg.get("base_model_name_or_path") or "").lower()
    return "gemma" in base_model and ("4" in base_model or "e2b" in base_model)


def _valid_adapter_dir(path: Path, lane: str = "gemma4_e2b") -> str:
    if _adapter_config_path(path).exists() and adapter_is_compatible(str(path), lane=lane):
        return str(path)
    checkpoint = _latest_checkpoint_dir(path)
    if checkpoint and adapter_is_compatible(str(checkpoint), lane=lane):
        return str(checkpoint)
    return ""


def adapter_path_for_user(user_id: str, lane: str = "gemma4_e2b") -> str:
    base = Path(settings.adapters_dir)
    prefix = _adapter_prefix(user_id, lane)
    # Checkpoint table is the source of truth for clone-tournament winners.
    # Directory suffix order is only a fallback for older installs.
    try:
        import sqlite3

        with sqlite3.connect(settings.sqlite_path) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT path FROM checkpoints WHERE user_id = ? AND lane = ? ORDER BY created_at DESC LIMIT 10",
                (user_id, lane),
            ).fetchall()
        for row in rows:
            p = Path(row["path"])
            if p.exists():
                valid = _valid_adapter_dir(p, lane)
                if valid:
                    return valid
    except Exception as e:
        log.debug("Checkpoint adapter lookup skipped user=%s lane=%s: %s", user_id, lane, e)

    for suffix in _ADAPTER_SUFFIX_PRIORITY:
        p = base / f"{prefix}{suffix}"
        if p.exists():
            valid = _valid_adapter_dir(p, lane)
            if valid:
                return valid
    return ""


def adapter_exists(user_id: str, lane: str = "gemma4_e2b") -> bool:
    return bool(adapter_path_for_user(user_id, lane=lane))


def _win_to_wsl(path: str) -> str:
    """Convert C:\\foo\\bar to /mnt/c/foo/bar for vLLM running in WSL."""
    from pathlib import Path as _Path
    p = _Path(path)
    parts = list(p.parts)
    if parts and len(parts[0]) == 3 and parts[0][1] == ":":
        drive = parts[0][0].lower()
        rest = "/".join(parts[1:]).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return path.replace("\\", "/")


async def hot_swap_adapter(
    user_id: str,
    adapter_path: str,
    record_checkpoint: bool = False,
    lane: str = "gemma4_e2b",
) -> bool:
    resolved_adapter_path = _valid_adapter_dir(Path(adapter_path), lane=lane)
    if not resolved_adapter_path:
        log.warning("Skipping incompatible LoRA user=%s lane=%s path=%s", user_id, lane, adapter_path)
        return False
    adapter_path = resolved_adapter_path
    from training.platform import get_training_runtime
    lora_name = lora_name_for_user(user_id)
    abs_path = str(Path(adapter_path).resolve())
    lora_path = _win_to_wsl(abs_path) if get_training_runtime() == "windows_wsl" else abs_path
    vllm_base_url = settings.gemma4_vllm_base_url
    vllm_root = vllm_base_url.rstrip("/")
    if vllm_root.endswith("/v1"):
        vllm_root = vllm_root[:-3]
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Unload only this user's adapter so vLLM picks up fresh weights.
            # Ignore errors — adapter may not be loaded yet on first training run.
            try:
                await client.post(
                    f"{vllm_root}/v1/unload_lora_adapter",
                    json={"lora_name": lora_name},
                )
            except Exception as unload_err:
                log.debug("Unload skipped user=%s: %s", user_id, unload_err)
            resp = await client.post(
                f"{vllm_root}/v1/load_lora_adapter",
                json={"lora_name": lora_name, "lora_path": lora_path},
            )
            if resp.status_code >= 400:
                log.error(
                    "LoRA load failed user=%s lane=%s status=%s body=%s",
                    user_id, lane, resp.status_code, resp.text[:1000],
                )
                resp.raise_for_status()
        log.info("Hot-swapped LoRA user=%s lane=%s lora=%s path=%s", user_id, lane, lora_name, adapter_path)
        if record_checkpoint:
            async with get_conn() as db:
                await db.execute(
                    "INSERT OR REPLACE INTO checkpoints (user_id, lane, path) VALUES (?, ?, ?)",
                    (user_id, lane, adapter_path),
                )
                await db.commit()
        return True
    except httpx.ConnectError as e:
        log.warning("LoRA hot-swap skipped user=%s lane=%s; vLLM unavailable: %s", user_id, lane, e)
        return False
    except Exception as e:
        log.error("LoRA hot-swap failed user=%s: %s", user_id, e)
        return False


async def vllm_model_loaded(user_id: str, lane: str = "gemma4_e2b") -> bool:
    lora_name = lora_name_for_user(user_id)
    vllm_base_url = settings.gemma4_vllm_base_url
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{vllm_base_url.rstrip('/')}/models")
            resp.raise_for_status()
            models = resp.json().get("data", [])
        return any(m.get("id") == lora_name for m in models if isinstance(m, dict))
    except Exception as e:
        log.warning("Could not check vLLM models lane=%s: %s", lane, e)
        return False


async def adapter_status(user_id: str, lane: str = "gemma4_e2b") -> dict:
    path = adapter_path_for_user(user_id, lane=lane)
    lora_name = lora_name_for_user(user_id)
    vllm_base_url = settings.gemma4_vllm_base_url
    status = {
        "lane": lane,
        "base_model": settings.gemma4_base_model,
        "lora_name": lora_name,
        "exists": bool(path),
        "compatible": bool(path and adapter_is_compatible(path, lane=lane)),
        "path": path,
        "loaded": False,
        "vllm": "unknown",
        "models": [],
        "serving_model": settings.gemma4_base_model,
    }
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{vllm_base_url.rstrip('/')}/models")
            status["vllm"] = "ok" if resp.status_code < 500 else f"http_{resp.status_code}"
            resp.raise_for_status()
            models = resp.json().get("data", [])
        ids = [m.get("id") for m in models if isinstance(m, dict) and m.get("id")]
        status["models"] = ids
        status["loaded"] = lora_name in ids
        if status["loaded"]:
            status["serving_model"] = lora_name
    except Exception as e:
        status["vllm"] = f"error: {e}"
    return status


async def ensure_adapter_loaded(user_id: str, lane: str = "gemma4_e2b") -> bool:
    path = adapter_path_for_user(user_id, lane=lane)
    if not path:
        return False
    if await vllm_model_loaded(user_id, lane=lane):
        return True
    return await hot_swap_adapter(user_id, path, record_checkpoint=False, lane=lane)


async def get_last_checkpoint(user_id: str, lane: str = "gemma4_e2b") -> str | None:
    async with get_conn() as db:
        async with db.execute(
            "SELECT path FROM checkpoints WHERE user_id = ? AND lane = ? ORDER BY created_at DESC LIMIT 10",
            (user_id, lane),
        ) as cur:
            rows = await cur.fetchall()
    for row in rows:
        path = row["path"]
        if path and Path(path).exists() and _valid_adapter_dir(Path(path), lane):
            return _valid_adapter_dir(Path(path), lane)
    return None


def adapter_user_from_dirname(name: str) -> str | None:
    for suffix in _ADAPTER_SUFFIX_PRIORITY:
        if name.endswith(suffix):
            user_id = name[: -len(suffix)]
            if user_id.startswith("gemma4_"):
                user_id = user_id[len("gemma4_"):]
            return user_id or None
    return None
