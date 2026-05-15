import asyncio
import logging
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from config import settings

log = logging.getLogger("echo.runtime")


async def http_json_health(url: str, timeout: float = 3.0) -> dict[str, Any]:
    """HTTP health with latency and parsed JSON when available."""
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
        latency_ms = int((time.perf_counter() - started) * 1000)
        data: Any = None
        try:
            data = resp.json()
        except Exception:
            data = None
        return {
            "ok": 200 <= resp.status_code < 500,
            "status": "ok" if resp.status_code < 500 else f"http_{resp.status_code}",
            "status_code": resp.status_code,
            "latency_ms": latency_ms,
            "data": data,
        }
    except Exception as e:
        return {
            "ok": False,
            "status": f"error: {e}",
            "status_code": None,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "data": None,
        }


async def vllm_models_health(timeout: float = 5.0) -> dict[str, Any]:
    url = settings.gemma4_vllm_base_url.rstrip("/") + "/models"
    health = await http_json_health(url, timeout=timeout)
    models: list[str] = []
    data = health.get("data")
    if isinstance(data, dict):
        models = [m.get("id") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
    health["models"] = models
    health["ready"] = bool(health["ok"] and models)
    if health["ok"] and not models:
        health["status"] = "no_models"
    return health


async def tcp_ok(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return bool(reader)
    except Exception:
        return False


async def run_process(args: list[str], timeout: int = 30) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        return 124, "", "timeout"
    return proc.returncode or 0, stdout.decode(errors="ignore"), stderr.decode(errors="ignore")


def _clean_process_text(text: str) -> str:
    if "\x00" in text:
        try:
            text = text.encode("utf-8", errors="ignore").decode("utf-16le", errors="ignore")
        except Exception:
            text = text.replace("\x00", "")
    return " ".join(text.replace("\x00", "").split())


async def wsl_distro_available(distro: str | None = None, timeout: int = 3) -> dict[str, Any]:
    distro = distro or settings.echo_wsl_distro or "Ubuntu-24.04"
    rc, out, err = await run_process(["wsl.exe", "-d", distro, "sh", "-lc", "printf ok"], timeout=timeout)
    available = rc == 0 and "ok" in out
    status = _clean_process_text(err or out or f"exit_{rc}")
    return {
        "available": available,
        "distro": distro,
        "status": "ready" if available else status[:500],
    }


async def stop_gemma_vllm_for_training() -> bool:
    from training.platform import get_training_runtime
    runtime = get_training_runtime()

    if settings.echo_vllm_externally_managed:
        log.info("vLLM is externally managed — skipping stop before training")
        return True

    if runtime == "disabled":
        return True

    if runtime == "windows_wsl":
        rc, _out, err = await run_process([
            "wsl.exe", "-d", settings.echo_wsl_distro, "sh", "-lc",
            "pkill -f 'vllm.*gemma.*e2b' || pkill -f 'start_gemma4_e2b_vllm' || true",
        ])
        if rc != 0:
            if err.strip():
                log.warning("Could not confirm Gemma vLLM stop before training: %s", err[-500:])
            else:
                log.info("Gemma vLLM stop command returned rc=%s; continuing", rc)
        else:
            log.info("Stopped Gemma vLLM before Gemma training")
        await asyncio.sleep(5)
        return True

    # linux_local
    stop_cmd = (settings.echo_vllm_stop_command or "").strip()
    args = shlex.split(stop_cmd) if stop_cmd else ["pkill", "-f", "vllm.*gemma"]
    rc, _out, err = await run_process(args)
    if rc not in (0, 1):  # rc=1 means no matching process — that is fine
        log.warning("Gemma vLLM stop returned rc=%s err=%s", rc, err[-300:])
    else:
        log.info("Stopped Gemma vLLM (linux_local) before training")
    await asyncio.sleep(5)
    return True


async def start_gemma_vllm_after_training() -> bool:
    from training.platform import get_training_runtime
    runtime = get_training_runtime()
    root = Path(__file__).resolve().parents[1]

    if settings.echo_vllm_externally_managed:
        log.info("vLLM is externally managed — skipping restart after training")
        return True

    if runtime == "disabled":
        return False

    if runtime == "windows_wsl":
        out_log = open(root / "gemma4_vllm.current.out.log", "ab")
        err_log = open(root / "gemma4_vllm.current.err.log", "ab")
        try:
            await asyncio.create_subprocess_exec(
                "wsl.exe", "-d", settings.echo_wsl_distro, "bash",
                "/mnt/c/Users/ASUS/Desktop/echo/start_gemma4_e2b_vllm.sh",
                stdout=out_log,
                stderr=err_log,
            )
        finally:
            out_log.close()
            err_log.close()
    else:
        # linux_local — use ECHO_VLLM_START_COMMAND or generate one from settings
        start_cmd = (settings.echo_vllm_start_command or "").strip()
        if not start_cmd:
            model_path = (settings.gemma4_training_model_path or settings.gemma4_base_model).strip()
            start_cmd = " ".join([
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--host", "0.0.0.0",
                "--port", "8003",
                "--model", model_path,
                "--served-model-name", settings.gemma4_base_model,
                "--max-model-len", str(settings.gemma4_max_model_len),
                "--trust-remote-code",
                "--enable-lora",
            ])
        import os
        env = os.environ.copy()
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
        out_log = open(root / "gemma4_vllm.current.out.log", "ab")
        err_log = open(root / "gemma4_vllm.current.err.log", "ab")
        try:
            await asyncio.create_subprocess_exec(
                *shlex.split(start_cmd),
                env=env,
                stdout=out_log,
                stderr=err_log,
            )
        finally:
            out_log.close()
            err_log.close()

    for _ in range(36):
        health = await vllm_models_health(timeout=5)
        if health.get("ready"):
            log.info("Gemma vLLM restarted after training models=%s", health.get("models"))
            return True
        await asyncio.sleep(5)
    log.warning("Gemma vLLM restart was requested but /models did not recover")
    return False
