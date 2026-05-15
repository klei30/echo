import os
import shutil
import sys
from pathlib import Path


def get_training_runtime() -> str:
    """
    Return the effective training runtime mode.

    Modes:
      windows_wsl  — training runs inside WSL Ubuntu on a Windows host (default on win32)
      linux_local  — training runs natively on Linux/macOS (Kaggle, cloud GPU, server)
      disabled     — training is administratively disabled; product still runs
      auto         — detect from platform (default)
    """
    from config import settings
    configured = (settings.echo_training_runtime or "auto").strip().lower()
    if configured not in ("auto", "windows_wsl", "linux_local", "disabled"):
        configured = "auto"
    if configured != "auto":
        return configured
    return "windows_wsl" if sys.platform == "win32" else "linux_local"


async def training_runtime_info() -> dict:
    """
    Return a structured dict describing the current training runtime.
    Used by training/summary.py and any API surface that exposes training state.
    """
    runtime = get_training_runtime()

    if runtime == "disabled":
        return {
            "mode": "disabled",
            "available": False,
            "status": "disabled",
            "python": None,
            "platform": sys.platform,
        }

    if runtime == "windows_wsl":
        from training.runtime import wsl_distro_available
        wsl = await wsl_distro_available()
        return {
            "mode": "windows_wsl",
            "available": bool(wsl.get("available")),
            "status": "ready" if wsl.get("available") else wsl.get("status", "unavailable"),
            "python": None,
            "platform": sys.platform,
            "wsl_distro": wsl.get("distro"),
        }

    # linux_local
    from config import settings
    python_override = (settings.echo_training_python or "").strip()
    python = python_override or sys.executable
    available = bool(python and (Path(python).is_file() or shutil.which(python)))
    return {
        "mode": "linux_local",
        "available": available,
        "status": "ready" if available else "training_python_missing",
        "python": python,
        "platform": sys.platform,
    }
