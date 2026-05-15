import importlib.metadata as md
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from training.platform import get_training_runtime


def win_to_wsl(path: str) -> str:
    p = path.replace("\\", "/")
    if len(p) >= 3 and p[1] == ":" and p[2] == "/":
        return f"/mnt/{p[0].lower()}/{p[3:]}"
    return p


def wsl_to_win(path: str) -> str:
    p = path.replace("\\", "/")
    if p.startswith("/mnt/") and len(p) > 6 and p[6] == "/":
        return f"{p[5].upper()}:/{p[7:]}"
    return path


def print_json(label: str, value) -> None:
    print(label, json.dumps(value, ensure_ascii=True, sort_keys=True))


def package_report() -> dict:
    required = ["unsloth", "transformers", "peft", "trl", "torch"]
    optional = ["llamafactory", "vllm"]
    out = {}
    for pkg in required + optional:
        try:
            out[pkg] = {"installed": True, "version": md.version(pkg), "required": pkg in required}
        except Exception as exc:
            out[pkg] = {"installed": False, "error": str(exc), "required": pkg in required}
    return out


def path_report(model_path: str) -> dict:
    current_path = Path(wsl_to_win(model_path) if sys.platform == "win32" else model_path)
    config_path = current_path / "config.json"
    report = {
        "configured": model_path,
        "current_os_path": str(current_path),
        "current_os_exists": current_path.exists(),
        "config_json_exists": config_path.exists(),
        "wsl_path": win_to_wsl(model_path),
        "wsl_exists": None,
    }

    wsl_exe = shutil.which("wsl.exe") or shutil.which("wsl")
    if wsl_exe:
        cmd = [
            wsl_exe,
            "-d",
            settings.echo_wsl_distro,
            "sh",
            "-lc",
            f"test -f '{win_to_wsl(model_path)}/config.json'",
        ]
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
            report["wsl_exists"] = completed.returncode == 0
            if completed.returncode != 0 and completed.stderr.strip():
                report["wsl_error"] = completed.stderr.strip()[:500]
        except Exception as exc:
            report["wsl_error"] = str(exc)
    return report


def torch_report() -> dict:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        arch_list = torch.cuda.get_arch_list() if cuda_available and hasattr(torch.cuda, "get_arch_list") else []
        capability = torch.cuda.get_device_capability(0) if cuda_available else None
        capability_tag = f"sm_{capability[0]}{capability[1]}" if capability else None
        return {
            "imported": True,
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "cuda_capability": capability_tag,
            "cuda_supported_arches": arch_list,
            "cuda_arch_supported": capability_tag in arch_list if capability_tag else None,
        }
    except Exception as exc:
        return {"imported": False, "error": repr(exc)}


def gemma_config_report(model_path: str) -> dict:
    try:
        from transformers import AutoConfig

        local_path = wsl_to_win(model_path) if sys.platform == "win32" else model_path
        cfg = AutoConfig.from_pretrained(local_path, trust_remote_code=True)
        return {
            "ok": True,
            "path_used": local_path,
            "config_class": type(cfg).__name__,
            "model_type": getattr(cfg, "model_type", None),
            "architectures": getattr(cfg, "architectures", None),
        }
    except Exception as exc:
        return {"ok": False, "path_used": model_path, "error": repr(exc)}


def wsl_training_report(model_path: str) -> dict:
    if get_training_runtime() != "windows_wsl":
        return {"checked": False, "reason": "runtime is not windows_wsl"}
    wsl_exe = shutil.which("wsl.exe") or shutil.which("wsl")
    if not wsl_exe:
        return {"checked": False, "ok": False, "error": "wsl.exe not found"}

    code = r"""
import importlib.metadata as md
import json
import sys

result = {"python": sys.executable, "packages": {}, "torch": {}, "gemma_config": {}}
for pkg in ["unsloth", "transformers", "peft", "trl", "torch", "vllm", "llamafactory"]:
    try:
        result["packages"][pkg] = {"installed": True, "version": md.version(pkg)}
    except Exception as exc:
        result["packages"][pkg] = {"installed": False, "error": str(exc)}

try:
    import torch
    cuda_available = bool(torch.cuda.is_available())
    arch_list = torch.cuda.get_arch_list() if cuda_available and hasattr(torch.cuda, "get_arch_list") else []
    capability = torch.cuda.get_device_capability(0) if cuda_available else None
    capability_tag = f"sm_{capability[0]}{capability[1]}" if capability else None
    result["torch"] = {
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_capability": capability_tag,
        "cuda_supported_arches": arch_list,
        "cuda_arch_supported": capability_tag in arch_list if capability_tag else None,
    }
except Exception as exc:
    result["torch"] = {"error": repr(exc)}

try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("__MODEL_PATH__", trust_remote_code=True)
    result["gemma_config"] = {
        "ok": True,
        "config_class": type(cfg).__name__,
        "model_type": getattr(cfg, "model_type", None),
        "architectures": getattr(cfg, "architectures", None),
    }
except Exception as exc:
    result["gemma_config"] = {"ok": False, "error": repr(exc)}

print(json.dumps(result, sort_keys=True))
""".replace("__MODEL_PATH__", win_to_wsl(model_path))

    cmd = [
        wsl_exe,
        "-d",
        settings.echo_wsl_distro,
        "sh",
        "-lc",
        (
            f"PYTHONPATH={shlex.quote(settings.echo_wsl_training_pythonpath)} "
            f"{shlex.quote(settings.echo_wsl_training_python)} -c {shlex.quote(code)}"
        ),
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if completed.returncode != 0:
            return {
                "checked": True,
                "ok": False,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-1000:],
                "stderr": completed.stderr[-1000:],
            }
        data = json.loads(completed.stdout.strip().splitlines()[-1])
        data["checked"] = True
        required = ["unsloth", "transformers", "peft", "trl", "torch"]
        data["ok"] = (
            all(data["packages"].get(pkg, {}).get("installed") for pkg in required)
            and bool(data.get("torch", {}).get("cuda_available"))
            and data.get("torch", {}).get("cuda_arch_supported") is not False
            and bool(data.get("gemma_config", {}).get("ok"))
        )
        return data
    except Exception as exc:
        return {"checked": True, "ok": False, "error": repr(exc)}


def main() -> int:
    model_path = (settings.gemma4_training_model_path or "").strip()
    packages = package_report()
    paths = path_report(model_path)
    config = gemma_config_report(model_path)
    torch = torch_report()
    wsl_training = wsl_training_report(model_path)
    runtime = {
        "mode": get_training_runtime(),
        "platform": sys.platform,
        "python": sys.executable,
        "model": settings.gemma4_base_model,
        "wsl_distro": settings.echo_wsl_distro,
        "wsl_training_python": settings.echo_wsl_training_python,
        "wsl_training_pythonpath": settings.echo_wsl_training_pythonpath,
    }

    print_json("training_runtime", runtime)
    print_json("packages", packages)
    print_json("model_path", paths)
    print_json("torch", torch)
    print_json("gemma_config", config)
    print_json("wsl_training", wsl_training)

    failures = []
    for pkg, data in packages.items():
        if data["required"] and not data["installed"]:
            failures.append(f"missing required package: {pkg}")
    if not paths["config_json_exists"]:
        failures.append("configured model path is missing config.json on current OS")
    if get_training_runtime() == "windows_wsl" and paths["wsl_exists"] is False:
        failures.append("configured model path is missing config.json inside WSL")
    if not config["ok"]:
        failures.append("transformers AutoConfig could not load configured Gemma path")
    if get_training_runtime() == "windows_wsl":
        if not wsl_training.get("ok"):
            failures.append("WSL training Python is not ready")
    elif not torch.get("cuda_available"):
        failures.append("torch CUDA is not available")
    elif torch.get("cuda_arch_supported") is False:
        failures.append("local torch build does not support this GPU CUDA capability")

    if failures:
        print_json("status", {"ok": False, "failures": failures})
        return 1
    print_json("status", {"ok": True, "failures": []})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
