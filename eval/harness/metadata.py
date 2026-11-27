"""Run metadata captured automatically (read-only git calls, host info, model settings)."""
from __future__ import annotations

import hashlib
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import psutil


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def cpu_name() -> str:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return out or platform.processor()
    except Exception:  # noqa: BLE001
        return platform.processor()


def file_sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def collect(repo: Path, th_path: Path, app_cfg_path: Path, app_cfg: dict, golden, args, run_id: str) -> dict:
    status = _git(repo, "status", "--porcelain")
    app_dirty = [l for l in status.splitlines() if l[3:].startswith(("backend/", "dashboard/", "scripts/"))]
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": "dev (smoke)" if args.smoke else args.split, "repeats": args.repeats_effective, "smoke": bool(args.smoke),
        "only": args.only or "", "seed": args.seed_effective,
        "host": {"os": platform.platform(), "python": platform.python_version(), "cpu": cpu_name(),
                 "cores_logical": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2 ** 30, 1)},
        "app_git": {"commit": _git(repo, "rev-parse", "HEAD"), "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
                    "dirty": bool(app_dirty), "dirty_files": app_dirty[:50]},
        "golden": {"version": golden.version.get("version"), "checksum": golden.version.get("checksum"),
                   "seed": golden.version.get("seed"), "built": golden.version.get("built_utc")},
        "thresholds_sha256": file_sha256(th_path), "app_config_sha256": file_sha256(app_cfg_path),
        "models": {}, "llm": {"upstream": app_cfg["llm"]["upstream"], "family": app_cfg["llm"].get("family"),
                              "proxy": app_cfg["llm"].get("proxy", True)},
        "env": app_cfg.get("env", {}), "video": app_cfg.get("video", {}),
    }


def models_from_run(app_cfg: dict, session: dict | None, app_log: str, llm_models: list | None) -> dict:
    """What the app actually used, from config + /api/session + its stdout lines."""
    m = {k: (v or "(app default)") for k, v in (app_cfg.get("models") or {}).items()}
    if session:
        m["session.tts"] = session.get("tts")
        m["session.stt"] = session.get("stt")
        m["session.embedder"] = session.get("embedder")
    g = re.search(r"\[stt\] loaded (\S+) \+ (\S+) VAD", app_log or "")
    if g:
        m["log.asr_model"], m["log.vad"] = g.group(1), g.group(2)
    g = re.search(r"temporal model=([^,]+)", app_log or "")
    if g:
        m["log.temporal_model"] = g.group(1).strip()
    if llm_models:
        m["llm_server.models"] = ",".join(llm_models)
    return m
