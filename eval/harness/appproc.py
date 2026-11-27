"""Launch the app as a black-box process: env vars in, HTTP/WS/logs out."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil


def free_port(start: int) -> int:
    for p in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("no free port")


class AppProcess:
    def __init__(self, repo: Path, app_cfg: dict, env: dict[str, str], port: int, log_path: Path):
        self.repo, self.cfg, self.port, self.log_path = repo, app_cfg, port, log_path
        base = {k: v for k, v in os.environ.items() if not k.startswith("DC_")}   # never inherit user DC_* settings
        self.env = base | {k: str(v) for k, v in env.items() if v not in (None, "")} | {
            "DC_PORT": str(port), "DC_HOST": "127.0.0.1",
            "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        self.proc: subprocess.Popen | None = None
        self._log = None
        self.started_at = 0.0
        self.stop_method = ""

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "wb")
        py = str((self.repo / self.cfg["python"]).resolve())
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        self.proc = subprocess.Popen([py, "-m", self.cfg["module"]], cwd=str(self.repo / self.cfg["cwd"]),
                                     env=self.env, stdout=self._log, stderr=subprocess.STDOUT,
                                     creationflags=flags)
        self.started_at = time.monotonic()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def get(self, path: str, timeout: float = 3.0) -> dict:
        with urllib.request.urlopen(self.url + path, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def wait_http(self, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            if not self.alive():
                raise RuntimeError(f"app exited during startup (code {self.proc.returncode}); see {self.log_path}")
            try:
                return self.get("/api/session", 2.0)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(0.3)
        raise TimeoutError(f"app HTTP not ready after {timeout:.0f}s ({last})")

    def vision_process(self) -> psutil.Process | None:
        """The vision worker: the app's multiprocessing (spawn) child."""
        if not self.alive():
            return None
        try:
            for c in psutil.Process(self.proc.pid).children(recursive=True):
                cmd = " ".join(c.cmdline())
                if "multiprocessing" in cmd and "resource_tracker" not in cmd:
                    return c
        except psutil.Error:
            return None
        return None

    def kill_vision(self) -> int | None:
        vp = self.vision_process()
        if vp is None:
            return None
        pid = vp.pid
        vp.kill()
        return pid

    def stop(self, timeout: float) -> str:
        """Graceful stop (CTRL_BREAK -> uvicorn shutdown -> app lifespan), else kill the tree."""
        if self.proc is None:
            return "never started"
        if self.alive():
            try:
                if sys.platform == "win32":
                    self.proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self.proc.send_signal(signal.SIGINT)
                self.proc.wait(timeout)
                self.stop_method = "graceful"
            except (subprocess.TimeoutExpired, OSError):
                self.stop_method = "killed"
        else:
            self.stop_method = f"already exited ({self.proc.returncode})"
        try:
            for c in psutil.Process(self.proc.pid).children(recursive=True):
                c.kill()
        except psutil.Error:
            pass
        if self.alive():
            self.proc.kill()
            self.proc.wait(5)
        if self._log:
            self._log.close()
        return self.stop_method

    def log_text(self) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
