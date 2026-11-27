"""Runs ONE case ONE time against a fresh app process and records every observable event.

Order: LLM proxy + loopback + virtual camera (calibration segment first) -> app process with
an empty data dir -> wait HTTP, first telemetry, STT listening, calibration -> switch the camera
to the test clip (clip t = 0) -> run the timeline on the wall clock -> wait the clip out plus
the settle time -> stop everything -> read the SQLite DB read-only -> write raw files.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf

from .appproc import AppProcess, free_port
from .audiofeed import AudioFeed
from .events import Clock, EventLog
from .llmproxy import LLMProxy
from .loopback import Loopback
from .noise import noise_rms_for_snr
from .videofeed import Segment, VideoFeed
from .wsrec import WsRecorder

LEVEL_RANK = {"ALERT": 0, "MILD": 1, "HIGH": 2}


class CaseError(Exception):
    pass


def read_db_readonly(db_path: Path) -> dict:
    if not db_path.is_file():
        return {"error": "no database file"}
    uri = f"file:{db_path.as_posix()}?mode=ro"
    out = {}
    with sqlite3.connect(uri, uri=True) as c:
        c.row_factory = sqlite3.Row
        for t in ("monitoring_records", "monitoring_logs", "calibration_profiles"):
            out[t] = [dict(r) for r in c.execute(f"SELECT * FROM {t}")]
    return out


def warm_llm(upstream: str, model: str) -> bool:
    import httpx
    try:
        r = httpx.post(upstream.rstrip("/") + "/chat/completions", timeout=180,
                       json={"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]})
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


class Executor:
    def __init__(self, repo: Path, app_cfg: dict, thresholds: dict, golden, camera_index: int | None):
        self.repo, self.app_cfg, self.th, self.golden = repo, app_cfg, thresholds, golden
        self.camera_index = camera_index

    # ------------------------------------------------------------------ helpers
    def _env(self, data_dir: Path, llm_url: str) -> dict:
        a = self.app_cfg
        env = dict(a.get("env", {}))
        env |= {k: v for k, v in (a.get("models") or {}).items() if v not in (None, "")}
        env |= {"DC_DATA_DIR": str(data_dir), "DC_LLM_URL": llm_url,
                "DC_ENABLE_MIC": "1", "DC_MIC_SOURCE": "stream"}
        if a["video"]["source"] == "virtualcam":
            env["DC_CAMERA_INDEX"] = str(self.camera_index)
        return env

    def _speak(self, feed: AudioFeed, ws: WsRecorder, log: EventLog, ev: dict, clip_t0: float) -> None:
        audio_id = ev["audio_id"]
        x = self.golden.load_audio(audio_id)
        gate = self.th["run"].get("speech_gate_max_wait_s", 20.0)
        waited = 0.0
        quiet = float(self.th["run"].get("speech_gate_quiet_s", 1.5))
        if not ev.get("overlap", False):        # natural turn-taking: don't talk over the companion
            t = time.monotonic()
            ok = ws.wait_quiet(quiet, gate)     # idle (no audio, no reply pending) for `quiet` s
            waited = time.monotonic() - t
            if not ok:
                log.add("harness", "speech_gate_timeout", audio_id=audio_id, waited_s=round(waited, 2))
        log.add("harness", "speak_request", audio_id=audio_id, event_id=ev.get("id"),
                clip_time=round(time.monotonic() - clip_t0, 3), gated_s=round(waited, 3),
                overlap=bool(ev.get("overlap", False)))
        feed.say(audio_id, x, event_id=ev.get("id"))

    def _timeline(self, case: dict, feed: AudioFeed, ws: WsRecorder, app: AppProcess,
                  video: VideoFeed, log: EventLog, clip_t0: float, stop: threading.Event) -> None:
        events = sorted(case.get("timeline", []), key=lambda e: e.get("at_s", 1e9))
        timed = [e for e in events if "at_s" in e]
        triggered = [e for e in events if "trigger" in e]
        for ev in triggered:
            threading.Thread(target=self._run_trigger, args=(ev, case, feed, ws, app, video, log, clip_t0, stop),
                             daemon=True).start()
        for ev in timed:
            due = clip_t0 + float(ev["at_s"])
            while not stop.is_set() and time.monotonic() < due:
                time.sleep(min(0.02, max(0.0, due - time.monotonic())))
            if stop.is_set():
                return
            self._do(ev, feed, ws, app, video, log, clip_t0)

    def _run_trigger(self, ev, case, feed, ws, app, video, log, clip_t0, stop) -> None:
        tr = ev["trigger"]
        not_before = clip_t0 + float(tr.get("not_before_s", 0.0))
        deadline = clip_t0 + float(tr.get("timeout_s", 1e9))
        while not stop.is_set() and time.monotonic() < not_before:
            time.sleep(0.05)
        if tr.get("after_event"):            # arm only after an earlier utterance has been spoken
            want = tr["after_event"]
            while not stop.is_set() and time.monotonic() < deadline and not any(
                    e["type"] == "speech_end" and e["data"].get("event_id") == want for e in log.select("harness")):
                time.sleep(0.05)
        if "level_at_least" in tr:
            need = LEVEL_RANK[tr["level_at_least"]]
            pred = lambda w: LEVEL_RANK.get(w.level or "ALERT", 0) >= need  # noqa: E731
        elif "companion_speaking" in tr:
            pred = lambda w: w.speaking and (w.speaking_priority or 0) > 0  # noqa: E731
        else:
            log.add("harness", "trigger_invalid", event_id=ev.get("id"), trigger=tr)
            return
        ok = False
        while not stop.is_set() and time.monotonic() < deadline:
            if ws.wait_for(pred, 0.5):
                ok = True
                break
        if not ok:
            log.add("harness", "trigger_timeout", event_id=ev.get("id"), trigger=tr)
            return
        fired = time.monotonic()
        log.add("harness", "trigger_fired", event_id=ev.get("id"), trigger=tr,
                clip_time=round(fired - clip_t0, 3))
        time.sleep(float(tr.get("after_s", 0.0)))
        if not stop.is_set():
            self._do(ev, feed, ws, app, video, log, clip_t0)

    def _do(self, ev, feed, ws, app, video, log, clip_t0) -> None:
        act = ev["action"]
        if act == "speak":
            self._speak(feed, ws, log, ev, clip_t0)
        elif act == "stop_video":
            video.stop_camera()
            log.add("harness", "stop_video", event_id=ev.get("id"))
        elif act == "kill_vision":
            pid = app.kill_vision()
            log.add("harness", "kill_vision", event_id=ev.get("id"), pid=pid)
        elif act == "blank_video":
            pass                                    # applied frame-accurately by the video feed
        else:
            log.add("harness", "action_unknown", event_id=ev.get("id"), action=act)

    # ------------------------------------------------------------------ main
    def run(self, case: dict, rep: int, run_dir: Path) -> dict:
        cid = case["case_id"]
        tag = f"{cid}__r{rep}"
        raw = run_dir / "raw"
        work = run_dir / "work" / tag
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
        (work / "data").mkdir(parents=True, exist_ok=True)
        clock = Clock()
        log = EventLog(raw / f"{tag}.jsonl", clock)
        log.add("harness", "case_start", case_id=cid, rep=rep, scenario=case["scenario"])
        result = {"case_id": cid, "rep": rep, "status": "RUN", "error": None, "raw": str(raw / f"{tag}.jsonl")}
        proxy = loop = video = app = ws = feed = None
        stop = threading.Event()
        try:
            clip = self.golden.clip(case["clip"]["clip_id"])
            # ---- LLM warm-up (directly to the upstream server, outside the timed window): a
            # local server may have unloaded the model between cases, which would add a cold
            # start that is a property of the server's idle policy, not of the app
            upstream = self.app_cfg["llm"]["upstream"]
            t = time.monotonic()
            warm = warm_llm(upstream, (self.app_cfg.get("models") or {}).get("DC_LLM_MODEL") or "local")
            log.add("harness", "llm_warmup", ok=warm, seconds=round(time.monotonic() - t, 2))
            if self.app_cfg["llm"].get("proxy", True):
                proxy = LLMProxy(upstream, free_port(18080), log)
                if not proxy.start():
                    raise CaseError(f"LLM proxy failed: {proxy.error}")
                llm_url = proxy.url
            else:
                llm_url = upstream
            # ---- loopback
            if self.th["loopback"].get("enabled", True):
                loop = Loopback(log, raw / f"{tag}.loopback.wav")
                if not loop.start():
                    log.add("harness", "loopback_unavailable", error=loop.error)
            # ---- video
            if self.app_cfg["video"]["source"] != "virtualcam":
                raise CaseError("video.source=file is not implemented in the runner yet; use virtualcam")
            cal = case.get("calibration") or {"method": "none"}
            c0, c1 = case["clip"].get("start_s", 0.0), case["clip"].get("end_s")
            blanks = [(float(e["at_s"]), float(e["at_s"]) + float(e["duration_s"]))
                      for e in case.get("timeline", []) if e.get("action") == "blank_video"]
            clip_seg = Segment(clip["abs_path"], c0, c1, loop=False, phase="clip", blanks=blanks)
            if cal["method"] == "prepend_neutral":
                cal_clip = self.golden.clip(cal["clip_id"])
                first = Segment(cal_clip["abs_path"], cal["start_s"], cal["end_s"], loop=True, phase="calibration")
            else:
                first = Segment(clip["abs_path"], c0, c0 + 0.5, loop=True, phase="prestart")
            video = VideoFeed(log, clip["width"], clip["height"], clip["fps"], self.app_cfg["video"].get("backend", "obs"))
            if not video.start(first):
                raise CaseError(video.error or "virtual camera failed")
            # ---- app
            port = free_port(int(self.app_cfg["app"]["base_port"]))
            app = AppProcess(self.repo, self.app_cfg["app"], self._env(work / "data", llm_url), port,
                             raw / f"{tag}.app.log")
            log.add("harness", "app_launch", port=port, env={k: v for k, v in app.env.items() if k.startswith("DC_")})
            app.start()
            session = app.wait_http(float(self.app_cfg["app"]["startup_timeout_s"]))
            log.add("harness", "app_http_ready", session=session)
            ws = WsRecorder(f"ws://127.0.0.1:{port}/ws", log)
            ws.start()
            if not ws.connected.wait(15):
                raise CaseError(f"WebSocket did not connect ({ws.error})")
            to = float(self.app_cfg["app"]["startup_timeout_s"])
            if not ws.wait_for(lambda w: w.telemetry_count > 0, to):
                raise CaseError("no telemetry from the vision process")
            log.add("harness", "vision_ready")
            if not ws.wait_for(lambda w: w.stt_status in ("listening", "no model", "no mic"), to):
                raise CaseError("STT never became ready")
            if ws.stt_status != "listening":
                raise CaseError(f"STT not usable: {ws.stt_status}")
            session = app.get("/api/session")              # engines as reported once everything is up
            log.add("harness", "app_ready", session=session)
            feed = AudioFeed(f"ws://127.0.0.1:{port}/ws/audio", log, int(self.app_cfg["audio"].get("chunk_ms", 50)))
            nb = (case.get("audio") or {}).get("noise_bed")
            if nb:
                ref = self.golden.load_audio(nb["ref_audio_id"])
                feed.set_noise_bed(nb["type"], noise_rms_for_snr(ref, float(nb["snr_db"])),
                                   int(self.th["run"]["seed"]) + rep)
            if (case.get("audio") or {}).get("echo") and loop is not None and loop.available:
                loop.set_echo_delay(float(self.th["echo"]["delay_ms"]))
                feed.set_echo(loop.echo_read, float(self.th["echo"]["gain_db"]))
                log.add("harness", "echo_enabled", gain_db=self.th["echo"]["gain_db"], delay_ms=self.th["echo"]["delay_ms"])
            feed.start()
            if not feed.connected.wait(10):
                raise CaseError(f"audio stream did not connect ({feed.error})")
            # ---- calibration
            if cal["method"] == "prepend_neutral":
                ok = ws.wait_for(lambda w: w.calibration_state in ("calibrated", "failed"), 90)
                if not ok or ws.calibration_state != "calibrated":
                    raise CaseError(f"calibration did not succeed (state={ws.calibration_state})")
                log.add("harness", "calibration_done", method="prepend_neutral")
            else:
                ws.wait_for(lambda w: w.calibration_state is not None, 5)
            video.queue_next(clip_seg)
            if not video.clip_started.wait(10):
                raise CaseError("clip did not start")
            clip_t0 = video.clip_t0
            tl = threading.Thread(target=self._timeline, args=(case, feed, ws, app, video, log, clip_t0, stop),
                                  daemon=True)
            tl.start()
            dur = (c1 if c1 is not None else clip["duration_s"]) - c0
            limit = clip_t0 + dur + 5
            while time.monotonic() < limit and not video.clip_finished.is_set():
                if not app.alive():
                    log.add("harness", "app_exited", code=app.proc.returncode)
                    break
                time.sleep(0.2)
            settle_end = time.monotonic() + float(self.th["run"]["settle_after_clip_s"])
            while time.monotonic() < settle_end or not feed.idle.is_set():
                if time.monotonic() > settle_end + 30:
                    break
                time.sleep(0.2)
            result["status"] = "DONE"
            result["clip_t0"] = clip_t0
            result["clip_duration_s"] = dur
            result["session"] = session
        except Exception as e:  # noqa: BLE001
            result["status"] = "ERROR"
            result["error"] = f"{type(e).__name__}: {e}"
            log.add("harness", "case_error", error=result["error"], tb=traceback.format_exc()[-3000:])
        finally:
            stop.set()
            for comp in (feed, ws):
                try:
                    comp and comp.stop()
                except Exception:  # noqa: BLE001
                    pass
            if app is not None:
                result["stop_method"] = app.stop(float(self.app_cfg["app"]["shutdown_timeout_s"]))
                full = app.log_text()
                result["app_log_tail"] = full[-4000:]
                result["app_model_lines"] = [ln for ln in full.splitlines()
                                             if ln.startswith(("[stt] loaded", "[vision] running", "[embed]", "[tts]"))][:20]
            for comp in (video, proxy):
                try:
                    comp and comp.stop()
                except Exception:  # noqa: BLE001
                    pass
            if loop is not None:
                loop.stop()
                (raw / f"{tag}.loopback.json").write_text(json.dumps(loop.summary()))
            if video is not None:
                (raw / f"{tag}.frames.json").write_text(json.dumps(
                    {"sent": video.sent, "skipped": video.skipped, "frames": video.frame_times[-20000:]}))
            try:
                result["db"] = read_db_readonly(work / "data" / "driver.db")
            except Exception as e:  # noqa: BLE001
                result["db"] = {"error": repr(e)}
            log.add("harness", "case_end", status=result["status"])
            log.close()
            (raw / f"{tag}.result.json").write_text(json.dumps(result, default=str, indent=1))
        return result
