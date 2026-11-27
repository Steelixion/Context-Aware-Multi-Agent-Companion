"""Wires everything together. All coordination is asyncio; heavy work lives in other
processes (vision, LLM server) or executor threads (STT/TTS/embeddings/LanceDB)."""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass, field

from .agents.companion import CompanionAgent
from .agents.memory import MemoryAgent
from .agents.safety import SafetyAgent, all_alert_lines
from .audio.arbiter import AudioArbiter
from .audio.emotion import EmotionClassifier
from .audio.stt import SpeechListener
from .audio.tts import TTS
from .bus import Bus
from .config import Config
from .db.relational import Database
from .db.vector_store import MemoryStore
from .llm.client import LLMClient
from .memory.embeddings import Embedder
from .state import Session
from .sync.manager import SyncManager
from .vision import worker as vision_worker


@dataclass
class Runtime:
    cfg: Config
    bus: Bus = field(default_factory=Bus)
    tasks: list = field(default_factory=list)

    async def start(self) -> None:
        cfg = self.cfg
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_running_loop()
        self.loop = loop

        self.db = Database(cfg.sqlite_path)
        await self.db.open()
        user = await self.db.first_user()
        uid = user["user_id"] if user else await self.db.create_user("Driver")
        user = await self.db.get_user(uid)
        rid = await self.db.start_record(uid)
        self.session = Session(user_id=uid, record_id=rid, name=user["display_name"], lang="en")
        self.profile = await self.db.latest_profile(uid)   # driver's last calibration, if any
        calibrated = bool(self.profile and self.profile.get("calibrated"))
        self.calibration: dict = dict(state="calibrated" if calibrated else "calibrating", frames=0,
                                      profile=self.profile, stored=self.profile is not None)

        self.embedder = await loop.run_in_executor(None, Embedder, cfg.embed_model)
        self.store = await loop.run_in_executor(None, MemoryStore, str(cfg.lance_path), self.embedder)
        self.llm = LLMClient(cfg.llm_url, cfg.llm_model)
        self.llm_online = False
        self.tts = TTS(cfg.piper_bin, cfg.piper_models)
        self.arbiter = AudioArbiter(self.bus, self.session, self.tts, cfg.enable_speaker, cfg.chat_mute_seconds)
        await self.arbiter.start()
        await self.arbiter.prerender(all_alert_lines())

        self.companion = CompanionAgent(self.bus, self.session, self.llm, self.store, self.arbiter)
        self.safety = SafetyAgent(cfg, self.bus, self.session, self.arbiter, self.db,
                                  self.companion.cancel, nudge=self.nudge_companion)
        self.memory = MemoryAgent(cfg, self.bus, self.session, self.llm, self.store)
        self.sync = SyncManager(cfg, self.bus, uid)
        for coro, name in [(self.safety.run(), "safety"), (self.companion.run(), "companion"),
                           (self.memory.run(), "memory"), (self.sync.run(), "sync"),
                           (self._vision_events(), "vision-events"), (self._llm_watch(), "llm-watch")]:
            self.tasks.append(asyncio.create_task(coro, name=name))

        # vision runs in a separate process; daemon threads bridge its queues to the bus
        ctx = mp.get_context("spawn")
        self.vq, self.pq, self.rq, self.cq, self.eq = (ctx.Queue(maxsize=8), ctx.Queue(maxsize=2),
                                                      ctx.Queue(maxsize=2), ctx.Queue(), ctx.Queue())
        vcfg = cfg.vision_dict() | dict(profile=self.profile)
        self.vproc = ctx.Process(target=vision_worker.main,
                                 args=(self.vq, self.pq, self.cq, self.eq, vcfg, self.rq),
                                 daemon=True, name="vision")
        self.vproc.start()
        self._stop = threading.Event()
        for q, topic in ((self.vq, "telemetry"), (self.pq, "frame"), (self.rq, "frame_raw"),
                         (self.eq, "vision_event")):
            threading.Thread(target=self._bridge, args=(q, topic), daemon=True).start()

        self.listener = None
        if cfg.enable_mic:
            self.listener = SpeechListener(self.bus, loop, self.arbiter, self.session,
                                           cfg.asr_model_dir, EmotionClassifier(cfg.ser_model),
                                           cfg.asr_threads, cfg.vad_model, cfg.vad_threshold,
                                           cfg.mic_source)
            self.listener.start()

    def _bridge(self, q, topic: str) -> None:
        while not self._stop.is_set():
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                continue
            except (EOFError, OSError):   # vision process gone; the safety watchdog reports it
                return
            try:
                self.loop.call_soon_threadsafe(self.bus.publish, topic, item)
            except RuntimeError:          # event loop closed during shutdown
                return

    async def _vision_events(self) -> None:
        """Calibration results from the vision process: store the profile, tell the dashboard."""
        q = self.bus.subscribe("vision_event")
        while True:
            ev = await q.get()
            if ev.get("type") != "calibration":
                continue
            if ev.get("ok"):
                try:
                    await self.db.save_profile(self.session.user_id, ev["profile"])
                    self.profile = ev["profile"]
                    self.sync.trigger()
                except Exception as e:  # noqa: BLE001
                    print(f"[runtime] could not save calibration: {e}", flush=True)
            self.calibration = dict(state="calibrated" if ev.get("ok") else "failed",
                                    frames=ev.get("frames", 0), profile=self.profile,
                                    stored=self.profile is not None, ts=time.time())
            self.bus.publish("calibration", self.calibration)

    async def _llm_watch(self, every: float = 10.0) -> None:
        """Keep the LLM server's availability current (dashboard chip) without blocking requests."""
        while True:
            online = await self.llm.available()
            if online != self.llm_online:
                self.llm_online = online
                self.bus.publish("llm", dict(online=online))
            await asyncio.sleep(every)

    def set_sim(self, mode: str) -> None:
        self.cq.put({"type": "sim", "mode": mode})

    def nudge_companion(self, prompt: str) -> None:
        self.bus.publish("speech", dict(text=prompt, lang="en", emotion="neutral", source="safety"))

    def calibrate(self) -> None:
        self.calibration = dict(self.calibration, state="calibrating")
        self.cq.put({"type": "calibrate"})
        self.bus.publish("calibration", self.calibration)

    async def stop(self) -> None:
        self._stop.set()
        if self.listener:
            self.listener.stop()
        try:
            await self.memory.flush()
        except Exception:  # noqa: BLE001
            pass
        # cancel an in-flight reply and every agent, and wait for them, before closing the
        # LLM client and the database they use
        self.companion.cancel()
        running = [t for t in (*self.tasks, self.companion.task) if t is not None]
        for t in running:
            t.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        await self.arbiter.stop()
        if self.vproc.is_alive():
            self.vproc.terminate()
            await asyncio.get_running_loop().run_in_executor(None, self.vproc.join, 5)
        await self.db.end_record(self.session.record_id)
        await self.llm.close()
        await self.db.close()
