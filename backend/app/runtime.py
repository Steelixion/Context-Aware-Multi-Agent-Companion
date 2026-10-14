"""Wires everything together. All coordination is asyncio; heavy work lives in other
processes (vision, LLM server) or executor threads (STT/TTS/embeddings/LanceDB)."""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import threading
from dataclasses import dataclass, field

from .agents.companion import CompanionAgent
from .agents.memory import MemoryAgent
from .agents.safety import SafetyAgent, WARNINGS
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
        self.session = Session(user_id=uid, record_id=rid, name=user["display_name"],
                               lang=user["preferred_lang"])

        self.embedder = await loop.run_in_executor(None, Embedder, cfg.embed_model)
        self.store = await loop.run_in_executor(None, MemoryStore, str(cfg.lance_path), self.embedder)
        self.llm = LLMClient(cfg.llm_url, cfg.llm_model)
        self.tts = TTS(cfg.piper_bin, cfg.piper_models)
        self.arbiter = AudioArbiter(self.bus, self.session, self.tts, cfg.enable_speaker, cfg.chat_mute_seconds)
        await self.arbiter.start()
        await self.arbiter.prerender(WARNINGS)

        self.companion = CompanionAgent(self.bus, self.session, self.llm, self.store, self.arbiter)
        self.safety = SafetyAgent(cfg, self.bus, self.session, self.arbiter, self.db, self.companion.cancel)
        self.memory = MemoryAgent(cfg, self.bus, self.session, self.llm, self.store)
        self.sync = SyncManager(cfg, self.bus, uid)
        for coro, name in [(self.safety.run(), "safety"), (self.companion.run(), "companion"),
                           (self.memory.run(), "memory"), (self.sync.run(), "sync")]:
            self.tasks.append(asyncio.create_task(coro, name=name))

        # vision runs in a separate process; two daemon threads bridge its queues to the bus
        ctx = mp.get_context("spawn")
        self.vq, self.pq, self.cq = ctx.Queue(maxsize=8), ctx.Queue(maxsize=2), ctx.Queue()
        self.vproc = ctx.Process(target=vision_worker.main,
                                 args=(self.vq, self.pq, self.cq, cfg.vision_dict()), daemon=True,
                                 name="vision")
        self.vproc.start()
        self._stop = threading.Event()
        for q, topic in ((self.vq, "telemetry"), (self.pq, "frame")):
            threading.Thread(target=self._bridge, args=(q, topic), daemon=True).start()

        self.listener = None
        if cfg.enable_mic:
            self.listener = SpeechListener(self.bus, loop, self.arbiter, cfg.whisper_model,
                                           EmotionClassifier(cfg.ser_model))
            self.listener.start()

    def _bridge(self, q, topic: str) -> None:
        import queue
        while not self._stop.is_set():
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                continue
            self.loop.call_soon_threadsafe(self.bus.publish, topic, item)

    def set_sim(self, mode: str) -> None:
        self.cq.put({"type": "sim", "mode": mode})

    async def stop(self) -> None:
        self._stop.set()
        if self.listener:
            self.listener.stop()
        try:
            await self.memory.flush()
        except Exception:  # noqa: BLE001
            pass
        for t in self.tasks:
            t.cancel()
        await self.arbiter.stop()
        if self.vproc.is_alive():
            self.vproc.terminate()
        await self.db.end_record(self.session.record_id)
        await self.llm.close()
        await self.db.close()
