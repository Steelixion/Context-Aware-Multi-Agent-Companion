from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import Config, ROOT
from ..runtime import Runtime

DIST = ROOT / "dashboard" / "dist"


class Profile(BaseModel):
    display_name: str
    lang: str = "en"


class SimMode(BaseModel):
    mode: str


class ChatIn(BaseModel):
    text: str
    lang: str | None = None
    emotion: str | None = None


class MemoryIn(BaseModel):
    text: str
    category: str = "other"
    topic: str = ""


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or Config()
    rt = Runtime(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await rt.start()
        yield
        await rt.stop()

    app = FastAPI(title="Driver Companion", lifespan=lifespan)

    @app.get("/api/session")
    async def session():
        s = rt.session
        return dict(user_id=s.user_id, name=s.name, lang=s.lang, record_id=s.record_id,
                    vision_mode=cfg.vision_mode, mic=cfg.enable_mic, tts=rt.tts.engine_name,
                    embedder=rt.embedder.name, llm_online=await rt.llm.available(),
                    ear_threshold=cfg.ear_threshold, ear_seconds=cfg.ear_seconds)

    @app.post("/api/profile")
    async def profile(p: Profile):
        await rt.db.update_user(rt.session.user_id, p.display_name, p.lang)
        rt.session.name, rt.session.lang = p.display_name, p.lang
        rt.sync.trigger()
        return {"ok": True}

    @app.get("/api/history")
    async def history():
        uid = rt.session.user_id
        return dict(records=await rt.db.list_records(uid), logs=await rt.db.recent_logs(uid, 60))

    @app.get("/api/memories")
    async def memories():
        return await rt.store.list(rt.session.user_id)

    @app.post("/api/memories")
    async def add_memory(m: MemoryIn):
        added = await rt.store.add(rt.session.user_id, m.text, m.category, m.topic)
        return {"added": added}

    @app.delete("/api/memories/{memory_id}")
    async def del_memory(memory_id: str):
        await rt.store.delete(rt.session.user_id, memory_id)
        return {"ok": True}

    @app.get("/api/sync")
    async def sync_status():
        return rt.sync.status

    @app.post("/api/sync/now")
    async def sync_now():
        rt.sync.trigger()
        return {"ok": True}

    @app.post("/api/sim")
    async def sim(m: SimMode):
        if cfg.vision_mode != "sim":
            raise HTTPException(400, "simulator only available when DC_VISION_MODE=sim")
        rt.set_sim(m.mode)
        return {"ok": True}

    @app.post("/api/chat")
    async def chat(c: ChatIn):
        rt.bus.publish("speech", dict(text=c.text, lang=c.lang or rt.session.lang,
                                      emotion=c.emotion or rt.session.emotion, source="text"))
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        q = rt.bus.subscribe("*", maxsize=200)
        s = rt.session
        await sock.send_text(json.dumps(dict(type="snapshot", chat=s.chat_log, sync=rt.sync.status,
                                             name=s.name, lang=s.lang)))

        async def sender():
            last_tel = 0.0
            while True:
                topic, msg = await q.get()
                if topic == "telemetry":
                    if msg["ts"] - last_tel < 0.09:
                        continue
                    last_tel = msg["ts"]
                await sock.send_text(json.dumps(dict(type=topic, data=msg)))

        send_task = asyncio.create_task(sender())
        try:
            while True:
                data = json.loads(await sock.receive_text())
                if data.get("type") == "chat" and data.get("text", "").strip():
                    rt.bus.publish("speech", dict(text=data["text"].strip(), lang=data.get("lang") or s.lang,
                                                  emotion=data.get("emotion") or s.emotion, source="text"))
                elif data.get("type") == "sim" and cfg.vision_mode == "sim":
                    rt.set_sim(data["mode"])
        except WebSocketDisconnect:
            pass
        finally:
            send_task.cancel()
            rt.bus.unsubscribe("*", q)

    if DIST.exists():
        app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

        @app.get("/{path:path}")
        async def spa(path: str):
            f = DIST / path
            return FileResponse(f if f.is_file() else DIST / "index.html")

    return app
