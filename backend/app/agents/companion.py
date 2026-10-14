"""Companion Agent: LangGraph pipeline  retrieve_memory -> build_prompt -> generate+speak.
Every await is cancellable, so the Safety Agent can kill a reply mid-token."""
from __future__ import annotations

import asyncio
import re
import time
import uuid
from typing import TypedDict

from ..audio.arbiter import AudioArbiter
from ..bus import Bus
from ..db.vector_store import MemoryStore
from ..llm.client import LLMClient, LLMUnavailable
from ..state import Session
from .tone_policy import build_system_prompt

_SENT_END = re.compile(r"(?<=[.!?。؟۔])\s+")
OFFLINE_REPLY = "My language model isn't running right now, but I'm here and I'm watching the road with you."


class GState(TypedDict, total=False):
    text: str
    lang: str
    emotion: str
    memories: list
    messages: list
    reply: str
    msg_id: str


class CompanionAgent:
    def __init__(self, bus: Bus, session: Session, llm: LLMClient, store: MemoryStore,
                 arbiter: AudioArbiter):
        self.bus, self.s, self.llm, self.store, self.arbiter = bus, session, llm, store, arbiter
        self.task: asyncio.Task | None = None
        self.graph = self._build_graph()

    # ---- lifecycle ----
    async def run(self) -> None:
        q = self.bus.subscribe("speech")
        while True:
            msg = await q.get()
            self.cancel()  # barge-in: new utterance replaces any running reply
            self.task = asyncio.create_task(self._turn(msg), name="companion-turn")

    def cancel(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
        self.arbiter.flush_chat()

    # ---- turn ----
    async def _turn(self, msg: dict) -> None:
        s = self.s
        s.lang = msg.get("lang") or s.lang
        s.emotion = msg.get("emotion") or s.emotion
        self._emit("user", msg["text"], str(uuid.uuid4()), True, emotion=s.emotion)
        mid = str(uuid.uuid4())
        reply = ""
        try:
            out = await self.graph.ainvoke(dict(text=msg["text"], lang=s.lang, emotion=s.emotion, msg_id=mid))
            reply = out.get("reply", "")
        except asyncio.CancelledError:
            self._emit("assistant", "", mid, True, interrupted=True)
            raise
        s.history += [{"role": "user", "content": msg["text"]},
                      {"role": "assistant", "content": reply}]
        s.history = s.history[-12:]
        s.turn_queue.append(dict(user=msg["text"], assistant=reply, ts=time.time(), lang=s.lang))

    def _emit(self, role, text, mid, final, **extra) -> None:
        m = dict(id=mid, role=role, text=text, final=final, ts=time.time(), **extra)
        if final:
            self.s.chat_log = (self.s.chat_log + [m])[-60:]
        self.bus.publish("chat", m)

    # ---- graph nodes ----
    async def _retrieve(self, st: GState) -> GState:
        try:
            mem = await self.store.search(self.s.user_id, st["text"], k=4)
        except Exception as e:  # noqa: BLE001
            print(f"[companion] memory search failed: {e}")
            mem = []
        return dict(memories=mem)

    async def _prompt(self, st: GState) -> GState:
        sys = build_system_prompt(self.s.name, st["lang"], self.s.level, st["emotion"],
                                  self.s.fatigue, st["memories"])
        msgs = [{"role": "system", "content": sys}, *self.s.history[-8:],
                {"role": "user", "content": st["text"]}]
        return dict(messages=msgs)

    async def _generate(self, st: GState) -> GState:
        mid, buf, full = st["msg_id"], "", ""
        try:
            async for tok in self.llm.stream(st["messages"]):
                buf += tok
                full += tok
                self._emit("assistant", full, mid, False)
                parts = _SENT_END.split(buf)
                for sent in parts[:-1]:
                    await self.arbiter.say(sent.strip(), st["lang"])
                buf = parts[-1]
            if buf.strip():
                await self.arbiter.say(buf.strip(), st["lang"])
        except LLMUnavailable:
            full = OFFLINE_REPLY
            await self.arbiter.say(full, "en")
        self._emit("assistant", full, mid, True)
        return dict(reply=full)

    def _build_graph(self):
        from langgraph.graph import StateGraph, END
        g = StateGraph(GState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("prompt", self._prompt)
        g.add_node("generate", self._generate)
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "prompt")
        g.add_edge("prompt", "generate")
        g.add_edge("generate", END)
        return g.compile()
