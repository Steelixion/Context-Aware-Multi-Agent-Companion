"""Memory Agent: background extraction of durable facts from the session turn queue."""
from __future__ import annotations

import asyncio
import json
import re
import time

from ..bus import Bus
from ..config import Config
from ..db.vector_store import CATEGORIES, MemoryStore
from ..llm.client import LLMClient, LLMUnavailable
from ..state import Session

EXTRACT_PROMPT = (
    "Extract durable facts and preferences about the USER from this conversation. "
    "Ignore small talk and anything temporary. Respond with ONLY a JSON array of objects "
    '{"text": "<third-person fact, e.g. The driver likes jazz>", "category": '
    f'"{"|".join(CATEGORIES)}", "topic": "<1-3 words>"}}. Return [] if there is nothing.'
)
_RULES = [  # offline fallback when the LLM is down
    (re.compile(r"\b(?i:my name is) (\w{2,20}(?: [A-Z]\w{1,20})?)"), "personal_fact", "name", "The driver's name is {}"),
    (re.compile(r"\bi (?:really )?(?:like|love|enjoy) ([\w ,']{3,60})", re.I), "preference", "likes", "The driver likes {}"),
    (re.compile(r"\bi (?:prefer) ([\w ,']{3,60})", re.I), "preference", "preference", "The driver prefers {}"),
    (re.compile(r"\bi (?:live|work) (?:in|at) ([\w ,']{2,40})", re.I), "personal_fact", "place", "The driver lives/works in {}"),
    (re.compile(r"\bi (?:hate|dislike) ([\w ,']{3,60})", re.I), "preference", "dislikes", "The driver dislikes {}"),
]


class MemoryAgent:
    def __init__(self, cfg: Config, bus: Bus, session: Session, llm: LLMClient, store: MemoryStore):
        self.cfg, self.bus, self.s, self.llm, self.store = cfg, bus, session, llm, store
        self.cursor = 0  # index into the turn queue snapshot we've processed
        self.processed = 0

    async def run(self) -> None:
        last = time.time()
        while True:
            await asyncio.sleep(2.0)
            pending = len(self.s.turn_queue) - self.cursor if self.cursor <= len(self.s.turn_queue) else 0
            if pending <= 0:
                continue
            idle = time.time() - last > self.cfg.memory_idle_seconds
            if pending >= self.cfg.memory_extract_every or idle:
                turns = list(self.s.turn_queue)[-pending:]
                self.cursor = len(self.s.turn_queue)
                last = time.time()
                try:
                    await self._extract(turns)
                except Exception as e:  # noqa: BLE001
                    print(f"[memory] extraction failed: {e}")

    async def flush(self) -> None:
        turns = list(self.s.turn_queue)[self.cursor:]
        self.cursor = len(self.s.turn_queue)
        if turns:
            await self._extract(turns)

    async def _extract(self, turns: list[dict]) -> None:
        convo = "\n".join(f"User: {t['user']}\nAssistant: {t['assistant']}" for t in turns)
        facts: list[dict] = []
        try:
            raw = await self.llm.complete([{"role": "system", "content": EXTRACT_PROMPT},
                                           {"role": "user", "content": convo}], max_tokens=300)
            m = re.search(r"\[.*\]", raw, re.S)
            facts = json.loads(m.group(0)) if m else []
        except (LLMUnavailable, json.JSONDecodeError):
            for t in turns:
                for rx, cat, topic, tpl in _RULES:
                    for hit in rx.findall(t["user"]):
                        facts.append(dict(text=tpl.format(hit.strip()), category=cat, topic=topic))
        for f in facts:
            if not isinstance(f, dict) or not f.get("text"):
                continue
            cat = f.get("category") if f.get("category") in CATEGORIES else "other"
            if await self.store.add(self.s.user_id, f["text"], cat, str(f.get("topic", ""))[:40]):
                self.processed += 1
                self.bus.publish("memory", dict(text=f["text"], category=cat, topic=f.get("topic", "")))
