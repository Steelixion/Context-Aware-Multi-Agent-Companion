"""Tiny in-process pub/sub. publish() never blocks: full queues drop their oldest item."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any


class Bus:
    def __init__(self) -> None:
        self._subs: dict[str, list[asyncio.Queue]] = defaultdict(list)

    def subscribe(self, topic: str, maxsize: int = 64) -> asyncio.Queue:
        """topic '*' receives (topic, message) tuples for everything."""
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subs[topic].append(q)
        return q

    def unsubscribe(self, topic: str, q: asyncio.Queue) -> None:
        if q in self._subs[topic]:
            self._subs[topic].remove(q)

    def publish(self, topic: str, msg: Any) -> None:
        """Call from the event-loop thread only (use loop.call_soon_threadsafe otherwise)."""
        for q in self._subs[topic]:
            self._put(q, msg)
        for q in self._subs["*"]:
            self._put(q, (topic, msg))

    @staticmethod
    def _put(q: asyncio.Queue, item: Any) -> None:
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        q.put_nowait(item)
