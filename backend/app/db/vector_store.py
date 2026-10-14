"""On-device vector memory (LanceDB). Blocking calls are pushed to a thread."""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Callable, Sequence

import lancedb
from lancedb.pydantic import LanceModel, Vector

EMBED_DIM = 384  # multilingual-e5-small / MiniLM class models
CATEGORIES = ("preference", "personal_fact", "routine", "health", "navigation", "other")


class Memory(LanceModel):
    memory_id: str
    user_id: str
    context_text: str
    embedding: Vector(EMBED_DIM)  # type: ignore[valid-type]
    category: str
    topic: str
    created_at: float
    updated_at: float


class MemoryStore:
    def __init__(self, path: str, embed: Callable[[Sequence[str]], list[list[float]]]):
        self._db = lancedb.connect(path)
        self._embed = embed  # sync fn; run in executor
        self._table = self._db.create_table("memories", schema=Memory, exist_ok=True)

    async def add(self, user_id: str, text: str, category: str = "other",
                  topic: str = "", dedup_threshold: float = 0.92) -> bool:
        """Insert a fact unless a near-duplicate exists. Returns True if inserted."""
        def _work() -> bool:
            vec = self._embed([text])[0]
            hits = (self._table.search(vec).where(f"user_id = '{_q(user_id)}'", prefilter=True)
                    .metric("cosine").limit(1).to_list())
            if hits and (1 - hits[0]["_distance"]) >= dedup_threshold:
                return False
            now = time.time()
            self._table.add([Memory(
                memory_id=str(uuid.uuid4()), user_id=user_id, context_text=text,
                embedding=vec, category=category, topic=topic,
                created_at=now, updated_at=now)])
            return True
        return await asyncio.to_thread(_work)

    async def search(self, user_id: str, query: str, k: int = 5,
                     min_score: float = 0.35) -> list[dict]:
        def _work() -> list[dict]:
            vec = self._embed([query])[0]
            rows = (self._table.search(vec).where(f"user_id = '{_q(user_id)}'", prefilter=True)
                    .metric("cosine").limit(k).to_list())
            return [{"text": r["context_text"], "category": r["category"],
                     "topic": r["topic"], "score": 1 - r["_distance"]}
                    for r in rows if (1 - r["_distance"]) >= min_score]
        return await asyncio.to_thread(_work)


    async def list(self, user_id: str, limit: int = 200) -> list[dict]:
        def _work() -> list[dict]:
            rows = [r for r in self._table.to_arrow().to_pylist() if r["user_id"] == user_id]
            rows.sort(key=lambda r: r["created_at"], reverse=True)
            return [{k: r[k] for k in ("memory_id", "context_text", "category", "topic", "created_at")}
                    for r in rows[:limit]]
        return await asyncio.to_thread(_work)

    async def delete(self, user_id: str, memory_id: str) -> None:
        await asyncio.to_thread(
            self._table.delete, f"memory_id = '{_q(memory_id)}' AND user_id = '{_q(user_id)}'")


def _q(s: str) -> str:
    return s.replace("'", "''")
