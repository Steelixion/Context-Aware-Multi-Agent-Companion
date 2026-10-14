"""Async SQLite access (aiosqlite). Never blocks the event loop."""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import aiosqlite

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._conn: aiosqlite.Connection | None = None

    async def open(self) -> None:
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()

    @property
    def conn(self) -> aiosqlite.Connection:
        assert self._conn, "Database not opened"
        return self._conn

    # ---- users ----
    async def create_user(self, display_name: str, lang: str = "en") -> str:
        uid, now = str(uuid.uuid4()), time.time()
        await self.conn.execute(
            "INSERT INTO users(user_id, display_name, preferred_lang, created_at, updated_at)"
            " VALUES (?,?,?,?,?)",
            (uid, display_name, lang, now, now),
        )
        await self.conn.commit()
        return uid

    async def get_user(self, user_id: str) -> aiosqlite.Row | None:
        cur = await self.conn.execute(
            "SELECT * FROM users WHERE user_id=? AND deleted_at IS NULL", (user_id,)
        )
        return await cur.fetchone()

    async def first_user(self) -> aiosqlite.Row | None:
        cur = await self.conn.execute(
            "SELECT * FROM users WHERE deleted_at IS NULL ORDER BY created_at LIMIT 1")
        return await cur.fetchone()

    async def update_user(self, user_id: str, display_name: str, lang: str) -> None:
        await self.conn.execute(
            "UPDATE users SET display_name=?, preferred_lang=?, updated_at=? WHERE user_id=?",
            (display_name, lang, time.time(), user_id))
        await self.conn.commit()

    async def list_records(self, user_id: str, limit: int = 20) -> list[dict]:
        cur = await self.conn.execute(
            "SELECT * FROM monitoring_records WHERE user_id=? ORDER BY started_at DESC LIMIT ?",
            (user_id, limit))
        return [dict(r) for r in await cur.fetchall()]

    async def recent_logs(self, user_id: str, limit: int = 100) -> list[dict]:
        cur = await self.conn.execute(
            """SELECT l.* FROM monitoring_logs l JOIN monitoring_records r USING(record_id)
               WHERE r.user_id=? ORDER BY l.ts DESC LIMIT ?""", (user_id, limit))
        return [dict(r) for r in await cur.fetchall()]

    # ---- monitoring ----
    async def start_record(self, user_id: str) -> str:
        rid, now = str(uuid.uuid4()), time.time()
        await self.conn.execute(
            "INSERT INTO monitoring_records(record_id, user_id, started_at, updated_at)"
            " VALUES (?,?,?,?)",
            (rid, user_id, now, now),
        )
        await self.conn.commit()
        return rid

    async def end_record(self, record_id: str) -> None:
        now = time.time()
        await self.conn.execute(
            """UPDATE monitoring_records SET ended_at=?, updated_at=?,
                 avg_fatigue=(SELECT AVG(fatigue_score) FROM monitoring_logs WHERE record_id=?),
                 max_fatigue=(SELECT MAX(fatigue_score) FROM monitoring_logs WHERE record_id=?)
               WHERE record_id=?""",
            (now, now, record_id, record_id, record_id),
        )
        await self.conn.commit()

    async def log_event(
        self, record_id: str, driver_state: str, drowsiness_level: str, *,
        fatigue_score: float | None = None, ear: float | None = None,
        mar: float | None = None, safety_prompt: str | None = None,
        emotion: str | None = None, ts: float | None = None,
    ) -> None:
        await self.conn.execute(
            "INSERT INTO monitoring_logs(record_id, ts, driver_state, drowsiness_level,"
            " fatigue_score, ear, mar, safety_prompt, emotion) VALUES (?,?,?,?,?,?,?,?,?)",
            (record_id, ts or time.time(), driver_state, drowsiness_level,
             fatigue_score, ear, mar, safety_prompt, emotion),
        )
        if safety_prompt:
            await self.conn.execute(
                "UPDATE monitoring_records SET alert_count=alert_count+1, updated_at=? WHERE record_id=?",
                (time.time(), record_id),
            )
        await self.conn.commit()
