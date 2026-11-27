"""Records every WebSocket message from the app with the harness's monotonic receive time,
and keeps a little live state (level, speaking, calibration, STT status) that the timeline
executor uses for triggers (e.g. 'speak when the level first becomes HIGH')."""
from __future__ import annotations

import asyncio
import json
import threading
import time

from .events import EventLog

DROP_PAYLOAD = {"frame", "frame_raw"}   # base64 JPEG previews: keep only the fact they arrived


class WsRecorder:
    def __init__(self, url: str, log: EventLog):
        self.url, self.log = url, log
        self.cond = threading.Condition()
        self.level: str | None = None
        self.speaking = False
        self.speaking_priority: int | None = None
        self.calibration_state: str | None = None
        self.stt_status: str | None = None
        self.telemetry_count = 0
        self.open_bubbles: set = set()          # assistant replies started but not finished
        self.busy_since_change = time.monotonic()
        self.connected = threading.Event()
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ws-recorder", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(5)

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
            self.log.add("harness", "ws_error", error=repr(e))

    async def _main(self) -> None:
        import websockets
        async with websockets.connect(self.url, max_size=None, ping_interval=None) as ws:
            self.connected.set()
            self.log.add("harness", "ws_connected", url=self.url)
            while not self._stop.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                t = time.monotonic()
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                typ = msg.get("type", "?")
                data = msg.get("data") if typ != "snapshot" else {k: v for k, v in msg.items() if k != "type"}
                if typ in DROP_PAYLOAD:
                    self.log.add("ws", typ, t=t, bytes=len(raw))
                    continue
                self.log.add("ws", typ, t=t, **(data if isinstance(data, dict) else {"value": data}))
                self._update(typ, data or {})

    @property
    def companion_busy(self) -> bool:
        """Talking, or a reply is being generated (bubble open), so it is about to talk."""
        return self.speaking or bool(self.open_bubbles)

    def _update(self, typ: str, d: dict) -> None:
        with self.cond:
            was_busy = self.companion_busy
            if typ == "state":
                self.level = d.get("level", self.level)
            elif typ == "telemetry":
                self.telemetry_count += 1
                self.level = d.get("drowsiness_level", self.level)
                cal = d.get("calibration") or {}
                if cal.get("state"):
                    self.calibration_state = cal["state"]
            elif typ == "audio":
                self.speaking = bool(d.get("speaking"))
                self.speaking_priority = d.get("priority") if self.speaking else None
            elif typ == "chat" and d.get("role") == "assistant":
                if d.get("final"):
                    self.open_bubbles.discard(d.get("id"))
                else:
                    self.open_bubbles.add(d.get("id"))
            elif typ == "calibration":
                self.calibration_state = d.get("state", self.calibration_state)
            elif typ == "snapshot":
                self.calibration_state = (d.get("calibration") or {}).get("state", self.calibration_state)
                self.stt_status = d.get("stt")
            elif typ == "stt":
                self.stt_status = d.get("status", self.stt_status)
            if self.companion_busy != was_busy:
                self.busy_since_change = time.monotonic()
            self.cond.notify_all()

    def wait_quiet(self, quiet_s: float, timeout: float) -> bool:
        """Wait until the companion has been idle (not talking, no reply pending) for quiet_s."""
        return self.wait_for(lambda w: not w.companion_busy and time.monotonic() - w.busy_since_change >= quiet_s,
                             timeout)

    def wait_for(self, pred, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self.cond:
            while not pred(self):
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self.cond.wait(min(left, 0.25))
            return True
