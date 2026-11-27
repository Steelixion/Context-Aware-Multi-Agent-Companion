"""The turn-taking gate: a sentence-boundary blip (audio stop + start at once) is not 'quiet'."""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.wsrec import WsRecorder  # noqa: E402


class _Log:
    def add(self, *a, **k):
        pass


def test_quiet_gate_ignores_blips_and_pending_replies():
    w = WsRecorder("ws://unused", _Log())
    w._update("audio", {"speaking": True, "priority": 5})
    w._update("audio", {"speaking": False})            # sentence boundary...
    w._update("audio", {"speaking": True, "priority": 5})  # ...next sentence at once
    assert not w.wait_quiet(0.3, 0.6)
    w._update("audio", {"speaking": False})
    w._update("chat", {"id": "r1", "role": "assistant", "final": False})   # reply being generated
    assert not w.wait_quiet(0.3, 0.6)
    w._update("chat", {"id": "r1", "role": "assistant", "final": True})
    t = time.monotonic()
    assert w.wait_quiet(0.3, 2.0)
    assert time.monotonic() - t >= 0.25                # had to wait for the quiet window
