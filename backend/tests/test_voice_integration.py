"""Whole app, voice in: fake mic -> VAD -> Omnilingual ASR -> bus -> Companion -> WebSocket."""
import time

import numpy as np
import pytest

from .conftest import REFERENCES, cer, load_wav, needs_model, needs_vad, resample


@pytest.fixture
def app_env(monkeypatch, tmp_path, fake_mic):
    monkeypatch.setenv("DC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DC_ENABLE_MIC", "1")
    monkeypatch.setenv("DC_ENABLE_SPEAKER", "0")
    monkeypatch.setenv("DC_VISION_MODE", "sim")
    monkeypatch.setenv("DC_LLM_URL", "http://127.0.0.1:9/v1")   # nothing there: offline reply
    monkeypatch.setenv("DC_SYNC_INTERVAL", "3600")
    return fake_mic


def _wait_for(ws, pred, timeout):
    """Read WebSocket messages until pred(msg) is true; return that message."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = ws.receive_json()
        if pred(msg):
            return msg
    raise AssertionError("timed out waiting for message")


@needs_model
@needs_vad
def test_spoken_utterance_reaches_companion_and_dashboard(app_env):
    from fastapi.testclient import TestClient

    from app.api.server import create_app
    from app.config import Config

    audio, sr = load_wav("es")
    app_env.audio = np.concatenate([np.zeros(16000, np.float32), resample(audio, sr, 16000),
                                    np.zeros(16000 * 2, np.float32)])
    with TestClient(create_app(Config())) as client:
        with client.websocket_connect("/ws") as ws:
            # connected while the model is still loading (~3 s): the snapshot says so, and an
            # 'stt' event on the same socket flips the dashboard chip once the mic is live
            snap = ws.receive_json()
            assert snap["type"] == "snapshot" and snap["stt"] in (
                "omnilingual (starting)", "omnilingual (loading)"), snap
            _wait_for(ws, lambda m: m["type"] == "stt" and m["data"]["engine"] == "omnilingual + silero VAD", 60)
            info = client.get("/api/session").json()
            assert info["mic"] is True and info["stt"] == "omnilingual + silero VAD"
            user = _wait_for(ws, lambda m: m["type"] == "chat" and m["data"]["role"] == "user", 60)
            assert cer(user["data"]["text"], REFERENCES["es"]) < 0.15, user
            reply = _wait_for(ws, lambda m: m["type"] == "chat" and m["data"]["role"] == "assistant"
                              and m["data"]["final"], 30)
            assert reply["data"]["text"] or reply["data"].get("interrupted")
        # the companion switched the session to the detected language
        assert client.get("/api/session").json()["lang"] == "es"


def test_mic_disabled_reports_stt_off(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from app.api.server import create_app
    from app.config import Config
    monkeypatch.setenv("DC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DC_ENABLE_MIC", "0")
    monkeypatch.setenv("DC_ENABLE_SPEAKER", "0")
    monkeypatch.setenv("DC_LLM_URL", "http://127.0.0.1:9/v1")
    with TestClient(create_app(Config())) as client:
        assert client.get("/api/session").json()["stt"] == "off"
        with client.websocket_connect("/ws") as ws:
            assert ws.receive_json()["stt"] == "off"
