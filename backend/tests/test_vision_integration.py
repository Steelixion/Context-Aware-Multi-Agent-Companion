"""Whole app, vision side: separate vision process -> bus -> Safety Agent / SQLite / WebSocket."""
import time

import cv2
import pytest

from .conftest import ASSETS, needs_vision


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("DC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DC_ENABLE_MIC", "0")
    monkeypatch.setenv("DC_ENABLE_SPEAKER", "0")
    monkeypatch.setenv("DC_LLM_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("DC_SYNC_INTERVAL", "3600")
    monkeypatch.setenv("DC_CALIBRATION_SECONDS", "2")
    return monkeypatch


def app():
    from fastapi.testclient import TestClient

    from app.api.server import create_app
    from app.config import Config
    return TestClient(create_app(Config()))


def wait_for(ws, pred, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = ws.receive_json()
        if pred(msg):
            return msg
    raise AssertionError("timed out")


def test_calibration_lifecycle_and_alerts_in_sim_mode(env):
    env.setenv("DC_VISION_MODE", "sim")
    with app() as client, client.websocket_connect("/ws") as ws:
        snap = ws.receive_json()
        assert snap["type"] == "snapshot" and snap["calibration"]["state"] == "calibrating"
        ev = wait_for(ws, lambda m: m["type"] == "calibration", 30)["data"]
        assert ev["state"] == "calibrated" and ev["stored"] and ev["frames"] >= 30
        stored = client.get("/api/calibration").json()
        assert stored["profile"]["base"]["ear"] == pytest.approx(0.32, abs=0.01)   # the simulated driver

        assert client.post("/api/calibrate").json() == {"ok": True}
        assert wait_for(ws, lambda m: m["type"] == "calibration", 5)["data"]["state"] == "calibrating"
        assert wait_for(ws, lambda m: m["type"] == "calibration", 30)["data"]["state"] == "calibrated"

        ws.send_json({"type": "sim", "mode": "nodding"})
        alert = wait_for(ws, lambda m: m["type"] == "alert" and m["data"]["severity"] == "CRITICAL", 20)
        assert alert["data"]["reason"] in ("head_drop", "sustained_fatigue")
        ws.send_text("not json")                      # malformed input must not drop the socket
        ws.send_json(["also", "not", "an", "object"])
        assert wait_for(ws, lambda m: m["type"] == "telemetry", 5)
        logged = client.get("/api/history").json()["logs"]
        assert any(log["safety_prompt"] for log in logged)      # the alert reached SQLite


def test_stored_calibration_is_used_from_the_first_frame_after_restart(env):
    env.setenv("DC_VISION_MODE", "sim")
    with app() as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()
        wait_for(ws, lambda m: m["type"] == "calibration", 30)
    with app() as client, client.websocket_connect("/ws") as ws:
        snap = ws.receive_json()
        assert snap["calibration"]["stored"]
        first = wait_for(ws, lambda m: m["type"] == "telemetry", 15)["data"]
        assert first["calibration"]["ear_threshold"] == pytest.approx(0.7 * 0.32, abs=0.01)   # not 0.21
        history = client.get("/api/history").json()
        assert len(history["records"]) == 2


@needs_vision
def test_camera_mode_runs_mediapipe_in_the_vision_process(env, tmp_path):
    im = cv2.resize(cv2.imread(str(ASSETS / "business-person.png")), (640, 906))
    path = str(tmp_path / "drive.avi")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 15, (640, 906))
    for _ in range(30):
        out.write(im)
    out.release()
    env.setenv("DC_VISION_MODE", "camera")
    env.setenv("DC_VIDEO_FILE", path)
    with app() as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()
        tel = wait_for(ws, lambda m: m["type"] == "telemetry" and m["data"]["face_detected"], 60)["data"]
        assert tel["source"] == "camera" and tel["posture"]["visible"] and tel["gaze"]["h"] is not None
        assert wait_for(ws, lambda m: m["type"] == "frame", 10)["data"][:4] == "/9j/"    # base64 JPEG
        ev = wait_for(ws, lambda m: m["type"] == "calibration", 60)["data"]
        assert ev["state"] == "calibrated"
        base = ev["profile"]["base"]
        assert 0.2 < base["ear"] < 0.4 and base["mar"] < 0.05 and 0.4 < base["delta"] < 0.8
        steady = wait_for(ws, lambda m: m["type"] == "telemetry" and m["data"]["calibration"]["state"] == "calibrated", 10)
        hp = steady["data"]["head_pose"]
        assert all(abs(v) < 2 for v in hp.values())          # a still photo: no movement vs. its own neutral
        assert steady["data"]["drowsiness_level"] == "ALERT" and steady["data"]["quality"] == 1.0
