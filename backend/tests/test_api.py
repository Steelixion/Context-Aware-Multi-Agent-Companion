"""HTTP API: dashboard serving and request validation."""
import pytest

from app.api.server import DIST


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from app.api.server import create_app
    from app.config import Config
    monkeypatch.setenv("DC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DC_ENABLE_SPEAKER", "0")
    monkeypatch.setenv("DC_LLM_URL", "http://127.0.0.1:9/v1")
    with TestClient(create_app(Config())) as c:
        yield c


needs_dist = pytest.mark.skipif(not (DIST / "index.html").is_file(), reason="dashboard not built (npm run build)")


@needs_dist
@pytest.mark.parametrize("url", ["/%2e%2e/%2e%2e/backend/app/config.py", "/..%2F..%2Fbackend%2Fapp%2Fconfig.py",
                                 "/%2e%2e/%2e%2e/README.md"])
def test_no_path_traversal_out_of_the_dashboard(client, url):
    r = client.get(url)
    assert "Central configuration" not in r.text and "<div id=" in r.text


@needs_dist
def test_index_is_revalidated_and_assets_are_served(client):
    r = client.get("/")
    assert r.headers["cache-control"] == "no-cache"
    asset = next((DIST / "assets").glob("*.js"))
    assert client.get(f"/assets/{asset.name}").status_code == 200


def test_session_and_calibration_endpoints(client):
    s = client.get("/api/session").json()
    assert s["stt"] == "off" and s["vision_mode"] == "sim" and "ear_threshold" not in s
    assert client.get("/api/calibration").json()["state"] in ("calibrating", "calibrated")
    assert client.post("/api/calibrate").json() == {"ok": True}
    assert client.get("/api/calibration").json()["state"] == "calibrating"


def test_chat_validation(client):
    assert client.post("/api/chat", json={"text": "hello"}).json() == {"ok": True}
    assert client.post("/api/chat", json={}).status_code == 422


def test_session_answers_fast_without_an_llm(client):
    import time
    t0 = time.perf_counter()
    assert client.get("/api/session").json()["llm_online"] is False
    assert time.perf_counter() - t0 < 0.5          # was ~2 s: it probed the dead LLM port per request


def test_llm_status_is_tracked_in_the_background(monkeypatch, tmp_path):
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from fastapi.testclient import TestClient

    from app.api.server import create_app
    from app.config import Config

    class Models(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/v1/models" else 404)
            self.end_headers()
            self.wfile.write(b'{"data": []}')

        def log_message(self, *a):
            pass

    stub = ThreadingHTTPServer(("127.0.0.1", 0), Models)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    monkeypatch.setenv("DC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DC_ENABLE_SPEAKER", "0")
    monkeypatch.setenv("DC_LLM_URL", f"http://127.0.0.1:{stub.server_address[1]}/v1")
    try:
        with TestClient(create_app(Config())) as c:
            deadline = time.time() + 5
            while not c.get("/api/session").json()["llm_online"] and time.time() < deadline:
                time.sleep(0.1)
            assert c.get("/api/session").json()["llm_online"] is True
            with c.websocket_connect("/ws") as ws:
                assert ws.receive_json()["llm_online"] is True
    finally:
        stub.shutdown()
