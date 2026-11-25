"""LLM streaming robustness, companion error containment and a clean shutdown mid-reply."""
import asyncio
import gc
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from app.agents.companion import CompanionAgent
from app.bus import Bus
from app.llm.client import LLMClient
from app.state import Session


def _chunk(text):
    return "data: " + json.dumps({"choices": [{"delta": {"content": text}}]}) + "\n\n"


def stub_llm(pause: float = 0.0):
    """OpenAI-style streaming server. Sends a malformed line and a usage-only chunk with
    "choices": [] (llama.cpp does with usage reporting), optionally pausing mid-reply."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/v1/models" else 404)
            self.end_headers()

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                self.wfile.write(_chunk("Hello").encode())
                self.wfile.flush()
                time.sleep(pause)
                for line in ("data: not json\n\n", 'data: {"choices": [], "usage": {"total_tokens": 9}}\n\n',
                             _chunk(" there."), "data: [DONE]\n\n"):
                    self.wfile.write(line.encode())
            except OSError:
                pass   # the client went away mid-reply

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


def test_stream_skips_malformed_and_usage_only_chunks():
    server, url = stub_llm()

    async def go():
        llm = LLMClient(url)
        try:
            return "".join([t async for t in llm.stream([{"role": "user", "content": "hi"}])])
        finally:
            await llm.close()
    try:
        assert asyncio.run(go()) == "Hello there."
    finally:
        server.shutdown()


def test_a_failed_reply_is_contained():
    session = Session(user_id="u", record_id="r")
    agent = CompanionAgent(Bus(), session, llm=None, store=None, arbiter=SimpleNamespace(flush_chat=lambda: None))

    async def boom(_):
        raise RuntimeError("model crashed")
    agent.graph = SimpleNamespace(ainvoke=boom)
    asyncio.run(agent._turn({"text": "hi"}))          # must not raise
    last = session.chat_log[-1]
    assert last["role"] == "assistant" and last["final"] and last["interrupted"]
    assert session.history == []                      # nothing half-done is remembered


def test_shutdown_in_the_middle_of_a_reply_is_clean(monkeypatch, tmp_path, caplog):
    """Regression: stop() closed the LLM client and database while a reply was still running,
    so the reply crashed ("client has been closed", "Task exception was never retrieved")."""
    from fastapi.testclient import TestClient

    import app.api.server as srv
    from app.api.server import create_app
    from app.config import Config

    finished_before_close = []

    class Recording(srv.Runtime):
        async def stop(self):
            await super().stop()
            task = self.companion.task      # the reply was in flight: it must have been cancelled
            last = self.session.chat_log[-1]
            finished_before_close.append(task is not None and task.cancelled()
                                         and last["role"] == "assistant" and last.get("interrupted"))
    monkeypatch.setattr(srv, "Runtime", Recording)
    server, url = stub_llm(pause=5.0)
    monkeypatch.setenv("DC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DC_ENABLE_SPEAKER", "0")
    monkeypatch.setenv("DC_LLM_URL", url)
    try:
        with caplog.at_level(logging.ERROR):
            with TestClient(create_app(Config())) as client, client.websocket_connect("/ws") as ws:
                ws.receive_json()
                client.post("/api/chat", json={"text": "hi"})
                deadline = time.time() + 20
                while time.time() < deadline:   # wait until the reply is streaming
                    m = ws.receive_json()
                    if m["type"] == "chat" and m["data"]["role"] == "assistant" and m["data"]["text"]:
                        break
                else:
                    pytest.fail("reply never started")
            t0 = time.time()
            gc.collect()
        assert time.time() - t0 < 5
        assert finished_before_close == [True]
        problems =[r.getMessage() for r in caplog.records if "never retrieved" in r.getMessage()
                    or "client has been closed" in r.getMessage()]
        assert problems == []
    finally:
        server.shutdown()
