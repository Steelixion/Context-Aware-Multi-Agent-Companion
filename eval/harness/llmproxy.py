"""Transparent recording proxy for the app's LLM server (DC_LLM_URL points here).

Forwards every request to the upstream OpenAI-compatible server unchanged and streams the
response back unchanged, while logging: request time, the messages sent (the app's system
prompt contains the detector state it passed to the LLM), time of the first streamed content
token, time of the last chunk, and the full streamed text. Nothing is modified.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time

from .events import EventLog


class LLMProxy:
    def __init__(self, upstream: str, port: int, log: EventLog):
        self.upstream, self.port, self.log = upstream.rstrip("/"), port, log
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner = None
        self._ready = threading.Event()
        self.error: str | None = None
        self._thread = threading.Thread(target=self._run, name="llm-proxy", daemon=True)
        self.n = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self) -> bool:
        self._thread.start()
        self._ready.wait(10)
        return self.error is None

    def stop(self) -> None:
        if self._loop and self._runner:
            fut = asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self._loop)
            try:
                fut.result(5)
            except Exception:  # noqa: BLE001
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(5)

    def _run(self) -> None:
        from aiohttp import ClientSession, ClientTimeout, web
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)

        async def handle(req: web.Request) -> web.StreamResponse:
            self.n += 1
            rid = self.n
            path = req.match_info["tail"]
            body = await req.read()
            t_req = time.monotonic()
            info: dict = {"rid": rid, "method": req.method, "path": path}
            stream = False
            if body:
                try:
                    j = json.loads(body)
                    stream = bool(j.get("stream"))
                    info |= {"model": j.get("model"), "stream": stream, "messages": j.get("messages"),
                             "max_tokens": j.get("max_tokens"), "temperature": j.get("temperature")}
                except ValueError:
                    pass
            if path.endswith("chat/completions"):
                self.log.add("llm", "llm_request", t=t_req, **info)
            headers = {k: v for k, v in req.headers.items()
                       if k.lower() not in ("host", "content-length", "transfer-encoding")}
            async with ClientSession(timeout=ClientTimeout(total=120)) as s:
                try:
                    async with s.request(req.method, f"{self.upstream}/{path}", data=body or None,
                                         headers=headers, params=req.query) as up:
                        resp = web.StreamResponse(status=up.status, headers={
                            k: v for k, v in up.headers.items()
                            if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")})
                        await resp.prepare(req)
                        first_tok, text, buf = None, [], b""
                        async for chunk in up.content.iter_any():
                            await resp.write(chunk)
                            if path.endswith("chat/completions"):
                                buf += chunk
                                *lines, buf = buf.split(b"\n")
                                for ln in lines:
                                    ln = ln.strip()
                                    if not ln.startswith(b"data:"):
                                        continue
                                    d = ln[5:].strip()
                                    if d == b"[DONE]":
                                        continue
                                    try:
                                        delta = json.loads(d)["choices"][0]["delta"].get("content")
                                    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                                        continue
                                    if delta:
                                        if first_tok is None:
                                            first_tok = time.monotonic()
                                            self.log.add("llm", "llm_first_token", t=first_tok, rid=rid)
                                        text.append(delta)
                        await resp.write_eof()
                        if path.endswith("chat/completions"):
                            self.log.add("llm", "llm_done", rid=rid, status=up.status,
                                         text="".join(text), first_token_s=None if first_tok is None
                                         else round(first_tok - t_req, 4))
                        return resp
                except Exception as e:  # noqa: BLE001
                    self.log.add("llm", "llm_error", rid=rid, error=repr(e))
                    return web.Response(status=502, text=f"proxy: upstream error {e!r}")

        async def boot():
            app = web.Application(client_max_size=64 * 1024 * 1024)
            app.router.add_route("*", "/v1/{tail:.*}", handle)
            self._runner = web.AppRunner(app, access_log=None)
            await self._runner.setup()
            site = web.TCPSite(self._runner, "127.0.0.1", self.port)
            await site.start()

        try:
            loop.run_until_complete(boot())
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
            self._ready.set()
            return
        self._ready.set()
        loop.run_forever()
