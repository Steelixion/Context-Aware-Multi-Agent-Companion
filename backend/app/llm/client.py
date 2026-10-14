"""Client for a local llama.cpp server (OpenAI-compatible /v1/chat/completions)."""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx


class LLMUnavailable(Exception):
    pass


class LLMClient:
    def __init__(self, base_url: str, model: str = "local"):
        self.base, self.model = base_url.rstrip("/"), model
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=2.0))

    async def available(self) -> bool:
        try:
            r = await self.http.get(self.base + "/models", timeout=2.0)
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    async def stream(self, messages: list[dict], max_tokens: int = 160,
                     temperature: float = 0.7) -> AsyncIterator[str]:
        body = dict(model=self.model, messages=messages, max_tokens=max_tokens,
                    temperature=temperature, stream=True)
        try:
            async with self.http.stream("POST", self.base + "/chat/completions", json=body) as r:
                if r.status_code != 200:
                    raise LLMUnavailable(f"HTTP {r.status_code}")
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta
        except httpx.HTTPError as e:
            raise LLMUnavailable(str(e)) from e

    async def complete(self, messages: list[dict], max_tokens: int = 300,
                       temperature: float = 0.1) -> str:
        return "".join([t async for t in self.stream(messages, max_tokens, temperature)])

    async def close(self) -> None:
        await self.http.aclose()
