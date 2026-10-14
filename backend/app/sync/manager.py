"""Background profile sync: encrypt locally, upload only when a network is reachable.

Bundle = zip(sqlite snapshot + LanceDB directory) -> AES-256-GCM. If DC_CLOUD_URL is unset,
the encrypted bundle is written to data/cloud_outbox/ (a stand-in for the cloud account)."""
from __future__ import annotations

import asyncio
import hashlib
import io
import os
import socket
import sqlite3
import tempfile
import time
import zipfile
from pathlib import Path

import httpx
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ..bus import Bus
from ..config import Config


def load_key(data_dir: Path) -> bytes:
    """32-byte key: OS keyring if available, else a 0600 file in the data dir."""
    try:
        import keyring
        k = keyring.get_password("driver-companion", "profile-key")
        if not k:
            k = os.urandom(32).hex()
            keyring.set_password("driver-companion", "profile-key", k)
        return bytes.fromhex(k)
    except Exception:  # noqa: BLE001
        p = data_dir / ".profile.key"
        if not p.exists():
            p.write_bytes(os.urandom(32))
            try:
                p.chmod(0o600)
            except OSError:
                pass
        return p.read_bytes()


def encrypt(key: bytes, plain: bytes) -> bytes:
    nonce = os.urandom(12)
    return b"DCP1" + nonce + AESGCM(key).encrypt(nonce, plain, b"DCP1")


def decrypt(key: bytes, blob: bytes) -> bytes:
    assert blob[:4] == b"DCP1"
    return AESGCM(key).decrypt(blob[4:16], blob[16:], b"DCP1")


class SyncManager:
    def __init__(self, cfg: Config, bus: Bus, user_id: str):
        self.cfg, self.bus, self.user_id = cfg, bus, user_id
        self.key = load_key(cfg.data_dir)
        self.last_hash: str | None = None
        self.status = dict(online=False, last_success=None, last_error=None, syncing=False, target="outbox")
        self._wake = asyncio.Event()
        self.status["target"] = cfg.cloud_url or str(cfg.data_dir / "cloud_outbox")

    def trigger(self) -> None:
        self._wake.set()

    async def run(self) -> None:
        backoff = 15.0
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.cfg.sync_interval)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            online = await asyncio.to_thread(self._probe)
            self.status["online"] = online
            if online:
                try:
                    await self.sync_once()
                    backoff = 15.0
                except Exception as e:  # noqa: BLE001
                    self.status["last_error"] = str(e)
                    print(f"[sync] failed: {e}")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 900)
            self._publish()

    def _probe(self) -> bool:
        host = self.cfg.probe_host
        try:
            with socket.create_connection((host, 53 if host.count(".") == 3 else 443), timeout=2):
                return True
        except OSError:
            return False

    def _bundle(self) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            snap = Path(tempfile.mktemp(suffix=".db"))
            try:
                src = sqlite3.connect(self.cfg.sqlite_path)
                dst = sqlite3.connect(snap)
                src.backup(dst)
                dst.close(); src.close()
                z.write(snap, "driver.db")
            finally:
                snap.unlink(missing_ok=True)
            root = self.cfg.lance_path
            if root.exists():
                for f in root.rglob("*"):
                    if f.is_file():
                        z.write(f, "vectors/" + f.relative_to(root).as_posix())
        return buf.getvalue()

    async def sync_once(self) -> None:
        self.status["syncing"] = True
        self._publish()
        try:
            plain = await asyncio.to_thread(self._bundle)
            digest = hashlib.sha256(plain).hexdigest()
            if digest == self.last_hash:
                return
            blob = await asyncio.to_thread(encrypt, self.key, plain)
            if self.cfg.cloud_url:
                async with httpx.AsyncClient(timeout=30) as c:
                    r = await c.put(f"{self.cfg.cloud_url.rstrip('/')}/profiles/{self.user_id}",
                                    content=blob, headers={"Authorization": f"Bearer {self.cfg.cloud_token}"})
                    r.raise_for_status()
            else:
                out = self.cfg.data_dir / "cloud_outbox"
                out.mkdir(exist_ok=True)
                (out / f"{self.user_id}.dcp").write_bytes(blob)
            self.last_hash = digest
            self.status.update(last_success=time.time(), last_error=None)
        finally:
            self.status["syncing"] = False

    def _publish(self) -> None:
        self.bus.publish("sync", dict(self.status))
