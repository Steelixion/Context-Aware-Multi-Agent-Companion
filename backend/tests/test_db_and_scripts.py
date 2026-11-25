"""Calibration profile storage and the model download script."""
import asyncio
import importlib.util
import io
import re
import tarfile

import pytest

from app.config import ROOT
from app.db.relational import Database


def test_calibration_profiles_store_newest_and_prune(tmp_path):
    async def go():
        db = Database(tmp_path / "t.db")
        await db.open()
        uid = await db.create_user("A")
        other = await db.create_user("B")
        assert await db.latest_profile(uid) is None
        for i in range(12):
            await db.save_profile(uid, {"base": {"ear": 0.2 + i / 100}, "frames": 150, "created_at": 1000.0 + i})
        await db.save_profile(other, {"base": {"ear": 0.5}, "frames": 150, "created_at": 5000.0})
        latest = await db.latest_profile(uid)
        cur = await db.conn.execute("SELECT COUNT(*) FROM calibration_profiles WHERE user_id=?", (uid,))
        count = (await cur.fetchone())[0]
        await db.conn.execute("UPDATE calibration_profiles SET profile_json='{oops' WHERE user_id=?", (other,))
        broken = await db.latest_profile(other)
        await db.close()
        return latest, count, broken
    latest, count, broken = asyncio.run(go())
    assert latest["base"]["ear"] == pytest.approx(0.31) and count == 10 and broken is None


def test_schema_upgrade_on_an_existing_database(tmp_path):
    """Opening an older database (no calibration table) must add it without losing data."""
    import sqlite3
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE users (user_id TEXT PRIMARY KEY, display_name TEXT NOT NULL,"
                      " preferred_lang TEXT NOT NULL DEFAULT 'en', cloud_account TEXT,"
                      " settings_json TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL,"
                      " updated_at REAL NOT NULL, deleted_at REAL);"
                      "INSERT INTO users VALUES ('u1','Old','en',NULL,'{}',1,1,NULL);")
    con.close()

    async def go():
        db = Database(path)
        await db.open()
        await db.save_profile("u1", {"base": {}, "frames": 1})
        user = await db.get_user("u1")
        await db.close()
        return user["display_name"]
    assert asyncio.run(go()) == "Old"


# --------------------------------------------------------------------------- scripts/download_models.py
@pytest.fixture(scope="module")
def dl():
    spec = importlib.util.spec_from_file_location("download_models", ROOT / "scripts" / "download_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_download_manifest_is_pinned(dl):
    entries = [e for group in dl.FILES.values() for e in group]
    assert entries and all(url.startswith("https://") and re.fullmatch(r"[0-9a-f]{64}", digest)
                           for url, _, digest in entries)
    assert all(re.fullmatch(r"[0-9a-f]{64}", d) for d in dl.STT_FILES.values())
    assert "/1/" in entries[1][0]                    # versioned MediaPipe URLs, not "latest"


def test_sha256(dl, tmp_path):
    f = tmp_path / "x"
    f.write_bytes(b"abc")
    assert dl.sha256(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def _tar(path, name, data=b"x"):
    with tarfile.open(path, "w:bz2") as t:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))


@pytest.mark.parametrize("name", ["../evil.txt", "../../evil.txt", "/abs/evil.txt"])
def test_safe_extract_never_writes_outside_the_target(dl, tmp_path, name):
    """'..' members are refused; absolute members are re-rooted inside the target (tarfile's
    'data' filter). Either way nothing may land outside it."""
    archive, out = tmp_path / "a.tar.bz2", tmp_path / "out"
    _tar(archive, name)
    try:
        dl.safe_extract(archive, out)
    except Exception:  # noqa: BLE001 - refusing is fine
        pass
    written = [p for p in tmp_path.rglob("evil.txt")]
    assert all(out in p.parents for p in written)
    if ".." in name:
        assert written == []


def test_safe_extract_normal_archive(dl, tmp_path):
    archive = tmp_path / "a.tar.bz2"
    _tar(archive, "model/tokens.txt", b"a 0\n")
    dl.safe_extract(archive, tmp_path / "out")
    assert (tmp_path / "out" / "model" / "tokens.txt").read_bytes() == b"a 0\n"
