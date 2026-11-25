#!/usr/bin/env python3
"""Download the model files the app needs into <repo>/models (git-ignored).

    python scripts/download_models.py                 # speech-to-text, VAD and vision models
    python scripts/download_models.py --test-assets   # also the sample images used by the tests
    python scripts/download_models.py --only vision   # just one group

Every file is checked against a pinned SHA-256; interrupted downloads resume; files that are
already present and correct are skipped. Standard library only, so it runs before any pip install.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
GH = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
MP = "https://storage.googleapis.com/mediapipe-models/"
MPA = "https://storage.googleapis.com/mediapipe-assets/"
STT_NAME = "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8-2026-02-05"

# group -> [(url, destination relative to models/, sha256)]
FILES = {
    "vad": [(GH + "silero_vad_v5.onnx", "vad/silero_vad_v5.onnx",
             "6b99cbfd39246b6706f98ec13c7c50c6b299181f2474fa05cbc8046acc274396")],
    "vision": [
        (MP + "face_landmarker/face_landmarker/float16/1/face_landmarker.task", "vision/face_landmarker.task",
         "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"),
        (MP + "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
         "vision/pose_landmarker_lite.task",
         "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"),
    ],
    "test-assets": [
        (MPA + "portrait.jpg", "test-assets/portrait.jpg",
         "a6f11efaa834706db23f275b6115058fa87fc7f14362681e6abe14e82749de3e"),
        (MPA + "business-person.png", "test-assets/business-person.png",
         "1f61cf0603cef77ffca4e24848ddf8290b5651d03b957e93b742c9ef963b5c11"),
        (MPA + "pose.jpg", "test-assets/pose.jpg",
         "c8a830ed683c0276d713dd5aeda28f415f10cd6291972084a40d0d8b934ed62b"),
    ],
}
# the speech-to-text model is an archive; its extracted files are what gets verified
STT_ARCHIVE = GH + STT_NAME + ".tar.bz2"
STT_FILES = {"model.int8.onnx": "e3042b2f3b3ef0af2211bf99d2b4bf94a21f5ac0e9898827e7dd6d003a860e91",
             "tokens.txt": "7d99997ef207ff14c2cfe825f2aa037528ea250113cc3c6392bfe49326884ba6"}


def set_models_dir(path: Path) -> None:
    global MODELS
    MODELS = path.resolve()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path, attempts: int = 5) -> None:
    """Download url -> dest, resuming a partial download after network errors."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "driver-companion-setup"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:      # server ignored the range: start over
                    have = 0
                total = have + int(r.headers.get("Content-Length") or 0)
                with part.open("ab" if have else "wb") as out:
                    done, shown = have, -1
                    while block := r.read(1 << 20):
                        out.write(block)
                        done += len(block)
                        pct = int(done * 100 / total) if total else -1
                        if pct // 10 != shown // 10 or pct == 100:
                            shown = pct
                            print(f"\r  {dest.name}: {done / 1e6:7.1f} MB" + (f" ({pct}%)" if total else ""),
                                  end="", flush=True)
            print()
            part.replace(dest)
            return
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print(f"\n  attempt {attempt}/{attempts} failed: {e}")
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)


def get_file(url: str, rel: str, digest: str) -> bool:
    dest = MODELS / rel
    if dest.is_file() and sha256(dest) == digest:
        print(f"ok        {rel}")
        return True
    print(f"download  {rel}")
    fetch(url, dest)
    if sha256(dest) != digest:
        dest.unlink()
        print(f"FAILED    {rel}: checksum mismatch (file removed)")
        return False
    return True


def safe_extract(archive: Path, target: Path) -> None:
    with tarfile.open(archive, "r:bz2") as tar:
        if hasattr(tarfile, "data_filter"):           # Python 3.12+ (and security backports)
            tar.extractall(target, filter="data")
            return
        root = target.resolve()
        for m in tar.getmembers():                    # older Pythons: reject paths outside target
            if not (m.isfile() or m.isdir()) or not (root / m.name).resolve().is_relative_to(root):
                raise ValueError(f"unsafe archive member: {m.name}")
        tar.extractall(target)


def get_stt() -> bool:
    model_dir = MODELS / "stt" / STT_NAME
    if all((model_dir / n).is_file() and sha256(model_dir / n) == d for n, d in STT_FILES.items()):
        print(f"ok        stt/{STT_NAME}")
        return True
    print(f"download  stt/{STT_NAME} (~290 MB)")
    with tempfile.TemporaryDirectory(dir=MODELS) as tmp:
        archive = Path(tmp) / "stt.tar.bz2"
        fetch(STT_ARCHIVE, archive)
        print("  extracting ...")
        safe_extract(archive, Path(tmp) / "x")
        extracted = Path(tmp) / "x" / STT_NAME
        bad = [n for n, d in STT_FILES.items() if not (extracted / n).is_file() or sha256(extracted / n) != d]
        if bad:
            print(f"FAILED    stt: checksum mismatch for {', '.join(bad)}")
            return False
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), str(model_dir))
    return True


def main() -> int:
    groups = ["stt", "vad", "vision", "test-assets"]
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", nargs="+", choices=groups, help="download only these groups")
    p.add_argument("--test-assets", action="store_true", help="also download the sample images for the tests")
    p.add_argument("--dest", type=Path, default=MODELS, help=argparse.SUPPRESS)  # for testing the script
    a = p.parse_args()
    set_models_dir(a.dest)
    wanted = a.only or (["stt", "vad", "vision"] + (["test-assets"] if a.test_assets else []))
    MODELS.mkdir(exist_ok=True)
    ok = True
    for g in wanted:
        try:
            if g == "stt":
                ok &= get_stt()
            else:
                for url, rel, digest in FILES[g]:
                    ok &= get_file(url, rel, digest)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED    {g}: {e}")
            ok = False
    print(f"\nAll models are ready in {MODELS}" if ok else
          "\nSome downloads failed. Run the script again to retry (finished files are kept).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
