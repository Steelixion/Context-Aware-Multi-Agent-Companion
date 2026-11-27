"""Builds the frozen golden dataset in eval/golden/. Nothing here runs implicitly: the runner only
READS and VERIFIES the golden files. Rebuilding is always an explicit command, and an existing
golden dataset is never overwritten without --force (which bumps the version).

  python eval/golden/build_golden.py --inspect                 # print the NTHU folder layout
  python eval/golden/build_golden.py --fetch-tts               # download + verify the reference TTS model
  python eval/golden/build_golden.py --build [--force]         # build manifest, GT, audio, scenarios
  python eval/golden/build_golden.py --verify                  # re-check every checksum
  python eval/golden/build_golden.py --selftest DIR            # tiny synthetic golden (no dataset) for harness checks

See eval/golden/README.md for the format.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
REPO = EVAL.parent
sys.path.insert(0, str(EVAL))

from harness import metrics as M  # noqa: E402
from harness.noise import NOISE_TYPES, mix_at_snr  # noqa: E402
from harness.speech import KOKORO_FILES, Synth, sha256_file  # noqa: E402

BUILDER_VERSION = "1.0"
DATASET_DIR = REPO / "Dataset" / "Testing_Dataset"
CACHE = EVAL / ".cache" / "kokoro"
SEED = 20261127             # dataset split / scenario selection / noise seeds
N_DEV = 10

# ---- reference speech (English, neutral; never reveals the drowsiness label) -------------
UTTERANCES = {
    "U01": "Can you suggest a good podcast for a long drive?",
    "U02": "Tell me an interesting fact about the moon.",
    "U03": "What should I pack for a weekend trip to the mountains?",
    "U04": "How many minutes are there in three and a half hours?",
    "U05": "Play something relaxing, maybe some soft jazz.",
    "F01": "I'm fine, I'm wide awake, don't worry about me.",
    "Q01": "What is the tire pressure on my front left wheel right now?",
    "Q02": "How much fuel is left in the tank right now?",
    "B01": "Wait, actually, never mind that.",
}
VOICES = ("af_heart", "am_michael", "bf_emma", "bm_george")
SPEED_VARIANTS = (0.85, 1.2)          # extra slow/fast renditions of U01-U05 (seeded voice)
SNRS = (20, 10, 5, 0)
NOISY_BASE = ("U01", "U02", "U03", "U04")
PAD_S = 0.6                           # noise lead/tail around noisy files (the live bed continues it)


def sha256(p: Path) -> str:
    return sha256_file(p)


def speed_variant_voice(uid: str, speed: float) -> str:
    """The one voice in which the slow/fast rendition of an utterance exists (deterministic)."""
    return VOICES[(sorted(UTTERANCES).index(uid) + SPEED_VARIANTS.index(speed)) % len(VOICES)]


def audio_id(uid: str, voice: str, speed: float, noise: str | None = None, snr: int | None = None) -> str:
    return f"{uid}_{voice}_s{int(round(speed * 100))}_" + ("clean" if not noise else f"{noise}{snr}")


# --------------------------------------------------------------------------- TTS model
def fetch_tts() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, (url, sha) in KOKORO_FILES.items():
        p = CACHE / name
        if p.is_file() and sha256(p) == sha:
            print(f"ok        {name}")
            continue
        print(f"download  {name}")
        urllib.request.urlretrieve(url, p)
        got = sha256(p)
        if got != sha:
            p.unlink()
            raise SystemExit(f"checksum mismatch for {name}: {got}")
        print(f"verified  {name}")


# --------------------------------------------------------------------------- audio
def tts_plan(uids=None) -> list[tuple[str, str, float]]:
    uids = list(uids or UTTERANCES)
    plan = [(u, v, 1.0) for u in uids for v in VOICES]
    return plan + [(u, speed_variant_voice(u, s), s) for u in uids if u.startswith("U") for s in SPEED_VARIANTS]


def planned_audio_ids(uids=None) -> set[str]:
    out = set()
    for u, v, s in tts_plan(uids):
        out.add(audio_id(u, v, s))
        if u in NOISY_BASE:
            out |= {audio_id(u, v, s, k, snr) for k in NOISE_TYPES for snr in SNRS}
    return out


def build_audio(out: Path, rows: list, synth: Synth, utter_ids=None) -> None:
    adir = out / "synthetic" / "audio"
    adir.mkdir(parents=True, exist_ok=True)
    uids = list(utter_ids or UTTERANCES)

    def write(aid: str, x: np.ndarray, **meta) -> None:
        p = adir / f"{aid}.wav"
        sf.write(str(p), x, 16000, subtype="PCM_16")
        rows.append(dict(audio_id=aid, path=p.relative_to(out).as_posix(), sha256=sha256(p),
                         duration_s=round(len(x) / 16000, 3), synthetic="true", **meta))

    clean: list[tuple[str, str, float, np.ndarray]] = []
    plan = tts_plan(uids)
    for k, (u, v, s) in enumerate(plan, 1):
        aid = audio_id(u, v, s)
        x = synth.say(UTTERANCES[u], v, s)
        clean.append((u, v, s, x))
        write(aid, x, utterance_id=u, text=UTTERANCES[u], voice=v, speed=s, noise="none", snr_db="",
              noise_seed="", generator=synth.engine, level_dbfs=-26, pad_s=0)
        print(f"  tts {k}/{len(plan)} {aid}", flush=True)
    for u, v, s, x in clean:
        if u not in NOISY_BASE:
            continue
        aid = audio_id(u, v, s)
        for kind in NOISE_TYPES:
            for snr in SNRS:
                seed = int(hashlib.sha256(f"{aid}|{kind}|{snr}|{SEED}".encode()).hexdigest()[:8], 16)
                y, _ = mix_at_snr(x, kind, snr, seed, PAD_S)
                write(audio_id(u, v, s, kind, snr), y, utterance_id=u, text=UTTERANCES[u], voice=v, speed=s,
                      noise=kind, snr_db=snr, noise_seed=seed, generator=f"{synth.engine} + harness.noise",
                      level_dbfs=-26, pad_s=PAD_S)


# --------------------------------------------------------------------------- writing / freezing
def write_csv(p: Path, rows: list[dict]) -> None:
    keys = list(rows[0].keys()) if rows else []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def freeze(out: Path, version_info: dict) -> str:
    """CHECKSUMS.sha256 over every golden file except the audio (which has its own SHA in
    audio_manifest.csv) and the checksum file itself; the dataset checksum is the SHA-256 of
    CHECKSUMS.sha256 + audio_manifest.csv."""
    lines = []
    for p in sorted(out.rglob("*")):
        if not p.is_file() or p.name in ("CHECKSUMS.sha256", "VERSION.yaml") or p.suffix in (".py", ".pyc", ".md"):
            continue                       # builder code and docs are not data
        rel = p.relative_to(out).as_posix()
        if rel.startswith("synthetic/audio/") or "__pycache__" in rel:
            continue
        lines.append(f"{sha256(p)}  {rel}")
    (out / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    h = hashlib.sha256((out / "CHECKSUMS.sha256").read_bytes() + (out / "audio_manifest.csv").read_bytes()).hexdigest()
    version_info["checksum"] = h
    (out / "VERSION.yaml").write_text(yaml.safe_dump(version_info, sort_keys=False), encoding="utf-8")
    lines.append(f"{sha256(out / 'VERSION.yaml')}  VERSION.yaml")
    (out / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return h


def clean_previous(out: Path, force: bool) -> int:
    """Refuse to overwrite a built golden dataset unless --force; returns the next version."""
    vf = out / "VERSION.yaml"
    if vf.is_file():
        old = yaml.safe_load(vf.read_text(encoding="utf-8"))
        if not force:
            raise SystemExit(f"golden dataset v{old.get('version')} already exists in {out}; "
                             "it is frozen. Use --force to rebuild (creates a new version).")
        for sub in ("gt", "scenarios", "synthetic"):
            shutil.rmtree(out / sub, ignore_errors=True)
        for f in ("manifest.csv", "audio_manifest.csv", "CHECKSUMS.sha256", "VERSION.yaml", "SUMMARY.md"):
            (out / f).unlink(missing_ok=True)
        return int(old.get("version", 0)) + 1
    for sub in ("gt", "scenarios", "synthetic"):     # leftovers of a failed, never-frozen build
        shutil.rmtree(out / sub, ignore_errors=True)
    for f in ("manifest.csv", "audio_manifest.csv", "CHECKSUMS.sha256", "SUMMARY.md"):
        (out / f).unlink(missing_ok=True)
    return 1


# --------------------------------------------------------------------------- scenarios
def case(case_id, scenario, title, split, clip_id, start_s, end_s, calibration, timeline=(), mode="scored",
         expected=(), judge=(), audio=None, score_exclude=None) -> dict:
    c = {"case_id": case_id, "scenario": scenario, "title": title, "mode": mode, "split": split,
         "clip": {"clip_id": clip_id, "start_s": round(start_s, 3), "end_s": None if end_s is None else round(end_s, 3)},
         "calibration": calibration, "timeline": list(timeline), "expected": list(expected)}
    if judge:
        c["judge"] = list(judge)
    if audio:
        c["audio"] = audio
    if score_exclude:
        c["score_exclude"] = score_exclude
    return c


def write_cases(out: Path, cases: list[dict]) -> None:
    d = out / "scenarios"
    d.mkdir(parents=True, exist_ok=True)
    for c in cases:
        (d / f"{c['case_id']}.yaml").write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8")


# --------------------------------------------------------------------------- inspect / NTHU
def inspect(ds: Path) -> None:
    files = sorted(p for p in ds.rglob("*") if p.is_file())
    print(f"{len(files)} files under {ds}")
    exts = Counter(p.suffix.lower() for p in files)
    print("extensions:", dict(exts))
    dirs = Counter(p.parent.relative_to(ds).as_posix() for p in files)
    for d, n in sorted(dirs.items())[:40]:
        print(f"  {n:4d}  {d}/")
    for p in files[:12]:
        print("  ", p.relative_to(ds).as_posix(), p.stat().st_size)
    for p in [f for f in files if f.suffix.lower() == ".txt"][:4]:
        t = p.read_text(errors="replace")
        print(f"--- {p.relative_to(ds)} ({len(t)} chars): {t[:200]!r}")
    vids = [p for p in files if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")]
    if vids:
        import cv2
        for p in vids[:3]:
            cap = cv2.VideoCapture(str(p))
            print(f"--- video {p.name}: fps={cap.get(5)} frames={cap.get(7)} size={cap.get(3)}x{cap.get(4)}")
            cap.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--fetch-tts", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--refreeze", action="store_true",
                    help="after editing/adding scenarios by hand: re-checksum and bump the version")
    ap.add_argument("--selftest", default=None, metavar="DIR")
    ap.add_argument("--dataset", default=str(DATASET_DIR))
    a = ap.parse_args()
    if a.inspect:
        inspect(Path(a.dataset))
    if a.fetch_tts:
        fetch_tts()
    if a.selftest:
        from selftest_golden import build_selftest
        build_selftest(Path(a.selftest))
    if a.build:
        from nthu import build_nthu
        build_nthu(Path(a.dataset), HERE, a.force)
    if a.refreeze:
        vf = HERE / "VERSION.yaml"
        if not vf.is_file():
            raise SystemExit("no golden dataset to refreeze; run --build first")
        info = yaml.safe_load(vf.read_text(encoding="utf-8"))
        old_v = info.get("version", 0)
        info["version"] = int(old_v) + 1
        info["refrozen_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        n_cases = len(list((HERE / "scenarios").glob("*.yaml")))
        info["cases"] = n_cases
        h = freeze(HERE, info)
        print(f"golden refrozen: v{old_v} -> v{info['version']}, {n_cases} cases, checksum {h[:12]}")
    if a.verify:
        from harness.golden import Golden
        g = Golden(HERE, REPO)
        probs = g.verify(True)
        print("OK: all checksums match" if not probs else "\n".join(probs))
        sys.exit(1 if probs else 0)


if __name__ == "__main__":
    main()
