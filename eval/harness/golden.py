"""Read-only access to the frozen golden dataset (eval/golden). Verifies checksums before use;
never regenerates anything (that is build_golden.py's explicit job)."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

GOLDEN = Path(__file__).resolve().parents[1] / "golden"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Golden:
    def __init__(self, root: Path = GOLDEN, repo: Path | None = None):
        self.root = root
        self.repo = repo or root.parents[1]
        if not (root / "VERSION.yaml").is_file():
            raise FileNotFoundError(f"golden dataset not built: run python eval/golden/build_golden.py --build")
        self.version = yaml.safe_load((root / "VERSION.yaml").read_text(encoding="utf-8"))
        with open(root / "manifest.csv", newline="", encoding="utf-8") as fh:
            self.clips = {r["clip_id"]: r for r in csv.DictReader(fh)}
        with open(root / "audio_manifest.csv", newline="", encoding="utf-8") as fh:
            self.audio = {r["audio_id"]: r for r in csv.DictReader(fh)}
        self.cases = {}
        for p in sorted((root / "scenarios").glob("*.yaml")):
            c = yaml.safe_load(p.read_text(encoding="utf-8"))
            self.cases[c["case_id"]] = c
        self._gt: dict[str, dict] = {}

    def verify(self, check_media: bool = True) -> list[str]:
        """Compare every file listed in CHECKSUMS.sha256 (golden files) and, optionally, the
        source videos and synthetic audio against their recorded SHA-256. Returns problems."""
        problems = []
        for line in (self.root / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, rel = line.split("  ", 1)
            p = self.root / rel
            if not p.is_file():
                problems.append(f"missing golden file {rel}")
            elif sha256(p) != digest:
                problems.append(f"golden file changed: {rel}")
        if check_media:
            for cid, r in self.clips.items():
                p = self.repo / r["source_path"]
                if not p.is_file():
                    problems.append(f"missing source video for {cid}: {r['source_path']}")
                elif sha256(p) != r["sha256"]:
                    problems.append(f"source video changed for {cid}")
            for aid, r in self.audio.items():
                p = self.root / r["path"]
                if not p.is_file():
                    problems.append(f"missing synthetic audio {aid}")
                elif sha256(p) != r["sha256"]:
                    problems.append(f"synthetic audio changed: {aid}")
        return problems

    def clip(self, clip_id: str) -> dict:
        r = dict(self.clips[clip_id])
        r["abs_path"] = str((self.repo / r["source_path"]).resolve())
        for k in ("fps", "duration_s"):
            r[k] = float(r[k])
        for k in ("width", "height", "frames"):
            r[k] = int(float(r[k]))
        return r

    def gt(self, clip_id: str) -> dict:
        if clip_id not in self._gt:
            self._gt[clip_id] = json.loads((self.root / "gt" / f"{clip_id}.json").read_text(encoding="utf-8"))
        return self._gt[clip_id]

    def load_audio(self, audio_id: str) -> np.ndarray:
        r = self.audio[audio_id]
        x, sr = sf.read(str(self.root / r["path"]), dtype="float32", always_2d=True)
        if sr != 16000:
            raise ValueError(f"{audio_id}: expected 16 kHz, got {sr}")
        return x[:, 0]

    def audio_meta(self) -> dict:
        out = {}
        for aid, r in self.audio.items():
            m = dict(r)
            m["snr_key"] = "clean" if r["noise"] in ("", "none") else str(int(float(r["snr_db"])))
            out[aid] = m
        return out
