"""Parser for the NTHU-DDD testing set exactly as it is in Dataset/Testing_Dataset (inspected
2026-11-27, see golden/README.md):

  <subject>_<condition>_mix.mp4                     70 videos, 14 subjects x 5 conditions,
                                                    H.264 640x480 @ 29.97 fps
  test_label_txt/<condition dir>/<subject>_<condition>_mixing_drowsiness.txt
  test_label_txt/wh/<subject>_<condition>_mixing_drowsiness.txt      (a flat copy of the same)

conditions: glasses, nightglasses, nightnoglasses, noglasses, sunglasses (the night_noglasses
folder spells its files 'night_noglasses'). Each label file is one character per frame,
'0' = non-drowsy, '1' = drowsy, plus a trailing newline. Only drowsiness is labelled in this
set (no eye / head / mouth labels). Every video has exactly 2 more frames than labels: we take
label i = frame i and leave the trailing frames unlabelled (None, never scored), and flag it.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

CONDITIONS = ("glasses", "nightglasses", "nightnoglasses", "noglasses", "sunglasses")
FOLDER = {"glasses": "glasses", "nightglasses": "night_glasses", "nightnoglasses": "night_noglasses",
          "noglasses": "noglasses", "sunglasses": "sunglasses"}
FILE_COND = {"nightnoglasses": ("nightnoglasses", "night_noglasses")}   # spellings seen in the folders


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _probe(p: Path) -> tuple[float, int, int, int]:
    """(fps, exact frame count, width, height) with ffprobe (counts packets: exact)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=avg_frame_rate,nb_read_packets,width,height", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True, check=True).stdout.strip().split(",")
    w, h, rate, n = int(out[0]), int(out[1]), out[2], int(out[3])
    a, b = rate.split("/")
    return round(int(a) / int(b), 4), n, w, h


def _read_labels(p: Path) -> str:
    return p.read_text(encoding="ascii").strip()


def discover_clips(ds: Path, repo: Path) -> list[dict]:
    lab_root = ds / "test_label_txt"
    clips = []
    for v in sorted(ds.glob("*_mix.mp4")):
        m = re.fullmatch(r"(\d{3})_([a-z]+)_mix\.mp4", v.name)
        if not m:
            raise SystemExit(f"unexpected video name {v.name}")
        subj, cond = m.groups()
        if cond not in CONDITIONS:
            raise SystemExit(f"unknown condition in {v.name}")
        flags: list[str] = []
        primary = None
        for spelling in FILE_COND.get(cond, (cond,)):
            p = lab_root / FOLDER[cond] / f"{subj}_{spelling}_mixing_drowsiness.txt"
            if p.is_file():
                primary = p
                break
        flat = lab_root / "wh" / f"{subj}_{cond}_mixing_drowsiness.txt"
        if primary is None and flat.is_file():
            primary = flat
            flags.append("label only in test_label_txt/wh")
        if primary is None:
            flags.append("NO LABEL FILE")
            text = ""
        else:
            text = _read_labels(primary)
            if flat.is_file() and flat != primary and _read_labels(flat) != text:
                flags.append("label copies in test_label_txt/wh and the condition folder DIFFER (condition folder used)")
        bad = set(text) - {"0", "1"}
        if bad:
            flags.append(f"unexpected label characters {sorted(bad)} (treated as unlabelled)")
        fps, frames, w, h = _probe(v)
        labels: list = [int(c) if c in "01" else None for c in text]
        if len(labels) < frames:
            flags.append(f"video has {frames - len(labels)} more frame(s) than labels: label i = frame i, "
                         "trailing frames unlabelled")
            labels += [None] * (frames - len(labels))
        elif len(labels) > frames:
            flags.append(f"labels have {len(labels) - frames} more entries than video frames: extra labels dropped")
            labels = labels[:frames]
        clips.append(dict(clip_id=f"{subj}_{cond}", source_path=v.resolve().relative_to(repo.resolve()).as_posix(),
                          sha256=_sha256(v), fps=fps, frames=frames, width=w, height=h, subject=subj, scenario=cond,
                          labels=labels, other_labels={},
                          label_files=[primary.relative_to(repo).as_posix()] if primary else [], flags=flags))
    return clips
