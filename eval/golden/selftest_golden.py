"""A tiny synthetic golden dataset for checking the harness itself without the NTHU videos.
Its clip is a still photo rendered to video (no blinks: it is NOT a detection benchmark) and it is
marked synthetic everywhere. Use:  run_eval.py --golden <DIR> --smoke"""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from build_golden import (CACHE, REPO, UTTERANCES, audio_id, case, freeze, sha256, write_cases, write_csv)
from harness.noise import mix_at_snr
from harness.speech import Synth


def build_selftest(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    media = out / "media"
    media.mkdir(exist_ok=True)
    clip = media / "selftest_portrait_45s.mp4"
    if not clip.is_file():
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-i",
                        str(REPO / "models" / "test-assets" / "portrait.jpg"), "-vf", "scale=640:480,format=yuv420p",
                        "-r", "30", "-t", "45", str(clip)], check=True)
    import cv2
    cap = cv2.VideoCapture(str(clip))
    fps, frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    cid = "selftest01"
    clips = [dict(clip_id=cid, source_path=clip.resolve().as_posix(), sha256=sha256(clip), fps=fps, frames=frames,
                  duration_s=round(frames / fps, 3), width=w, height=h, subject="synthetic", scenario="still photo",
                  split="dev", drowsy_fraction=0.0, n_events=0, label_flags="synthetic clip, all frames labelled non-drowsy",
                  calibration_method="prepend_neutral (same clip 0-12 s)", cases="", synthetic="true")]
    (out / "gt").mkdir(exist_ok=True)
    gt = dict(clip_id=cid, fps=fps, frames=frames, duration_s=round(frames / fps, 3), frame_labels=[0] * frames,
              events=[], other_labels={}, label_source="synthetic (selftest)", flags=["synthetic"])
    (out / "gt" / f"{cid}.json").write_text(__import__("json").dumps(gt), encoding="utf-8")

    synth = Synth(CACHE)
    rows = []
    adir = out / "synthetic" / "audio"
    adir.mkdir(parents=True, exist_ok=True)
    for u in ("U02", "Q01"):
        x = synth.say(UTTERANCES[u], "am_michael", 1.0)
        aid = audio_id(u, "am_michael", 1.0)
        p = adir / f"{aid}.wav"
        sf.write(str(p), x, 16000, subtype="PCM_16")
        rows.append(dict(audio_id=aid, path=p.relative_to(out).as_posix(), sha256=sha256(p), duration_s=round(len(x) / 16000, 3),
                         synthetic="true", utterance_id=u, text=UTTERANCES[u], voice="am_michael", speed=1.0,
                         noise="none", snr_db="", noise_seed="", generator=synth.engine, level_dbfs=-26, pad_s=0))
        if u == "U02":
            y, _ = mix_at_snr(x, "cabin", 10, 1234, 0.6)
            aid2 = audio_id(u, "am_michael", 1.0, "cabin", 10)
            p2 = adir / f"{aid2}.wav"
            sf.write(str(p2), y, 16000, subtype="PCM_16")
            rows.append(dict(rows[-1] | dict(audio_id=aid2, path=p2.relative_to(out).as_posix(), sha256=sha256(p2),
                                              duration_s=round(len(y) / 16000, 3), noise="cabin", snr_db=10,
                                              noise_seed=1234, pad_s=0.6)))
    write_csv(out / "audio_manifest.csv", rows)
    cal = {"method": "prepend_neutral", "clip_id": cid, "start_s": 0.0, "end_s": 12.0,
           "note": "selftest: same still clip"}
    a = audio_id("U02", "am_michael", 1.0)
    q = audio_id("Q01", "am_michael", 1.0)
    cases = [
        case("ST_S03", "S03", "Self-test: awake driver asks two questions", "dev", cid, 0, 40, cal,
             [{"id": "e1", "at_s": 6.0, "action": "speak", "audio_id": a},
              {"id": "e2", "at_s": 22.0, "action": "speak", "audio_id": q}],
             expected=["On-topic reply to each question", "No drowsiness warning"],
             judge=["coherent", "grounded", "appropriate_action", "safe"]),
        case("ST_S11", "S11", "Self-test: face hidden for 4 s", "dev", cid, 0, 30, cal,
             [{"id": "b1", "at_s": 12.0, "action": "blank_video", "duration_s": 4.0}],
             expected=["No drowsiness warning while the face is hidden"]),
        case("ST_S12", "S12", "Self-test: vision process stops", "dev", cid, 0, 25, cal,
             [{"id": "k1", "at_s": 10.0, "action": "kill_vision"}],
             expected=["'Camera signal lost' warning about 3 s after telemetry stops"]),
    ]
    for c in cases:
        clips[0]["cases"] = ",".join(x["case_id"] for x in cases)
    write_csv(out / "manifest.csv", clips)
    write_cases(out, cases)
    info = dict(version=1, name="selftest (synthetic, not a benchmark)", built_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                builder="selftest_golden.py", seed=0, smoke_clip=cid, smoke_max_cases=3, synthetic_only=True)
    h = freeze(out, info)
    print(f"selftest golden written to {out} (checksum {h[:12]})")
