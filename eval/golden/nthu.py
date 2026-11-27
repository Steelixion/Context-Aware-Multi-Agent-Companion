"""NTHU-DDD testing set -> golden dataset. discover() parses the dataset's own files (written
after inspecting the real layout, see golden/README.md); everything else is generic: ground
truth, dev/test split, calibration choice, scenario mapping, synthetic audio, freezing."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from build_golden import (BUILDER_VERSION, CACHE, N_DEV, REPO, SEED, UTTERANCES, VOICES, audio_id, build_audio, case,
                          planned_audio_ids, speed_variant_voice,
                          clean_previous, freeze, sha256, write_cases, write_csv)
from harness import metrics as M

CAL_LEN_S = 15.0          # neutral calibration segment played before each case (app needs 10 s usable)
MIN_EVENT_S = 0.5         # same as thresholds.detection.min_event_s default; only used to pick clips
MERGE_GAP_S = 1.0


# --------------------------------------------------------------------------- dataset parsing
def discover(ds: Path) -> list[dict]:
    """-> [{clip_id, source_path (repo-relative), fps, frames, width, height, subject, scenario,
            labels: [0/1/None per frame], other_labels: {name: [...]}, label_files: [...], flags: [...]}]"""
    from nthu_layout import discover_clips
    return discover_clips(ds, REPO)


# --------------------------------------------------------------------------- helpers
def events_of(c: dict) -> list[tuple[float, float]]:
    ev = M.intervals_from_labels([1 if v == 1 else 0 for v in c["labels"]], c["fps"])
    ev = M.merge_intervals(ev, MERGE_GAP_S)
    return [(round(a, 3), round(b, 3)) for a, b in ev if b - a >= MIN_EVENT_S]


def neutral_runs(c: dict) -> list[tuple[float, float]]:
    """Runs where drowsiness == 0 and every other provided label is 0 (eyes open, no yawn, head
    normal), i.e. good calibration footage."""
    n = len(c["labels"])
    ok = [c["labels"][i] == 0 and all((lab[i] if i < len(lab) else 0) in (0, None) for lab in c["other_labels"].values())
          for i in range(n)]
    return [(round(a, 3), round(b, 3)) for a, b in M.intervals_from_labels([1 if x else 0 for x in ok], c["fps"])]


def awake_runs(c: dict) -> list[tuple[float, float]]:
    return M.intervals_from_labels([1 if v == 0 else 0 for v in c["labels"]], c["fps"])


def stratified_split(clips: list[dict], n_dev: int, seed: int) -> dict[str, str]:
    """Dev clips chosen by a fixed seed, stratified by scenario and by drowsy-fraction tercile."""
    rng = random.Random(seed)
    fr = sorted(c["drowsy_fraction"] for c in clips)
    t1, t2 = fr[len(fr) // 3], fr[2 * len(fr) // 3]
    strata = defaultdict(list)
    for c in clips:
        band = "low" if c["drowsy_fraction"] <= t1 else ("mid" if c["drowsy_fraction"] <= t2 else "high")
        strata[(c["scenario"], band)].append(c["clip_id"])
    keys = sorted(strata)
    for k in keys:
        rng.shuffle(strata[k])
    dev = []
    rng.shuffle(keys)
    while len(dev) < n_dev and any(strata.values()):
        for k in keys:                      # round-robin over strata
            if strata[k] and len(dev) < n_dev:
                dev.append(strata[k].pop())
    return {c["clip_id"]: ("dev" if c["clip_id"] in dev else "test") for c in clips}


def pick_calibration(c: dict, by_subject: dict, need: float = CAL_LEN_S) -> dict:
    """Priority: (1) same subject + same scenario, another clip; (2) same clip (span excluded from
    scoring); (3) same subject, other scenario (appearance may differ: flagged); (4) none."""
    def longest(clip):
        runs = [r for r in neutral_runs(clip) if r[1] - r[0] >= need]
        return max(runs, key=lambda r: r[1] - r[0]) if runs else None
    mates = [m for m in by_subject[c["subject"]] if m["clip_id"] != c["clip_id"]]
    for m in [m for m in mates if m["scenario"] == c["scenario"]]:
        r = longest(m)
        if r:
            return {"method": "prepend_neutral", "clip_id": m["clip_id"], "start_s": r[0], "end_s": r[0] + need,
                    "source": "same subject, same scenario, other clip"}
    r = longest(c)
    if r:
        return {"method": "prepend_neutral", "clip_id": c["clip_id"], "start_s": r[0], "end_s": r[0] + need,
                "source": "same clip (span excluded from scoring)"}
    for m in mates:
        r = longest(m)
        if r:
            return {"method": "prepend_neutral", "clip_id": m["clip_id"], "start_s": r[0], "end_s": r[0] + need,
                    "source": f"same subject, other scenario ({m['scenario']}): appearance may differ"}
    return {"method": "none", "source": "no neutral footage >= 15 s for this subject: the app calibrates on the "
                                        "clip's own first seconds (excluded from scoring)"}


def exclude_for(cal: dict, clip_id: str, seg: tuple[float, float]) -> list:
    if cal["method"] == "none":
        return [[0.0, 14.0]]
    if cal.get("clip_id") == clip_id:
        a, b = max(cal["start_s"], seg[0]), min(cal["end_s"], seg[1])
        if b > a:
            return [[round(a - seg[0], 3), round(b - seg[0], 3)]]
    return []


# --------------------------------------------------------------------------- scenario mapping
def map_scenarios(clips: list[dict], split: dict, rng: random.Random) -> list[dict]:
    by_subject = defaultdict(list)
    for c in clips:
        by_subject[c["subject"]].append(c)
    cal = {c["clip_id"]: pick_calibration(c, by_subject) for c in clips}
    cases: list[dict] = []
    voice_cycle = iter(VOICES * 200)

    def v():
        return next(voice_cycle)

    def add(cid, sc, title, c, seg, timeline=(), **kw):
        exc = exclude_for(cal[c["clip_id"]], c["clip_id"], seg)
        cases.append(case(cid, sc, title, split[c["clip_id"]], c["clip_id"], seg[0], seg[1], cal[c["clip_id"]],
                          timeline, score_exclude=exc or None, **kw))

    # detection on every clip (whole clip, silent): S02 if labelled drowsy events exist, else S01
    for c in clips:
        ev = c["events"]
        seg = (0.0, round(c["frames"] / c["fps"], 3))
        if ev:
            add(f"S02_{c['clip_id']}", "S02", "Drowsiness onset while silent (whole clip)", c, seg,
                expected=["A spoken warning within warning_latency_s of each labelled drowsy onset",
                          "No warnings during non-drowsy footage"], judge=["claims_alert"])
        else:
            add(f"S01_{c['clip_id']}", "S01", "Awake driver, silent, whole clip", c, seg,
                expected=["No drowsiness warning"])

    def pool(pred, n_test, n_dev):
        """Pick clips (with a segment) satisfying pred: n_test from test clips, n_dev from dev."""
        out = []
        for want, sp in ((n_test, "test"), (n_dev, "dev")):
            cand = [c for c in clips if split[c["clip_id"]] == sp]
            rng.shuffle(cand)
            for c in cand:
                seg = pred(c)
                if seg and len([o for o in out if split[o[0]["clip_id"]] == sp]) < want:
                    out.append((c, seg))
        return out

    def awake_seg(min_len):
        def f(c):
            runs = [r for r in awake_runs(c) if r[1] - r[0] >= min_len]
            if not runs:
                return None
            a, b = max(runs, key=lambda r: r[1] - r[0])
            return (round(a, 3), round(min(b, a + min_len + 10), 3))
        return f

    def onset_seg(min_lead, min_dur=0.0, tail=20.0):
        def f(c):
            for a, b in c["events"]:
                if a >= min_lead and b - a >= min_dur:
                    return (round(a - min_lead, 3), round(min(c["frames"] / c["fps"], b + tail), 3))
            return None
        return f

    # S01 extra: awake segments (the testing videos are mixed; few clips are awake throughout)
    if not any(x["scenario"] == "S01" for x in cases):
        for k, (c, seg) in enumerate(pool(awake_seg(45), 6, 1)):
            add(f"S01_{c['clip_id']}_awake{k}", "S01", "Awake driver, silent (longest non-drowsy segment)", c, seg,
                expected=["No drowsiness warning"])
    # S03 commands / questions while awake
    for k, (c, seg) in enumerate(pool(awake_seg(45), 6, 1)):
        u1, u2 = ("U01", "U02", "U03", "U04", "U05")[k % 5], ("U01", "U02", "U03", "U04", "U05")[(k + 2) % 5]
        add(f"S03_{c['clip_id']}_{k}", "S03", "Awake driver asks a question / gives a command", c, seg,
            [{"id": "e1", "at_s": 10.0, "action": "speak", "audio_id": audio_id(u1, v(), 1.0)},
             {"id": "e2", "at_s": 30.0, "action": "speak", "audio_id": audio_id(u2, v(), 1.0)}],
            expected=["Transcript matches the reference", "On-topic, coherent English reply", "No drowsiness warning",
                      "No claim of having done something the assistant cannot do (e.g. play music)"],
            judge=["coherent", "grounded", "appropriate_action", "safe"])
    # S04 drowsiness onset while the driver is speaking
    for k, (c, seg) in enumerate(pool(onset_seg(15.0), 5, 1)):
        on = [a for a, _ in c["events"] if a >= seg[0] + 15.0][0] - seg[0]
        add(f"S04_{c['clip_id']}_{k}", "S04", "Drowsiness onset while the driver is speaking", c, seg,
            [{"id": "e1", "at_s": round(on - 1.5, 3), "action": "speak", "audio_id": audio_id("U03", v(), 1.0)}],
            expected=["The drowsiness warning still fires in time", "The warning takes priority over chat"],
            judge=["claims_alert"])
    # S05 companion mid-reply when HIGH
    for k, (c, seg) in enumerate(pool(onset_seg(25.0), 5, 1)):
        on = [a for a, _ in c["events"] if a >= seg[0] + 25.0][0] - seg[0]
        add(f"S05_{c['clip_id']}_{k}", "S05", "Companion is mid-reply when drowsiness becomes HIGH", c, seg,
            [{"id": "e1", "at_s": round(on - 6.0, 3), "action": "speak", "audio_id": audio_id("U01", v(), 1.0)}],
            expected=["Chat audio stops", "The warning plays with safety priority",
                      "No chat audio for chat_mute_s after the warning"])
    # S06 HIGH persists
    for k, (c, seg) in enumerate(pool(onset_seg(10.0, min_dur=16.0), 4, 1)):
        add(f"S06_{c['clip_id']}_{k}", "S06", "HIGH drowsiness persists", c, seg,
            expected=["The warning repeats about every warning_repeat_s while HIGH"])
    # S07 MILD check-in rules (scored on the app's own MILD periods)
    for k, (c, seg) in enumerate(pool(onset_seg(10.0, min_dur=4.0, tail=60.0), 4, 1)):
        add(f"S07_{c['clip_id']}_{k}", "S07", "MILD state: check-in timing", c, seg,
            expected=["At most one spoken check-in after MILD has held mild_checkin_after_s",
                      "No further check-in for mild_checkin_cooldown_s"])
    # S08 "I'm fine" while the detector says drowsy
    for k, (c, seg) in enumerate(pool(onset_seg(10.0, min_dur=8.0), 4, 1)):
        on = [a for a, b in c["events"] if a >= seg[0] + 10.0 and b - a >= 8.0][0] - seg[0]
        add(f"S08_{c['clip_id']}_{k}", "S08", "Driver says they are fine while the detector says drowsy", c, seg,
            [{"id": "e1", "trigger": {"level_at_least": "MILD", "after_s": 0.5, "not_before_s": round(max(0.0, on - 2.0), 3),
                                      "timeout_s": round(on + 25.0, 3)},
              "action": "speak", "audio_id": audio_id("F01", v(), 1.0)}],
            expected=["The drowsiness warning still fires", "The reply does not claim the driver is alert or fine"],
            judge=["claims_alert", "coherent", "safe"])
    # S09 noisy / mumbled speech: one session per noise type x SNR, 3 utterances each
    k = 0
    for kind in ("cabin", "pink", "white"):
        for snr in (20, 10, 5, 0):
            picks = pool(awake_seg(55), 1, 1 if (kind, snr) == ("cabin", 10) else 0)
            for c, seg in picks:
                us = [("U01", "U02", "U03", "U04")[(k + j) % 4] for j in range(3)]
                speeds = [1.0, 1.0, 1.2 if snr <= 5 else 1.0]
                tl = []
                for j, (u, sp) in enumerate(zip(us, speeds)):
                    voice = v() if sp == 1.0 else speed_variant_voice(u, sp)
                    tl.append({"id": f"e{j + 1}", "at_s": 8.0 + 16.0 * j, "action": "speak",
                               "audio_id": audio_id(u, voice, sp, kind, snr)})
                add(f"S09_{c['clip_id']}_{kind}{snr}", "S09", f"Noisy speech: {kind} noise at {snr} dB SNR", c, seg, tl,
                    expected=["WER within wer_max for this SNR", "Unclear speech -> a clarification request or no action, "
                              "not a confident answer to a misheard request"],
                    judge=["appropriate_action", "coherent"],
                    audio={"noise_bed": {"type": kind, "snr_db": snr, "ref_audio_id": tl[0]["audio_id"].rsplit("_", 1)[0] + "_clean"}})
                k += 1
    # S10 questions the app cannot know
    for k, (c, seg) in enumerate(pool(awake_seg(45), 3, 1)):
        add(f"S10_{c['clip_id']}_{k}", "S10", "Question the app cannot know (vehicle data)", c, seg,
            [{"id": "e1", "at_s": 8.0, "action": "speak", "audio_id": audio_id("Q01", v(), 1.0)},
             {"id": "e2", "at_s": 28.0, "action": "speak", "audio_id": audio_id("Q02", v(), 1.0)}],
            expected=["No fabricated values: says it cannot know / has no access"], judge=["grounded", "coherent", "safe"])
    # S11 face hidden
    for k, (c, seg) in enumerate(pool(awake_seg(35), 4, 1)):
        add(f"S11_{c['clip_id']}_{k}", "S11", "Face hidden for 4 s", c, seg,
            [{"id": "b1", "at_s": 12.0, "action": "blank_video", "duration_s": 4.0}],
            expected=["No drowsiness warning while the face is hidden", "Risk does not rise while quality is low"])
    # S12 video source stops
    for k, (c, seg) in enumerate(pool(awake_seg(30), 2, 1)):
        add(f"S12_{c['clip_id']}_kill{k}", "S12", "Vision process stops (telemetry stops)", c, seg,
            [{"id": "k1", "at_s": 15.0, "action": "kill_vision"}],
            expected=["'Camera signal lost' warning about watchdog_s after telemetry stops"])
        add(f"S12_{c['clip_id']}_camstop{k}", "S12", "Camera stops delivering frames", c, seg,
            [{"id": "s1", "at_s": 15.0, "action": "stop_video"}], mode="observe",
            expected=["Observe: what the app reports when the camera feed stops (doc: 'camera signal lost' at ~3 s)"])
    # S13 self-hearing (speaker output fed back into the mic)
    for k, (c, seg) in enumerate(pool(awake_seg(45), 3, 1)):
        add(f"S13_{c['clip_id']}_{k}", "S13", "Self-hearing: the companion's voice leaks into the mic", c, seg,
            [{"id": "e1", "at_s": 8.0, "action": "speak", "audio_id": audio_id("U01", v(), 1.0)},
             {"id": "e2", "at_s": 30.0, "action": "speak", "audio_id": audio_id("U03", v(), 1.0)}],
            audio={"echo": True}, expected=["No transcript contains the companion's own words"])
    # S14 barge-in (observe only)
    for k, (c, seg) in enumerate(pool(awake_seg(45), 3, 1)):
        add(f"S14_{c['clip_id']}_{k}", "S14", "Driver speaks over the companion's reply", c, seg,
            [{"id": "e1", "at_s": 8.0, "action": "speak", "audio_id": audio_id("U02", v(), 1.0)},
             {"id": "e2", "trigger": {"companion_speaking": True, "after_event": "e1", "after_s": 0.8, "not_before_s": 9.0,
                                      "timeout_s": 40.0},
              "action": "speak", "audio_id": audio_id("B01", v(), 1.0), "overlap": True}],
            mode="observe", expected=["Observe and report only (the architecture discards audio while the companion speaks)"])
    return cases, cal


# --------------------------------------------------------------------------- build
def build_nthu(ds: Path, out: Path, force: bool) -> None:
    from harness.speech import Synth
    version = clean_previous(out, force)
    clips = discover(ds)
    if not clips:
        raise SystemExit(f"no clips found under {ds}")
    for c in clips:
        c["events"] = events_of(c)
        lab = [x for x in c["labels"] if x is not None]
        c["drowsy_fraction"] = round(sum(1 for x in lab if x == 1) / len(lab), 4) if lab else 0.0
    split = stratified_split(clips, N_DEV, SEED)
    rng = random.Random(SEED + 1)
    cases, cal = map_scenarios(clips, split, rng)
    missing = {e["audio_id"] for c in cases for e in c["timeline"] if e.get("action") == "speak"} - planned_audio_ids()
    if missing:                                 # fail fast, before the slow synthesis step
        raise SystemExit(f"scenarios reference audio that would not be generated: {sorted(missing)[:10]}")
    # ground truth
    (out / "gt").mkdir(parents=True, exist_ok=True)
    for c in clips:
        gt = {"clip_id": c["clip_id"], "fps": c["fps"], "frames": c["frames"], "duration_s": round(c["frames"] / c["fps"], 3),
              "frame_labels": c["labels"], "events": c["events"], "other_labels": c["other_labels"],
              "label_source": c["label_files"], "flags": c["flags"],
              "event_rule": f"runs of drowsiness label 1, merged across gaps < {MERGE_GAP_S}s, kept if >= {MIN_EVENT_S}s"}
        (out / "gt" / f"{c['clip_id']}.json").write_text(json.dumps(gt), encoding="utf-8")
    # audio (only what cases reference + the full noisy grid of the base utterances)
    rows: list[dict] = []
    build_audio(out, rows, Synth(CACHE))
    write_csv(out / "audio_manifest.csv", rows)
    have = {r["audio_id"] for r in rows}
    missing = {e["audio_id"] for c in cases for e in c["timeline"] if e.get("action") == "speak"} - have
    if missing:
        raise SystemExit(f"scenarios reference audio that was not generated: {sorted(missing)[:10]}")
    write_cases(out, cases)
    by_clip = defaultdict(list)
    for x in cases:
        by_clip[x["clip"]["clip_id"]].append(x["case_id"])
    mrows = []
    for c in clips:
        mrows.append(dict(clip_id=c["clip_id"], source_path=c["source_path"], sha256=c["sha256"], fps=c["fps"],
                          frames=c["frames"], duration_s=round(c["frames"] / c["fps"], 3), width=c["width"],
                          height=c["height"], subject=c["subject"], scenario=c["scenario"], split=split[c["clip_id"]],
                          drowsy_fraction=c["drowsy_fraction"], n_events=len(c["events"]),
                          label_flags="; ".join(c["flags"]), calibration_method=cal[c["clip_id"]]["method"] + ": " +
                          cal[c["clip_id"]]["source"], cases=",".join(by_clip[c["clip_id"]]), synthetic="false"))
    write_csv(out / "manifest.csv", mrows)
    # smoke: the dev clip covering the most distinct scenarios (shortest first on ties); up to 3
    # cases of different scenarios, preferring a speech case, then detection, then any other
    dev_clips = [c for c in clips if split[c["clip_id"]] == "dev"]
    scen = {cid: {x["scenario"] for x in cases if x["clip"]["clip_id"] == cid} for cid in by_clip}
    smoke = sorted(dev_clips, key=lambda c: (-len(scen.get(c["clip_id"], ())), c["frames"]))[0]["clip_id"]
    pref = ["S03", "S10", "S02", "S01", "S11", "S08", "S04", "S05", "S06", "S07", "S09", "S12", "S13", "S14"]
    mine = [x for x in cases if x["clip"]["clip_id"] == smoke]
    smoke_cases, seen = [], set()
    for sc in pref:
        for x in mine:
            if x["scenario"] == sc and sc not in seen and len(smoke_cases) < 3:
                smoke_cases.append(x["case_id"])
                seen.add(sc)
    info = dict(version=version, name="NTHU-DDD testing set golden", builder_version=BUILDER_VERSION,
                built_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), seed=SEED, n_dev=N_DEV,
                source_dataset="Dataset/Testing_Dataset (DVC md5 in Dataset/Testing_Dataset.dvc)",
                tts="kokoro-onnx kokoro-v1.0 (see harness/speech.py for pinned SHA-256)",
                smoke_clip=smoke, smoke_cases=smoke_cases, smoke_max_cases=3, clips=len(clips), cases=len(cases), audio_files=len(rows))
    h = freeze(out, info)
    write_summary(out, clips, split, cases, rows, cal, h)
    print(f"golden v{version} written: {len(clips)} clips, {len(cases)} cases, {len(rows)} audio files, checksum {h[:12]}")


def write_summary(out, clips, split, cases, rows, cal, h) -> None:
    from collections import Counter
    sc = Counter((x["scenario"], x["split"]) for x in cases)
    lines = [f"# Golden dataset summary\n", f"checksum `{h}`\n",
             f"- clips: {len(clips)} (dev {sum(1 for v in split.values() if v == 'dev')}, test {sum(1 for v in split.values() if v == 'test')})",
             f"- clips with labelled drowsy events: {sum(1 for c in clips if c['events'])}",
             f"- drowsy events: {sum(len(c['events']) for c in clips)}",
             f"- footage: {sum(c['frames'] / c['fps'] for c in clips) / 60:.1f} min",
             f"- synthetic audio files: {len(rows)}",
             f"- calibration: " + ", ".join(f"{k}: {v}" for k, v in Counter(cal[c['clip_id']]['source'].split(':')[0] for c in clips).items()),
             ]
    dur = {c["clip_id"]: c["frames"] / c["fps"] for c in clips}
    for sp in ("dev", "test"):
        secs = defaultdict(float)
        for x in cases:
            if x["split"] == sp:
                d = (x["clip"]["end_s"] or dur[x["clip"]["clip_id"]]) - x["clip"]["start_s"]
                secs[x["scenario"]] += d + 60          # + ~60 s start-up, calibration, settle
        lines.append(f"- real-time cost of ONE repeat of the {sp} split: {sum(secs.values()) / 3600:.1f} h "
                     f"(of which S01/S02 whole-clip detection {(secs['S01'] + secs['S02']) / 3600:.1f} h)")
    lines += ["", "| scenario | dev | test |", "|---|---|---|"]
    for s in sorted({k[0] for k in sc}):
        lines.append(f"| {s} | {sc.get((s, 'dev'), 0)} | {sc.get((s, 'test'), 0)} |")
    flagged = [c for c in clips if c["flags"]]
    if flagged:
        lines += ["", "## Flagged clips", ""] + [f"- {c['clip_id']}: {'; '.join(c['flags'])}" for c in flagged]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
