"""Run the whole evaluation against the running app and write the Excel report.

  python eval/run_eval.py --config eval/config/thresholds.yaml --split test --out results/<timestamp>/
  python eval/run_eval.py --smoke                      # one dev clip, one repeat, full report
  python eval/run_eval.py --only S03_xxx,S05_yyy       # selected cases
  python eval/run_eval.py --resume --out results/<run> # continue an interrupted run
  python eval/run_eval.py --rescore --out results/<run># re-score existing raw logs, no app runs
  python eval/run_eval.py --baseline results/<older>   # adds the Compare sheet

Cases that crash are logged and marked ERROR (never FAIL, never dropped); the run continues.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

EVAL = Path(__file__).resolve().parent
REPO = EVAL.parent
sys.path.insert(0, str(EVAL))

from harness import checks as C  # noqa: E402
from harness import metrics as M  # noqa: E402
from harness import report  # noqa: E402
from harness.events import load_events  # noqa: E402
from harness.golden import Golden  # noqa: E402
from harness.judge import NOT_CONFIGURED, Judge, JudgeConfig  # noqa: E402
from harness.metadata import collect, models_from_run  # noqa: E402
from harness.observe import build as build_obs, first_audio_after  # noqa: E402

log = logging.getLogger("eval")

AVAILABLE_DATA = ("The assistant can only know: the driver-monitoring state it is told (alert / mild / "
                  "high drowsiness, fatigue score, voice emotion), what the driver says, the "
                  "conversation so far and its stored memories of the driver. It has NO access to "
                  "vehicle sensors (fuel, tyre pressure, speed), GPS/location, maps, weather, the "
                  "clock, or the internet.")


# --------------------------------------------------------------------------- setup
def find_obs_camera_index(backend: str = "obs") -> tuple[int | None, str]:
    """Index of the OBS Virtual Camera for OpenCV's DirectShow backend, verified by sending a
    test pattern through pyvirtualcam and reading it back."""
    import cv2
    import numpy as np
    import pyvirtualcam
    order = []
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                             capture_output=True, text=True, timeout=20).stderr
        for line in out.splitlines():
            m = re.search(r'"([^"]+)" \((video|none)\)', line)
            if m:
                order.append(m.group(1))
    except Exception:  # noqa: BLE001
        pass
    cands = [i for i, n in enumerate(order) if "OBS" in n] + [i for i in range(6) if i not in
                                                               [j for j, n in enumerate(order) if "OBS" in n]]
    pattern = np.zeros((480, 640, 3), np.uint8)
    pattern[:, :, 1] = 200          # green (BGR)
    with pyvirtualcam.Camera(640, 480, 15, fmt=pyvirtualcam.PixelFormat.BGR, backend=backend) as cam:
        import threading
        stop = threading.Event()

        def pump():
            while not stop.is_set():
                cam.send(pattern)
                cam.sleep_until_next_frame()
        th = threading.Thread(target=pump, daemon=True)
        th.start()
        try:
            for i in cands:
                if i < len(order) and "OBS" not in order[i] and order:
                    continue                      # never open the physical webcam
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                ok, fr = False, None
                for _ in range(20):
                    ok, fr = cap.read()
                    if ok and fr is not None and fr[:, :, 1].mean() > 150 and fr[:, :, 2].mean() < 60:
                        cap.release()
                        return i, f"verified (device list: {order})"
                    time.sleep(0.05)
                cap.release()
        finally:
            stop.set()
            th.join(2)
    return None, f"OBS Virtual Camera not found (device list: {order})"


def llm_server_models(upstream: str) -> list[str] | None:
    import httpx
    try:
        r = httpx.get(upstream.rstrip("/") + "/models", timeout=5)
        return [m.get("id") for m in r.json().get("data", [])]
    except Exception:  # noqa: BLE001
        return None


def plan(golden: Golden, split: str, only: list[str] | None, smoke: bool) -> list[dict]:
    cases = list(golden.cases.values())
    if smoke and golden.version.get("smoke_cases"):
        return [golden.cases[c] for c in golden.version["smoke_cases"] if c in golden.cases]
    if smoke:
        sc = golden.version.get("smoke_clip")
        cases = [c for c in cases if c["clip"]["clip_id"] == sc and c["split"] == "dev"]
        return sorted(cases, key=lambda c: c["case_id"])[: golden.version.get("smoke_max_cases", 3)]
    if only:
        return [golden.cases[c] for c in only if c in golden.cases]
    if split != "all":
        cases = [c for c in cases if c["split"] == split]
    return sorted(cases, key=lambda c: c["case_id"])


# --------------------------------------------------------------------------- scoring
def score_run(case: dict, res: dict, golden: Golden, th: dict, run_dir: Path) -> dict:
    tag = f"{case['case_id']}__r{res['rep']}"
    out = {"case": case, "res": res, "checks": [], "obs": None, "status": res["status"]}
    if res["status"] != "DONE":
        out["checks"] = [C.chk("run", "ERROR", res.get("error"), "completed run", severity=3)]
        out["status"] = "ERROR"
        return out
    events = load_events(run_dir / "raw" / f"{tag}.jsonl")
    lb_path = run_dir / "raw" / f"{tag}.loopback.json"
    lb = json.loads(lb_path.read_text()) if lb_path.is_file() else None
    obs = build_obs(events, res["clip_t0"], res["clip_duration_s"], lb, th)
    gt = golden.gt(case["clip"]["clip_id"])
    try:
        out["checks"] = C.score(case, obs, gt, th, golden.audio_meta())
    except Exception as e:  # noqa: BLE001
        out["checks"] = [C.chk("scoring", "ERROR", repr(e), "scored", severity=3)]
        out["status"] = "ERROR"
        return out
    out["obs"] = obs
    out["status"] = C.case_status(out["checks"], "DONE")
    return out


def judge_items(case: dict, sr: dict, golden: Golden) -> list[tuple[str, str, dict]]:
    """(turn label, question_id, facts) for the judge; one case per call."""
    obs = sr["obs"]
    qs = case.get("judge") or []
    if obs is None or not qs:
        return []
    gt = golden.gt(case["clip"]["clip_id"])
    c0 = case["clip"].get("start_s", 0.0)
    gt_ev = C.gt_events(gt, {"detection": {"merge_event_gap_s": 1.0, "min_event_s": 0.0}}, c0, c0 + obs.duration)
    meta = golden.audio_meta()
    items = []
    sys_prompts = {d.get("rid"): (d.get("messages") or [{}])[0].get("content", "") for _, ty, d in obs.llm
                   if ty == "llm_request"}
    for u in obs.turns:
        if u.reply_text is None:
            continue
        lvl = C.level_at(obs, u.reply_t or u.speech_end)
        facts = {"scenario": case["title"], "expected_behaviour": case.get("expected", []),
                 "driver_actually_said": meta[u.audio_id]["text"], "speech_recognition_heard": u.transcript,
                 "detector_level_when_replying": lvl,
                 "ground_truth_driver_drowsy": M.in_any(u.speech_end, gt_ev),
                 "assistant_reply": u.reply_text, "assistant_can_know": AVAILABLE_DATA}
        rid = next((d.get("rid") for t, ty, d in obs.llm if ty == "llm_request" and u.llm_request_t == t), None)
        if rid in sys_prompts:
            facts["system_prompt_given_to_assistant"] = sys_prompts[rid][:3000]
        for q in qs:
            if q == "claims_alert" and lvl == "ALERT" and not facts["ground_truth_driver_drowsy"]:
                continue
            items.append((u.event_id or u.audio_id, q, facts))
    if "claims_alert" in qs:                          # the app's own spoken drowsiness warnings
        for ta in obs.drowsy_warnings:
            rep = [(t, d) for t, d in obs.chats if d.get("role") == "assistant" and d.get("final")
                   and ta <= t <= ta + 30 and d.get("text")]
            if rep:
                t, d = rep[0]
                items.append((f"warning@{ta:.1f}s", "claims_alert", {
                    "scenario": case["title"], "detector_level_when_replying": C.level_at(obs, t),
                    "ground_truth_driver_drowsy": True, "assistant_reply": d["text"],
                    "context": "This reply is what the assistant said right after its driver-monitoring "
                               "detected high drowsiness.", "assistant_can_know": AVAILABLE_DATA}))
    return items


# --------------------------------------------------------------------------- aggregation
def aggregate(scored: list[dict], golden: Golden, th: dict, judge_rows: list, judge_note: str,
              proxy_on: bool = True) -> dict:
    det_rows, safety_rows, asr_rows, lat_rows, case_rows, fail_rows = [], [], [], [], [], []
    conf = {"mild_or_high": M.Confusion(), "high_only": M.Confusion()}
    ev_total = ev_hit_high = ev_hit_mild = 0
    fa_total, nondrowsy_s = 0, 0.0
    delays_high, warn_lat, lat = [], [], {}
    lost_talkover = 0
    meta = golden.audio_meta()
    D = th["detection"]

    def add_lat(stage, v, source):
        if v is not None:
            lat.setdefault(stage, {"vals": [], "src": set()})
            lat[stage]["vals"].append(v)
            lat[stage]["src"].add(source)

    for sr in scored:
        case, res, obs = sr["case"], sr["res"], sr["obs"]
        cid, rep, sc = case["case_id"], res["rep"], case["scenario"]
        raw = res.get("raw", "")
        fails = [c for c in sr["checks"] if c["status"] in ("FAIL", "ERROR")]
        f0 = fails[0] if fails else (next((c for c in sr["checks"] if c["status"] == "INCONCLUSIVE"), None))
        case_rows.append([cid, rep, sc, case["split"], case["clip"]["clip_id"], sr["status"], len(fails),
                          f0["check"] if f0 else "", f0["observed"] if f0 else "", f0["expected"] if f0 else "",
                          res.get("error") or "", raw])
        for c in sr["checks"]:
            if c["status"] in ("FAIL", "ERROR", "INCONCLUSIVE"):
                fail_rows.append([c["severity"], cid, rep, sc, c["status"], c["check"], c["observed"], c["expected"],
                                  c["detail"], raw])
            if sc in ("S02", "S04", "S05", "S06", "S07", "S08", "S11", "S12", "S01"):
                safety_rows.append([cid, rep, sc, c["check"], c["status"], c["observed"], c["expected"], c["detail"]])
        if obs is None:
            continue
        gt = golden.gt(case["clip"]["clip_id"])
        c0 = case["clip"].get("start_s", 0.0)
        events = C.gt_events(gt, th, c0, c0 + obs.duration)
        # ---- detection (silent, unmodified clips only)
        if sc in ("S01", "S02"):
            fps = float(gt["fps"])
            f0i, f1i = int(round(c0 * fps)), int(round((c0 + obs.duration) * fps))
            labels = gt["frame_labels"][f0i:f1i]
            valid = [(0.0, obs.duration)]
            ex = case.get("score_exclude") or []
            if ex:
                valid = M.intervals_complement([(a, b) for a, b in ex], 0.0, obs.duration)
            hm = M.match_events(events, obs.level_changes, 2, D["event_early_s"], D["event_late_s"])
            mm = M.match_events(events, obs.level_changes, 1, D["event_early_s"], D["event_late_s"])
            nd = M.intervals_complement(events, 0.0, obs.duration)
            fa = M.warnings_outside(obs.drowsy_warnings, events, D["false_alarm_margin_before_s"], D["false_alarm_margin_after_s"])
            fa = [t for t in fa if 0 <= t <= obs.duration]
            ev_total += len(events)
            ev_hit_high += sum(e.detected for e in hm)
            ev_hit_mild += sum(e.detected for e in mm)
            fa_total += len(fa)
            nondrowsy_s += M.total_length(nd)
            delays_high += [e.delay for e in hm if e.delay is not None]
            for name, rank in (("mild_or_high", 1), ("high_only", 2)):
                cf = M.frame_confusion(labels, fps, obs.level_changes, rank, D["frame_lag_s"], valid)
                conf[name] = conf[name] + cf
                det_rows.append([cid, rep, case["clip"]["clip_id"], name, cf.tp, cf.fp, cf.fn, cf.tn, cf.precision,
                                 cf.recall, cf.f1, len(events), sum(e.detected for e in (hm if rank == 2 else mm)),
                                 sum(not e.detected for e in (hm if rank == 2 else mm)),
                                 [None if e.delay is None else round(e.delay, 2) for e in (hm if rank == 2 else mm)],
                                 len(fa), round(M.total_length(nd), 1), sr["status"]])
        # ---- warning latency (onset -> audible warning) for events with a warning
        if sc in ("S02", "S04"):
            for ev in events:
                ws = C.warnings_for_event(obs, ev, th)
                if ws:
                    tau, src = first_audio_after(obs, ws[0])
                    if tau is not None:
                        warn_lat.append(tau - ev[0])
                        add_lat("warning: GT onset -> warning audio", tau - ev[0], src)
                        add_lat("warning: decision (alert) -> warning audio", tau - ws[0], src)
                    hi = M.first_crossing(obs.level_changes, 2, ev[0] - D["event_early_s"], ev[1] + D["event_late_s"])
                    if hi is not None:
                        add_lat("warning: level HIGH -> decision (alert)", ws[0] - hi, "ws state + alert ts")
        # ---- speech turns
        def asr_status(u, wer, m):
            if not u.transcripts:
                return "INCONCLUSIVE (lost to companion talk-over)" if C.talked_over(obs, u) else "FAIL (no transcript)"
            return "PASS" if wer <= th["speech"]["wer_max"].get(m.get("snr_key", "clean"), 1e9) else "FAIL"

        for u in obs.turns:
            m = meta.get(u.audio_id, {})
            wer = M.word_error_rate(m.get("text", ""), u.transcript).wer if u.transcripts else None
            t_tr = u.transcripts[-1][0] if u.transcripts else None
            asr_rows.append([cid, rep, u.audio_id, m.get("voice"), m.get("speed"), m.get("noise") or "none",
                             m.get("snr_db") or "clean", m.get("text"), u.transcript if u.transcripts else "(none)",
                             wer, len(u.transcripts), None if t_tr is None else round(t_tr - u.speech_end, 3),
                             asr_status(u, wer, m)])
            if not u.transcripts and C.talked_over(obs, u):
                lost_talkover += 1
            turn = u.event_id or u.audio_id

            def row(stage, a, b, src):
                v = None if a is None or b is None else b - a
                lat_rows.append([cid, rep, turn, stage, v, src if v is not None else f"NOT OBSERVABLE ({src})"])
                add_lat(stage, v, src)
            row("speech end -> transcript", u.speech_end, t_tr, "harness audio clock -> ws speech event")
            if proxy_on:
                row("transcript -> LLM request", t_tr, u.llm_request_t, "LLM proxy")
                row("transcript -> first LLM token", t_tr, u.llm_first_token_t, "LLM proxy")
            else:
                for st in ("transcript -> LLM request", "transcript -> first LLM token"):
                    lat_rows.append([cid, rep, turn, st, None, "NOT OBSERVABLE (llm.proxy is off)"])
            row("transcript -> final reply text", t_tr, u.reply_t, "ws chat event")
            row("first LLM token -> first audio event", u.llm_first_token_t, u.first_audio_event_t, "LLM proxy -> ws audio event")
            row("first audio event -> audible", u.first_audio_event_t, u.first_audible_t, "ws audio event -> loopback")
            if u.first_audible_t is not None:
                row("END-TO-END speech end -> first audio", u.speech_end, u.first_audible_t, "harness audio clock -> loopback")
            else:
                row("END-TO-END speech end -> first audio", u.speech_end, u.first_audio_event_t,
                    "harness audio clock -> ws audio event (loopback unavailable)")
    # ---- tables
    lat_table = []
    for stage, d in sorted(lat.items()):
        s = M.latency_summary(d["vals"])
        lat_table.append([stage, s["n"], s["p50"], s["p95"], s["p99"], s["max"], "; ".join(sorted(d["src"]))])
    wer_by = {}
    for r in asr_rows:
        if r[8] == "(none)":
            continue
        k = str(r[6])
        wer_by.setdefault(k, []).append((r[7], r[8]))
    order = ["clean", "20", "10", "5", "0"]
    wer_by_snr = [(k, M.corpus_wer(wer_by[k])) for k in order if k in wer_by] + \
                 [(k, M.corpus_wer(v)) for k, v in wer_by.items() if k not in order]
    counts = {}
    for sr in scored:
        counts[sr["status"]] = counts.get(sr["status"], 0) + 1
    hours = nondrowsy_s / 3600
    claims = [r for r in judge_rows if r[3] == "claims_alert" and r[7] in ("yes", "no")]
    e2e = lat.get("END-TO-END speech end -> first audio", {"vals": [], "src": set()})
    headline = {
        "event_recall_high": (ev_hit_high / ev_total if ev_total else None, f"{ev_hit_high}/{ev_total} GT events reached HIGH"),
        "event_recall_mild_or_high": (ev_hit_mild / ev_total if ev_total else None, f"{ev_hit_mild}/{ev_total}"),
        "frame_f1_high": (conf["high_only"].f1, "frame-level, level == HIGH"),
        "frame_f1_mild_or_high": (conf["mild_or_high"].f1, "frame-level, level >= MILD"),
        "detection_delay_high_p50_s": (M.percentile(delays_high, 50), "first HIGH - GT onset"),
        "false_alarms_per_hour": (fa_total / hours if hours > 0 else None, f"{fa_total} in {hours:.2f} h of non-drowsy footage"),
        "warning_latency_p95_s": (M.percentile(warn_lat, 95), "GT onset -> warning audio"),
        "asr_wer_clean": (dict(wer_by_snr).get("clean"), "corpus WER, clean speech"),
        "asr_wer_overall": (M.corpus_wer([(r[7], r[8]) for r in asr_rows if r[8] != "(none)"]), "all SNRs"),
        "utterances_lost_to_talkover": (lost_talkover if asr_rows else None,
                                        f"of {len(asr_rows)} utterances: the companion spoke over them, the app muted its mic"),
        "e2e_first_audio_p95_s": (M.percentile(e2e["vals"], 95), "; ".join(sorted(e2e["src"])) or "no turns"),
        "judge_claims_alert_rate": ((sum(r[7] == "yes" for r in claims) / len(claims)) if claims else None,
                                    judge_note or f"{len(claims)} judged replies"),
    }
    gates = []
    for g in th.get("release_gates", []):
        actual = headline.get(g["metric"], (None, ""))[0]
        if actual is None:
            res = "NOT OBSERVABLE"
        else:
            ok = {"<=": actual <= g["value"], ">=": actual >= g["value"], "<": actual < g["value"],
                  ">": actual > g["value"], "==": actual == g["value"]}[g["op"]]
            res = "PASS" if ok else "FAIL"
        gates.append(dict(metric=g["metric"], op=g["op"], value=g["value"], actual=actual, result=res))
    fail_rows.sort(key=lambda r: (r[0], r[1], r[2]))
    return dict(detection_rows=det_rows, safety_rows=safety_rows, asr_rows=asr_rows, latency_rows=lat_rows,
                latency_table=lat_table, wer_by_snr=wer_by_snr, case_rows=case_rows, failure_rows=fail_rows,
                summary=dict(status_counts=counts, headline=headline, gates=gates,
                             top_problems=top_problems(scored, gates, headline)))


def top_problems(scored: list[dict], gates: list[dict], headline: dict) -> list[str]:
    """Plain-English problems computed from the data (not written by an LLM)."""
    groups: dict[tuple, dict] = {}
    for sr in scored:
        for c in sr["checks"]:
            if c["status"] not in ("FAIL", "ERROR"):
                continue
            base = re.sub(r"\[.*?\]", "", c["check"].split(":")[-1])
            k = (c["severity"], sr["case"]["scenario"], base, c["status"])
            g = groups.setdefault(k, {"n": 0, "obs": [], "cases": set()})
            g["n"] += 1
            g["cases"].add(sr["case"]["case_id"])
            if c["observed"] not in (None, ""):
                g["obs"].append(c["observed"])
    runs_per_sc: dict[str, int] = {}
    for sr in scored:
        runs_per_sc[sr["case"]["scenario"]] = runs_per_sc.get(sr["case"]["scenario"], 0) + 1
    out = []
    for (sev, sc, base, st), g in sorted(groups.items(), key=lambda kv: (kv[0][0], -kv[1]["n"])):
        ex = ", ".join(str(o) for o in g["obs"][:3])
        out.append(f"[{sc}] '{base}' {st} {g['n']}x across {len(g['cases'])} case(s) "
                   f"({runs_per_sc.get(sc, 0)} {sc} runs). Observed e.g.: {ex[:160]}")
    for g in gates:
        if g["result"] == "FAIL":
            out.append(f"Release gate {g['metric']} failed: {g['actual']:.3g} vs required {g['op']} {g['value']}")
    return out[:5]


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(EVAL / "config" / "thresholds.yaml"))
    ap.add_argument("--app-config", default=str(EVAL / "config" / "app.yaml"))
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="comma-separated case ids")
    ap.add_argument("--resume", action="store_true", help="skip case runs that already have a result in --out")
    ap.add_argument("--smoke", action="store_true", help="one dev clip, one repeat, full report")
    ap.add_argument("--rescore", action="store_true", help="re-score raw logs in --out without running the app")
    ap.add_argument("--baseline", default=None, help="earlier results folder for the Compare sheet")
    ap.add_argument("--repeats", type=int, default=None, help="override run.repeats")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--skip-media-verify", action="store_true", help="only verify golden JSON/CSV checksums")
    ap.add_argument("--golden", default=None, help="golden dataset folder (default eval/golden)")
    args = ap.parse_args()

    th_path, app_path = Path(args.config).resolve(), Path(args.app_config).resolve()
    th = yaml.safe_load(th_path.read_text(encoding="utf-8"))
    app_cfg = yaml.safe_load(app_path.read_text(encoding="utf-8"))
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + ("-smoke" if args.smoke else "")
    out = Path(args.out).resolve() if args.out else (REPO / "results" / run_id)
    if args.out and (out / "run_state.json").is_file():
        run_id = json.loads((out / "run_state.json").read_text())["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out / "eval.log", encoding="utf-8")])
    args.repeats_effective = 1 if args.smoke else (args.repeats or int(th["run"]["repeats"]))
    args.seed_effective = th["run"]["seed"]

    golden = Golden(Path(args.golden).resolve(), REPO) if args.golden else Golden(repo=REPO)
    problems = golden.verify(check_media=not args.skip_media_verify)
    if problems:
        log.error("golden dataset verification failed (%d problems), refusing to run:\n  %s",
                  len(problems), "\n  ".join(problems[:20]))
        return 2
    log.info("golden dataset %s verified (checksum %s)", golden.version.get("version"), golden.version.get("checksum"))

    cases = plan(golden, args.split, args.only.split(",") if args.only else None, args.smoke)
    if not cases:
        log.error("no cases selected")
        return 2
    by_sc = th["run"].get("repeats_by_scenario") or {}

    def reps(case: dict) -> int:
        if args.smoke or args.repeats:
            return args.repeats_effective
        return int(by_sc.get(case["scenario"], args.repeats_effective))

    est = sum(reps(c) * (((c["clip"]["end_s"] or float(golden.clip(c["clip"]["clip_id"])["duration_s"]))
                          - c["clip"]["start_s"]) + 60) for c in cases)
    md = collect(REPO, th_path, app_path, app_cfg, golden, args, run_id)
    md["repeats_by_scenario"] = by_sc
    md["estimated_runtime_h"] = round(est / 3600, 2)
    (out / "run_state.json").write_text(json.dumps({"run_id": run_id, "args": vars(args)}, default=str, indent=1))
    log.info("run %s: %d cases, %d case runs, estimated %.1f h (real time) -> %s", run_id, len(cases),
             sum(reps(c) for c in cases), est / 3600, out)

    results: list[dict] = []
    if not args.rescore:
        from harness.executor import Executor
        cam = None
        if app_cfg["video"]["source"] == "virtualcam":
            ci = app_cfg["video"].get("camera_index", "auto")
            if ci == "auto":
                cam, how = find_obs_camera_index(app_cfg["video"].get("backend", "obs"))
                log.info("virtual camera index: %s (%s)", cam, how)
                if cam is None:
                    log.error("cannot find the OBS Virtual Camera; start OBS once or set video.camera_index")
                    return 2
            else:
                cam = int(ci)
            md["video"]["camera_index_used"] = cam
        ex = Executor(REPO, app_cfg, th, golden, cam)
        for case in cases:
            for rep in range(1, reps(case) + 1):
                tag = f"{case['case_id']}__r{rep}"
                rp = out / "raw" / f"{tag}.result.json"
                if args.resume and rp.is_file():
                    log.info("resume: %s already done", tag)
                    continue
                log.info("running %s (%s)", tag, case["title"])
                t0 = time.monotonic()
                try:
                    r = ex.run(case, rep, out)
                except Exception as e:  # noqa: BLE001 - one broken case never stops the run
                    r = {"case_id": case["case_id"], "rep": rep, "status": "ERROR", "error": repr(e)}
                    rp.write_text(json.dumps(r))
                log.info("  %s in %.0fs%s", r["status"], time.monotonic() - t0, f": {r.get('error')}" if r.get("error") else "")
    # ---- score
    for case in cases:
        for rep in range(1, reps(case) + 1):
            rp = out / "raw" / f"{case['case_id']}__r{rep}.result.json"
            if rp.is_file():
                results.append(json.loads(rp.read_text()))
            else:
                results.append({"case_id": case["case_id"], "rep": rep, "status": "ERROR",
                                "error": "no result (not run or crashed before writing)"})
    by_id = {c["case_id"]: c for c in cases}
    scored = [score_run(by_id[r["case_id"]], r, golden, th, out) for r in results]
    # ---- models actually used (from the first completed run)
    done = [r for r in results if r.get("status") == "DONE"]
    app_log = ("\n".join(done[0].get("app_model_lines", [])) + "\n" + done[0].get("app_log_tail", "")) if done else ""
    md["models"] = models_from_run(app_cfg, done[0].get("session") if done else None, app_log,
                                   llm_server_models(app_cfg["llm"]["upstream"]))
    # ---- judge
    judge_rows: list = []
    jcfg, why = (None, "NOT RUN (--no-judge)") if args.no_judge else JudgeConfig.from_thresholds(th.get("judge"), app_cfg["llm"].get("family", ""))
    md["judge"] = {"provider": jcfg.provider if jcfg else "none", "model": jcfg.model if jcfg else "",
                   "runs": jcfg.runs if jcfg else 0, "temperature": jcfg.temperature if jcfg else None,
                   "status": "configured" if jcfg else why}
    judge = Judge(jcfg) if jcfg else None
    for sr in scored:
        for turn, q, facts in judge_items(sr["case"], sr, golden):
            cid, rep = sr["case"]["case_id"], sr["res"]["rep"]
            if judge is None:
                judge_rows.append([cid, rep, turn, q, why, why, why, why, None, "NOT RUN", "", ""])
                continue
            v = judge.ask(q, facts)
            rv = [r["verdict"] or f"ERR: {r['error']}" for r in v["runs"]] + [""] * 3
            res = "NOT RUN" if v["passed"] is None else ("PASS" if v["passed"] else "FAIL")
            ev = next((r["evidence"] for r in v["runs"] if r["verdict"] == v["majority"]), "")
            judge_rows.append([cid, rep, turn, q, rv[0], rv[1], rv[2], v["majority"] or "no majority",
                               round(v["agreement"], 3), res, ev, f"{v['provider']}/{v['model']}"])
            if res == "FAIL":
                sr["checks"].append(C.chk(f"{turn}:judge:{q}", "FAIL", v["majority"], v["good_verdict"], ev,
                                          1 if q == "claims_alert" else 5))
                sr["status"] = C.case_status(sr["checks"], "DONE" if sr["status"] != "ERROR" else "ERROR")
    agg = aggregate(scored, golden, th, judge_rows, "" if judge else why, app_cfg["llm"].get("proxy", True))
    manifest_headers = list(next(iter(golden.clips.values())).keys()) if golden.clips else []
    mrows = [[r.get(h, "") for h in manifest_headers] for r in golden.clips.values()]
    data = agg | {"metadata": md, "judge_rows": judge_rows, "manifest_headers": manifest_headers, "manifest_rows": mrows}
    baseline = None
    if args.baseline:
        bp = Path(args.baseline) / "summary.json"
        baseline = json.loads(bp.read_text()) if bp.is_file() else None
        if baseline is None:
            log.warning("baseline %s has no summary.json; Compare sheet skipped", args.baseline)
    path = report.write(out, data, baseline)
    summary = {"metadata": md, "summary": agg["summary"],
               "cases": [dict(zip(["case_id", "rep", "scenario", "split", "clip_id", "status", "failed_checks",
                                   "first_failure", "observed", "expected", "error", "raw_log"], r)) for r in agg["case_rows"]]}
    (out / "summary.json").write_text(json.dumps(summary, default=str, indent=1))
    log.info("status counts: %s", agg["summary"]["status_counts"])
    log.info("report: %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
