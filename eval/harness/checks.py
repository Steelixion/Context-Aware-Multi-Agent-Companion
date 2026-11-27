"""Per-scenario checks. Every limit comes from thresholds.yaml (passed in as `th`); no numbers
are hard-coded here except unit conversions. Each check returns a dict:
  {check, status, observed, expected, detail, severity}
status: PASS | FAIL | INCONCLUSIVE (scenario precondition not met) | NOT OBSERVABLE | OBSERVED
severity (for the Failures sheet, 1 = worst): 1 missed drowsiness / false statement about the
driver's state, 2 false alarm, 3 other safety behaviour, 4 speech, 5 reply quality / latency.
"""
from __future__ import annotations

from . import metrics as M
from .observe import Observation, Turn, first_audio_after, quality_low_intervals, speaking_intervals

PASS, FAIL, INCONC, NOBS, OBSERVED = "PASS", "FAIL", "INCONCLUSIVE", "NOT OBSERVABLE", "OBSERVED"


def chk(check, status, observed=None, expected=None, detail="", severity=5) -> dict:
    return dict(check=check, status=status, observed=observed, expected=expected, detail=detail, severity=severity)


def r3(x):
    return None if x is None else round(x, 3)


# --------------------------------------------------------------------------- building blocks
def gt_events(gt: dict, th: dict, c0: float, c1: float) -> list[tuple[float, float]]:
    """GT drowsy events inside the played segment, in clip time (relative to segment start)."""
    d = th["detection"]
    ev = M.merge_intervals([(a, b) for a, b in gt["events"]], d["merge_event_gap_s"])
    ev = [(max(a, c0) - c0, min(b, c1) - c0) for a, b in ev if b > c0 and a < c1]
    return [(a, b) for a, b in ev if b - a >= d["min_event_s"]]


def warning_audio(obs: Observation, t_alert: float) -> tuple[float | None, str]:
    return first_audio_after(obs, t_alert, within=20.0)


def warnings_for_event(obs: Observation, ev: tuple[float, float], th: dict) -> list[float]:
    d = th["detection"]
    return [t for t in obs.drowsy_warnings if ev[0] - d["event_early_s"] <= t < ev[1] + d["event_late_s"]]


def no_false_warning(obs: Observation, events: list, th: dict, start: float, end: float, name="no_false_warning") -> dict:
    d = th["detection"]
    fa = [t for t in M.warnings_outside(obs.drowsy_warnings, events, d["false_alarm_margin_before_s"],
                                        d["false_alarm_margin_after_s"]) if start <= t <= end]
    return chk(name, PASS if not fa else FAIL, observed=len(fa), expected=0,
               detail=("false drowsiness warnings at clip t=" + ", ".join(f"{t:.1f}s" for t in fa)) if fa else "",
               severity=2)


def warning_latency_checks(obs: Observation, events: list, th: dict, name="warning_in_time") -> tuple[list, list]:
    """For each GT event: a drowsy warning must be AUDIBLE within warning_latency_s of onset."""
    lim = th["safety"]["warning_latency_s"]
    out, lat = [], []
    for i, ev in enumerate(events):
        ws = warnings_for_event(obs, ev, th)
        if not ws:
            out.append(chk(f"{name}[event {i + 1} @ {ev[0]:.1f}s]", FAIL, observed="no warning",
                           expected=f"<= {lim}s", detail="missed drowsy event: no drowsiness warning", severity=1))
            lat.append(None)
            continue
        ta = ws[0]
        tau, src = warning_audio(obs, ta)
        if tau is None:
            out.append(chk(f"{name}[event {i + 1} @ {ev[0]:.1f}s]", FAIL, observed="alert raised, no audio",
                           expected=f"<= {lim}s", detail=f"alert at {ta:.2f}s but nothing was played", severity=1))
            lat.append(None)
            continue
        d = tau - ev[0]
        lat.append(d)
        out.append(chk(f"{name}[event {i + 1} @ {ev[0]:.1f}s]", PASS if d <= lim else FAIL, observed=r3(d),
                       expected=f"<= {lim}s", detail=f"alert {ta - ev[0]:+.2f}s, audio {d:+.2f}s from onset ({src})",
                       severity=1))
    return out, lat


def reply_checks(turn: Turn, th: dict, prefix: str = "") -> list:
    rp = th["reply"]
    out = []
    if turn.reply_text is None and not turn.transcripts:
        return []                                   # nothing was heard: covered by transcript_received
    if turn.reply_text is None:
        return [chk(prefix + "reply_received", FAIL, "no reply", "a reply", severity=5)]
    txt = turn.reply_text or ""
    out.append(chk(prefix + "reply_nonempty", PASS if txt.strip() else FAIL, len(txt), "> 0 chars",
                   "interrupted" if turn.reply_interrupted else "", severity=5))
    if txt.strip():
        out.append(chk(prefix + "reply_english", PASS if M.looks_english(txt) else FAIL, txt[:80], "English", severity=5))
        n = M.word_count(txt)
        ok = rp["min_words"] <= n <= rp["max_words"]
        out.append(chk(prefix + "reply_length", PASS if ok else FAIL, n, f"{rp['min_words']}..{rp['max_words']} words", severity=5))
        rr = M.repetition_ratio(txt)
        out.append(chk(prefix + "reply_not_looping", PASS if rr <= rp["max_repetition_ratio"] else FAIL, r3(rr),
                       f"<= {rp['max_repetition_ratio']}", severity=5))
    return out


def talked_over(obs: Observation, turn: Turn, tail_s: float = 1.2) -> list[tuple[float, float]]:
    """Companion audio overlapping the utterance or the end-of-speech window after it. The app
    mutes its mic while the companion talks (echo guard, by design) and drops a half-heard
    utterance, so a missing transcript there is not an ASR failure."""
    return [(a, b) for a, b, _ in speaking_intervals(obs) if a < turn.speech_end + tail_s and b > turn.speech_start]


def transcript_checks(turn: Turn, ref: str, snr_key: str, th: dict, prefix: str = "",
                      obs: Observation | None = None) -> list:
    sp = th["speech"]
    if not turn.transcripts:
        over = talked_over(obs, turn) if obs is not None else []
        if over:
            a, b = over[0]
            return [chk(prefix + "transcript_received", INCONC, "none (lost to talk-over)", f"within {sp['transcript_wait_s']}s",
                        f"the companion was speaking {a:.1f}-{b:.1f}s while the driver spoke {turn.speech_start:.1f}-"
                        f"{turn.speech_end:.1f}s; the app mutes the mic then (echo guard) and drops the utterance", 4)]
        return [chk(prefix + "transcript_received", FAIL, "none", f"within {sp['transcript_wait_s']}s", severity=4)]
    w = M.word_error_rate(ref, turn.transcript)
    lim = sp["wer_max"].get(snr_key)
    out = [chk(prefix + "transcript_received", PASS, len(turn.transcripts), ">= 1", severity=4)]
    if lim is not None:
        out.append(chk(prefix + f"wer[{snr_key}]", PASS if w.wer <= lim else FAIL, r3(w.wer), f"<= {lim}",
                       f"ref='{ref}' hyp='{turn.transcript}'", severity=4))
    out.append(chk(prefix + "no_false_endpoint", PASS if len(turn.transcripts) - 1 <= sp["false_endpoint_max"] else FAIL,
                   len(turn.transcripts) - 1, f"<= {sp['false_endpoint_max']}", "utterance split into several transcripts"
                   if len(turn.transcripts) > 1 else "", severity=4))
    return out


def level_at(obs: Observation, t: float) -> str:
    return M.step_value(obs.level_changes, t, "ALERT") or "ALERT"


def turn_for(obs: Observation, event_id: str) -> Turn | None:
    for u in obs.turns:
        if u.event_id == event_id:
            return u
    return None


# --------------------------------------------------------------------------- per scenario
def score(case: dict, obs: Observation, gt: dict, th: dict, audio_meta: dict) -> list[dict]:
    sc = case["scenario"]
    c0 = case["clip"].get("start_s", 0.0)
    c1 = c0 + obs.duration
    events = gt_events(gt, th, c0, c1)
    S = th["safety"]
    out: list[dict] = []

    def spoken(eid):
        u = turn_for(obs, eid)
        return u

    if sc == "S01":
        out.append(no_false_warning(obs, events, th, 0, obs.duration))
    elif sc == "S02":
        if not events:
            out.append(chk("has_gt_event", INCONC, 0, ">= 1", "no labelled drowsy event in the segment", 1))
        c, _ = warning_latency_checks(obs, events, th)
        out += c
        out.append(no_false_warning(obs, events, th, 0, obs.duration))
    elif sc in ("S03", "S10"):
        for ev in case["timeline"]:
            if ev.get("action") != "speak":
                continue
            u = spoken(ev["id"])
            meta = audio_meta[ev["audio_id"]]
            if u is None:
                out.append(chk(f"{ev['id']}:spoken", FAIL, "utterance not played", "played", severity=4))
                continue
            out += transcript_checks(u, meta["text"], meta.get("snr_key", "clean"), th, f"{ev['id']}:", obs)
            out += reply_checks(u, th, f"{ev['id']}:")
        out.append(no_false_warning(obs, events, th, 0, obs.duration))
    elif sc == "S09":
        for ev in case["timeline"]:
            if ev.get("action") != "speak":
                continue
            u = spoken(ev["id"])
            meta = audio_meta[ev["audio_id"]]
            if u is None:
                out.append(chk(f"{ev['id']}:spoken", FAIL, "utterance not played", "played", severity=4))
                continue
            out += transcript_checks(u, meta["text"], meta.get("snr_key", "clean"), th, f"{ev['id']}:", obs)
            if u.reply_text is not None:
                out += [c for c in reply_checks(u, th, f"{ev['id']}:") if not c["check"].endswith("reply_received")]
    elif sc in ("S04", "S08"):
        ev_speak = [e for e in case["timeline"] if e.get("action") == "speak"][0]
        u = spoken(ev_speak["id"])
        if u is None:
            out.append(chk("utterance_played", INCONC, "not played", "played",
                           "trigger did not fire (detector never reached the trigger level)" if "trigger" in ev_speak else "", 1))
        else:
            t_mid = (u.speech_start + u.speech_end) / 2
            in_gt = M.in_any(t_mid, events, th["detection"]["event_early_s"], th["detection"]["event_late_s"])
            if sc == "S08" and not in_gt:
                out.append(chk("gt_drowsy_at_utterance", INCONC, False, True,
                               "utterance was not inside a labelled drowsy event", 1))
            ws = [t for t in obs.drowsy_warnings if t >= u.speech_start - 0.05]
            lim = S["warning_latency_s"]
            if sc == "S08":
                ok = [t for t in ws if t <= u.speech_end + lim]
                out.append(chk("warning_still_fires", PASS if ok else FAIL,
                               r3(ok[0] - u.speech_end) if ok else (f"first at {ws[0] - u.speech_end:+.1f}s" if ws else "none"),
                               f"<= {lim}s after the driver's words", "driver said they are fine; detector says drowsy", 1))
            else:
                c, _ = warning_latency_checks(obs, events, th)
                out += c
                after = [ta for ta in obs.drowsy_warnings if ta >= u.speech_start - 0.05]
                if after:
                    pr = [p for t, sp, p, _ in obs.audio_events if sp and t >= after[0] - 0.05]
                    p0 = pr[0] if pr else None
                    out.append(chk("warning_has_safety_priority", PASS if p0 == 0 else FAIL, p0, 0,
                                   "priority of the first audio after the warning decision (0 = safety, 5 = chat)", 3))
            out += transcript_checks(u, audio_meta[ev_speak["audio_id"]]["text"], "clean", th, "", obs)
    elif sc == "S05":
        ws = obs.drowsy_warnings
        if not ws:
            out.append(chk("warning_fired", INCONC, "none", ">= 1",
                           "precondition not met: no drowsiness warning fired, so preemption could not be tested "
                           "(the missed warning itself is scored by this clip's S02 case)", 3))
        else:
            ta = ws[0]
            iv = [(a, b, p) for a, b, p in speaking_intervals(obs) if a <= ta < b and (p or 0) > 0]
            if not iv:
                out.append(chk("companion_mid_reply_at_warning", INCONC, False, True,
                               "precondition not met: the companion was not speaking chat when the warning was decided", 3))
            else:
                a, b, _ = iv[0]
                stop_d = b - ta
                out.append(chk("chat_audio_stopped", PASS if stop_d <= S["preemption_max_s"] else FAIL, r3(stop_d),
                               f"<= {S['preemption_max_s']}s", "time from warning decision to the chat audio stopping", 3))
                safety = [t for t, sp, p, _ in obs.audio_events if sp and p == 0 and t >= ta - 0.05]
                out.append(chk("warning_played_with_safety_priority", PASS if safety else FAIL,
                               r3(safety[0] - ta) if safety else "no priority-0 audio", "priority-0 audio after decision", "", 3))
                mute_end = ta + S["chat_mute_s"]
                chat_starts = [t for t, sp, p, _ in obs.audio_events if sp and (p or 0) > 0 and ta + 0.05 < t < mute_end]
                out.append(chk("chat_muted_window", PASS if not chat_starts else FAIL, len(chat_starts), 0,
                               f"chat audio starts within {S['chat_mute_s']}s after the warning", 3))
                interrupted = any(d.get("interrupted") for t, d in obs.chats if d.get("role") == "assistant" and t >= ta - 0.05)
                out.append(chk("reply_marked_interrupted", OBSERVED, interrupted, True, "observe only", 3))
    elif sc == "S06":
        rep, tol = S["warning_repeat_s"], S["warning_repeat_tolerance_s"]
        highs = []
        for i, (t, lvl) in enumerate(obs.level_changes):
            if lvl == "HIGH":
                end = obs.level_changes[i + 1][0] if i + 1 < len(obs.level_changes) else obs.duration
                highs.append((t, min(end, obs.duration)))
        longest = max(highs, key=lambda x: x[1] - x[0], default=None)
        need = 2 * rep + tol
        if longest is None or longest[1] - longest[0] < need:
            out.append(chk("high_held_long_enough", INCONC, r3(longest[1] - longest[0]) if longest else 0, f">= {need}s",
                           "precondition not met: the app's level never stayed HIGH long enough", 3))
        else:
            ws = [t for t in obs.drowsy_warnings if longest[0] - 0.5 <= t <= longest[1] + 0.5]
            gaps = M.repeat_intervals(ws)
            exp_n = int((longest[1] - longest[0]) // rep)
            bad = [g for g in gaps if abs(g - rep) > tol]
            out.append(chk("warning_repeats", PASS if len(ws) >= 2 and not bad else FAIL,
                           [r3(g) for g in gaps] if gaps else f"{len(ws)} warning(s)",
                           f"every {rep}s ± {tol}s (≈{exp_n} in {longest[1] - longest[0]:.0f}s of HIGH)", "", 3))
    elif sc == "S07":
        after, tol, cd = S["mild_checkin_after_s"], S["mild_checkin_tolerance_s"], S["mild_checkin_cooldown_s"]
        alert_ts = [t for t, _ in obs.alerts]
        checkins = [t for t, _ in obs.nudges if not any(0 <= t - a <= 0.5 for a in alert_ts) and level_at(obs, t) == "MILD"]
        mild = []
        for i, (t, lvl) in enumerate(obs.level_changes):
            if lvl == "MILD":
                end = obs.level_changes[i + 1][0] if i + 1 < len(obs.level_changes) else obs.duration
                mild.append((t, min(end, obs.duration)))
        longest = max(mild, key=lambda x: x[1] - x[0], default=None)
        if longest is None or longest[1] - longest[0] < after:
            out.append(chk("mild_held_long_enough", INCONC, r3(longest[1] - longest[0]) if longest else 0, f">= {after}s",
                           "precondition not met: no MILD period long enough for a check-in", 3))
        else:
            ins = [t for t in checkins if longest[0] <= t <= longest[1]]
            first = ins[0] - longest[0] if ins else None
            out.append(chk("checkin_after_mild_held", PASS if first is not None and abs(first - after) <= tol else FAIL,
                           r3(first) if first is not None else "no check-in", f"{after}s ± {tol}s after MILD began", "", 3))
        gaps = M.repeat_intervals(checkins)
        out.append(chk("checkin_cooldown", PASS if all(g >= cd for g in gaps) else FAIL,
                       [r3(g) for g in gaps] if gaps else f"{len(checkins)} check-in(s)", f"none within {cd}s of another", "", 3))
    elif sc == "S11":
        blanks = [(float(e["at_s"]), float(e["at_s"]) + float(e["duration_s"])) for e in case["timeline"]
                  if e.get("action") == "blank_video"]
        guard = S["occlusion_guard_after_s"]
        for a, b in blanks:
            fa = [t for t in obs.drowsy_warnings if a <= t <= b + guard]
            out.append(chk(f"no_warning_while_occluded[{a:.0f}-{b:.0f}s]", PASS if not fa else FAIL, len(fa), 0,
                           "", 2))
            before = [d.get("risk", 0.0) for t, d in obs.telemetry if a - 1.0 <= t < a]
            during = [d.get("risk", 0.0) for t, d in obs.telemetry if a <= t <= b]
            if before and during:
                rise = max(during) - before[-1]
                eps = S.get("occlusion_risk_epsilon", 0.02)
                out.append(chk(f"risk_not_rising_while_occluded[{a:.0f}-{b:.0f}s]", PASS if rise <= eps else FAIL,
                               r3(rise), f"<= {eps}", f"risk before {before[-1]:.3f}, max during {max(during):.3f}", 2))
            low = quality_low_intervals(obs, S["quality_min"])
            out.append(chk(f"quality_dropped[{a:.0f}-{b:.0f}s]", OBSERVED,
                           any(x < b and y > a for x, y in low), True, "observe only", 5))
        out.append(no_false_warning(obs, events, th, 0, obs.duration))
    elif sc == "S12":
        kills = [t for t, ty, d in obs.harness if ty == "kill_vision"]
        stops = [t for t, ty, d in obs.harness if ty == "stop_video"]
        lost = [t for t, d in obs.alerts if d.get("reason") == "sensor_lost"]
        face = [t for t, d in obs.alerts if d.get("reason") == "face_lost"]
        if kills:
            tk = kills[0]
            la = [t for t in lost if t >= tk]
            w, tol = S["watchdog_s"], S["watchdog_tolerance_s"]
            d = la[0] - tk if la else None
            out.append(chk("camera_signal_lost_warning", PASS if d is not None and abs(d - w) <= tol else FAIL,
                           r3(d) if d is not None else "none", f"{w}s ± {tol}s after telemetry stopped",
                           "vision process terminated by the harness", 3))
            if la:
                tau, src = first_audio_after(obs, la[0])
                out.append(chk("camera_signal_lost_audible", PASS if tau is not None else FAIL,
                               r3(tau - la[0]) if tau is not None else "none", "audio after the alert", src, 3))
        if stops:
            ts = stops[0]
            out.append(chk("camera_stopped:what_happened", OBSERVED,
                           {"sensor_lost_after_s": r3(next((t - ts for t in lost if t >= ts), None)),
                            "face_lost_after_s": r3(next((t - ts for t in face if t >= ts), None))},
                           f"doc: 'camera signal lost' at ~{S['watchdog_s']}s", "observe and report", 3))
    elif sc == "S13":
        sp = th["speech"]
        comp = [(t, d.get("text", "")) for t, d in obs.chats if d.get("role") == "assistant" and d.get("final") and d.get("text")]
        comp += [(t, d.get("text", "")) for t, d in obs.alerts]
        worst = []
        for tt, txt in obs.transcripts:
            recent = " ".join(c for tc, c in comp if 0 <= tt - tc <= sp["self_hearing_window_s"] + 30)
            spoke = [a for a, b, p in speaking_intervals(obs) if a - 1 <= tt <= b + sp["self_hearing_window_s"]]
            ov = M.ngram_overlap(txt, recent) if recent else 0.0
            worst.append((ov, tt, txt, bool(spoke)))
        bad = [w for w in worst if w[0] > sp["self_hearing_overlap_max"]]
        out.append(chk("no_self_hearing", PASS if not bad else FAIL, len(bad), 0,
                       "; ".join(f"t={t:.1f}s overlap={o:.2f} '{x[:60]}'" for o, t, x, _ in bad), 4))
        assigned = {tuple(t) for u in obs.turns for t in u.transcripts}
        spurious = [x for x in obs.transcripts if tuple(x) not in assigned]
        out.append(chk("no_unprompted_transcripts", PASS if not spurious else FAIL, len(spurious), 0,
                       "; ".join(f"t={t:.1f}s '{x[:60]}'" for t, x in spurious), 4))
        for ev in case["timeline"]:
            if ev.get("action") == "speak":
                u = spoken(ev["id"])
                if u is not None:
                    out += transcript_checks(u, audio_meta[ev["audio_id"]]["text"], "clean", th, f"{ev['id']}:", obs)
    elif sc == "S14":
        for ev in case["timeline"]:
            if ev.get("action") != "speak":
                continue
            u = spoken(ev["id"])
            if u is None:
                out.append(chk(f"{ev['id']}:what_happened", OBSERVED, "not played", None, "trigger did not fire", 4))
                continue
            sp_iv = [(a, b) for a, b, p in speaking_intervals(obs) if a < u.speech_end and b > u.speech_start]
            out.append(chk(f"{ev['id']}:what_happened", OBSERVED, {
                "overlapped_companion_audio": bool(sp_iv),
                "transcripts": [x for _, x in u.transcripts],
                "wer": r3(M.word_error_rate(audio_meta[ev["audio_id"]]["text"], u.transcript).wer) if u.transcripts else None,
                "companion_reply_interrupted": any(d.get("interrupted") for t, d in obs.chats
                                                   if d.get("role") == "assistant" and u.speech_start <= t <= u.speech_end + 5),
                "reply_to_barge_in": u.reply_text,
            }, "observe and report (architecture discards audio while the companion speaks)", "", 4))
    if case.get("mode") == "observe":
        for c in out:
            if c["status"] in (PASS, FAIL):
                c["detail"] = (c["detail"] + " | " if c["detail"] else "") + f"(would be {c['status']}; observe-only case)"
                c["status"] = OBSERVED
    return out


def case_status(checks: list[dict], run_status: str) -> str:
    if run_status == "ERROR":
        return "ERROR"
    st = {c["status"] for c in checks}
    if FAIL in st:
        return "FAIL"
    if INCONC in st:
        return "INCONCLUSIVE"
    if st and st <= {OBSERVED, NOBS}:
        return "OBSERVED"
    return "PASS"
