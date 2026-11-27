"""Turns one run's raw event log into clip-time observations (all times = seconds since the
test clip's first frame was sent to the virtual camera, on the harness monotonic clock)."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import metrics as M


@dataclass
class Turn:
    audio_id: str
    event_id: str | None
    speech_start: float
    speech_end: float
    transcripts: list = field(default_factory=list)       # [(t, text)]
    user_chat_t: float | None = None
    llm_request_t: float | None = None
    llm_first_token_t: float | None = None
    llm_done_t: float | None = None
    llm_text: str | None = None
    reply_t: float | None = None
    reply_text: str | None = None
    reply_interrupted: bool = False
    first_audio_event_t: float | None = None
    first_audible_t: float | None = None

    @property
    def transcript(self) -> str:
        return " ".join(t for _, t in self.transcripts)


@dataclass
class Observation:
    clip_t0: float
    duration: float
    level_changes: list            # [(t, level)] from 'state' (unthrottled)
    telemetry: list                # [(t, dict)] throttled telemetry, t from its ts
    alerts: list                   # [(t, dict)]
    drowsy_warnings: list          # [t] alert reason == drowsy
    nudges: list                   # [(t, text)] speech source=safety
    audio_events: list             # [(t, speaking, priority, text)]
    audible: list | None           # [(start, end)] loopback segments, None if no loopback
    chats: list                    # [(t, dict)]
    transcripts: list              # [(t, text)] speech source=voice
    turns: list                    # [Turn]
    llm: list                      # [(t, type, data)]
    harness: list                  # [(t, type, data)]
    calibration_end: float | None
    stats: dict


def build(events: list[dict], clip_t0: float, duration: float, loopback: dict | None, th: dict) -> Observation:
    # wall -> mono offsets from 'clock' events (all on this host)
    offs = [(e["data"]["wall"] - e["data"]["mono"]) for e in events if e["type"] == "clock" and e["src"] == "harness"]
    off = sum(offs) / len(offs) if offs else 0.0

    def rel(t: float) -> float:
        return round(t - clip_t0, 4)

    def wall(ts: float) -> float:
        return rel(ts - off)

    lv, tel, alerts, warns, nudges, aud, chats, trans, llm, har = [], [], [], [], [], [], [], [], [], []
    cal_end = None
    for e in events:
        s, ty, d, t = e["src"], e["type"], e["data"], e["t"]
        if s == "ws":
            if ty == "state":
                lv.append((rel(t), d.get("level")))
            elif ty == "telemetry":
                tel.append((wall(d["ts"]) if "ts" in d else rel(t), d))
            elif ty == "alert":
                at = wall(d["ts"]) if "ts" in d else rel(t)
                alerts.append((at, d))
                if d.get("reason") == "drowsy":
                    warns.append(at)
            elif ty == "speech":
                if d.get("source") == "safety":
                    nudges.append((rel(t), d.get("text", "")))
                elif d.get("source") == "voice":
                    trans.append((rel(t), d.get("text", "")))
            elif ty == "audio":
                aud.append((rel(t), bool(d.get("speaking")), d.get("priority"), d.get("text", "")))
            elif ty == "chat":
                chats.append((rel(t), d))
            elif ty == "calibration" and d.get("state") in ("calibrated", "failed"):
                cal_end = rel(t)
        elif s == "llm":
            llm.append((rel(t), ty, d))
        elif s == "harness":
            har.append((rel(t), ty, d))
    lv = M.compress_changes(lv)

    audible = None
    if loopback and loopback.get("available") and loopback.get("t0") is not None:
        lb = th["loopback"]
        audible = [(round(a - clip_t0, 4), round(b - clip_t0, 4)) for a, b in M.active_segments(
            loopback["db"], loopback["hop_s"], loopback["t0"], lb["threshold_dbfs"], lb["min_segment_s"], lb["merge_gap_s"])]

    # ---- turns: harness utterances -> transcript -> LLM -> reply -> audio
    starts = {d["audio_id"] + str(d.get("event_id")): t for t, ty, d in har if ty == "speech_start"}
    turns = []
    for t, ty, d in har:
        if ty == "speech_end":
            key = d["audio_id"] + str(d.get("event_id"))
            turns.append(Turn(d["audio_id"], d.get("event_id"), starts.get(key, t), t))
    turns.sort(key=lambda u: u.speech_start)
    wait = float(th["speech"]["transcript_wait_s"])
    # a transcript is emitted only after its speaker stops (VAD end-of-speech), so it belongs to
    # the most recent utterance that had ENDED when it arrived (and only within transcript_wait_s);
    # a transcript arriving while an utterance is still being spoken belongs to that one (VAD
    # split it: a false endpoint)
    for tt, txt in trans:
        cands = [u for u in turns if u.speech_start < tt <= u.speech_end + wait]
        if not cands:
            continue
        ended = [u for u in cands if u.speech_end <= tt + 0.05]
        owner = max(ended, key=lambda u: u.speech_end) if ended else max(cands, key=lambda u: u.speech_start)
        owner.transcripts.append((tt, txt))
    for i, u in enumerate(turns):
        if not u.transcripts:
            continue
        t_tr = u.transcripts[-1][0]
        horizon = turns[i + 1].speech_start if i + 1 < len(turns) else float("inf")
        texts = {txt for _, txt in u.transcripts}
        # user bubble = the app's echo of this transcript
        for tc, d in chats:
            if d.get("role") == "user" and d.get("text") in texts and tc >= u.transcripts[0][0] - 0.05:
                u.user_chat_t = tc
                break
        # LLM request = the one whose last user message IS this transcript (not a safety nudge)
        rid = None
        for tl, ty, d in llm:
            if ty != "llm_request" or tl < u.transcripts[0][0] - 0.05:
                continue
            msgs = d.get("messages") or []
            last = msgs[-1] if msgs else {}
            if last.get("role") == "user" and last.get("content") in texts:
                u.llm_request_t, rid = tl, d.get("rid")
                break
        if rid is not None:
            for tl, ty, d in llm:
                if d.get("rid") == rid and ty == "llm_first_token":
                    u.llm_first_token_t = tl
                if d.get("rid") == rid and ty == "llm_done":
                    u.llm_done_t, u.llm_text = tl, d.get("text")
        # reply = the assistant bubble opened right after the user bubble, followed by its id
        if u.user_chat_t is not None:
            bubble = next((d.get("id") for tc, d in chats if d.get("role") == "assistant" and not d.get("final")
                           and tc >= u.user_chat_t - 0.01), None)
            fin = next(((tc, d) for tc, d in chats if d.get("id") == bubble and d.get("final")), None)
            if fin:
                u.reply_t, u.reply_text = fin[0], fin[1].get("text", "")
                u.reply_interrupted = bool(fin[1].get("interrupted"))
        # first audio = first 'audio speaking' whose text is part of this reply
        if u.reply_text:
            a = [ta for ta, sp, pr, txt in aud if sp and ta >= t_tr and txt and txt.strip() in u.reply_text]
            if a:
                u.first_audio_event_t = a[0]
                if audible is not None:
                    on = [s0 for s0, _ in audible if s0 >= a[0] - 0.05]
                    if on and on[0] < a[0] + 15:
                        u.first_audible_t = on[0]

    stats = {}
    for t, ty, d in har:
        if ty == "audio_stats":
            stats["audio"] = d
        if ty == "clip_end":
            stats["video"] = d
    return Observation(clip_t0, duration, lv, tel, alerts, warns, nudges, aud, audible, chats, trans,
                       turns, llm, har, cal_end, stats)


def first_audio_after(obs: Observation, t: float, within: float = 20.0) -> tuple[float | None, str]:
    """First audible onset (loopback) after t, else first 'audio speaking' event; with source."""
    if obs.audible is not None:
        on = [a for a, _ in obs.audible if t - 0.05 <= a <= t + within]
        if on:
            return on[0], "loopback"
    ev = [ta for ta, sp, _, _ in obs.audio_events if sp and t - 0.05 <= ta <= t + within]
    if ev:
        return ev[0], "audio event (not audible onset)"
    return None, "none"


def speaking_intervals(obs: Observation) -> list[tuple[float, float, int | None]]:
    """[(start, end, priority)] from 'audio' events."""
    out, cur = [], None
    for t, sp, pr, _ in obs.audio_events:
        if sp:
            cur = (t, pr)
        elif cur is not None:
            out.append((cur[0], t, cur[1]))
            cur = None
    if cur is not None:
        out.append((cur[0], float("inf"), cur[1]))
    return out


def quality_low_intervals(obs: Observation, qmin: float) -> list[tuple[float, float]]:
    out, start = [], None
    for t, d in obs.telemetry:
        low = d.get("quality", 1.0) < qmin
        if low and start is None:
            start = t
        elif not low and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, obs.telemetry[-1][0]))
    return out
