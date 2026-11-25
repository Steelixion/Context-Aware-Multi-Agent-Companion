import React, { useRef } from "react";
import { Bar, Card, Chip, LEVEL } from "./ui.jsx";

function Gauge({ value, level }) {
  const r = 54, c = 2 * Math.PI * r, off = c * (1 - Math.min(1, value));
  return (
    <div className="relative h-36 w-36 shrink-0">
      <svg viewBox="0 0 128 128" className="-rotate-90">
        <circle cx="64" cy="64" r={r} fill="none" strokeWidth="10" className="stroke-slate-800" />
        <circle cx="64" cy="64" r={r} fill="none" strokeWidth="10" strokeLinecap="round"
          strokeDasharray={c} strokeDashoffset={off}
          className={`transition-all duration-300 ${LEVEL[level].ring} stroke-current`} />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-3xl font-bold tabular-nums">{Math.floor(value * 100 + 1e-9)}</span>
        <span className="text-[10px] uppercase tracking-widest text-slate-400">risk</span>
      </div>
    </div>
  );
}

function Tiles({ title, items, note, cols = 3 }) {
  return (
    <div className="mt-4">
      <div className="mb-1.5 flex items-center justify-between">
        <h3 className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">{title}</h3>
        {note}
      </div>
      <dl className={`grid ${cols === 4 ? "grid-cols-4" : "grid-cols-3"} gap-2 text-center text-sm`}>
        {items.map(([k, v, bad]) => (
          <div key={k} className="rounded-lg bg-slate-800/60 py-1.5">
            <dt className="text-[10px] uppercase tracking-wider text-slate-400">{k}</dt>
            <dd className={`font-mono text-base ${bad ? "text-red-400" : ""}`}>{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function CalibrationBar({ tel, calibration, calibrate }) {
  const live = tel?.calibration;
  const busy = live?.state === "calibrating";
  const failed = !busy && calibration?.state === "failed";
  const text = busy ? "Calibrating: look ahead normally"
    : failed ? "Calibration failed (face not visible enough). Using the previous thresholds."
    : live?.state === "calibrated" ? "Calibrated to this driver"
    : "Using default thresholds";
  return (
    <div className="mt-4 rounded-lg border border-slate-800 bg-slate-900/60 p-3">
      <div className="flex items-center justify-between gap-3 text-sm">
        <span className={failed ? "text-amber-300" : busy ? "text-sky-300" : "text-slate-300"}>{text}</span>
        <button onClick={calibrate} disabled={busy}
          className="shrink-0 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1 text-xs hover:border-sky-500 disabled:cursor-not-allowed disabled:opacity-40">
          Recalibrate
        </button>
      </div>
      {busy && (
        <div className="mt-2 h-1.5 overflow-hidden rounded bg-slate-800">
          <div className="h-full bg-sky-500 transition-all" style={{ width: `${Math.round((live.progress || 0) * 100)}%` }} />
        </div>
      )}
    </div>
  );
}

const unsign0 = (s) => s.replace(/^-(0(.0+)?)$/, "$1");   // "-0" / "-0.00" -> "0" / "0.00"
const deg = (v) => `${unsign0((v ?? 0).toFixed(0))}°`;
const HOLD_MS = 1000;   // gaze is unmeasurable while the eyes are closed: bridge blinks, show real gaps

function useHeldGaze(g, ts) {
  const last = useRef(null);
  if (g?.h != null) last.current = { h: g.h, v: g.v, ts };
  const held = last.current && ts != null && (ts - last.current.ts) * 1000 <= HOLD_MS ? last.current : null;
  return g?.h != null ? g : held;
}
const num = (v, d = 2) => (v == null ? "—" : unsign0(v.toFixed(d)));

export default function DriverState({ tel, calibration, calibrate }) {
  const level = LEVEL[tel?.drowsiness_level] ? tel.drowsiness_level : "ALERT";
  const cal = tel?.calibration;
  const earThr = cal?.ear_threshold ?? 0.21, marThr = cal?.mar_threshold ?? 0.45;
  const q = tel?.quality ?? 1, g = tel?.gaze, p = tel?.posture, h = tel?.head_pose;
  const gaze = useHeldGaze(g, tel?.ts);
  const win = Math.round(tel?.window_s ?? 60);
  return (
    <Card title="Driver state" right={<Chip tone={q >= 0.5 ? "slate" : "amber"}>tracking {Math.round(q * 100)}%</Chip>}>
      <div className="flex items-center gap-5">
        <Gauge value={tel?.risk || 0} level={level} />
        <div className="flex-1 space-y-3">
          <span className={`inline-block rounded-lg border px-3 py-1 text-lg font-bold ${LEVEL[level].bg}`}>
            {LEVEL[level].label}
          </span>
          {tel && !tel.face_detected && <p className="text-sm text-amber-300">No face detected{tel.note && ` (${tel.note})`}</p>}
          <Bar label={`EAR (closed < ${earThr.toFixed(2)})`} value={tel?.ear || 0} max={0.45} warnBelow={earThr} />
          <Bar label={`MAR (yawn > ${marThr.toFixed(2)})`} value={tel?.mar || 0} max={1} danger={(v) => v > marThr} />
        </div>
      </div>
      <CalibrationBar tel={tel} calibration={calibration} calibrate={calibrate} />
      <Tiles title={`Eyes and mouth (last ${win} s)`} items={[
        ["PERCLOS", `${Math.round((tel?.perclos || 0) * 100)}%`, (tel?.perclos || 0) >= 0.15],
        ["Eyes closed", `${((tel?.eyes_closed_ms || 0) / 1000).toFixed(1)}s`, (tel?.eyes_closed_ms || 0) >= 1500],
        ["Blinks", tel?.blinks ?? 0],
        ["Microsleeps", tel?.long_closures ?? 0, (tel?.long_closures || 0) > 0],
        ["Yawns", tel?.yawns ?? 0, tel?.yawning],
        ["Head drops", tel?.head_drops ?? 0, tel?.head_drop],
      ]} />
      <Tiles title="Head pose and gaze" note={g && !g.on_road && <Chip tone="amber">eyes off road</Chip>} items={[
        ["Yaw", deg(h?.yaw), Math.abs(h?.yaw || 0) >= 20],
        ["Pitch", deg(h?.pitch), (h?.pitch || 0) <= -20],
        ["Roll", deg(h?.roll)],
        ["Gaze H", num(gaze?.h), Math.abs(gaze?.h || 0) >= 0.15],
        ["Gaze V", num(gaze?.v)],
        ["Off road", `${((g?.off_road_ms || 0) / 1000).toFixed(1)}s`, g?.distracted],
      ]} />
      <Tiles title="Posture" cols={4} note={p && !p.visible && <Chip>shoulders not visible</Chip>} items={[
        ["Tilt", p?.visible ? deg(p.shoulder_tilt) : "—"],
        ["Lean", p?.visible ? num(p.lean) : "—"],
        ["Slump", p?.visible ? `${Math.round((p.slump_score || 0) * 100)}%` : "—"],
        ["Deviation", p?.visible ? num(p.deviation, 1) : "—", p?.unstable],
      ]} />
    </Card>
  );
}
