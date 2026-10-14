import React from "react";
import { Bar, Card, LEVEL } from "./ui.jsx";

function Gauge({ score, level }) {
  const r = 54, c = 2 * Math.PI * r, off = c * (1 - Math.min(1, score));
  return (
    <div className="relative h-36 w-36 shrink-0">
      <svg viewBox="0 0 128 128" className="-rotate-90">
        <circle cx="64" cy="64" r={r} fill="none" strokeWidth="10" className="stroke-slate-800" />
        <circle cx="64" cy="64" r={r} fill="none" strokeWidth="10" strokeLinecap="round"
          strokeDasharray={c} strokeDashoffset={off}
          className={`transition-all duration-300 ${LEVEL[level].ring} stroke-current`} />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-3xl font-bold tabular-nums">{Math.round(score * 100)}</span>
        <span className="text-[10px] uppercase tracking-widest text-slate-400">fatigue</span>
      </div>
    </div>
  );
}

export default function DriverState({ tel, info }) {
  const level = tel?.drowsiness_level || "ALERT";
  const earThr = info?.ear_threshold ?? 0.2;
  return (
    <Card title="Driver state">
      <div className="flex items-center gap-5">
        <Gauge score={tel?.fatigue_score || 0} level={level} />
        <div className="flex-1 space-y-3">
          <span className={`inline-block rounded-lg border px-3 py-1 text-lg font-bold ${LEVEL[level].bg}`}>
            {LEVEL[level].label}
          </span>
          {tel && !tel.face_detected && <p className="text-sm text-amber-300">No face detected {tel.note && `— ${tel.note}`}</p>}
          <Bar label={`EAR (thr ${earThr})`} value={tel?.ear || 0} max={0.45} warnBelow={earThr} />
          <Bar label="MAR (yawn > 0.6)" value={tel?.mar || 0} max={1} danger={(v) => v > 0.6} />
        </div>
      </div>
      <dl className="mt-4 grid grid-cols-3 gap-3 text-center text-sm">
        {[
          ["PERCLOS", `${Math.round((tel?.perclos_30s || 0) * 100)}%`],
          ["Eyes closed", `${((tel?.eyes_closed_ms || 0) / 1000).toFixed(1)}s`],
          ["Yawns / 30s", tel?.yawns_30s ?? 0],
          ["Yaw", `${tel?.head_pose?.yaw?.toFixed?.(0) ?? 0}°`],
          ["Pitch", `${tel?.head_pose?.pitch?.toFixed?.(0) ?? 0}°`],
          ["Slump", `${Math.round((tel?.posture?.slump_score || 0) * 100)}%`],
        ].map(([k, v]) => (
          <div key={k} className="rounded-lg bg-slate-800/60 py-2">
            <dt className="text-[10px] uppercase tracking-wider text-slate-400">{k}</dt>
            <dd className="font-mono text-base">{v}</dd>
          </div>
        ))}
      </dl>
    </Card>
  );
}
