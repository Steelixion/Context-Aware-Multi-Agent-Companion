import React from "react";

export const LEVEL = {
  ALERT:    { label: "Alert",    ring: "text-emerald-400", bg: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40" },
  MILD:     { label: "Mild",     ring: "text-yellow-300",  bg: "bg-yellow-500/15 text-yellow-200 border-yellow-500/40" },
  MODERATE: { label: "Moderate", ring: "text-orange-400",  bg: "bg-orange-500/15 text-orange-300 border-orange-500/40" },
  HIGH:     { label: "DROWSY",   ring: "text-red-500",     bg: "bg-red-500/20 text-red-300 border-red-500/60" },
};

export function Card({ title, right, children, className = "" }) {
  return (
    <section className={`rounded-2xl border border-slate-800 bg-slate-900/70 p-4 ${className}`}>
      {(title || right) && (
        <header className="mb-3 flex items-center justify-between">
          <h2 className="text-xs font-semibold uppercase tracking-widest text-slate-400">{title}</h2>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

export function Chip({ children, tone = "slate" }) {
  const t = {
    slate: "border-slate-700 text-slate-300",
    green: "border-emerald-600/50 text-emerald-300",
    red: "border-red-600/50 text-red-300",
    amber: "border-amber-600/50 text-amber-300",
  }[tone];
  return <span className={`rounded-full border px-2.5 py-0.5 text-xs ${t}`}>{children}</span>;
}

export function Bar({ label, value, max = 1, warnBelow, danger, fmt = (v) => v.toFixed(2) }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  const bad = danger ? danger(value) : warnBelow != null && value < warnBelow;
  return (
    <div>
      <div className="mb-1 flex justify-between text-sm">
        <span className="text-slate-400">{label}</span>
        <span className={`font-mono ${bad ? "text-red-400" : "text-slate-100"}`}>{fmt(value)}</span>
      </div>
      <div className="h-2 overflow-hidden rounded bg-slate-800">
        <div className={`h-full transition-all duration-150 ${bad ? "bg-red-500" : "bg-sky-500"}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
