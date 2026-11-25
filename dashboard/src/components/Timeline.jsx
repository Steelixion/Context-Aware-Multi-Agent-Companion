import React from "react";
import { Card } from "./ui.jsx";

function path(series, key, max, w, h) {
  if (series.length < 2) return "";
  return series.map((p, i) => {
    const x = (i / (series.length - 1)) * w;
    const y = h - Math.min(1, (p[key] || 0) / max) * h;
    return `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
}

export default function Timeline({ series }) {
  const w = 600, h = 120;
  const earThreshold = (series.length && series[series.length - 1].thr) || 0.21;
  const thrY = h - (earThreshold / 0.45) * h;
  return (
    <Card title="Last 30 seconds" right={
      <div className="flex gap-3 text-xs">
        <span className="text-sky-400">● EAR</span><span className="text-orange-400">● Drowsiness risk</span>
      </div>}>
      <svg viewBox={`0 0 ${w} ${h}`} className="h-32 w-full" preserveAspectRatio="none">
        <line x1="0" x2={w} y1={thrY} y2={thrY} stroke="#ef4444" strokeDasharray="4 4" strokeOpacity=".6" />
        <path d={path(series, "ear", 0.45, w, h)} fill="none" stroke="#38bdf8" strokeWidth="1.8" vectorEffect="non-scaling-stroke" />
        <path d={path(series, "risk", 1, w, h)} fill="none" stroke="#fb923c" strokeWidth="2.2" vectorEffect="non-scaling-stroke" />
      </svg>
      <p className="mt-1 text-xs text-slate-500">Red dashed line = this driver&apos;s eye-closure threshold ({earThreshold.toFixed(2)}).</p>
    </Card>
  );
}
