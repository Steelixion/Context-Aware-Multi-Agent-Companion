import React from "react";
import { useCompanion } from "./useCompanion.js";
import { Card, Chip } from "./components/ui.jsx";
import DriverState from "./components/DriverState.jsx";
import Timeline from "./components/Timeline.jsx";
import Camera from "./components/Camera.jsx";
import ChatPane from "./components/ChatPane.jsx";
import { History, Memories, SyncPanel } from "./components/Panels.jsx";

const LEGEND = [
  ["bg-emerald-500", "face mesh (478)"],
  ["bg-cyan-400", "eyes and gaze"],
  ["bg-pink-400", "mouth / MAR"],
  ["bg-red-500", "head axes"],
  ["bg-amber-300", "pose and shoulders"],
];

const num = (v) => (typeof v === "number" ? v.toFixed(2) : "—");
const deg = (v) => (typeof v === "number" ? `${v.toFixed(0)}°` : "—");

function LandmarkView({ frame, tel }) {
  const posture = tel?.posture;
  const gaze = tel?.gaze;
  const head = tel?.head_pose;
  return (
    <Card title="MediaPipe landmarks"
      right={<Chip tone={!tel ? "slate" : posture?.visible ? "green" : "amber"}>
        {!tel ? "waiting" : posture?.visible ? "pose visible" : "shoulders not visible"}
      </Chip>}>
      <div className="flex aspect-video items-center justify-center overflow-hidden rounded-lg bg-black">
        {frame ? <img alt="mediapipe landmarks" className="h-full w-full object-cover"
          src={`data:image/jpeg;base64,${frame}`} />
          : <span className="px-6 text-center text-sm text-slate-500">Waiting for camera landmarks…</span>}
      </div>
      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-400">
        {LEGEND.map(([color, label]) => (
          <span key={label} className="inline-flex items-center gap-1.5">
            <span className={`h-2 w-2 rounded-full ${color}`} />{label}
          </span>
        ))}
      </div>
      <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
        <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-2">
          <div className="mb-1 text-[10px] uppercase tracking-widest text-slate-500">Head</div>
          <div className="font-mono">yaw {deg(head?.yaw)} · pitch {deg(head?.pitch)} · roll {deg(head?.roll)}</div>
        </div>
        <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-2">
          <div className="mb-1 text-[10px] uppercase tracking-widest text-slate-500">Gaze</div>
          <div className="font-mono">h {num(gaze?.h)} · v {num(gaze?.v)}</div>
        </div>
        <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-2">
          <div className="mb-1 text-[10px] uppercase tracking-widest text-slate-500">Posture</div>
          <div className="font-mono">tilt {deg(posture?.shoulder_tilt)} · lean {num(posture?.lean)}</div>
        </div>
      </div>
    </Card>
  );
}

export default function App() {
  const c = useCompanion();

  return (
    <div className="mx-auto max-w-7xl p-4 lg:p-6">
      <header className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-xl font-bold">Driver Companion</h1>
          <p className="text-xs text-slate-400">Fully on-device · {c.info?.name || "Driver"}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Chip tone={c.connected ? "green" : "red"}>{c.connected ? "connected" : "reconnecting…"}</Chip>
          <Chip>vision: {c.info?.vision_mode || "—"}</Chip>
          <Chip>stt: {c.stt || "—"}</Chip>
          <Chip>tts: {c.info?.tts || "—"}</Chip>
          <Chip>embeddings: {c.info?.embedder?.split("/").pop() || "—"}</Chip>
        </div>
      </header>

      <main className="grid gap-4 lg:grid-cols-3">
        <div className="grid gap-4 md:grid-cols-2 lg:col-span-3">
          <Camera frame={c.frameRaw} info={c.info} tel={c.tel} setSim={c.setSim} title="Raw camera" />
          <LandmarkView frame={c.frame} tel={c.tel} />
        </div>
        <div className="lg:col-span-1"><DriverState tel={c.tel} calibration={c.calibration} calibrate={c.calibrate} /></div>
        <div className="lg:col-span-2"><Timeline series={c.series} /></div>
        <div className="h-[26rem] lg:col-span-3">
          <ChatPane chat={c.chat} speaking={c.speaking} llmOnline={c.llmOnline} />
        </div>
        <div className="lg:col-span-3"><History alerts={c.alerts} /></div>
        <div className="lg:col-span-1"><Memories tick={c.memTick} /></div>
        <div className="lg:col-span-2"><SyncPanel sync={c.sync} /></div>
      </main>
    </div>
  );
}
