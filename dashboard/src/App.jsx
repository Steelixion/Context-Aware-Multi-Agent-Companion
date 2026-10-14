import React from "react";
import { useCompanion } from "./useCompanion.js";
import { Chip } from "./components/ui.jsx";
import DriverState from "./components/DriverState.jsx";
import Timeline from "./components/Timeline.jsx";
import Camera from "./components/Camera.jsx";
import ChatPane from "./components/ChatPane.jsx";
import { History, Memories, SyncPanel } from "./components/Panels.jsx";

export default function App() {
  const c = useCompanion();
  const critical = c.alert?.severity === "CRITICAL";

  return (
    <div className="mx-auto max-w-7xl p-4 lg:p-6">
      {critical && <div className="flash pointer-events-none fixed inset-0 z-40 bg-red-600" />}
      {c.alert && (
        <div className={`fixed inset-x-0 top-0 z-50 px-6 py-4 text-center text-xl font-bold shadow-lg
          ${critical ? "bg-red-600 text-white" : "bg-amber-500 text-black"}`} role="alert">
          {c.alert.text}
        </div>
      )}

      <header className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-xl font-bold">Driver Companion</h1>
          <p className="text-xs text-slate-400">Fully on-device · {c.info?.name || "Driver"}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Chip tone={c.connected ? "green" : "red"}>{c.connected ? "connected" : "reconnecting…"}</Chip>
          <Chip>vision: {c.info?.vision_mode || "—"}</Chip>
          <Chip>tts: {c.info?.tts || "—"}</Chip>
          <Chip>embeddings: {c.info?.embedder?.split("/").pop() || "—"}</Chip>
        </div>
      </header>

      <main className="grid gap-4 lg:grid-cols-3">
        <div className="space-y-4 lg:col-span-1">
          <Camera frame={c.frame} info={c.info} tel={c.tel} setSim={c.setSim} />
          <DriverState tel={c.tel} info={c.info} />
        </div>
        <div className="space-y-4 lg:col-span-2">
          <Timeline series={c.series} earThreshold={c.info?.ear_threshold} />
          <div className="h-[28rem]">
            <ChatPane chat={c.chat} sendChat={c.sendChat} speaking={c.speaking} info={c.info} />
          </div>
        </div>
        <div className="lg:col-span-1"><Memories tick={c.memTick} /></div>
        <div className="lg:col-span-1"><SyncPanel sync={c.sync} /></div>
        <div className="lg:col-span-1"><History alerts={c.alerts} /></div>
      </main>
    </div>
  );
}
