import React from "react";
import { Card, Chip } from "./ui.jsx";

const MODES = [["normal", "Alert"], ["yawning", "Yawning"], ["nodding", "Head nod"], ["drowsy", "Drowsy"], ["no_face", "No face"]];

export default function Camera({ frame, info, tel, setSim }) {
  const sim = info?.vision_mode === "sim";
  return (
    <Card title="In-cabin camera" right={<Chip tone={sim ? "amber" : "green"}>{sim ? "simulator" : "live camera"}</Chip>}>
      <div className="flex aspect-video items-center justify-center overflow-hidden rounded-lg bg-black">
        {frame ? <img alt="driver" className="h-full w-full object-cover" src={`data:image/jpeg;base64,${frame}`} />
          : <span className="px-6 text-center text-sm text-slate-500">
              {sim ? "Simulator mode: no video. Use the scenarios below to drive the pipeline." : "Waiting for camera…"}
            </span>}
      </div>
      {sim && (
        <div className="mt-3 flex flex-wrap gap-2">
          {MODES.map(([m, label]) => (
            <button key={m} onClick={() => setSim(m)}
              className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm hover:border-sky-500 hover:bg-slate-700">
              {label}
            </button>
          ))}
        </div>
      )}
    </Card>
  );
}
