import React, { useCallback, useEffect, useState } from "react";
import { Card, Chip } from "./ui.jsx";

const fmt = (ts) => (ts ? new Date(ts * 1000).toLocaleTimeString() : "—");

export function Memories({ tick }) {
  const [items, setItems] = useState([]);
  const [text, setText] = useState("");
  const load = useCallback(() => fetch("/api/memories").then((r) => r.json()).then(setItems).catch(() => {}), []);
  useEffect(() => { load(); }, [tick, load]);

  const add = async (e) => {
    e.preventDefault();
    if (!text.trim()) return;
    await fetch("/api/memories", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
    setText(""); load();
  };
  const del = async (id) => { await fetch(`/api/memories/${id}`, { method: "DELETE" }); load(); };

  return (
    <Card title="Long-term memory" right={<Chip>{items.length} facts</Chip>}>
      <ul className="max-h-56 space-y-1.5 overflow-y-auto text-sm">
        {items.length === 0 && <li className="text-slate-500">Nothing yet — the Memory Agent learns from your conversations.</li>}
        {items.map((m) => (
          <li key={m.memory_id} className="group flex items-start justify-between gap-2 rounded-lg bg-slate-800/60 px-3 py-1.5">
            <span>{m.context_text} <span className="text-xs text-slate-500">· {m.category}</span></span>
            <button onClick={() => del(m.memory_id)} className="text-slate-500 opacity-0 hover:text-red-400 group-hover:opacity-100">✕</button>
          </li>
        ))}
      </ul>
      <form onSubmit={add} className="mt-3 flex gap-2">
        <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Add a fact manually…"
          className="min-w-0 flex-1 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm outline-none focus:border-sky-500" />
        <button className="rounded-lg border border-slate-700 px-3 text-sm hover:bg-slate-800">Add</button>
      </form>
    </Card>
  );
}

export function SyncPanel({ sync }) {
  const trigger = () => fetch("/api/sync/now", { method: "POST" });
  return (
    <Card title="Profile sync" right={<Chip tone={sync?.online ? "green" : "amber"}>{sync?.online ? "online" : "offline — local only"}</Chip>}>
      <dl className="space-y-1 text-sm">
        <div className="flex justify-between"><dt className="text-slate-400">Last encrypted sync</dt><dd>{fmt(sync?.last_success)}</dd></div>
        <div className="flex justify-between gap-4"><dt className="text-slate-400">Target</dt><dd className="truncate font-mono text-xs">{sync?.target || "—"}</dd></div>
        {sync?.last_error && <p className="text-xs text-red-400">{sync.last_error}</p>}
      </dl>
      <button onClick={trigger} disabled={sync?.syncing}
        className="mt-3 w-full rounded-lg border border-slate-700 py-1.5 text-sm hover:bg-slate-800 disabled:opacity-50">
        {sync?.syncing ? "Syncing…" : "Sync now (when online)"}
      </button>
      <p className="mt-2 text-xs text-slate-500">AES-256-GCM encrypted before leaving the device. The car works fully offline.</p>
    </Card>
  );
}

export function History({ alerts }) {
  const [data, setData] = useState({ records: [], logs: [] });
  useEffect(() => { fetch("/api/history").then((r) => r.json()).then(setData).catch(() => {}); }, [alerts.length]);
  return (
    <Card title="Monitoring log" right={<Chip>{data.records.length} sessions</Chip>}>
      <div className="max-h-56 overflow-y-auto">
        <table className="w-full text-left text-xs">
          <thead className="sticky top-0 bg-slate-900 text-slate-400">
            <tr><th className="py-1">Time</th><th>State</th><th>Level</th><th>Fatigue</th><th>Safety prompt</th></tr>
          </thead>
          <tbody>
            {data.logs.map((l) => (
              <tr key={l.log_id} className="border-t border-slate-800">
                <td className="py-1 font-mono">{fmt(l.ts)}</td><td>{l.driver_state}</td>
                <td className={l.drowsiness_level === "HIGH" ? "text-red-400" : ""}>{l.drowsiness_level}</td>
                <td>{l.fatigue_score?.toFixed(2)}</td><td className="max-w-[14rem] truncate">{l.safety_prompt || ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
