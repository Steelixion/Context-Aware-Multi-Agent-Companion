import React, { useCallback, useEffect, useState } from "react";

export default function SettingsSheet({ open, onClose, memTick, sync, savedBaseline, onRecalibrate }) {
  const [items, setItems] = useState([]);
  const load = useCallback(() => fetch("/api/memories").then((r) => r.json()).then(setItems).catch(() => {}), []);

  useEffect(() => { if (open) load(); }, [open, memTick, load]);
  if (!open) return null;

  const remove = async (id) => { await fetch(`/api/memories/${id}`, { method: "DELETE" }); load(); };
  const syncNow = () => fetch("/api/sync/now", { method: "POST" });

  return (
    <div className="sheet-backdrop" onMouseDown={onClose}>
      <section className="settings-sheet" onMouseDown={(e) => e.stopPropagation()}>
        <div className="sheet-head">
          <h2>Local settings</h2>
          <button className="sheet-close" onClick={onClose} aria-label="Close settings">×</button>
        </div>
        <div className="sheet-body">
          <div className="sheet-row">
            <span>Driver baseline</span>
            <strong>{savedBaseline ? "Saved" : "Not set"}</strong>
          </div>
          <button className="sheet-action" onClick={onRecalibrate}>Recalibrate baseline</button>
          <h3>Saved facts</h3>
          <ul className="memory-list">
            {items.length === 0 && <li className="memory-empty">No saved facts yet.</li>}
            {items.map((m) => (
              <li key={m.memory_id}>
                <span>{m.context_text}</span>
                <button onClick={() => remove(m.memory_id)} aria-label={`Delete ${m.context_text}`}>×</button>
              </li>
            ))}
          </ul>
          <button className="sheet-action" onClick={syncNow} disabled={sync?.syncing}>
            {sync?.syncing ? "Syncing…" : "Sync local data"}
          </button>
          <p className="sheet-note">{sync?.online ? "Cloud target ready." : "Offline — data stays on this device."}</p>
        </div>
      </section>
    </div>
  );
}
