import React from "react";
import SettingsSheet from "./SettingsSheet.jsx";

function GearIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9 7 7M17 17l2.1 2.1M19.1 4.9 17 7M7 17l-2.1 2.1" />
    </svg>
  );
}

function CancelIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
      <path d="M6 6l12 12M18 6 6 18" />
    </svg>
  );
}

function MicIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
    </svg>
  );
}

export default function CockpitView({ state, status, alertText, connected, canSettings, settingsOpen,
  onOpenSettings, onCloseSettings, onCancel, onMic, micArmed, memTick, sync, savedBaseline, onRecalibrate }) {
  return (
    <main className="cockpit-view">
      <header className="edge-bar">
        <span className={`status-dot${connected ? " is-active" : " is-offline"}`} aria-label={connected ? "Monitoring active" : "Monitoring offline"} />
        <button className="gear-button" onClick={onOpenSettings} disabled={!canSettings}
          title={canSettings ? "Local settings" : "Settings available when risk is low"}
          aria-label="Open local settings">
          <GearIcon />
        </button>
      </header>

      <section className="cockpit-stage">
        {state === "alert" && <div className="alert-banner">{alertText}</div>}
        <div className="stage-center">
          <div className={`orb orb--${state}`} aria-hidden="true">
            <span className="orb-layer orb-layer-a" />
            <span className="orb-layer orb-layer-b" />
            <span className="orb-core" />
          </div>
          <p className="cockpit-status">{status}</p>
        </div>
      </section>

      <footer className="dock">
        <button className="dock-button dock-button--cancel" onClick={onCancel} aria-label="Dismiss or cancel">
          <CancelIcon />
        </button>
        <button className={`dock-button dock-button--mic${micArmed ? " is-active" : ""}`} onClick={onMic}
          aria-label="Toggle microphone" aria-pressed={micArmed}>
          <MicIcon />
        </button>
      </footer>

      <SettingsSheet open={settingsOpen} onClose={onCloseSettings} memTick={memTick} sync={sync}
        savedBaseline={savedBaseline} onRecalibrate={onRecalibrate} />
    </main>
  );
}
