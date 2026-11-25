import React from "react";

export default function CalibrationView({ progress, blocked, savedBaseline, status, onSkip, onRecalibrate }) {
  const sweep = `${Math.round(Math.max(0, Math.min(1, progress)) * 360)}deg`;

  return (
    <main className="calibration-view">
      <div className="calibration-stack">
        <div className={`reticle${blocked ? " reticle--blocked" : ""}`} style={{ "--sweep": sweep }} aria-hidden="true">
          <span className="reticle-tick tick-tl" />
          <span className="reticle-tick tick-tr" />
          <span className="reticle-tick tick-bl" />
          <span className="reticle-tick tick-br" />
          <div className="reticle-progress" />
          <div className="reticle-core" />
          <div className="reticle-cross" />
        </div>
        <p className="calibration-line">Look straight ahead in a natural driving posture</p>
        <p className="calibration-sub">{status}</p>
      </div>
      <button className="pill-action" onClick={savedBaseline ? onSkip : onRecalibrate}>
        {savedBaseline ? "Skip" : "Recalibrate"}
      </button>
    </main>
  );
}
