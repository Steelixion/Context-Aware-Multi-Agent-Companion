"""Central configuration. Every value can be overridden with a DC_<NAME> environment variable.

Model files live in <repo>/models (downloaded by scripts/download_models.py) and are found
automatically; the DC_*_MODEL variables override that.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"


def _env(name: str, default, cast=str):
    v = os.getenv(f"DC_{name}")
    return default if v is None or v == "" else cast(v)


def _bool(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def _first(*candidates: Path) -> str:
    return next((str(p) for p in candidates if p.is_file()), "")


def _discover_voices() -> str:
    """Map '<lang>_<REGION>-name-quality.onnx' Piper voices to 'lang=path' (models/tts or tts/voices)."""
    found: dict[str, str] = {}
    for d in (MODELS / "tts", ROOT / "tts" / "voices"):
        for f in sorted(d.glob("*.onnx")):
            found.setdefault(f.name.split("_")[0], str(f))
    return ",".join(f"{k}={v}" for k, v in found.items())


def _discover_asr() -> str:
    """First sherpa-onnx Parakeet TDT model dir (encoder/decoder/joiner/tokens) in models/stt."""
    needed = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
    dirs = [d for d in (MODELS / "stt").glob("*")
            if d.is_dir() and all((d / name).is_file() for name in needed)]
    dirs.sort(key=lambda d: d.name)
    return str(dirs[0]) if dirs else ""


@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", ROOT / "data", Path)))
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env("PORT", 8000, int))

    # --- vision: source ---
    vision_mode: str = field(default_factory=lambda: _env("VISION_MODE", "sim"))  # camera | sim
    camera_index: int = field(default_factory=lambda: _env("CAMERA_INDEX", 0, int))
    video_file: str = field(default_factory=lambda: _env("VIDEO_FILE", ""))  # camera mode: play a file instead
    vision_fps: int = field(default_factory=lambda: _env("VISION_FPS", 15, int))
    face_model: str = field(default_factory=lambda: _env(
        "FACE_MODEL", _first(MODELS / "vision" / "face_landmarker.task")))
    pose_model: str = field(default_factory=lambda: _env(
        "POSE_MODEL", _first(MODELS / "vision" / "pose_landmarker_lite.task",
                             MODELS / "vision" / "pose_landmarker_full.task")))
    pose_every: int = 5                    # posture changes slowly: run the pose model on every 5th frame
    temporal_model: str = field(default_factory=lambda: _env("TEMPORAL_MODEL", ""))  # trained TCN (.onnx)

    # --- vision: calibration and personalised thresholds (Implementation ch. 6.2) ---
    calibration_seconds: float = field(default_factory=lambda: _env("CALIBRATION_SECONDS", 10.0, float))
    calibration_timeout: float = 60.0      # give up if this long passes without enough usable frames
    ear_kappa: float = 0.80                # tau_EAR = kappa * EAR_base          (6.3)
    mar_delta: float = 0.25                # tau_MAR = MAR_base + delta_m        (6.4)

    # --- vision: events (6.3) ---
    perclos_seconds: float = 60.0          # PERCLOS / event-counter window      (6.15)
    blink_max_seconds: float = 0.5         # shorter closures are ordinary blinks
    long_closure_seconds: float = 1.5      # longer closures are safety events (microsleep)
    yawn_seconds: float = field(default_factory=lambda: _env("YAWN_SECONDS", 1.2, float))
    head_drop_degrees: float = 15.0        # downward pitch below the baseline ...
    head_drop_seconds: float = 1.0         # ... held this long is a head drop
    gaze_tolerance: float = 0.10           # |g_h| road-attentive bound          (6.11)
    gaze_v_tolerance: float = 0.20         # |g_v| road-attentive bound
    yaw_tolerance: float = 15.0            # |yaw| road-attentive bound (degrees)
    distraction_seconds: float = field(default_factory=lambda: _env("DISTRACTION_SECONDS", 1.2, float))
    posture_tolerances: tuple = (8.0, 0.15, 0.20)  # tau_theta (deg), tau_lambda, tau_delta (6.14)
    posture_seconds: float = 2.5           # D_t > 1 this long = posture instability
    quality_seconds: float = 2.0           # telemetry quality = share of usable frames over this span
    quality_min: float = 0.5               # below this, no drowsiness alerts (no false alarms)

    # --- vision: temporal estimation and risk (6.4) ---
    ema_alpha: float = 0.30                # p_hat = a * p + (1 - a) * p_hat     (6.16)
    risk_mild: float = 0.50                # R >= 0.50 -> MILD                   (Table 6.2)
    risk_high: float = 0.85                # R >= 0.85 -> HIGH

    # --- safety agent ---
    watchdog_seconds: float = 3.0          # no telemetry for this long -> sensor lost warning
    face_lost_seconds: float = 6.0
    warning_repeat_seconds: float = 6.0
    chat_mute_seconds: float = 10.0        # chat stays muted this long after a safety warning
    mild_checkin_seconds: float = 20.0     # MILD this long -> companion check-in  (Table 6.2)
    mild_checkin_cooldown: float = 60.0    # ... then no check-in for this long
    distraction_cooldown: float = 10.0
    posture_cooldown: float = 60.0

    # --- LLM (llama.cpp server, OpenAI-compatible) ---
    llm_url: str = field(default_factory=lambda: _env("LLM_URL", "http://127.0.0.1:8080/v1"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "local"))

    # --- audio (speech-to-text runs on the CPU only) ---
    enable_mic: bool = field(default_factory=lambda: _env("ENABLE_MIC", False, _bool))
    enable_speaker: bool = field(default_factory=lambda: _env("ENABLE_SPEAKER", True, _bool))
    asr_model_dir: str = field(default_factory=lambda: _env("ASR_MODEL_DIR", _discover_asr()))
    asr_threads: int = field(default_factory=lambda: _env("ASR_THREADS", 2, int))
    vad_model: str = field(default_factory=lambda: _env(
        "VAD_MODEL", _first(MODELS / "vad" / "silero_vad_v5.onnx", MODELS / "vad" / "silero_vad.onnx")))
    vad_threshold: float = field(default_factory=lambda: _env("VAD_THRESHOLD", 0.5, float))
    piper_bin: str = field(default_factory=lambda: _env("PIPER_BIN", "piper"))
    # "en=path.onnx,es=path.onnx"; if unset, voices in models/tts or tts/voices are auto-detected
    piper_models: str = field(default_factory=lambda: _env("PIPER_MODELS", _discover_voices()))
    ser_model: str = field(default_factory=lambda: _env("SER_MODEL", ""))  # HF audio-classification id

    # --- memory ---
    embed_model: str = field(default_factory=lambda: _env(
        "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
    memory_extract_every: int = 4          # turns
    memory_idle_seconds: float = 45.0

    # --- sync ---
    cloud_url: str = field(default_factory=lambda: _env("CLOUD_URL", ""))
    cloud_token: str = field(default_factory=lambda: _env("CLOUD_TOKEN", ""))
    probe_host: str = field(default_factory=lambda: _env("PROBE_HOST", "1.1.1.1"))
    sync_interval: float = field(default_factory=lambda: _env("SYNC_INTERVAL", 300.0, float))

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "driver.db"

    @property
    def lance_path(self) -> Path:
        return self.data_dir / "vectors"

    def vision_dict(self) -> dict:
        """Picklable settings for the vision process."""
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}
