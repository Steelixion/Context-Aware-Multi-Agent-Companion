"""Central configuration. Every value can be overridden with a DC_* env var."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default, cast=str):
    v = os.getenv(f"DC_{name}")
    return default if v is None else cast(v)


def _discover_voices() -> str:
    """Map '<lang>_<REGION>-name-quality.onnx' files in tts/voices to 'lang=path'."""
    found: dict[str, str] = {}
    for f in sorted((ROOT / "tts" / "voices").glob("*.onnx")):
        found.setdefault(f.name.split("_")[0], str(f))
    return ",".join(f"{k}={v}" for k, v in found.items())


def _bool(v: str) -> bool:
    return v.lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", ROOT / "data", Path)))
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env("PORT", 8000, int))

    # --- vision ---
    vision_mode: str = field(default_factory=lambda: _env("VISION_MODE", "sim"))  # camera | sim
    camera_index: int = field(default_factory=lambda: _env("CAMERA_INDEX", 0, int))
    vision_fps: int = field(default_factory=lambda: _env("VISION_FPS", 15, int))
    window_seconds: int = 30
    temporal_weights: str = field(default_factory=lambda: _env("TEMPORAL_WEIGHTS", ""))

    # --- safety rule (spec: EAR < 0.20 for > 1.5 s) ---
    ear_threshold: float = field(default_factory=lambda: _env("EAR_THRESHOLD", 0.20, float))
    ear_seconds: float = field(default_factory=lambda: _env("EAR_SECONDS", 1.5, float))
    fatigue_high: float = 0.80
    watchdog_seconds: float = 3.0       # no telemetry for this long -> sensor lost warning
    face_lost_seconds: float = 6.0
    warning_repeat_seconds: float = 6.0
    chat_mute_seconds: float = 10.0     # chat stays muted this long after a safety warning

    # --- LLM (llama.cpp server, OpenAI-compatible) ---
    llm_url: str = field(default_factory=lambda: _env("LLM_URL", "http://127.0.0.1:8080/v1"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "local"))

    # --- audio ---
    enable_mic: bool = field(default_factory=lambda: _env("ENABLE_MIC", False, _bool))
    enable_speaker: bool = field(default_factory=lambda: _env("ENABLE_SPEAKER", True, _bool))
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "small"))
    piper_bin: str = field(default_factory=lambda: _env("PIPER_BIN", "piper"))
    # "en=path.onnx,es=path.onnx"; if unset, voices in <repo>/tts/voices are auto-detected
    piper_models: str = field(default_factory=lambda: _env("PIPER_MODELS", _discover_voices()))
    ser_model: str = field(default_factory=lambda: _env("SER_MODEL", ""))  # HF audio-classification id

    # --- memory ---
    embed_model: str = field(default_factory=lambda: _env(
        "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
    memory_extract_every: int = 4       # turns
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
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}
