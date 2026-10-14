# Context-Aware Multi-Agent Companion for Drivers

Fully local driver-monitoring dashboard + voice companion (final year project). A vision pipeline estimates fatigue from the driver's face and posture, a Safety Agent preempts with alerts, and a Companion Agent (local LLM) chats by voice or text with long-term memory. Everything runs on-device; cloud sync is optional and encrypted.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design.

## Stack

- **Backend:** Python, FastAPI + WebSocket, asyncio event bus, SQLite, LanceDB
- **Vision:** MediaPipe FaceMesh/Pose, temporal fatigue model
- **Audio:** faster-whisper (STT), Piper/pyttsx3 (TTS), speech-emotion model
- **LLM:** llama.cpp server (OpenAI-compatible)
- **Dashboard:** React + Tailwind (Vite)

## Prerequisites

Python 3.10-3.12, Node.js 18+. Large assets (llama.cpp binaries, Piper voices) are not in the repo; download them separately.

## Run (works out of the box in simulator mode)

```powershell
# backend
cd backend
python -m venv .venv ; .\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m app.main          # http://127.0.0.1:8000

# dashboard (dev, hot reload on :5173)
cd dashboard ; npm install ; npm run dev
# or: npm run build   -> the backend then serves the dashboard itself at :8000
```

With no extra setup you get the simulated camera (scenario buttons in the UI), the Safety Agent,
text chat, memory, SQLite logging, LanceDB memory and encrypted sync (to `data/cloud_outbox/`).

## Going fully real (all optional, all local)

| Feature | Setup |
|---|---|
| Live camera | `pip install opencv-python mediapipe` (Python 3.10-3.12), then `DC_VISION_MODE=camera` |
| LLM | Run a llama.cpp server, e.g. `llama-server -m qwen2.5-7b-instruct-q4.gguf --port 8080` (default `DC_LLM_URL=http://127.0.0.1:8080/v1`) |
| Voice in | `pip install faster-whisper`, `DC_ENABLE_MIC=1` (`DC_WHISPER_MODEL=small`) |
| Speech emotion | `pip install transformers torch`, `DC_SER_MODEL=superb/wav2vec2-base-superb-er` (else a crude loudness heuristic) |
| Voice out | default pyttsx3 (OS voices); better: Piper, `DC_PIPER_MODELS="en=C:/piper/en.onnx,es=C:/piper/es.onnx"` |
| Real embeddings | `pip install fastembed` (else hashed-trigram fallback, weaker semantic search) |
| Cloud sync | `DC_CLOUD_URL=https://…` + `DC_CLOUD_TOKEN` (PUT `/profiles/<user_id>` with the encrypted blob) |
| Trained temporal model | `DC_TEMPORAL_WEIGHTS=model.pt` (GRU, see `vision/temporal_model.py`); otherwise a window-statistics heuristic is used |

## Known limitations

- No trained GRU checkpoint is included; the default fatigue score is a heuristic over the 30 s window (PERCLOS, longest closure, yawns, head-down, slump).
- Speech emotion without a model is only a rough energy heuristic.
- Posture needs shoulders in frame; otherwise it reads 0.
- Sync is upload-only; restoring a profile from the cloud is not implemented yet.
- Tested here: backend in simulator mode (safety trigger, chat, memory, DB) and the dashboard build. The camera, mic, Piper and a real LLM were not tested.
