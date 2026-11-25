# Commands

How to set up, run and test the project. Commands are for Windows PowerShell; macOS/Linux equivalents are at the end.

## 1. Prerequisites

- Python 3.10 or newer (tested on 3.14)
- Node.js 18 or newer (for the dashboard)
- Git
- Optional: a [llama.cpp](https://github.com/ggml-org/llama.cpp) build and a GGUF model for the companion's replies
- Optional: a webcam and a microphone (the simulator works without either)

## 2. One-time setup

From the repository root:

```powershell
python scripts/download_models.py --test-assets
```

This downloads about 300 MB into `models/` (git-ignored). Every file is checked against a pinned SHA-256; run it again to resume or repair. `--test-assets` adds the three sample photos the vision tests use.

| Folder | Contents |
|---|---|
| `models/stt/` | Omnilingual ASR 300M CTC v2 int8 (speech-to-text) |
| `models/vad/` | Silero VAD v5 |
| `models/vision/` | MediaPipe Face Landmarker and Pose Landmarker (lite) |
| `models/test-assets/` | sample photos for the tests |

Backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\pip install -r requirements-dev.txt
cd ..
```

Use `requirements.txt` instead of `requirements-dev.txt` if you don't need the tests.

Dashboard:

```powershell
cd dashboard
npm install
npm run build
cd ..
```

After `npm run build`, the backend serves the dashboard itself at http://127.0.0.1:8000.

## 3. Run

All run commands start in `backend/`. Settings are environment variables (section 6). Set them in the same PowerShell window before starting.

**Simulator** (no camera or microphone needed; scenario buttons in the dashboard):

```powershell
.\.venv\Scripts\python -m app.main
```

Open http://127.0.0.1:8000. Stop with `Ctrl+C`.

**Live camera:**

```powershell
$env:DC_VISION_MODE = "camera"
.\.venv\Scripts\python -m app.main
```

Sit in front of the camera and look ahead normally for the first 10 seconds while the dashboard says "Calibrating". Your baselines are saved and reused next time; press **Recalibrate** after moving the camera or changing drivers.

**Live camera and voice:**

```powershell
$env:DC_VISION_MODE = "camera"
$env:DC_ENABLE_MIC = "1"
.\.venv\Scripts\python -m app.main
```

**A video file instead of the webcam** (demos, recorded drives):

```powershell
$env:DC_VISION_MODE = "camera"
$env:DC_VIDEO_FILE = "C:\path\to\drive.mp4"
.\.venv\Scripts\python -m app.main
```

**Local LLM** (in a second window; the companion uses it automatically, and the dashboard's "LLM online" chip turns green within 10 s):

```powershell
llama-server -m C:\models\your-model-Q4.gguf --port 8080 -c 4096
```

**Dashboard with hot reload** (for UI work; keep the backend running, then in another window):

```powershell
cd dashboard
npm run dev
```

Open http://localhost:5173. API and WebSocket calls are proxied to port 8000.

**Clear the settings** in the current window:

```powershell
Remove-Item Env:DC_* -ErrorAction SilentlyContinue
```

## 4. Tools

Transcribe audio files (any sample rate), or the live microphone, without the full app:

```powershell
.\.venv\Scripts\python -m app.audio.stt ..\models\stt\sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8-2026-02-05\test_wavs\en.wav
```

```powershell
.\.venv\Scripts\python -m app.audio.stt --mic
```

Start a new calibration while the app runs (same as the dashboard button):

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/calibrate
```

Download only one model group (`stt`, `vad`, `vision` or `test-assets`):

```powershell
python ..\scripts\download_models.py --only vision
```

## 5. Tests

```powershell
.\.venv\Scripts\python -m pytest
```

There are about 280 tests, taking roughly 70 s. Tests that need a model or the sample photos are skipped, with the reason, if those files are missing. Useful variants:

```powershell
.\.venv\Scripts\python -m pytest tests/test_monitor.py -v
```

```powershell
.\.venv\Scripts\python -m pytest -k "vad or stt"
```

## 6. Configuration

Every setting is an environment variable named `DC_<NAME>`.

| Variable | Default | Meaning |
|---|---|---|
| `DC_VISION_MODE` | `sim` | `sim` (simulator) or `camera` |
| `DC_CAMERA_INDEX` | `0` | webcam index |
| `DC_VIDEO_FILE` | | in camera mode, read this video file instead of a webcam |
| `DC_VISION_FPS` | `15` | vision frame rate |
| `DC_CALIBRATION_SECONDS` | `10` | length of the calibration phase |
| `DC_YAWN_SECONDS` | `3.0` | how long the mouth must stay open for a yawn |
| `DC_DISTRACTION_SECONDS` | `2.0` | eyes off the road this long counts as distraction |
| `DC_TEMPORAL_MODEL` | | trained TCN (`.onnx`); see `app/vision/temporal_model.py` for the input contract |
| `DC_FACE_MODEL`, `DC_POSE_MODEL` | `models/vision/...` | MediaPipe model files |
| `DC_ENABLE_MIC` | `0` | `1` turns on voice input |
| `DC_ENABLE_SPEAKER` | `1` | `0` disables audio output |
| `DC_ASR_MODEL_DIR` | `models/stt/...` | speech-to-text model folder |
| `DC_ASR_THREADS` | `4` | CPU threads for speech-to-text |
| `DC_VAD_MODEL` | `models/vad/silero_vad_v5.onnx` | VAD model |
| `DC_VAD_THRESHOLD` | `0.5` | speech probability threshold, 0-1 (higher = stricter) |
| `DC_LLM_URL` | `http://127.0.0.1:8080/v1` | OpenAI-compatible LLM server |
| `DC_LLM_MODEL` | `local` | model name sent to the server |
| `DC_PIPER_MODELS` | auto | `en=C:/voices/en.onnx,ur=...`; auto-detected in `models/tts/` or `tts/voices/` |
| `DC_SER_MODEL` | | Hugging Face speech-emotion model id (needs `transformers torch`) |
| `DC_EMBED_MODEL` | multilingual MiniLM | embedding model (needs `fastembed`) |
| `DC_DATA_DIR` | `data/` | SQLite database, vector memory, sync outbox |
| `DC_HOST`, `DC_PORT` | `127.0.0.1`, `8000` | server address |
| `DC_CLOUD_URL`, `DC_CLOUD_TOKEN` | | optional encrypted profile sync target |

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `stt: omnilingual (no model)` in the header | run `python scripts/download_models.py` |
| `stt: omnilingual (no mic)` | no input device found, or another app holds it exclusively |
| `stt: omnilingual + energy VAD` | the Silero model is missing; the energy fallback only works in a quiet room |
| "camera pipeline unavailable" / no face | check the camera isn't in use by another app and isn't covered, and the face is lit and in frame; try `DC_CAMERA_INDEX=1` |
| "Calibration failed" | the face wasn't visible for enough of the 10 s; face the camera and press Recalibrate |
| Posture shows "shoulders not visible" | move the camera so both shoulders are in frame; posture is then monitored too |
| "LLM offline" | start `llama-server` on port 8080 (or set `DC_LLM_URL`) |
| Port 8000 in use | `$env:DC_PORT = "8001"` |

## 8. macOS / Linux

The same steps with POSIX paths:

```bash
python3 scripts/download_models.py --test-assets
cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt && cd ..
cd dashboard && npm install && npm run build && cd ..
cd backend && DC_VISION_MODE=camera DC_ENABLE_MIC=1 .venv/bin/python -m app.main
```
