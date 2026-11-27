# Phase 0 findings: how the app can be tested as a black box

Investigated at app commit `164079f` (`main`, fast-forwarded from `origin/main` on 2026-11-27). Everything below comes from reading the code under `backend/`, `dashboard/`, `scripts/` and the docs, plus three short probe runs of the unmodified app (described in §8). Nothing in the app tree was changed.

Each item is labelled:
- **[code]**: read from the source, with the file and line.
- **[measured]**: seen in a probe run on this machine.
- **[assumption]**: my decision where the code doesn't settle it. I'll go ahead on these unless you say otherwise.
- **[question]**: I'm blocked until you answer. These are collected in §9.

---

## 1. Starting the app headless

- Entry point: `cd backend; .\.venv\Scripts\python -m app.main`. It runs uvicorn on `DC_HOST:DC_PORT` (default `127.0.0.1:8000`) **[code: app/main.py]**. No browser is needed. The dashboard is optional static hosting.
- All configuration comes from `DC_*` environment variables read once in `Config()` **[code: app/config.py]**. The ones the harness will set per test:

| Variable | Harness value | Why |
|---|---|---|
| `DC_DATA_DIR` | `results/<run>/work/<case>/<rep>/data` | isolates SQLite, LanceDB, outbox, key file |
| `DC_PORT` | a free port per test | parallel-safe, no clash with a dev instance |
| `DC_VISION_MODE` | `camera` | real MediaPipe path |
| `DC_CAMERA_INDEX` or `DC_VIDEO_FILE` | see §2 | video input |
| `DC_VISION_FPS` | from `thresholds.yaml` (default 15, the app default) | the app's processing rate |
| `DC_ENABLE_MIC` | `1` for audio scenarios, `0` otherwise | voice input |
| `DC_ENABLE_SPEAKER` | `1` (audio cases) | real playback, so the loopback capture can time it |
| `DC_LLM_URL` | the harness's recording proxy (§6) in front of the real LLM server | timestamps LLM tokens without code changes |
| `DC_LLM_MODEL`, `DC_ASR_MODEL_DIR`, `DC_VAD_MODEL`, `DC_TEMPORAL_MODEL`, `DC_PIPER_MODELS` | passed through from the eval config | the models under test. Recorded in run metadata |
| `DC_PROBE_HOST` | `127.0.0.1` | stops the sync manager probing `1.1.1.1` (a network call) |

- **Readiness** (no `/health` route exists) **[code: app/api/server.py, runtime.py]**:
  1. HTTP is up once `GET /api/session` returns 200. The lifespan hook (`rt.start()`) finishes before uvicorn accepts requests, and that includes pre-rendering the alert lines.
  2. Vision is up at the first `telemetry` WebSocket message. The vision process is spawned in `rt.start()` and loads MediaPipe after HTTP is already up.
  3. STT is up when a WS `stt` event arrives with `status == "listening"`. `"no model"` or `"no mic"` means it is not usable, and the case is marked ERROR.
  4. The LLM is up when `GET /api/session` shows `llm_online == true`. It is polled every 10 s **[code: runtime.py `_llm_watch`]**.
- **Shutdown.** **[measured]** A hard kill (the probe used `timeout`) skips the lifespan shutdown, and `monitoring_records.ended_at` stays NULL. **[assumption]** The harness will start the app in a new process group, send `CTRL_BREAK_EVENT`, wait up to 15 s, then kill it. I'll confirm in Phase 2 that uvicorn handles CTRL_BREAK on Windows and record the result. The `ended_at` column isn't scored anyway.
- stdout/stderr carry useful lines: `[vision] running: source=..., temporal model=<name>, profile=...`, `[stt] loaded <asr dir> + <vad> VAD`, `[stt] 2.1s audio -> 640 ms ...: <text>`, `[embed] ...`. They are saved per test and parsed for metadata only (model names, ASR compute time).

## 2. Feeding a specific video

**Option A: `DC_VIDEO_FILE` (built in)** **[code: vision/worker.py `VideoFileReader`, `main`]**
- `DC_VISION_MODE=camera` + `DC_VIDEO_FILE=<path>`.
- **It is not wall-clock playback.** The loop reads the *next* frame on every tick and sleeps to `1/DC_VISION_FPS`. Playback speed is therefore `DC_VISION_FPS` frames per second, whatever the file's own FPS. It drops no frames, so if a tick takes longer than the period, playback falls behind real time. At the end of the file it **loops back to frame 0**.
- **[measured]** A 640x480 30 fps clip with `DC_VISION_FPS=30` on an idle CPU: telemetry timestamps were about 0.098 s apart over the WS (every 3rd frame, see §4), so the app kept up at about 30 fps. Under load (LLM + ASR on the same CPU) it can't be guaranteed, and the drift isn't visible to the harness, because telemetry carries no frame index.

**Option B: virtual camera (recommended, will be the default)** **[measured: `OBS Virtual Camera` is installed here (DirectShow device list)]**
- The harness pushes frames at the clip's original FPS on its own monotonic clock with `pyvirtualcam` (OBS backend). The app reads it as a webcam: `DC_CAMERA_INDEX=<obs index>` and no `DC_VIDEO_FILE`.
- The app's webcam reader always takes the **newest** frame and drops stale ones **[code: `LatestFrameReader`]**, exactly as with a real camera. So input is real time, the harness knows exactly which source frame was on screen at every instant, and scoring maps telemetry time to video time with no drift.
- The harness can also stop or blank the feed (S11, S12) and prepend the calibration segment without writing any files.
- **[assumption]** I'll find the OBS camera index by probing indices with OpenCV and matching the test pattern, and record it. Option A stays available as a fallback (`video_source: file` in config), flagged in the report as "not wall-clock".
- `DC_VISION_FPS` is a property of the app under test (default 15), not of the input. Frames *arrive* at the original FPS either way.

## 3. Feeding audio without code changes

- **The app has no audio-file input or audio simulator.** The mic is `sounddevice.InputStream` on the **system default input device**, 16 kHz mono, 50 ms blocks **[code: audio/stt.py `SpeechListener._run`]**. There is no env var to pick a device. TTS also plays on the system default output **[code: audio/arbiter.py `_play`]**.
- **[measured]** This machine has no virtual audio cable. The devices are the Intel mic array and the Realtek speaker.
- Plan: install **VB-CABLE** (free) and make `CABLE Output` the default *recording* device while evals run. The harness plays scenario WAVs into `CABLE Input` in real-time 50 ms chunks. For the speaker side, the harness records the default output with **WASAPI loopback**, which gives true audible onset and offset of every reply and warning, and also the bytes of what the companion said.
- Self-hearing (S13) needs the companion's voice to reach the "mic". With a cable there is no acoustic path, so the harness will **feed the loopback capture back into `CABLE Input`** at a configured gain and delay (`echo.gain_db`, `echo.delay_ms` in thresholds.yaml). That simulates cabin echo deterministically. **[assumption]**
- Changing the Windows default recording device is a system setting, so it needs your OK (§9 Q2). The harness will check it before a run and stop with a clear message if it's wrong, rather than change it silently.
- Fallback that needs no device change: `POST /api/chat {"text": ...}` (or a WS `{"type":"chat"}` message) publishes the same `speech` bus event that ASR would **[code: api/server.py]**. It **bypasses VAD and ASR**, so cases run this way report ASR/endpointing as NOT OBSERVABLE (text injection), but the companion and safety logic is still exercised. Every result records which input path was used.

## 4. What is observable from outside

**WebSocket `/ws`**: forwards every bus topic as `{"type": topic, "data": msg}`, after a `snapshot` message **[code: api/server.py `ws`]**.

| WS `type` | Content | Timestamp in message | Notes |
|---|---|---|---|
| `telemetry` | full per-frame contract (`ts, face_detected, quality, ear, mar, head_pose, gaze, posture, perclos, eyes_closed_ms, head_drop_ms, fatigue_score, risk, drowsiness_level, calibration{state,progress,...}, source, note`) | `ts` = `time.time()` in the vision process | **throttled**: dropped if under 0.09 s after the last one sent. **[measured]** At 15 fps you get every 2nd frame (0.134 s gaps). At 30 fps, every 3rd |
| `state` | `{level, fatigue, risk, safety_active}`, one per telemetry message the Safety Agent evaluated | **none** (receive time only) | not throttled. Gives every level change, so level transitions are never missed |
| `alert` | `{ts, severity, reason, text}`. `reason` is one of `drowsy`, `distracted`, `posture`, `face_lost`, `sensor_lost` | `ts` = `time.time()` | the Safety Agent's decision instant |
| `speech` | `{text, lang, emotion, source, ts?}`. `source`: `voice` (ASR), `text` (API), `safety` (internal nudge prompt to the LLM) | `ts` only for `voice` (after ASR finishes) | **this is the ASR transcript event** |
| `chat` | `{id, role, text, final, ts, interrupted?, source?}`. User line, then an empty assistant bubble at generation start, then the final assistant text | `ts` = `time.time()` | **no token streaming**: only the full reply text at the end |
| `audio` | `{speaking: true/false, text, priority}` (0 = safety, 5 = chat) | **none** | `speaking:true` is sent **before** TTS synthesis of a non-cached line, so it isn't audible onset |
| `calibration`, `vision_event` | calibration start and result, frames used, profile | `ts` on the result | |
| `stt`, `llm`, `sync` | engine and status chips | none | readiness and metadata |
| `frame`, `frame_raw` | base64 JPEG previews (every 3rd frame) | none | ignored except for counting |

**HTTP**: `GET /api/session` (engines: `tts`, `stt`, `embedder`, `llm_online`), `/api/calibration`, `/api/history` (DB logs, last 60), `/api/memories`. `POST /api/chat`, `/api/calibrate`, `/api/sim`.

**SQLite (read-only, `mode=ro` URI)**: `monitoring_logs` has state changes (debounced 1 s) plus every `_raise` alert with its text. `calibration_profiles` has the stored profile. Advisories routed through the LLM (`drowsy`, `distracted`, `posture`) are **not** written with a prompt: only `face_lost` and `sensor_lost` go through `_raise`. **[code: agents/safety.py]**

**Clocks.** All app timestamps are `time.time()` on the same host. The harness records one `time.monotonic()` and `time.time()` pair at start and every 10 s, and converts with it. Message receive times are taken with `time.monotonic()`. **[measured]** Median WS delivery lag (receive time minus `telemetry.ts`) was 2.5 ms.

## 5. Clean state per test

- `DC_DATA_DIR` holds **all** per-driver state: `driver.db` (users, records, logs, calibration profiles), `vectors/` (LanceDB memories), `cloud_outbox/`, and `.profile.key` (used only if the OS keyring fails) **[code: config.py, sync/manager.py]**. A new empty temp directory per test and repeat gives a new user, a new session record, no memories and no stored calibration. **[measured]** The probe run created exactly those files under the temp dir.
- **Not isolated by `DC_DATA_DIR`**:
  - the OS keyring entry `driver-companion/profile-key`, which is shared but only encrypts sync bundles, so it doesn't matter here;
  - the LLM server's own state (llama.cpp slot/prompt cache). **[assumption]** The proxy (§6) logs every request, so cross-test leakage would be visible. The config can also restart the LLM server per case (off by default);
  - the fastembed model cache. **[measured]** fastembed isn't installed in the backend venv, so the app uses its hashed-trigram embedder, and the metadata records `embedder=hash-fallback`. If you install fastembed it downloads its model from Hugging Face on first use, which is a network call by the app, not the harness;
  - Windows default audio devices (§3).
- **Calibration** **[code: runtime.py, vision/worker.py, vision/calibration.py]**: with an empty data dir there is no stored profile, so the vision process calibrates on the first `DC_CALIBRATION_SECONDS × DC_VISION_FPS` usable frames (face found, |yaw| ≤ 35°, |pitch| ≤ 40°, timeout 60 s). The WS `calibration` event reports when it finishes and whether it succeeded. **Monitoring and alerts keep running during calibration with the default profile**, so the harness excludes that whole span from scoring, including any alerts in it.
- **[assumption]** Default calibration method: the virtual camera plays a neutral segment from the **same subject's** non-drowsy footage, then the harness waits for `calibration.state == calibrated` and then starts the test clip. If calibration fails, the case is ERROR. Alternative (`calibration: stored_profile`): run a calibration-only session once per subject into a template data dir, which the app writes itself, and copy that dir for each test. The test then starts calibrated from frame 0. The method used is recorded per case.

## 6. What can be timed externally

| Stage | Observable? | How |
|---|---|---|
| Harness speech start/end | yes | harness audio clock (chunks written to `CABLE Input`) |
| VAD end-of-utterance | **NOT OBSERVABLE** | not published. Only implied: the transcript arrives after 0.8 s of silence plus ASR time |
| Transcript ready | yes | WS `speech` (`source=voice`): `ts` and receive time |
| ASR compute time | partially | stdout `[stt] ... -> N ms`. Reported as app-logged, not measured |
| LLM request sent / first token / last token | yes, **via proxy only** | the harness runs a small pass-through HTTP proxy on `DC_LLM_URL` that forwards to the real server and timestamps the request and every streamed chunk. Not observable without the proxy |
| Reply text (full) | yes | WS `chat` final assistant message |
| First reply sentence queued for TTS | yes (approximately) | first WS `audio{speaking:true, priority:5}` after the turn. Includes nothing of synthesis |
| First audio actually audible | yes, **with loopback** | WASAPI loopback onset detection. NOT OBSERVABLE if the loopback is unavailable (then only the `audio` event time is reported, labelled as such) |
| Risk crosses HIGH | yes | first `state`/`telemetry` with `drowsiness_level == HIGH` |
| Safety decision | yes | WS `alert` `ts` |
| Warning audible | yes, with loopback | as above |
| Barge-in / preemption internals (queue flush, cancel) | partially | `chat` with `interrupted:true`, `audio{speaking:false}`, loopback silence. The queue contents themselves are NOT OBSERVABLE |

---

## 7. Behaviour that differs from the architecture document (from code reading, to be confirmed by runs)

You asked me not to change the app, so I've only noted these. Several scenarios will test them directly, and they will probably **fail** against the documented expectations. That's the eval doing its job, not a harness bug.

1. **HIGH drowsiness does not play the pre-rendered warning, and does not preempt.** On HIGH the Safety Agent publishes an `alert` and then sends a prompt to the **LLM companion** (`_advisory` → `_nudge` → `nudge_companion`). The spoken output is whatever the LLM generates, queued at **chat priority 5** through `arbiter.say`. `AudioArbiter.preempt()` (stop playback, mute chat for 10 s) is **never called anywhere** **[code: grep `preempt` finds only its definition]**. The `_cancel_companion` argument of `SafetyAgent` is unused, and the companion explicitly does *not* cancel a running reply for `source == "safety"` messages **[code: agents/companion.py `run`]**. Consequences: S05 preemption and the chat-mute window, and the "chat stopped and muted" claim, are not implemented.
2. **With the LLM offline, the HIGH "warning" is the offline reply** ("My language model isn't running right now, but I'm here…") and not the warning text **[code: companion.py `OFFLINE_REPLY`]**.
3. **The warning repeat is about 60 s, not 6 s.** HIGH is gated on `now - last_checkin >= mild_checkin_cooldown` (60 s). `warning_repeat_seconds` (6 s) is read nowhere **[code: safety.py:124]**.
4. **The MILD check-in timing differs.** The code nudges immediately on entering MILD (re-arm cooldown `min(60, 15)` = 15 s). The doc says "after 20 s, then not again for 120 s", and config has `mild_checkin_cooldown = 60`. `mild_checkin_seconds` (20 s) is unused.
5. **Any driver speech suppresses HIGH warnings for 60 s.** `user_spoke()` sets `last_checkin = now` on every non-safety `speech` event, so a driver who talks (or says "I'm fine") right before or during drowsiness gets no warning for 60 s. So do any MILD nudge and any visual-cue nudge. This bears directly on S02, S04 and S08.
6. **The watchdog only fires if the vision process stops sending telemetry.** A camera that stops delivering frames still produces telemetry every tick (`face_detected:false`, with a `note`). So the "camera signal lost" warning comes only from a vision-process crash or hang. A dead camera gives the `face_lost` advisory after 6 s instead **[code: worker.py, safety.py]**. With `DC_VIDEO_FILE` the video loops and never "stops".
7. The `sensor_lost` and `face_lost` lines go through `arbiter.say` at **chat priority**, not safety priority.

## 8. Probe runs (what I ran and saw)

All three used the backend venv (Python 3.14.0), a temp `DC_DATA_DIR` in the session scratchpad, and an unreachable `DC_LLM_URL`. Scripts were in the scratchpad, not the repo.

1. Sim mode, `DC_ENABLE_SPEAKER=0`, port 8765, WS listened for 25 s. Seen: `snapshot 1, telemetry 151, state 300, vision_event 1, calibration 1, sync 2`. Telemetry gaps were 0.134 s and delivery lag 2.5 ms. `/api/session` reported `tts=pyttsx3, stt=off, embedder=hash-fallback`. Calibration finished with 150 frames.
2. After the forced stop: the DB opened read-only showed `monitoring_records` 1 row with `ended_at = NULL`, `calibration_profiles` 1, `monitoring_logs` 0.
3. Camera mode, `DC_VIDEO_FILE` = a 40 s 640x480 30 fps clip built with ffmpeg from `models/test-assets/portrait.jpg`, `DC_VISION_FPS=30`. Telemetry gaps had a median of 0.098 s (about 30 fps processing). Also seen: `speech 1, chat 2, audio 2`, a Safety nudge on the static portrait.

**Environment facts that affect the eval:**
- `models/stt/` holds the **old Omnilingual** model. The Parakeet model the current code expects is not downloaded yet: `python scripts/download_models.py --only stt` (about 460 MB).
- No llama.cpp server is on this machine (`command.txt` refers to `~/llama.cpp`, which isn't here). Ollama is installed, with `qwen2.5:7b`.
- No Piper voices are installed, so the app's TTS is `pyttsx3` (Windows SAPI voices). The harness's own speech TTS must therefore not be SAPI.
- The dataset isn't local: `Dataset/Testing_Dataset` is a DVC pointer (210 files, 1.16 GB) on a Google Drive remote, and DVC isn't installed.

## 9. Open questions (blocking)

1. **Dataset access.** Should I install `dvc[gdrive]` into an isolated `eval/.venv` and run the three `dvc remote modify` commands you pasted, then `dvc pull Dataset/Testing_Dataset.dvc`? The `--local` flag writes the client ID and secret only to `.dvc/config.local`, which Git ignores. The first pull opens a Google sign-in in your browser, which you'd have to complete. Or, if you already have the videos on disk, give me the folder path and I'll skip DVC.
2. **Audio routing.** May I rely on VB-CABLE (you install it from vb-audio.com, and set "CABLE Output" as the default recording device during eval runs)? If not, audio-heavy cases (S03/S04/S08/S09/S13/S14 via mic) will use text injection, with ASR, endpointing and self-hearing marked NOT OBSERVABLE. ASR WER would then be measured only offline, through the app's own CLI `python -m app.audio.stt <wav>`, without VAD.
3. **LLM under test, and the judge.** Which LLM server and model should runs use: llama.cpp with Gemma (as `command.txt` suggests, which I'd need you to install or point me to), or the Ollama `qwen2.5:7b` already here (`DC_LLM_URL=http://127.0.0.1:11434/v1`)? For the judge (a different family from the app's LLM), which provider do you want, for example Anthropic with `ANTHROPIC_API_KEY`, or OpenAI? I'll build it pluggable either way, and it's skipped cleanly when no key is set.

## 10. Assumptions I'll go ahead with unless you object

- A1. Virtual camera (OBS) is the default video source. `DC_VIDEO_FILE` is a fallback, flagged as not wall-clock.
- A2. Calibration uses a prepended neutral segment from the same subject's non-drowsy footage. Stored-profile mode is optional. The calibration span is never scored.
- A3. An LLM recording proxy on `DC_LLM_URL` makes the LLM stages observable.
- A4. The harness speech TTS is Kokoro-82M (ONNX, offline, Apache-2.0, several English voices and speeds), downloaded once by the explicit build command with a pinned SHA-256. It's independent of the app's SAPI or Piper voice. If you'd rather avoid a ~330 MB download, the alternative is Piper with voices different from the app's.
- A5. Cabin noise is synthesised with fixed seeds (engine harmonics plus brown/road rumble, pink, white), not taken from recordings, to avoid licensing and download issues. This is documented as a limitation.
- A6. The golden dataset stores **references** to the NTHU videos (relative path plus SHA-256) and derived JSON. It doesn't copy the videos, which respects the dataset licence and keeps `eval/golden/` small. Synthetic audio lives in `eval/golden/synthetic/`, separate from the originals.
- A7. S12 is tested two ways: (a) virtual camera stops or blanks, expecting `face_lost` after about 6 s, as observe-and-report against the doc's 3 s claim; (b) the harness terminates the app's `vision` child process, which is the only externally reachable way to trigger the telemetry watchdog, with the 3 s check scored. Both are recorded.
- A8. Documented behaviour (the architecture doc) is the expected behaviour. Where code and doc disagree (§7), the case is scored against the doc and the report cites the discrepancy. Thresholds default to the documented values and live in `thresholds.yaml`.
- A9. The eval has its own `eval/.venv` and pinned `eval/requirements.txt`, separate from the backend venv. The app is always launched with the backend's own venv interpreter (path in config).

---

## 11. Decisions after review (2026-11-27)

| Question | Your answer | What was done |
|---|---|---|
| Q1 dataset | pull via DVC with the given remote settings | `eval/.venv` with `dvc[gdrive]`; credentials only in `.dvc/config.local` (git-ignored); the documented pyOpenSSL/cryptography pin fixed the `GEN_EMAIL` error; Google sign-in done by you in the browser pane; one network timeout, resumed by re-running `dvc pull` |
| Q2 audio | stream audio in real time like a natural speaker, app code may change; never send chat as text | new opt-in app input `DC_MIC_SOURCE=stream` + WebSocket `/ws/audio` (16 kHz int16 PCM). The harness streams 50 ms chunks on a monotonic schedule; the app re-blocks them to 50 ms and applies the echo-guard flag on arrival, so VAD -> ASR -> echo guard are the same code path as a microphone. `/api/chat` is never used. No virtual audio cable is needed, so §3's VB-CABLE plan is dropped |
| Q3 judge | Groq gpt-oss-120b or Gemini Flash | `judge.provider: groq|gemini` in thresholds.yaml, keys from `GROQ_API_KEY` / `GEMINI_API_KEY`; a judge of the same family as the app LLM is refused (Gemini vs a Gemma app LLM) |
| App LLM | not specified | assumption: the Ollama `qwen2.5:7b` already installed (`eval/config/app.yaml`, family `qwen`) |

App change details: `backend/app/config.py` (`mic_source`), `backend/app/audio/stt.py`
(`SpeechListener(source=...)`, `feed()`), `backend/app/runtime.py` (passes it through),
`backend/app/api/server.py` (`/ws/audio`), docs in `COMMANDS.md` and `docs/ARCHITECTURE.md`, 7 new
tests in `backend/tests/test_stt.py` and `test_voice_integration.py` (all pass). The full backend suite
has 29 failures that exist identically without this change (checked by stashing it).

Measured since Phase 0:
- OBS Virtual Camera index 1, verified by sending a test pattern through pyvirtualcam and reading it
  back with OpenCV/DirectShow.
- WASAPI loopback works on the default speaker: a 0.8 s tone came back as a 0.76 s segment at -8 dBFS.
- uvicorn stops gracefully on CTRL_BREAK (lifespan shutdown ran: `stop_method = graceful`).
- In a live self-test run, the app's "camera signal lost" alert came 2.91 s after the harness
  terminated the vision process.
- Ollama `qwen2.5:7b` here: ~10 s for a short warm answer, ~48 s when cold (model unloaded after
  5 min idle): the harness warms it before each case, outside the timed window.

## 12. First results on real data (smoke, dev clip 021_nightnoglasses, 1 repeat each, judge not configured)

Two smoke runs of the same three cases (`results/smoke1`, `results/smoke`), app commit 164079f + the
streamed-mic change, heuristic temporal model, Parakeet + Silero, pyttsx3, Ollama qwen2.5:7b:

| | run 1 | run 2 |
|---|---|---|
| level HIGH reached after GT onset (event 1 / event 2) | +14.5 s / +11.7 s | similar (frame F1 HIGH 0.86 both) |
| first drowsiness warning (event 1, onset 5.6 s) | 82.2 s | 136.7 s (after the labelled event ended: missed) |
| why so late | MILD check-ins at 7.1 s and 22.1 s reset the 60 s gate | the same, plus a posture nudge at 76.7 s resetting it again |
| warnings repeat | every ~60 s | every ~60 s |
| S10 transcripts (2 utterances) | exact (WER 0) | exact (WER 0) |
| S10 speech end -> audible reply p95 | 7.9 s | 5.2 s |
| S11 occlusion (4 s blank) | no warning, risk held | no warning, risk held |

This confirms §7 items 3 and 5 on real footage: HIGH drowsiness is detected within ~15 s, but the
spoken warning is gated by the 60 s check-in cooldown that every nudge (MILD, posture, distraction)
and every driver utterance resets, so it can come a minute or more late, or not at all within the
event. The warning is LLM-generated and takes ~4 s from decision to audio.

### Dev check (7 dev cases, 1 repeat, `results/devcheck2` + `results/s14check`, golden v2)

| Case | Result | What happened |
|---|---|---|
| S03 questions | PASS | both transcripts exact; replies on topic ("There are 210 minutes in three and a half hours.") |
| S05 preemption | INCONCLUSIVE | on this sunglasses clip the app was HIGH for only 0.1 s inside the clip, so no warning fired to preempt anything |
| S08 "I'm fine" | FAIL | transcript exact, but no drowsiness warning followed: the driver's words reset the 60 s gate (§7 item 5) |
| S09 cabin noise 10 dB | PASS | 3/3 utterances exact (WER 0) |
| S12 camera stops | OBSERVED | no "camera signal lost"; the "can't see your face" advisory came 6.1 s after the feed stopped |
| S13 self-hearing (echo on) | FAIL | no self-hearing; one ASR word split ("week end" vs "weekend", WER 0.18) |
| S14 barge-in | OBSERVED | the driver's interruption over a reply was dropped (mic muted by design); the reply was not interrupted |

Harness bugs found by these live runs and fixed (each now has a unit test): turn-taking gate
treating the instant between two reply sentences as silence; transcripts compared by identity in
S13; transcript ownership when a barge-in precedes a question; the camera-stop case ending early;
S05 counting a missing warning as a preemption failure.
