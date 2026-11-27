# Evaluation harness

A black-box evaluation of the driver companion. It starts the app as a separate process, feeds
it NTHU-DDD test videos through a virtual camera and synthetic English speech through a
real-time audio stream, records everything the app does from the outside (WebSocket, HTTP,
its LLM traffic through a recording proxy, the speaker output, the SQLite DB read-only), scores
it against the dataset's own labels and the documented behaviour, and writes an Excel report.

The harness lives entirely in `eval/`. The only app change it needs is the opt-in streamed
microphone input (`DC_MIC_SOURCE=stream`, WebSocket `/ws/audio`), which you approved; with the
default `DC_MIC_SOURCE=device` the app is unchanged. Investigation notes: [FINDINGS.md](FINDINGS.md).

## Contents

| Path | What it is |
|---|---|
| `run_eval.py` | the one command: runs cases, scores, judges, writes the report |
| `config/thresholds.yaml` | **every** pass/fail limit, latency budget, judge setting, repeats |
| `config/app.yaml` | how to launch the app and which models it uses (edit this to swap a model) |
| `golden/` | the frozen golden dataset (manifest, ground truth, scenarios, synthetic audio) and its builder |
| `harness/` | small modules: app process, WS recorder, virtual camera, audio stream, loopback, LLM proxy, observation, checks, metrics, judge, report |
| `judge/calibrate.py` | judge-vs-human agreement (Cohen's kappa) |
| `tests/` | unit tests: metrics (WER, events, delays, percentiles, kappa), scenario checks on synthetic event logs, judge plumbing |

## Install (once)

Prerequisites on this machine: the app's backend venv and models (`python scripts/download_models.py`;
see `COMMANDS.md`), OBS
Studio (for its **OBS Virtual Camera**), ffmpeg on PATH, and an OpenAI-compatible LLM server for
the app (default in `config/app.yaml`: Ollama at `http://127.0.0.1:11434/v1`, `qwen2.5:7b`).

```powershell
python -m venv eval\.venv
eval\.venv\Scripts\pip install -r eval\requirements.txt
```

```powershell
eval\.venv\Scripts\python eval\golden\build_golden.py --fetch-tts
```

Get the dataset (DVC, Google Drive; the first pull opens a Google sign-in in your browser):

```powershell
eval\.venv\Scripts\dvc pull Dataset/Testing_Dataset.dvc
```

## Run

```powershell
eval\.venv\Scripts\python eval\run_eval.py --config eval\config\thresholds.yaml --split test --out results\run1
```

| Option | Meaning |
|---|---|
| `--smoke` | one dev clip, one repeat, full Excel report (quick pipeline check) |
| `--split dev\|test\|all` | which golden cases to run |
| `--only ID1,ID2` | just these case ids |
| `--resume` | continue an interrupted run in the same `--out` folder (finished case runs are kept) |
| `--rescore` | re-score the raw logs in `--out` without running the app (e.g. after changing thresholds) |
| `--baseline results\old` | adds the **Compare** sheet (metric deltas and which model settings changed) |
| `--repeats N` | override `run.repeats` (and `run.repeats_by_scenario`) |
| `--no-judge` | skip the LLM judge |

**How long it takes.** Everything runs in real time. One repeat of the test split is about **9.3 h**
(6.9 h of it is whole-clip detection, S01/S02), the dev split about 1.7 h, `--smoke` about 12 min
(numbers from `golden/SUMMARY.md`; the log prints an estimate at start). With the default
`repeats: 3` the test split is about 28 h; to cut it, set e.g. `run.repeats_by_scenario: {S01: 1, S02: 1}`
(detection is the least variable part), or run it overnight with `--resume` after any interruption.

The judge needs an API key in the environment (never in a file):

```powershell
$env:GROQ_API_KEY = "<your key>"
```

or set `judge.provider: gemini` in `thresholds.yaml` and `$env:GEMINI_API_KEY`. Without a key the
judge cells say `NOT RUN (no judge configured)` and everything else is still produced. The
judge is refused (NOT RUN) if its family equals the app LLM's `llm.family` in `app.yaml`.

**While a run is going:** the app's voice comes out of your speakers (it is captured by WASAPI
loopback to time the audible onset), the OBS Virtual Camera is in use, and the CPU is busy.
Don't play other sounds; they would be counted as app audio. Runs are real time: a case takes
its clip length plus about 30-60 s of start-up, calibration and settling.

Output in `results/<run>/`: `report.xlsx`, `summary.json`, `eval.log`, and `raw/` with, per case
run, the event log (`<case>__r<k>.jsonl`), app stdout/stderr (`.app.log`), speaker capture
(`.loopback.wav` + `.loopback.json`), frame timing (`.frames.json`) and the result incl. a
read-only DB dump (`.result.json`). Work data dirs are in `work/` (safe to delete).

## Unit tests

```powershell
eval\.venv\Scripts\python -m pytest eval\tests -q
```

## Swapping a model and comparing

1. Edit `config/app.yaml` only: e.g. `models.DC_ASR_MODEL_DIR`, `DC_VAD_MODEL`, `DC_TEMPORAL_MODEL`,
   `DC_PIPER_MODELS`, `DC_LLM_MODEL`, or `llm.upstream`/`llm.family` for another LLM server.
2. Run the same split with the same `thresholds.yaml` into a new folder, with the old run as
   baseline:

```powershell
eval\.venv\Scripts\python eval\run_eval.py --split test --out results\after_swap --baseline results\run1
```

The **Compare** sheet lists every headline metric for both runs, the delta, status counts, and
the model settings that differ (taken from `app.yaml`, `/api/session` and the app's own log
lines). Test cases need no edits; the golden checksum in Run_Metadata proves both runs used
the same dataset.

## Changing thresholds

All limits are in `config/thresholds.yaml`; values marked `# DEFAULT` are my starting points for
you to review, `# DOC` ones come from `docs/ARCHITECTURE.md`. After editing, re-score without
re-running the app:

```powershell
eval\.venv\Scripts\python eval\run_eval.py --rescore --out results\run1
```

## Adding a scenario

Write a YAML file in `golden/scenarios/` (then re-freeze with `build_golden.py --refreeze`; see
`golden/README.md`). A case names a clip segment, a calibration method, a timeline and checks:

```yaml
case_id: S03_my_case
scenario: S03                # selects the check set in harness/checks.py
title: Awake driver asks about a podcast
mode: scored                 # or observe (never PASS/FAIL)
split: test
clip: {clip_id: nthu_007_glasses, start_s: 0.0, end_s: 60.0}
calibration: {method: prepend_neutral, clip_id: nthu_007_glasses, start_s: 2.0, end_s: 14.0}
timeline:
  - {id: e1, at_s: 12.0, action: speak, audio_id: U01_af_heart_s100_clean}
  - {id: e2, trigger: {level_at_least: HIGH, after_s: 0.5, timeout_s: 60}, action: speak, audio_id: F01_bf_emma_s100_clean}
  - {id: e3, trigger: {companion_speaking: true, after_event: e1, after_s: 0.8, timeout_s: 40}, action: speak,
     audio_id: B01_bm_george_s100_clean, overlap: true}           # barge-in: talk over the reply to e1
  - {id: b1, at_s: 30.0, action: blank_video, duration_s: 4.0}     # also: stop_video, kill_vision
expected: ["On-topic reply"]
judge: [coherent, grounded, appropriate_action, safe]          # optional; claims_alert = safety question
audio: {noise_bed: {type: cabin, snr_db: 10, ref_audio_id: U01_af_heart_s100_clean}, echo: false}
```

Cases contain no thresholds and never name a model. Trigger keys: `level_at_least` (ALERT/MILD/HIGH,
from the app's own level) or `companion_speaking`; optional `after_event` (arm only after that
utterance was spoken), `not_before_s`, `timeout_s`, `after_s`.

## How things are measured

- **Clock.** One `time.monotonic()` clock in the harness. App timestamps (`time.time()`, same host)
  are converted with the wall/mono offset sampled every 10 s (logged as `clock` events).
  Clip time 0 = the moment the first test-clip frame was sent to the virtual camera.
- **Video** runs on the wall clock at the clip's own FPS; if the harness falls behind, frames are
  skipped (counted in `.frames.json`), never slowed. The app samples the camera at its own
  `DC_VISION_FPS`, as with a real camera.
- **Audio** is 16 kHz int16 PCM, 50 ms per WebSocket message on a monotonic deadline schedule,
  silence (or the scenario's noise bed) between utterances. Before speaking, the harness waits
  until the companion has been idle (no audio and no reply being generated) for
  `run.speech_gate_quiet_s`, up to `run.speech_gate_max_wait_s` (natural turn-taking), unless the event
  has `overlap: true`. If the app still talks over the driver (it starts a check-in of its own), the
  app's echo guard drops the utterance by design; that utterance is scored INCONCLUSIVE ("lost to
  companion talk-over"), counted in the headline `utterances_lost_to_talkover`, and kept out of WER.
- **Transcript ownership:** a transcript belongs to the most recent utterance that had ended when it
  arrived (ASR only emits after end-of-speech), within `speech.transcript_wait_s`.
- **Calibration.** Each case first shows a neutral segment from the same subject's non-drowsy
  footage until the app reports `calibrated`, then switches to the test clip. Nothing before clip
  time 0 is scored.
- **Text normalisation for WER:** NFKC, lowercase, curly quotes straightened, hyphens/slashes to
  spaces, all punctuation removed, apostrophes removed ("it's" = "its"), digits 0-10 spelled out,
  British spellings mapped to American ("tyre" = "tire"; list in `harness/metrics.py`).
  Corpus WER = total errors / total reference words.
- **Frame-level detection:** for every ground-truth frame, the app's level at that instant (a step
  function built from the unthrottled WS `state` events) is compared with the label; positive
  class = level >= MILD, and separately level == HIGH.
- **Events:** GT drowsy runs (merged across short gaps, minimum length) are detected if the level
  reaches the class in [onset - `event_early_s`, offset + `event_late_s`); delay = first crossing - onset.
- **Warnings** are the app's `alert` events with reason `drowsy`; "warning audio" is the first
  loopback onset after it (or, if loopback is unavailable, the first `audio` event, labelled so).
- **Latency percentiles** use linear interpolation (numpy's default).

## Known limitations (what is NOT OBSERVABLE and why)

- **VAD end-of-utterance** is not published by the app; only the transcript event is. Speech end
  -> transcript therefore includes the 0.8 s end-of-speech silence plus ASR time.
- **First LLM token** is observable only through the recording proxy (`llm.proxy: true`). With it
  off, those stages are NOT OBSERVABLE.
- **TTS start** in the app's `audio` event is sent *before* synthesis, so it is not audible onset;
  the audible onset comes from loopback capture, which includes the sound device's own output
  latency (tens of ms, constant). If loopback is unavailable, end-to-end latency falls back to the
  `audio` event and every such number is labelled with that source.
- **Arbiter queue contents** (what was flushed) cannot be seen; preemption is judged from audio
  start/stop events, priorities and the loopback.
- **Tool calls**: the app exposes none, so tool-call checks do not apply.
- **The watchdog** can only be triggered externally by terminating the app's vision process; a
  camera that stops delivering frames does not trigger it (see FINDINGS §7).
- **The LLM server** is shared across cases; the harness warms it before each case (outside the
  timed window) so a server that unloads idle models does not add cold starts.
- **Noise** is synthetic (seeded cabin/pink/white), not recorded cabin audio.
- **Self-hearing** is simulated by feeding the speaker loopback back into the mic stream at
  `echo.gain_db` / `echo.delay_ms`; it is not a real acoustic path.
- **The judge** is unvalidated until you run `judge/calibrate.py` on your own labels.
