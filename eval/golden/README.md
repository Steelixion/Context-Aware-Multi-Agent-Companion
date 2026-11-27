# Golden dataset

A frozen, versioned, checksummed test set built from the NTHU-DDD **testing** videos. The runner
only reads and verifies it (`run_eval.py` refuses to run if any checksum differs). It is never
regenerated implicitly; rebuilding is an explicit command that creates a new version.

## Source data as found (inspected 2026-11-27)

`Dataset/Testing_Dataset` (DVC, 210 files, 1.16 GB):

| Item | Finding |
|---|---|
| Videos | 70 × `<subject>_<condition>_mix.mp4`: 14 subjects (003 010 011 014 016 017 018 019 021 025 027 028 029 037) × 5 conditions (glasses, nightglasses, nightnoglasses, noglasses, sunglasses); H.264, 640×480, 29.97 fps, 2236-21486 frames (1.2-12 min); 409 min in total |
| Labels | `test_label_txt/<condition>/<subject>_<condition>_mixing_drowsiness.txt`, one character per frame, `0` = non-drowsy, `1` = drowsy, trailing newline |
| Duplicate labels | `test_label_txt/wh/` is a flat copy of all 70 label files. Contents are identical; the `night_noglasses` folder spells its files `night_noglasses` instead of `nightnoglasses` (names differ, contents equal) |
| Other labels | none: this testing set labels **drowsiness only** (no eye / head / mouth labels) |
| Frame count vs labels | every video has exactly **2 more frames** than its label file. We take label *i* = frame *i*; the last 2 frames are unlabelled (`null`) and never scored. Flagged per clip in the manifest |
| Clips with no drowsy frames | 9 (used for S01 whole-clip false-alarm checks) |

## Files

| File | Content |
|---|---|
| `VERSION.yaml` | version, build time, builder version, seed, dataset checksum, smoke clip |
| `manifest.csv` | one row per clip: id, source path, SHA-256, fps, frames, duration, size, subject, condition, split, drowsy fraction, number of events, label flags, calibration method, case ids, `synthetic=false` |
| `gt/<clip_id>.json` | `frame_labels` (0/1/null per frame), `events` [[onset_s, offset_s]], label source file, flags, the event rule |
| `audio_manifest.csv` | one row per synthetic WAV: id, path, SHA-256, duration, text, voice, speed, noise type, SNR, noise seed, generator, `synthetic=true` |
| `synthetic/audio/*.wav` | 16 kHz mono speech (git-ignored; verified by SHA-256 against `audio_manifest.csv`) |
| `scenarios/*.yaml` | one test case each (see `eval/README.md` for the schema) |
| `CHECKSUMS.sha256` | SHA-256 of every data file above except the WAVs (they are in `audio_manifest.csv`) |
| `SUMMARY.md` | counts per scenario and split, footage, real-time cost estimates, flagged clips |

The **dataset checksum** in `VERSION.yaml` is SHA-256(`CHECKSUMS.sha256` + `audio_manifest.csv`), so it
covers every golden file and every WAV. Original videos are not copied; the manifest points to
them in `Dataset/Testing_Dataset` and records their SHA-256 (also verified before each run).

## How it is derived

- **Ground truth** comes only from the dataset's own drowsiness labels. Drowsy **events** are runs
  of label `1`, merged across gaps under 1 s and kept if at least 0.5 s long (the scorer re-applies
  the `detection.*` settings in `thresholds.yaml`, so these can be tuned without rebuilding).
- **Split:** 10 dev clips, the rest test. Chosen with seed `20261127`, stratified by condition and by
  drowsy-fraction tercile (round-robin over the strata).
- **Calibration** (the app needs ~10 s of neutral footage before personal thresholds apply): each
  case first plays a 15 s neutral segment (drowsiness `0`) until the app reports `calibrated`.
  Source, in order of preference: (1) the same subject and condition in another clip (does not
  exist in this set: one clip per subject and condition); (2) the longest non-drowsy run of the
  same clip, and that span is then excluded from scoring if the case plays it; (3) the same
  subject in another condition (flagged: appearance may differ); (4) none, so the app calibrates on
  the clip's first seconds, which are excluded from scoring. The method used is in the manifest
  and in each case's `calibration` block.
- **Synthetic speech** (the videos are silent): English only, neutral content that never mentions
  tiredness (except S08's scripted "I'm fine"), generated with Kokoro-82M (kokoro-onnx, offline,
  model SHA-256 pinned in `harness/speech.py`), independent of the app's TTS. 4 voices (US/UK,
  female/male), speeds 0.85 / 1.0 / 1.2, normalised to -26 dBFS. Repeated synthesis of the same
  text gave bit-identical audio on this machine; another machine or ONNX Runtime version may
  differ slightly, which is why the frozen WAVs (not the generator) are the reference.
- **Noise variants** of U01-U04 for every voice and speed: cabin (synthetic engine harmonics + road
  rumble), pink and white noise at 20, 10, 5 and 0 dB SNR (SNR over active speech samples), 0.6 s of
  noise before and after; per-file seeds are recorded. During a run the same noise type continues
  as a live bed between utterances at the same level.

## Commands

```powershell
eval\.venv\Scripts\python eval\golden\build_golden.py --inspect      # print the dataset layout
eval\.venv\Scripts\python eval\golden\build_golden.py --fetch-tts    # reference TTS model (pinned SHA-256)
eval\.venv\Scripts\python eval\golden\build_golden.py --build        # first build (refuses if one exists)
eval\.venv\Scripts\python eval\golden\build_golden.py --build --force   # rebuild => version + 1
eval\.venv\Scripts\python eval\golden\build_golden.py --verify       # check every checksum
eval\.venv\Scripts\python eval\golden\build_golden.py --refreeze     # after hand-editing/adding scenarios: re-checksum, version + 1
```

`--selftest DIR` builds a tiny synthetic golden set (a still photo rendered to video) for checking
the harness without the dataset; it is not a benchmark.

To share the synthetic audio with teammates, track it with DVC like the videos
(`dvc add eval/golden/synthetic/audio`). A rebuild elsewhere is likely but not guaranteed to be
bit-identical; `--verify` will say.
