# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FastAPI backend for NeuroSense, a Multiple Sclerosis screening app. `README.md` holds the full API contract for the frontend. Keep it in sync when request/response shapes change.

Everything runs locally; there is no deployment target.
- **Frontend:** the React/Vite source lives in a separate project, `D:\STEM-Project\NeuroSense-Project`. It runs on its own dev server (`npm run dev`, port 8080) alongside this API (`fastapi dev main.py`, port 8000).
- **Cross-origin calls:** the frontend calls the API across origins, and the wildcard CORS middleware in `main.py` allows this. Don't tighten CORS without also changing the frontend.
- **No SPA serving:** this backend no longer serves a frontend build. The SPA-serving code in `main.py` is leftover: with no `dist/`, it logs a warning and the app runs as API-only.

The app runs five tests in a fixed order, and each test page only starts its test and stores the backend's result:
1. **Survey** (software): `/api/survey`.
2. **Vision** (software): `/api/vision`.
3. **Video analysis** (hardware, camera): `/api/video`.
4. **Force / pressure** (hardware sensor): `/api/sensors/force`.
5. **Vibration** (hardware sensor): `/api/sensors/vibration`.

When all five are done, the frontend posts one field per test to `/api/results/combined`, which returns the MS likelihood %. The frontend only stores results; all scoring happens here.

**Suspended:** motion and speech. Don't work on them unless asked.
- Motion is mounted only if `data/motion/*.json` exists, because `MotionTest` trains at import time.
- The speech router import is commented out in `main.py`, because faster-whisper/torch fails to load.

`controllers/survey_questions.py` is an unused offline RandomForest training script; it needs an `MS_Survey.csv` that isn't in the repo.

## Commands

```sh
pip install -r requirements.txt                 # Python 3.13; ffmpeg on PATH for odd webm files
fastapi dev main.py                             # dev server
python scripts/build_vision_dataset.py --source <path>/101_ObjectCategories --out data/vision
python scripts/download_video_models.py         # optional; models also download on first use

python -m unittest discover -s tests -t . -v                                   # all tests
python -m unittest tests.test_video_timeline -v                                # one module
python -m unittest tests.test_video_timeline.<TestClass>.<test_method> -v      # one test
```

- `tests/test_video_timeline.py`, `test_survey.py`, `test_sensors.py` and `test_combined.py` run anywhere, in well under a second. On Windows set `PYTHONUTF8=1` (also needed for `fastapi dev` when its output is redirected).
- `tests/test_video_pipeline.py` is skipped unless `NEUROSENSE_TEST_FACE` points to a frontal face photo. It builds synthetic clips from that photo (`tests/video_fixtures.py`) and runs the real MediaPipe pipeline and HTTP API. `NEUROSENSE_TEST_DUMP=out.json` saves the response.
- There is no linter or formatter config.

## Environment and untracked data

`data/` and `models/` are gitignored and must exist locally:
- `data/vision/` holds `manifest.json`, `images/`, and `responses.jsonl`, an append-only log of every evaluation. It is built by the dataset script from Caltech-101.
- `models/` holds the MediaPipe `.task` files. Override the folder with `NEUROSENSE_MODELS_DIR`.

Library constraints:
- MediaPipe is a recent version without `mp.solutions`. Use only the **Tasks API** (`mediapipe.tasks.python.vision`).
- Install `opencv-contrib-python`, never `opencv-python`. Both ship `cv2`, and having both installed breaks imports.

## Architecture

The code is split into layers:
- **Routes** (`routes/*_routes.py`) are thin. They validate input with Pydantic or form parsing.
- **Controllers** (`controllers/*.py`) hold all the logic. They raise domain exceptions, and the routes map them to HTTP errors:
  - 400 for bad input, e.g. `VisionInputError`.
  - 503 when data or models are missing, e.g. `VisionDatasetUnavailable`, `ModelUnavailableError`.
- **`main.py`** mounts the routers under `/api/...` and enables CORS. The SPA-serving code after the routers is leftover from an older setup.

All user-facing strings are bilingual, as `*_en` / `*_ar` pairs. Every result includes a "screening aid, not a diagnosis" note.

### Vision (`controllers/vision.py`, `scripts/build_vision_dataset.py`)

**Dataset build.** The build script degrades Caltech-101 images with MS-relevant distortions: `blur`, `contrast`, `double`, `fog`, `desaturation`, plus `none` for controls. It maps SSIM degradation to an IRT difficulty `b` per item and writes the result to the manifest. The script is deterministic per `--seed`, and `DATASET_VERSION` lives there.

**Exercise selection.** `select_exercises` picks one control item plus items spread across distortion types and target difficulties, each with a different object class. The correct answer is never sent to the client.

**Scoring.**
- The model is 3PL IRT with guessing fixed at `1/num_options`.
- Ability θ is the EAP estimate on a θ grid with an N(0,1) prior.
- The percentile is exact: it enumerates every answer pattern on the *same items* for normal observers assumed N(0,1).
- `BANDS` maps the percentile to a label.

**Statelessness and norms.** The test is stateless between requests; the only side effect is appending to `responses.jsonl`, which is guarded by a lock. The norms are model-based, not clinically validated, so keep that caveat when changing scoring.

### Video analysis (`controllers/video_analysis.py`, `routes/video_analysis_routes.py`)

The pipeline runs once per clip, with one instructed action per clip:

1. **Decode.** Frames are sampled at `TARGET_FPS` (~12) with OpenCV, capped at 30 s and downscaled to a 640 px width. If OpenCV fails or yields no frames, the clip is transcoded with ffmpeg (`extract_features_with_fallback`).
2. **Extract features.** MediaPipe landmarkers run in VIDEO mode. The face landmarker always runs; the pose landmarker runs only for hand actions. Each frame becomes a set of raw features (`FEATURE_NAMES`): yaw, pitch, iris gaze, blink, and wrist raise.
3. **Score actions.** `action_score_series` turns the features into a per-frame score in [0,1], measured relative to a baseline neutral pose taken from the first `BASELINE_WINDOW_S` of video.
4. **Detect onset.** `detect_action_onset` finds the onset using hysteresis (`SCORE_THRESHOLD`/`SCORE_RELEASE`), a minimum hold (`MIN_HOLD_S`), and dropout bridging. Steps 3–4 are pure functions on numpy arrays, and the unit tests exercise them directly.
5. **Summarize.** `summarize_results` gives `video_score = 100 × (0.6·completion + 0.4·response-time score)`.

Conventions and invariants:
- **Left/right:** actions use the *person's* left and right. Clips are assumed not mirrored unless `mirrored=true`. Raw yaw and gaze signs are image-based: positive means toward the image right.
- **Action kinds:** `ActionSpec.kind` is `TRANSITION` (the person must move away from neutral) or `STATE` (the target may already hold, which gives `already_in_state` and a response time of 0).
- **Action aliases:** the action registry is `ACTIONS`, and friendly names resolve through `normalize_action`.
- **Thread safety:** landmarkers are created per `extract_features` call because MediaPipe instances are neither thread-safe nor reusable across timelines. The routes run clips on a shared 3-worker `ThreadPoolExecutor`, which caps concurrency server-wide.
- **Model downloads:** `ensure_model` downloads models lazily, guarded by a lock.
- **Per-clip errors:** a clip that fails to decode gets an `error` field instead of failing the whole request.
- **Tuning:** the detection thresholds are module-level constants at the top of the controller, and they have only been validated on synthetic clips.

### Survey (`controllers/survey.py`)

`SECTIONS` holds the questions with their bilingual texts; the frontend renders them from `GET /questions`. A "yes" to a main question scores 2 points, and each "yes" follow-up scores 1, counted only under a "yes". Levels come from the share of each section's `max_score`; those maxima are deliberately above the reachable totals. `symptom_score` is 0–100, and higher means more symptoms.

### Sensors (`controllers/sensors.py`)

- **Measuring:** `measure(sensor)` asks a `SensorReader` for samples, summarizes them with `SensorSpec.summarize`, and maps the value to an impairment level `low`/`medium`/`high`. `higher_is_worse` sets the direction.
- **Reader choice:** `NEUROSENSE_SENSOR_MODE` picks `SimulatedReader` (default; random values, results flagged `simulated`) or `RaspberryPiReader`. `RaspberryPiReader.read` is the single place to add the Raspberry Pi code.
- **One at a time:** a per-sensor lock rejects concurrent measurements with 409.
- **Placeholders:** the units and thresholds are not calibrated yet.

### Combined result (`controllers/combined.py`)

- **Per-test risk:** each test becomes a risk in [0,1]. Survey is `score/100`. Vision and video are `1 − score/100`. For the sensors, low / medium / high become 0 / 0.5 / 1.
- **Likelihood:** `ms_likelihood` is the `WEIGHTS`-weighted mean of those risks, in percent; the weights are equal by default.
- **Keep in sync:** the request model in `routes/results_routes.py` names one field per test, and it must match what the frontend stores (`TestData` in the frontend's `src/contexts/userTypes.ts`).

`utils.py` is fully commented-out legacy code.
