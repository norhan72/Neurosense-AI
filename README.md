# NeuroSense AI: backend API

FastAPI backend for the NeuroSense Multiple Sclerosis screening app. Runs locally next to the frontend's dev server.

The app runs five tests in this order, then combines them:

| # | Test | Kind | Endpoint |
|---|---|---|---|
| 1 | Symptom survey | software | `/api/survey` |
| 2 | Vision (distorted-image recognition, IRT scoring) | software | `/api/vision` |
| 3 | Video analysis (camera; instructed actions, response time) | hardware | `/api/video` |
| 4 | Force / pressure sensor | hardware | `/api/sensors/force` |
| 5 | Vibration sensor | hardware | `/api/sensors/vibration` |
| | Combined MS likelihood | | `/api/results/combined` |

Suspended (code kept): motion (`/api/motion`, mounted only if `data/motion/*.json` exists) and speech (not mounted; faster-whisper/torch import disabled in `main.py`).

These are screening aids, not diagnoses. The vision norms are model-based, the video thresholds are heuristics, the sensor thresholds are placeholders and the combined weights are equal by default (see "Limitations" below).

## Setup and run

```sh
pip install -r requirements.txt          # Python 3.13; needs ffmpeg on PATH for odd webm files
python scripts/download_video_models.py  # optional: the API also downloads models on first use
python scripts/build_vision_dataset.py --source <path>/101_ObjectCategories --out data/vision
fastapi dev main.py
```

- **`data/` and `models/` are not in git** (gitignored). Rebuild `data/vision` with the script above from Caltech-101 (https://data.caltech.edu/records/mzrjq-6wc02, CC BY 4.0; attribution in `data/vision/README.md`). MediaPipe models download into `models/`; `NEUROSENSE_MODELS_DIR` overrides the folder.
- **Frontend:** the React app is a separate project (`D:\STEM-Project\NeuroSense-Project`). It runs on its own Vite dev server (`npm run dev`, port 8080) next to this API (`fastapi dev main.py`, port 8000). CORS allows any origin, so the frontend calls the API directly. This backend does not serve a frontend build.
- **Sensors:** `NEUROSENSE_SENSOR_MODE=simulated` (default) returns random plausible readings, flagged `simulated: true`. Set it to `raspberry_pi` once `RaspberryPiReader.read` in `controllers/sensors.py` talks to the device.
- **Tests:** run `python -m unittest discover -s tests -t . -v`. The video integration tests need `NEUROSENSE_TEST_FACE=<frontal face photo>`.

## API contract (for the frontend)

All labels are bilingual: `*_en` / `*_ar`. Errors use FastAPI's `{"detail": "..."}` shape: 400 for bad input, 503 when the dataset or models are missing.

### Survey

**1. Get the questions.** `GET /api/survey/questions` returns `{sections: [{key, title_en, title_ar, questions: [{id, text_en, text_ar, follow_ups: [{id, text_en, text_ar}]}]}], note_en, note_ar}`. Follow-ups are asked only after a "yes" to their question.

**2. Evaluate.** `POST /api/survey/evaluate` with `{"answers": {"q1": true, "q1a": true}}` (ids not sent count as "no"; unknown ids return 400). Response:

```json
{"symptom_score": 3.8, "level": "low", "label_en": "Few reported symptoms", "label_ar": "أعراض قليلة",
 "points": 3, "max_points": 79,
 "sections": [{"key": "voice", "title_en": "…", "title_ar": "…", "points": 3, "max_points": 12, "level": "medium"}],
 "note_en": "…", "note_ar": "…"}
```

- Each "yes" to a main question scores 2 points, and each "yes" follow-up scores 1.
- `symptom_score` is the points as a share of the maximum, from 0 to 100; higher means more symptoms.
- Levels by share of points: high ≥ 50%, medium ≥ 25%, otherwise low.

### Vision test

**1. Get exercises.** Call `GET /api/vision/exercises?count=6` (`count` must be 3–10, 5–7 recommended). Each set contains one clean control image and the rest spread across distortion types and difficulty, each with a different object class. The correct answer is never sent.

```json
{
  "dataset_version": "1.0.0",
  "count": 6,
  "num_options": 4,
  "exercises": [
    {
      "image_id": "v0267",
      "image_url": "/api/vision/images/v0267",
      "options": [
        {"key": "Motorbikes", "label_en": "Motorbike", "label_ar": "دراجة نارية"},
        {"key": "butterfly", "label_en": "Butterfly", "label_ar": "فراشة"},
        {"key": "lamp", "label_en": "Lamp", "label_ar": "مصباح"},
        {"key": "cup", "label_en": "Cup", "label_ar": "كوب"}
      ]
    }
  ]
}
```

**2. Show images.** `GET /api/vision/images/{image_id}` returns a 320×320 PNG; use `image_url` as-is.

**3. Evaluate.** `POST /api/vision/evaluate` (JSON). Send one answer per shown image; `answer` is the chosen option `key`, and `response_time_ms` is optional. Duplicate or unknown `image_id`s, or fewer than 3 / more than 10 answers, return 400.

```json
{"answers": [{"image_id": "v0267", "answer": "Motorbikes", "response_time_ms": 1800}]}
```

Response:

```json
{
  "session_id": "83e6309b-…",
  "dataset_version": "1.0.0",
  "theta": 1.035, "theta_se": 0.876,
  "percentile": 85.0,
  "vision_score": 85.0,
  "label_en": "Normal", "label_ar": "طبيعي",
  "accuracy": 1.0, "num_correct": 6, "num_items": 6,
  "breakdown": [
    {"distortion": "fog", "count": 1, "correct": 1, "accuracy": 1.0,
     "symptom_en": "Hazy vision", "symptom_ar": "رؤية ضبابية (غشاوة)"}
  ],
  "items": [
    {"image_id": "v0267", "answer": "Motorbikes", "correct_class": "Motorbikes",
     "label_en": "Motorbike", "label_ar": "دراجة نارية", "correct": true}
  ],
  "note_en": "This is a screening aid, not a diagnosis. …", "note_ar": "…"
}
```

- `vision_score` (0–100, higher is better) is the percentile against normal observers answering the **same items**: the share of normal observers who would score lower (ties count half). It is computed exactly by enumerating every possible answer pattern.
- Labels by percentile: Normal (≥ 15), Borderline (5–15), Mild impairment (1–5), Significant impairment (< 1).
- Distortion types, each with the symptom it relates to: `none` (control), `blur` (optic neuritis), `contrast` (reduced contrast sensitivity), `double` (diplopia/INO), `fog` (hazy vision), `desaturation` (colour vision loss).
- Every evaluation is appended to `data/vision/responses.jsonl`, so item difficulties can be recalibrated from real users later.

### Video analysis test

**1. List actions.** `GET /api/video/actions` returns `{"actions": [{"action", "label_en", "label_ar", "instruction_en", "instruction_ar", "kind"}]}`.

| Kind | Actions |
|---|---|
| Movement | `look_left`, `look_right`, `look_up`, `look_down`, `close_eyes`, `raise_hand`, `raise_left_hand`, `raise_right_hand` |
| State (may already be true) | `look_straight`, `open_eyes`, `lower_hand` |

Friendly names such as "look left", "close eyes" and "put hand down" are also accepted. Left and right are the **person's own** left and right.

**2. Analyze.** `POST /api/video/analyze` (multipart/form-data):

| Field | Required | Description |
|---|---|---|
| `videos` | yes | One file per action (repeat the field), up to 10 files, each ≤ 50 MB. mp4 or webm; clips are analysed up to 30 s. |
| `actions` | yes | JSON list (`["look left","close eyes"]`) or comma-separated, same order and length as `videos`. |
| `instruction_offsets` | no | Seconds from clip start to the moment the instruction was shown: one number or a JSON list (0–30). Defaults to 0. |
| `mirrored` | no | `true` if the clips were recorded mirrored (selfie view). Defaults to `false`; MediaRecorder output is normally not mirrored even when the preview is. |

```js
const form = new FormData();
clips.forEach(c => form.append("videos", c.blob, c.name));
form.append("actions", JSON.stringify(clips.map(c => c.action)));
form.append("instruction_offsets", JSON.stringify(clips.map(c => c.offsetSeconds)));
await fetch("/api/video/analyze", { method: "POST", body: form });
```

Response (trimmed). Results come back in upload order. A clip that can't be decoded gets an `error` instead of failing the whole request.

```json
{
  "results": [
    {
      "filename": "clip1.webm", "action": "look_left",
      "label_en": "Look left", "label_ar": "…",
      "detected": true, "response_time_s": 1.139, "onset_s": 1.639,
      "instruction_offset_s": 0.5, "confidence": 0.859, "hold_duration_s": 2.281,
      "already_in_state": false, "frames_analyzed": 48, "duration_s": 4.0,
      "face_detected_ratio": 1.0, "pose_detected_ratio": null,
      "quality_warning": null, "note": null, "error": null
    }
  ],
  "summary": {
    "num_videos": 4, "num_analyzed": 4, "num_detected": 3,
    "completion_rate": 0.75, "mean_response_time_s": 1.066,
    "response_time_score": 1.0, "video_score": 85.0,
    "label_en": "Normal responsiveness", "label_ar": "استجابة طبيعية",
    "note_en": "Screening heuristic, not a diagnosis. …", "note_ar": "…"
  }
}
```

How the fields are computed:

- **`detected`:** the action held for at least 0.4 s after the instruction.
- **`response_time_s`:** the time from the instruction to the onset of the action; `null` when not detected.
- **State actions:** if the person was already in the requested state at the instruction moment, the result has `already_in_state: true` and a response time of 0.
- **`quality_warning`:** set when the face (or pose) was found in fewer than 60% of frames.
- **`video_score`:** `100 × (0.6 × completion_rate + 0.4 × response_time_score)`. Response-time score is 1.0 up to 1.5 s and falls to 0 at 5 s.

### Sensor tests (force, vibration)

`POST /api/sensors/{force|vibration}/measure` (no body) runs one measurement and waits for it to finish, which takes a few seconds. `GET /api/sensors` lists the sensors and the current mode.

```json
{"sensor": "force", "label_en": "Grip force", "label_ar": "قوة القبضة",
 "level": "low", "level_label_en": "Typical", "level_label_ar": "طبيعي",
 "value": 32.89, "unit": "kg", "num_readings": 100, "duration_s": 5.0,
 "simulated": true, "note_en": "…", "note_ar": "…"}
```

- **`level`** is an impairment level: `low` is typical, `high` is impaired.
  - **Force:** uses the peak reading, and a lower value is worse.
  - **Vibration:** uses the median reading, and a higher perception threshold is worse.
  - **Thresholds:** in `SENSORS` in `controllers/sensors.py`; they are placeholders until calibrated.
- **Errors:** 409 if that sensor is already measuring, 503 if it can't be read, 422 for an unknown sensor.

### Combined result

`POST /api/results/combined` takes the relevant field of each test's response:

```json
{"survey": {"symptom_score": 20}, "vision": {"vision_score": 85}, "video": {"video_score": 90},
 "force": {"level": "low"}, "vibration": {"level": "medium"}}
```

Response:

```json
{"ms_likelihood": 19.0, "label_en": "Low likelihood", "label_ar": "احتمال منخفض",
 "tests": [{"test": "survey", "risk": 0.2, "weight": 0.2}],
 "note_en": "…", "note_ar": "…"}
```

- Each test becomes a risk between 0 and 1:
  - survey: `symptom_score/100`
  - vision: `1 − vision_score/100`
  - video: `1 − video_score/100`
  - force and vibration: low / medium / high become 0 / 0.5 / 1
- `ms_likelihood` is the weighted mean of the risks, in percent. The weights are equal by default and live in `WEIGHTS` in `controllers/combined.py`.
- Labels: high from 60%, moderate from 30%, otherwise low.
- Missing tests or out-of-range values return 422.

## Limitations

- **Vision norms are model-based.** Item difficulty comes from image degradation (SSIM). Until it is recalibrated from `responses.jsonl` with real normative data, percentiles are not clinically validated.
- **Vision results are coarse with few items.** With 6 items the ability estimate's SE is about 0.7–0.9. With 6 items the score moves in coarse steps: in simulation, a perfect run lands around the 95th percentile and 4/6 correct around the 27th.
- **The sensor levels are not calibrated.** The units, summary statistics and thresholds in `controllers/sensors.py` are placeholders. In simulated mode the readings are random.
- **The combined likelihood is not a calibrated probability.** It is an equally weighted mean of per-test risks. Replace `WEIGHTS` (or the whole function) with a model fitted on labelled data when that data exists.
- **Video was validated only on synthetic clips** made from a still photo. Thresholds (head turn 10°, nod 8°, blink, hold 0.4 s), eye-only gaze, `look_up`/`look_down`, the hand actions and the `mirrored` default all still need checking on real webcam recordings.
