# NeuroSense AI: backend API

FastAPI backend for the NeuroSense Multiple Sclerosis screening app. It also serves the prebuilt React SPA.

Software tests:

| Test | Status | Endpoint prefix |
|---|---|---|
| Vision (distorted-image recognition, IRT scoring) | active | `/api/vision` |
| Video analysis (instructed actions, response time) | active | `/api/video` |
| Motion (phone accelerometer) | suspended; enabled only if `data/motion/*.json` exists | `/api/motion` |
| Speech | suspended (faster-whisper/torch import disabled in `main.py`) | not mounted |

These are screening aids, not diagnoses. The vision norms are model-based and the video thresholds are heuristics (see "Limitations" below).

## Setup and run

```sh
pip install -r requirements.txt          # Python 3.13; needs ffmpeg on PATH for odd webm files
python scripts/download_video_models.py  # optional: the API also downloads models on first use
python scripts/build_vision_dataset.py --source <path>/101_ObjectCategories --out data/vision
fastapi dev main.py
```

- **`data/` and `models/` are not in git** (gitignored). Rebuild `data/vision` with the script above from Caltech-101 (https://data.caltech.edu/records/mzrjq-6wc02, CC BY 4.0; attribution in `data/vision/README.md`). MediaPipe models download into `models/`; `NEUROSENSE_MODELS_DIR` overrides the folder.
- **Frontend:** the SPA build directory is `dist/` by default. Override it with `NEUROSENSE_DIST_DIR`. The current build lives at `D:\STEM-Project\NeuroSense-Project\dist`, which also has the frontend source:
  ```powershell
  $env:NEUROSENSE_DIST_DIR = "D:\STEM-Project\NeuroSense-Project\dist"; fastapi dev main.py
  ```
- **Tests:** run `python -m unittest discover -s tests -t . -v`. The video integration tests need `NEUROSENSE_TEST_FACE=<frontal face photo>`.

## API contract (for the frontend)

All labels are bilingual: `*_en` / `*_ar`. Errors use FastAPI's `{"detail": "..."}` shape: 400 for bad input, 503 when the dataset or models are missing.

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

## Limitations

- **Vision norms are model-based.** Item difficulty comes from image degradation (SSIM). Until it is recalibrated from `responses.jsonl` with real normative data, percentiles are not clinically validated.
- **Vision results are coarse with few items.** With 6 items the ability estimate's SE is about 0.7–0.9. With 6 items the score moves in coarse steps: in simulation, a perfect run lands around the 95th percentile and 4/6 correct around the 27th.
- **Video was validated only on synthetic clips** made from a still photo. Thresholds (head turn 10°, nod 8°, blink, hold 0.4 s), eye-only gaze, `look_up`/`look_down`, the hand actions and the `mirrored` default all still need checking on real webcam recordings.
