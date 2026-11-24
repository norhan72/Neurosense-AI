"""Video analysis test: did the person perform an instructed action, and how fast?

Each uploaded clip shows one person sitting in front of a camera and comes with
one instructed action (``look_left``, ``close_eyes``, ``raise_hand``, ...). For
every clip we

1. decode it at ~12 fps (OpenCV, with an ffmpeg transcode fallback for files
   OpenCV cannot read, e.g. some browser ``webm`` recordings),
2. run MediaPipe Tasks landmarkers in VIDEO mode (face always, pose only for hand
   actions) and turn each frame into raw signals (head yaw/pitch, iris gaze,
   eye-blink, wrist height),
3. map those signals to a per-frame *action score* in ``[0, 1]``,
4. find the action onset on the score timeline with hysteresis and a minimum hold
   time (pure functions, see ``detect_action_onset``), and
5. combine all clips into an overall score (``summarize_results``).

Conventions
-----------
* Actions are in the PERSON's frame of reference ("look left" = the person's own
  left). Recordings are assumed NOT mirrored (a plain camera recording, where the
  person's left appears on the image right). Pass ``mirrored=True`` for clips
  recorded from a mirrored selfie preview; left and right are then swapped.
* Raw signals are image based: positive yaw/gaze = toward the image right (the
  nose moves toward face landmark 454), positive pitch = looking up. This was
  verified empirically: horizontally flipping a frame flips the yaw sign, and
  compressing the image-right half of a face (that cheek turning away) raises it.
* "State" actions (look_straight, open_eyes, lower_hand) can already be true when
  the instruction is given. The clip is then reported as detected with a response
  time of 0 s and an explanatory ``note``.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import tempfile
import threading
import urllib.request
from contextlib import ExitStack
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

logger = logging.getLogger("neurosense.video")

# --- Tunables -----------------------------------------------------------------

TARGET_FPS = 12.0  # frames per second actually analysed
MAX_DURATION_S = 30.0  # longer clips are truncated (with a warning)
MAX_FRAME_WIDTH = 640  # frames are downscaled to this width before inference
FFMPEG_TIMEOUT_S = 120

# Timeline / onset detection.
SCORE_THRESHOLD = 0.5  # score at which an action counts as "on"
SCORE_RELEASE = 0.35  # hysteresis: it stays "on" until the score drops below this
MIN_HOLD_S = 0.4  # minimum duration (rejects blinks and jitter)
MAX_DROPOUT_S = 0.25  # short dips / missing frames inside a hold are bridged
BASELINE_WINDOW_S = 0.5  # neutral pose is estimated from this much video

# Signal scaling: the deviation that maps to a score of 1.0 (0.5 = threshold).
YAW_FULL_DEG = 20.0  # head turn: threshold at 10 deg from neutral
PITCH_FULL_DEG = 16.0  # head nod: threshold at 8 deg from neutral
GAZE_H_FULL = 0.6  # iris offset (-1..1 across the eye width): threshold at 0.3
GAZE_V_FULL = 0.5  # eyeLookUp - eyeLookDown blendshapes: threshold at 0.25
# eyeBlink blendshapes rarely reach 1.0 even for firmly closed eyes, so they are
# rescaled: score 0 at BLINK_OPEN, 1 at BLINK_CLOSED (threshold at 0.35).
BLINK_OPEN = 0.1
BLINK_CLOSED = 0.6
BASELINE_CAP_DEG = 10.0  # a neutral pose further than this is not "neutral"
BASELINE_CAP_GAZE = 0.25

# "Look straight" tolerances (absolute, not baseline relative).
STRAIGHT_YAW_TOL_DEG = 12.0
STRAIGHT_PITCH_TOL_DEG = 15.0
STRAIGHT_GAZE_TOL = 0.35

# Hand raise: wrist height above the shoulder, in shoulder widths.
HAND_RAISE_ZERO = -0.2  # score 0 (wrist slightly below the shoulder line)
HAND_RAISE_FULL = 0.4  # score 1 (wrist around head height)
MIN_LANDMARK_VISIBILITY = 0.5

# Quality / scoring.
MIN_DETECTION_RATIO = 0.6  # warn when the face/pose is found in fewer frames
RT_FULL_SCORE_S = 1.5  # response time score is 1.0 up to here ...
RT_ZERO_SCORE_S = 5.0  # ... and falls linearly to 0.0 here
COMPLETION_WEIGHT = 0.6
RESPONSE_TIME_WEIGHT = 0.4

# --- Models ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("NEUROSENSE_MODELS_DIR", PROJECT_ROOT / "models"))
_MODEL_BASE_URL = "https://storage.googleapis.com/mediapipe-models/"
MODEL_URLS = {
    "face_landmarker.task": _MODEL_BASE_URL
    + "face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "pose_landmarker_lite.task": _MODEL_BASE_URL
    + "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}
FACE_MODEL = "face_landmarker.task"
POSE_MODEL = "pose_landmarker_lite.task"

_download_lock = threading.Lock()


class ModelUnavailableError(RuntimeError):
    """A landmarker model is missing and could not be downloaded."""


class VideoDecodeError(RuntimeError):
    """The uploaded file could not be decoded as a video."""


def ensure_model(name: str) -> Path:
    """Return the local path of a model, downloading it on first use.

    The download goes to a temp file in ``MODELS_DIR`` that is atomically renamed,
    so a crashed or concurrent download never leaves a truncated model behind.
    """
    path = MODELS_DIR / name
    if path.is_file():
        return path

    with _download_lock:
        if path.is_file():  # another thread finished the download meanwhile
            return path
        url = MODEL_URLS[name]
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=MODELS_DIR, suffix=".part")
        try:
            logger.info("Downloading %s from %s", name, url)
            with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as resp:
                shutil.copyfileobj(resp, out)
            os.replace(tmp_name, path)
        except Exception as exc:
            Path(tmp_name).unlink(missing_ok=True)
            raise ModelUnavailableError(
                f"Model '{name}' is not available at {path} and could not be "
                f"downloaded from {url}: {exc}. Run scripts/download_video_models.py."
            ) from exc
    return path


@cache
def _mediapipe():
    """Import MediaPipe lazily: it is slow to import and noisy on stderr."""
    # Silence per-landmarker INFO/WARNING spam (model load, XNNPACK, sparsity).
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    return mp, vision, BaseOptions


# --- Actions --------------------------------------------------------------------

STATE = "state"  # the target state may already hold at the instruction moment
TRANSITION = "transition"  # the person has to move away from neutral


@dataclass(frozen=True)
class ActionSpec:
    key: str
    label_en: str
    label_ar: str
    instruction_en: str
    instruction_ar: str
    kind: str
    needs_pose: bool = False


ACTIONS: dict[str, ActionSpec] = {
    spec.key: spec
    for spec in (
        ActionSpec("look_right", "Look right", "النظر إلى اليمين",
                   "Turn your head and look to your right.",
                   "أدر رأسك وانظر إلى يمينك.", TRANSITION),
        ActionSpec("look_left", "Look left", "النظر إلى اليسار",
                   "Turn your head and look to your left.",
                   "أدر رأسك وانظر إلى يسارك.", TRANSITION),
        ActionSpec("look_up", "Look up", "النظر إلى الأعلى",
                   "Look up.", "انظر إلى الأعلى.", TRANSITION),
        ActionSpec("look_down", "Look down", "النظر إلى الأسفل",
                   "Look down.", "انظر إلى الأسفل.", TRANSITION),
        ActionSpec("look_straight", "Look straight", "النظر إلى الأمام",
                   "Look straight at the camera.",
                   "انظر مباشرة إلى الكاميرا.", STATE),
        ActionSpec("close_eyes", "Close eyes", "إغلاق العينين",
                   "Close your eyes.", "أغلق عينيك.", TRANSITION),
        ActionSpec("open_eyes", "Open eyes", "فتح العينين",
                   "Open your eyes.", "افتح عينيك.", STATE),
        ActionSpec("raise_hand", "Raise a hand", "رفع اليد",
                   "Raise one hand above your shoulder.",
                   "ارفع إحدى يديك فوق مستوى كتفك.", TRANSITION, needs_pose=True),
        ActionSpec("raise_left_hand", "Raise left hand", "رفع اليد اليسرى",
                   "Raise your left hand above your shoulder.",
                   "ارفع يدك اليسرى فوق مستوى كتفك.", TRANSITION, needs_pose=True),
        ActionSpec("raise_right_hand", "Raise right hand", "رفع اليد اليمنى",
                   "Raise your right hand above your shoulder.",
                   "ارفع يدك اليمنى فوق مستوى كتفك.", TRANSITION, needs_pose=True),
        ActionSpec("lower_hand", "Lower your hand", "إنزال اليد",
                   "Put your hand down.", "أنزل يدك.", STATE, needs_pose=True),
    )
}

_EXTRA_ALIASES = {
    "look forward": "look_straight",
    "look ahead": "look_straight",
    "look at the camera": "look_straight",
    "look straight ahead": "look_straight",
    "close your eyes": "close_eyes",
    "open your eyes": "open_eyes",
    "raise a hand": "raise_hand",
    "raise your hand": "raise_hand",
    "raise one hand": "raise_hand",
    "hand up": "raise_hand",
    "raise your left hand": "raise_left_hand",
    "raise your right hand": "raise_right_hand",
    "put hand down": "lower_hand",
    "put your hand down": "lower_hand",
    "put down hand": "lower_hand",
    "hand down": "lower_hand",
    "lower hand": "lower_hand",
    "lower a hand": "lower_hand",
}


def _alias_key(text: str) -> str:
    return " ".join(text.strip().lower().replace("_", " ").replace("-", " ").split())


_ALIASES = {_alias_key(key): key for key in ACTIONS} | {
    _alias_key(alias): key for alias, key in _EXTRA_ALIASES.items()
}


def normalize_action(name: str) -> str | None:
    """Map a friendly action name ("Look Left", "raise a hand") to its key."""
    return _ALIASES.get(_alias_key(str(name)))


def list_actions() -> list[dict[str, Any]]:
    return [
        {
            "action": spec.key,
            "label_en": spec.label_en,
            "label_ar": spec.label_ar,
            "instruction_en": spec.instruction_en,
            "instruction_ar": spec.instruction_ar,
            "kind": spec.kind,
        }
        for spec in ACTIONS.values()
    ]


def required_models(action_keys: list[str]) -> list[str]:
    models = [FACE_MODEL]
    if any(ACTIONS[key].needs_pose for key in action_keys):
        models.append(POSE_MODEL)
    return models


# --- Decoding -------------------------------------------------------------------


def _ffmpeg_executable() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in (r"C:\Program Files\ffmpeg\bin\ffmpeg.exe", "/usr/bin/ffmpeg"):
        if os.path.isfile(candidate):
            return candidate
    return None


def transcode_to_mp4(src: str, dst: str, fps: float = TARGET_FPS) -> None:
    """Re-encode ``src`` to a constant-fps H.264 mp4 that OpenCV can read."""
    ffmpeg = _ffmpeg_executable()
    if ffmpeg is None:
        raise VideoDecodeError("video could not be decoded and ffmpeg is not installed")
    cmd = [
        ffmpeg, "-y", "-v", "error", "-i", src, "-an",
        "-t", str(MAX_DURATION_S),
        "-vf", f"fps={fps:g},scale='min({MAX_FRAME_WIDTH},iw)':-2",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", dst,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise VideoDecodeError("ffmpeg timed out while converting the video") from exc
    if proc.returncode != 0:
        message = proc.stderr.decode(errors="replace").strip().splitlines()
        raise VideoDecodeError(
            "video could not be decoded" + (f": {message[-1]}" if message else "")
        )


def iter_sampled_frames(path: str, target_fps: float = TARGET_FPS,
                        stats: dict[str, Any] | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(timestamp_ms, rgb_frame)`` at roughly ``target_fps``.

    Timestamps come from ``CAP_PROP_POS_MSEC`` (falling back to index/fps, which
    browser webm files often report as 0 or 1000) and are forced to be strictly
    increasing, as MediaPipe VIDEO mode requires. Reads until EOF because the
    frame count is unreliable for webm. ``stats["truncated"]`` is set when the
    video was longer than ``MAX_DURATION_S``.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoDecodeError("video could not be opened")
    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if not 1.0 <= native_fps <= 240.0:
            native_fps = 0.0
        step_ms = 1000.0 / target_fps
        next_sample_ms = 0.0
        last_t = -1.0
        last_ts = -1
        index = -1
        while cap.grab():  # grab() is cheap; only sampled frames are decoded
            index += 1
            t = cap.get(cv2.CAP_PROP_POS_MSEC)
            if not t > last_t:
                by_index = index * 1000.0 / native_fps if native_fps else -1.0
                t = by_index if by_index > last_t else last_t + 1000.0 / (native_fps or 30.0)
            last_t = t
            if t > MAX_DURATION_S * 1000.0:
                if stats is not None:
                    stats["truncated"] = True
                break
            if t < next_sample_ms:
                continue
            next_sample_ms += step_ms
            if next_sample_ms <= t:  # skipped ahead (variable frame rate): resync
                next_sample_ms = t + step_ms

            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            if width > MAX_FRAME_WIDTH:
                scale = MAX_FRAME_WIDTH / width
                frame = cv2.resize(frame, (MAX_FRAME_WIDTH, round(height * scale)),
                                   interpolation=cv2.INTER_AREA)
            timestamp_ms = max(int(round(t)), last_ts + 1)
            last_ts = timestamp_ms
            yield timestamp_ms, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


# --- Per-frame features -----------------------------------------------------------

# Face mesh indices (MediaPipe 478-point topology, refined with irises).
NOSE_TIP, CHEEK_A, CHEEK_B = 1, 234, 454  # 454 is on the image right for a camera view
EYES = (  # (corner, corner, iris centre) per eye
    (33, 133, 468),
    (362, 263, 473),
)
# Pose landmark indices.
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 11, 12, 15, 16

FEATURE_NAMES = ("yaw", "pitch", "gaze_h", "gaze_v", "blink", "raise_l", "raise_r")


@dataclass
class FrameSeries:
    """Raw per-frame signals; NaN where the detector found nothing."""

    times_s: np.ndarray
    features: dict[str, np.ndarray]
    face_found: np.ndarray
    pose_found: np.ndarray | None  # None when the pose model was not run
    truncated: bool = False  # the video was longer than MAX_DURATION_S


def _head_angles(matrix: np.ndarray) -> tuple[float, float]:
    """(yaw, pitch) in degrees: yaw > 0 toward image right, pitch > 0 looking up."""
    rot = np.asarray(matrix)[:3, :3]
    yaw = math.degrees(math.atan2(-rot[2, 0], math.hypot(rot[0, 0], rot[1, 0])))
    # The matrix pitch decreases when the head tilts back (verified against the
    # nose-between-forehead-and-chin geometry), so negate it to make "up" positive.
    pitch = -math.degrees(math.atan2(rot[2, 1], rot[2, 2]))
    return yaw, pitch


def _iris_offset(landmarks) -> float:
    """Mean horizontal iris position over both eyes, -1 (image left) .. 1 (image right)."""
    offsets = []
    for corner_a, corner_b, iris in EYES:
        x_a, x_b = landmarks[corner_a].x, landmarks[corner_b].x
        width = abs(x_b - x_a)
        if width > 1e-6:
            offsets.append(2.0 * (landmarks[iris].x - min(x_a, x_b)) / width - 1.0)
    return float(np.mean(offsets)) if offsets else math.nan


def _face_features(result) -> dict[str, float] | None:
    if not result.face_landmarks:
        return None
    landmarks = result.face_landmarks[0]
    blend = {c.category_name: c.score for c in result.face_blendshapes[0]}

    if result.facial_transformation_matrixes:
        yaw, pitch = _head_angles(result.facial_transformation_matrixes[0])
    else:  # geometric fallback: nose position between the cheeks
        span = landmarks[CHEEK_B].x - landmarks[CHEEK_A].x
        ratio = (landmarks[NOSE_TIP].x - landmarks[CHEEK_A].x) / span if span else 0.5
        yaw, pitch = math.degrees(math.asin(np.clip(2 * ratio - 1, -1, 1))), math.nan

    return {
        "yaw": yaw,
        "pitch": pitch,
        "gaze_h": _iris_offset(landmarks),
        "gaze_v": (blend["eyeLookUpLeft"] + blend["eyeLookUpRight"]
                   - blend["eyeLookDownLeft"] - blend["eyeLookDownRight"]) / 2.0,
        # Symmetric mean, so the blendshape left/right naming does not matter.
        "blink": (blend["eyeBlinkLeft"] + blend["eyeBlinkRight"]) / 2.0,
    }


def _wrist_raise_score(landmarks, wrist: int, shoulder: int, shoulder_width: float,
                       aspect: float) -> float:
    if landmarks[wrist].visibility < MIN_LANDMARK_VISIBILITY:
        return 0.0  # wrist out of view: for a seated person this means "down"
    # y grows downward; scale y by the aspect ratio so units match x.
    height = (landmarks[shoulder].y - landmarks[wrist].y) / aspect / shoulder_width
    return float(np.clip((height - HAND_RAISE_ZERO) / (HAND_RAISE_FULL - HAND_RAISE_ZERO), 0, 1))


def _pose_features(result, aspect: float) -> dict[str, float] | None:
    """Anatomical (person's own) left/right raise scores, or None without a person."""
    if not result.pose_landmarks:
        return None
    lm = result.pose_landmarks[0]
    if min(lm[L_SHOULDER].visibility, lm[R_SHOULDER].visibility) < MIN_LANDMARK_VISIBILITY:
        return None
    shoulder_width = math.hypot(lm[L_SHOULDER].x - lm[R_SHOULDER].x,
                                (lm[L_SHOULDER].y - lm[R_SHOULDER].y) / aspect)
    if shoulder_width < 1e-3:
        return None
    return {
        "raise_l": _wrist_raise_score(lm, L_WRIST, L_SHOULDER, shoulder_width, aspect),
        "raise_r": _wrist_raise_score(lm, R_WRIST, R_SHOULDER, shoulder_width, aspect),
    }


def extract_features(path: str, use_pose: bool) -> FrameSeries:
    """Run the landmarkers over a video. Landmarkers are created per call because
    MediaPipe instances are neither thread-safe nor reusable across timelines."""
    mp, vision, BaseOptions = _mediapipe()
    face_options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model(FACE_MODEL))),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    times, face_found, pose_found = [], [], []
    stats: dict[str, Any] = {}
    rows: dict[str, list[float]] = {name: [] for name in FEATURE_NAMES}

    with ExitStack() as stack:
        face_lm = stack.enter_context(vision.FaceLandmarker.create_from_options(face_options))
        pose_lm = None
        if use_pose:
            pose_options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(ensure_model(POSE_MODEL))),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
            )
            pose_lm = stack.enter_context(vision.PoseLandmarker.create_from_options(pose_options))

        for timestamp_ms, rgb in iter_sampled_frames(path, stats=stats):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            values = _face_features(face_lm.detect_for_video(image, timestamp_ms)) or {}
            face_found.append(bool(values))
            if pose_lm is not None:
                aspect = rgb.shape[1] / rgb.shape[0]  # normalised y -> x units
                pose_values = _pose_features(pose_lm.detect_for_video(image, timestamp_ms), aspect)
                pose_found.append(pose_values is not None)
                values |= pose_values or {}
            times.append(timestamp_ms / 1000.0)
            for name in FEATURE_NAMES:
                rows[name].append(values.get(name, math.nan))

    return FrameSeries(
        times_s=np.asarray(times, dtype=float),
        features={name: np.asarray(vals, dtype=float) for name, vals in rows.items()},
        face_found=np.asarray(face_found, dtype=bool),
        pose_found=np.asarray(pose_found, dtype=bool) if use_pose else None,
        truncated=stats.get("truncated", False),
    )


def extract_features_with_fallback(path: str, use_pose: bool) -> tuple[FrameSeries, bool]:
    """Decode with OpenCV; if that fails or yields no frames, transcode with ffmpeg.

    Returns the series and whether the ffmpeg fallback was used.
    """
    try:
        series = extract_features(path, use_pose)
        if series.times_s.size:
            return series, False
    except VideoDecodeError:
        pass

    fd, converted = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        transcode_to_mp4(path, converted)
        series = extract_features(converted, use_pose)
    finally:
        Path(converted).unlink(missing_ok=True)
    if not series.times_s.size:
        raise VideoDecodeError("video contains no decodable frames")
    return series, True


# --- Action scores (raw signals -> [0, 1]) ----------------------------------------


def _baseline(times_s: np.ndarray, values: np.ndarray, start_s: float, cap: float) -> float:
    """Robust neutral value: median of the first valid samples around the start.

    Clamped to ``±cap`` so a person who is already clearly turned is not treated
    as neutral (which would hide a real movement).
    """
    valid = ~np.isnan(values)
    window = valid & (times_s >= start_s) & (times_s <= start_s + BASELINE_WINDOW_S)
    if window.sum() < 3:  # too little data in the window: use the first valid samples
        window = valid & (np.cumsum(valid) <= 5)
    if not window.any():
        return 0.0
    return float(np.clip(np.median(values[window]), -cap, cap))


def _directional(values: np.ndarray, times_s: np.ndarray, baseline_start_s: float,
                 sign: float, full: float, cap: float) -> np.ndarray:
    base = _baseline(times_s, values, baseline_start_s, cap)
    return np.clip(sign * (values - base) / full, 0.0, 1.0)


def action_score_series(action: str, series: FrameSeries, mirrored: bool = False,
                        instruction_offset_s: float = 0.0) -> np.ndarray:
    """Per-frame score in [0, 1] (NaN where undetermined) for one action."""
    f, t = series.features, series.times_s
    # Person's left appears on the image right in a non-mirrored recording.
    person_left = -1.0 if mirrored else 1.0
    # Neutral pose: the half second just before the instruction (or the start).
    base_start = max(0.0, instruction_offset_s - BASELINE_WINDOW_S)

    eyes_closed = np.clip((f["blink"] - BLINK_OPEN) / (BLINK_CLOSED - BLINK_OPEN), 0.0, 1.0)

    def look(sign_head: float, head: str, head_full: float,
             gaze: str, gaze_full: float) -> np.ndarray:
        head_score = _directional(f[head], t, base_start, sign_head, head_full, BASELINE_CAP_DEG)
        gaze_score = _directional(f[gaze], t, base_start, sign_head, gaze_full, BASELINE_CAP_GAZE)
        return np.fmax(head_score, gaze_score)  # fmax: one NaN signal does not hide the other

    match action:
        case "look_left":
            return look(person_left, "yaw", YAW_FULL_DEG, "gaze_h", GAZE_H_FULL)
        case "look_right":
            return look(-person_left, "yaw", YAW_FULL_DEG, "gaze_h", GAZE_H_FULL)
        case "look_up":
            return look(1.0, "pitch", PITCH_FULL_DEG, "gaze_v", GAZE_V_FULL)
        case "look_down":
            return look(-1.0, "pitch", PITCH_FULL_DEG, "gaze_v", GAZE_V_FULL)
        case "look_straight":
            deviation = np.fmax.reduce([
                np.abs(f["yaw"]) / STRAIGHT_YAW_TOL_DEG,
                np.abs(f["pitch"]) / STRAIGHT_PITCH_TOL_DEG,
                np.abs(f["gaze_h"]) / STRAIGHT_GAZE_TOL,
            ])
            return np.clip(1.0 - deviation / 2.0, 0.0, 1.0)  # deviation 1 -> 0.5
        case "close_eyes":
            return eyes_closed
        case "open_eyes":
            return 1.0 - eyes_closed
        case "raise_hand":
            return np.fmax(f["raise_l"], f["raise_r"])
        case "raise_left_hand":
            return (f["raise_r"] if mirrored else f["raise_l"]).copy()
        case "raise_right_hand":
            return (f["raise_l"] if mirrored else f["raise_r"]).copy()
        case "lower_hand":
            return 1.0 - np.fmax(f["raise_l"], f["raise_r"])
    raise ValueError(f"unsupported action: {action}")


# --- Timeline -> onset (pure functions) --------------------------------------------


@dataclass(frozen=True)
class OnsetResult:
    detected: bool
    onset_s: float | None = None
    response_time_s: float | None = None
    hold_duration_s: float = 0.0
    already_in_state: bool = False
    peak_score: float = 0.0  # max smoothed score after the instruction
    hold_mean_score: float = 0.0  # mean smoothed score during the detected hold


def prepare_scores(times_s: np.ndarray, scores: np.ndarray,
                   max_gap_s: float = MAX_DROPOUT_S) -> np.ndarray:
    """Fill short detector dropouts and remove single-frame spikes.

    Interior NaN runs (between two valid samples) lasting up to ``max_gap_s`` are
    filled with the preceding value; everything else that is missing counts as 0.
    Then a 3-sample median filter is applied (it does not shift step edges).
    """
    scores = np.asarray(scores, dtype=float).copy()
    valid = np.flatnonzero(~np.isnan(scores))
    for prev, nxt in zip(valid[:-1], valid[1:]):
        if nxt - prev > 1 and times_s[nxt] - times_s[prev + 1] <= max_gap_s + 1e-9:
            scores[prev + 1:nxt] = scores[prev]
    scores = np.nan_to_num(scores, nan=0.0)
    if scores.size >= 3:
        padded = np.concatenate(([scores[0]], scores, [scores[-1]]))
        scores = np.median(np.lib.stride_tricks.sliding_window_view(padded, 3), axis=1)
    return scores


def _hysteresis_segments(scores: np.ndarray, threshold: float,
                         release: float) -> list[tuple[int, int]]:
    """Index ranges [start, end) where the score is "on" with hysteresis."""
    segments, start = [], None
    for i, score in enumerate(scores):
        if start is None and score >= threshold:
            start = i
        elif start is not None and score < release:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(scores)))
    return segments


def _merge_close(segments: list[tuple[int, int]], times_s: np.ndarray,
                 max_gap_s: float) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in segments:
        # gap = from the first "off" sample to the next "on" sample
        if merged and times_s[start] - times_s[merged[-1][1]] <= max_gap_s + 1e-9:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _crossing_time(times_s: np.ndarray, scores: np.ndarray, index: int,
                   threshold: float) -> float:
    """Linearly interpolated time at which the score crossed ``threshold``."""
    if index == 0 or scores[index] <= scores[index - 1]:
        return float(times_s[index])
    frac = (threshold - scores[index - 1]) / (scores[index] - scores[index - 1])
    frac = min(max(frac, 0.0), 1.0)
    return float(times_s[index - 1] + frac * (times_s[index] - times_s[index - 1]))


def detect_action_onset(
    times_s: np.ndarray,
    scores: np.ndarray,
    instruction_offset_s: float = 0.0,
    threshold: float = SCORE_THRESHOLD,
    release: float = SCORE_RELEASE,
    min_hold_s: float = MIN_HOLD_S,
    max_gap_s: float = MAX_DROPOUT_S,
) -> OnsetResult:
    """Find when the action started.

    The action counts as performed once the (smoothed) score rises above
    ``threshold`` and stays above ``release`` for at least ``min_hold_s``. A hold
    cut off by the end of the video needs only half of ``min_hold_s``. Holds that
    end before the instruction are ignored; a hold already running at the
    instruction moment is reported with ``already_in_state`` and response time 0.
    """
    times_s = np.asarray(times_s, dtype=float)
    if times_s.size == 0:
        return OnsetResult(detected=False)
    smooth = prepare_scores(times_s, scores, max_gap_s)
    after = times_s >= instruction_offset_s
    peak = float(smooth[after].max()) if after.any() else 0.0
    frame_s = float(np.median(np.diff(times_s))) if times_s.size > 1 else 0.0

    segments = _merge_close(_hysteresis_segments(smooth, threshold, release), times_s, max_gap_s)
    for start, end in segments:
        truncated = end == len(smooth)
        end_s = float(times_s[end - 1]) + (0.0 if truncated else frame_s)
        if end_s <= instruction_offset_s:
            continue  # finished before the instruction was given
        onset = _crossing_time(times_s, smooth, start, threshold)
        hold = end_s - max(onset, instruction_offset_s)
        if hold < (min_hold_s / 2 if truncated else min_hold_s) - 1e-9:
            continue

        # "Already" = on at the instruction, or on from the first frame we saw
        # (allowing for a camera that starts a fraction of a second late).
        already = onset <= instruction_offset_s or (
            start == 0 and times_s[0] <= instruction_offset_s + 0.25
        )
        response = 0.0 if already else onset - instruction_offset_s
        return OnsetResult(
            detected=True,
            onset_s=round(onset, 3),
            response_time_s=round(max(response, 0.0), 3),
            hold_duration_s=round(hold, 3),
            already_in_state=already,
            peak_score=round(peak, 3),
            hold_mean_score=round(float(smooth[start:end].mean()), 3),
        )
    return OnsetResult(detected=False, peak_score=round(peak, 3))


def detection_confidence(onset: OnsetResult, coverage: float,
                         threshold: float = SCORE_THRESHOLD) -> float:
    """Confidence (0..1) in the reported decision, detected or not.

    Signal part: how far the score was from the threshold (mean score during the
    hold when detected, peak score when not), mapped to 0.55..1.0. It is scaled
    down when the face/pose was found in fewer than 80% of the frames.
    """
    if onset.detected:
        margin = (onset.hold_mean_score - threshold) / (1.0 - threshold)
    else:
        margin = (threshold - onset.peak_score) / threshold
    signal = 0.55 + 0.45 * float(np.clip(margin, 0.0, 1.0))
    return round(signal * float(np.clip(coverage / 0.8, 0.0, 1.0)), 3)


# --- Scoring (pure functions) -------------------------------------------------------


def response_time_score(response_time_s: float) -> float:
    """1.0 up to RT_FULL_SCORE_S, linear down to 0.0 at RT_ZERO_SCORE_S.

    Heuristic: a healthy adult usually reacts to a simple spoken/written
    instruction within ~1-2 s; MS often slows information processing speed.
    """
    span = RT_ZERO_SCORE_S - RT_FULL_SCORE_S
    return float(np.clip((RT_ZERO_SCORE_S - response_time_s) / span, 0.0, 1.0))


_SCORE_BANDS = (  # (minimum score, label_en, label_ar)
    (80.0, "Normal responsiveness", "استجابة طبيعية"),
    (60.0, "Mildly slowed responses", "بطء خفيف في الاستجابة"),
    (40.0, "Moderately slowed or incomplete responses", "بطء متوسط أو استجابات غير مكتملة"),
    (0.0, "Markedly impaired responses", "ضعف ملحوظ في الاستجابة"),
)

SUMMARY_NOTE_EN = (
    "Screening heuristic, not a diagnosis. Score = 100 x (0.6 x completion rate + "
    "0.4 x response-time score); response time scores 1.0 up to 1.5 s and drops "
    "linearly to 0 at 5 s. Videos already in the requested state at the instruction "
    "moment count as completed but are left out of the response-time average."
)
SUMMARY_NOTE_AR = (
    "هذه نتيجة فحص أولي تقريبية وليست تشخيصاً طبياً. النتيجة = 100 × (0.6 × نسبة إنجاز "
    "الحركات + 0.4 × درجة زمن الاستجابة)؛ تكون درجة زمن الاستجابة كاملة حتى 1.5 ثانية "
    "وتنخفض تدريجياً حتى الصفر عند 5 ثوانٍ. المقاطع التي كان فيها الشخص في الوضع المطلوب "
    "لحظة التعليمات تُحتسب منجزة لكنها لا تدخل في متوسط زمن الاستجابة."
)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Overall score over all analysed (non-error) videos."""
    analysed = [r for r in results if not r.get("error")]
    detected = [r for r in analysed if r.get("detected")]
    timed = [r for r in detected if not r.get("already_in_state")]

    summary: dict[str, Any] = {
        "num_videos": len(results),
        "num_analyzed": len(analysed),
        "num_detected": len(detected),
        "completion_rate": None,
        "mean_response_time_s": None,
        "response_time_score": None,
        "video_score": None,
        "label_en": "Not enough data",
        "label_ar": "لا توجد بيانات كافية",
        "note_en": SUMMARY_NOTE_EN,
        "note_ar": SUMMARY_NOTE_AR,
    }
    if not analysed:
        return summary

    completion = len(detected) / len(analysed)
    if timed:
        times = [r["response_time_s"] for r in timed]
        rt_score = float(np.mean([response_time_score(t) for t in times]))
        summary["mean_response_time_s"] = round(float(np.mean(times)), 3)
        summary["response_time_score"] = round(rt_score, 3)
        score = 100.0 * (COMPLETION_WEIGHT * completion + RESPONSE_TIME_WEIGHT * rt_score)
    else:  # no measurable response time: completion decides alone
        score = 100.0 * completion

    label_en, label_ar = next((en, ar) for low, en, ar in _SCORE_BANDS if score >= low)
    summary.update(
        completion_rate=round(completion, 3),
        video_score=round(score, 1),
        label_en=label_en,
        label_ar=label_ar,
    )
    return summary


# --- Per-video pipeline ---------------------------------------------------------------

_NOTE_ALREADY = {
    STATE: ("Already in the requested state at the instruction moment; "
            "response time reported as 0 s.",
            "كان الشخص في الوضع المطلوب لحظة إعطاء التعليمات؛ تم احتساب زمن الاستجابة 0 ثانية."),
    TRANSITION: ("The action was already in progress at the instruction moment; "
                 "response time reported as 0 s.",
                 "كانت الحركة قد بدأت قبل لحظة إعطاء التعليمات؛ تم احتساب زمن الاستجابة 0 ثانية."),
}


def _quality_warning(ratio: float, what_en: str, what_ar: str) -> tuple[str | None, str | None]:
    if ratio >= MIN_DETECTION_RATIO:
        return None, None
    percent = round(100 * ratio)
    return (
        f"The {what_en} was detected in only {percent}% of frames; "
        "check lighting and that the person is fully in view.",
        f"تم اكتشاف {what_ar} في {percent}% فقط من الإطارات؛ تأكد من الإضاءة ومن ظهور الشخص بالكامل.",
    )


def _empty_result(action: str, instruction_offset_s: float) -> dict[str, Any]:
    spec = ACTIONS.get(action)
    return {
        "action": action,
        "label_en": spec.label_en if spec else action,
        "label_ar": spec.label_ar if spec else action,
        "instruction_offset_s": instruction_offset_s,
        "detected": False,
        "response_time_s": None,
        "onset_s": None,
        "confidence": 0.0,
        "hold_duration_s": 0.0,
        "already_in_state": False,
        "frames_analyzed": 0,
        "duration_s": 0.0,
        "face_detected_ratio": 0.0,
        "pose_detected_ratio": None,
        "quality_warning": None,
        "quality_warning_ar": None,
        "note": None,
        "note_ar": None,
        "transcoded": False,
        "error": None,
    }


def analyze_video(path: str, action: str, instruction_offset_s: float = 0.0,
                  mirrored: bool = False) -> dict[str, Any]:
    """Analyse one clip for one action. Never raises; failures go to ``error``."""
    result = _empty_result(action, instruction_offset_s)
    spec = ACTIONS.get(action)
    if spec is None:
        result["error"] = f"unsupported action '{action}'"
        return result

    try:
        series, transcoded = extract_features_with_fallback(path, spec.needs_pose)
    except ModelUnavailableError:
        raise  # a server problem, not a problem with this video
    except VideoDecodeError as exc:
        result["error"] = str(exc)
        return result
    except Exception as exc:  # keep one bad clip from failing the whole request
        logger.exception("video analysis failed for %s", path)
        result["error"] = f"analysis failed: {exc}"
        return result

    times = series.times_s
    frame_s = float(np.median(np.diff(times))) if times.size > 1 else 0.0
    duration = float(times[-1] + frame_s) if times.size else 0.0
    face_ratio = float(series.face_found.mean()) if times.size else 0.0
    pose_ratio = float(series.pose_found.mean()) if series.pose_found is not None and times.size else None
    result.update(
        frames_analyzed=int(times.size),
        duration_s=round(duration, 3),
        face_detected_ratio=round(face_ratio, 3),
        pose_detected_ratio=None if pose_ratio is None else round(pose_ratio, 3),
        transcoded=transcoded,
    )

    if instruction_offset_s >= duration:
        result["error"] = (f"instruction offset {instruction_offset_s:g} s is beyond "
                           f"the end of the video ({duration:.2f} s)")
        return result

    coverage = pose_ratio if spec.needs_pose else face_ratio
    what = ("person (body pose)", "جسم الشخص") if spec.needs_pose else ("face", "الوجه")
    result["quality_warning"], result["quality_warning_ar"] = _quality_warning(coverage, *what)

    scores = action_score_series(action, series, mirrored, instruction_offset_s)
    onset = detect_action_onset(times, scores, instruction_offset_s)
    result.update(
        detected=onset.detected,
        response_time_s=onset.response_time_s,
        onset_s=onset.onset_s,
        confidence=detection_confidence(onset, coverage),
        hold_duration_s=onset.hold_duration_s,
        already_in_state=onset.already_in_state,
        peak_score=onset.peak_score,
    )
    if onset.already_in_state:
        result["note"], result["note_ar"] = _NOTE_ALREADY[spec.kind]
    if series.truncated:
        result["note"] = " ".join(filter(None, [
            result["note"], f"Only the first {MAX_DURATION_S:g} s were analysed."]))
        result["note_ar"] = " ".join(filter(None, [
            result["note_ar"], f"تم تحليل أول {MAX_DURATION_S:g} ثانية فقط."]))
    return result
