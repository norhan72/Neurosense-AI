"""Synthetic test clips for the video analysis test, built from one face photo.

No real recordings are needed (or committed): clips are generated at runtime
into a temp dir from any frontal face photo, e.g. a public-domain NASA portrait.

* ``turn_head(frame, s)`` fakes a head turn by compressing one half of the face
  horizontally (s > 0: the image-right half, i.e. toward the person's left in a
  non-mirrored recording). It moves the MediaPipe yaw by about +20 deg at s=0.6.
* ``close_eyes(frame)`` paints skin over both eyes and draws a lid line.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from controllers import video_analysis as va

EYE_RINGS = (
    (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
    (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398),
)
FRAME_SIZE = (640, 480)


@dataclass
class FaceScene:
    frame: np.ndarray  # BGR, FRAME_SIZE
    landmarks: list  # normalised face landmarks of ``frame``

    @property
    def nose_x(self) -> float:
        return self.landmarks[va.NOSE_TIP].x * self.frame.shape[1]

    @property
    def face_half_width(self) -> float:
        lm = self.landmarks
        return abs(lm[va.CHEEK_B].x - lm[va.CHEEK_A].x) * self.frame.shape[1] / 2


def _detect_landmarks(bgr: np.ndarray) -> list:
    mp, vision, BaseOptions = va._mediapipe()
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(va.ensure_model(va.FACE_MODEL))))
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        raise ValueError("no face found in the test photo")
    return result.face_landmarks[0]


def load_scene(photo: str | Path) -> FaceScene:
    """Crop the photo to a 4:3 webcam-like frame where the face is ~30% wide."""
    image = cv2.imread(str(photo))
    if image is None:
        raise FileNotFoundError(photo)
    lm = _detect_landmarks(image)
    h, w = image.shape[:2]
    xs = np.array([p.x for p in lm]) * w
    ys = np.array([p.y for p in lm]) * h
    crop_w = min((xs.max() - xs.min()) / 0.3, w)
    crop_h = min(crop_w * 3 / 4, h)
    crop_w = crop_h * 4 / 3
    cx, cy = xs.mean(), ys.mean() + 0.15 * crop_h  # leave room for shoulders
    x0 = int(np.clip(cx - crop_w / 2, 0, w - crop_w))
    y0 = int(np.clip(cy - crop_h / 2, 0, h - crop_h))
    frame = cv2.resize(image[y0:y0 + int(crop_h), x0:x0 + int(crop_w)], FRAME_SIZE,
                       interpolation=cv2.INTER_AREA)
    return FaceScene(frame, _detect_landmarks(frame))


def turn_head(scene: FaceScene, s: float) -> np.ndarray:
    """Compress one face half by ``|s|`` (s > 0: image-right half)."""
    frame = scene.frame
    if s == 0:
        return frame
    if s < 0:  # mirror the frame, compress its right half, mirror back
        mirrored_nose_x = frame.shape[1] - 1 - scene.nose_x
        return cv2.flip(_compress_right(cv2.flip(frame, 1), mirrored_nose_x,
                                        scene.face_half_width, -s), 1)
    return _compress_right(frame, scene.nose_x, scene.face_half_width, s)


def _compress_right(frame: np.ndarray, cx: float, half: float, s: float) -> np.ndarray:
    """Squeeze columns [cx, cx + half] by (1 - s) and stretch the rest to fill."""
    height, width = frame.shape[:2]
    xs = np.arange(width, dtype=np.float32)
    k = 1.0 - s
    right_end = min(cx + half, width - 2)
    src = np.where(
        xs < cx, xs,
        np.where(xs < cx + k * (right_end - cx), cx + (xs - cx) / k,
                 right_end + (xs - cx - k * (right_end - cx))
                 * (width - right_end) / (width - cx - k * (right_end - cx))))
    map_x = np.tile(src.astype(np.float32), (height, 1))
    map_y = np.tile(np.arange(height, dtype=np.float32)[:, None], (1, width))
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def close_eyes(scene: FaceScene) -> np.ndarray:
    frame = scene.frame.astype(np.float32)
    height, width = frame.shape[:2]
    for ring in EYE_RINGS:
        pts = np.array([[scene.landmarks[i].x * width, scene.landmarks[i].y * height]
                        for i in ring], np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        skin = scene.frame[y + h + 4:y + h + 10, x:x + w].reshape(-1, 3).mean(axis=0)
        mask = np.zeros((height, width), np.uint8)
        cv2.ellipse(mask, ((x + w / 2, y + h / 2), (w + 8, h + 8), 0), 255, -1)
        alpha = cv2.GaussianBlur(mask, (7, 7), 0)[..., None] / 255.0
        frame = frame * (1 - alpha) + skin * alpha
        cv2.line(frame, (x, y + h // 2 + 1), (x + w, y + h // 2 + 1), (40, 40, 60), 2)
    return frame.astype(np.uint8)


def write_clip(path: Path, frame_at: Callable[[float], np.ndarray],
               duration_s: float, fps: float = 25.0) -> Path:
    """Render ``frame_at(t)`` for t in [0, duration) to an mp4 file."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, FRAME_SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"cannot write {path}")
    try:
        for i in range(int(round(duration_s * fps))):
            writer.write(frame_at(i / fps))
    finally:
        writer.release()
    return path


def to_webm(src: Path, dst: Path) -> Path:
    ffmpeg = va._ffmpeg_executable()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found")
    subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(src), "-c:v", "libvpx-vp9",
                    "-b:v", "1M", "-deadline", "realtime", "-an", str(dst)], check=True)
    return dst


def ramp(t: float, start: float, rise: float = 0.2) -> float:
    """0 before ``start``, linear to 1 over ``rise`` seconds, then 1."""
    return float(np.clip((t - start) / rise, 0.0, 1.0))


def build_clips(photo: str | Path, out_dir: Path) -> dict[str, Path]:
    """Generate the standard set of synthetic clips into ``out_dir``."""
    scene = load_scene(photo)
    out_dir.mkdir(parents=True, exist_ok=True)
    closed = close_eyes(scene)
    turn_frames = {}

    def turned(t: float, start: float, sign: float) -> np.ndarray:
        level = round(ramp(t, start), 2)  # cache: few distinct warps
        key = (sign, level)
        if key not in turn_frames:
            turn_frames[key] = turn_head(scene, sign * 0.6 * level)
        return turn_frames[key]

    clips = {
        "static": write_clip(out_dir / "static.mp4", lambda t: scene.frame, 3.0),
        "static_flipped": write_clip(out_dir / "static_flipped.mp4",
                                     lambda t: cv2.flip(scene.frame, 1), 3.0),
        # Person turns to their own LEFT (image right) starting at 1.5 s.
        "turn_left": write_clip(out_dir / "turn_left.mp4", lambda t: turned(t, 1.5, 1.0), 4.0),
        "turn_right": write_clip(out_dir / "turn_right.mp4", lambda t: turned(t, 2.0, -1.0), 4.0),
        # Eyes close at 1.0 s.
        "close_eyes": write_clip(out_dir / "close_eyes.mp4",
                                 lambda t: closed if t >= 1.0 else scene.frame, 3.5),
    }
    if shutil.which("ffmpeg") or va._ffmpeg_executable():
        clips["turn_left_webm"] = to_webm(clips["turn_left"], out_dir / "turn_left.webm")
    return clips
