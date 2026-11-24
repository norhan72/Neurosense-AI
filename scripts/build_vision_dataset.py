"""Build the NeuroSense vision-test dataset from Caltech-101.

Each item is an object photo (one of 16 everyday classes) that is either left
clean ("control") or degraded with an MS-relevant visual distortion at a graded
severity.  The user must name the object among four options; accuracy on
degraded items measures visual function (see ``controllers/vision.py``).

Usage (from the repository root)::

    python scripts/build_vision_dataset.py --source <path>/101_ObjectCategories

Outputs ``data/vision/images/<id>.png`` and ``data/vision/manifest.json``.
The run is fully deterministic for a given ``--seed``.

Images: Caltech 101, Fei-Fei Li, Marco Andreetto, Marc'Aurelio Ranzato,
Pietro Perona, CaltechDATA, 2022, CC BY 4.0, doi:10.22002/D1.20086.
"""

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DATASET_VERSION = "1.0.0"
IMAGE_SIZE = 320
NUM_OPTIONS = 4
MIN_SOURCE_SIDE = 180
MID_GRAY = 128.0

# Caltech-101 folder name -> (label_en, label_ar).
CLASSES = {
    "airplanes": ("Airplane", "طائرة"),
    "Motorbikes": ("Motorbike", "دراجة نارية"),
    "car_side": ("Car", "سيارة"),
    "watch": ("Wristwatch", "ساعة يد"),
    "umbrella": ("Umbrella", "مظلة"),
    "chair": ("Chair", "كرسي"),
    "laptop": ("Laptop", "حاسوب محمول"),
    "cup": ("Cup", "كوب"),
    "scissors": ("Scissors", "مقص"),
    "butterfly": ("Butterfly", "فراشة"),
    "elephant": ("Elephant", "فيل"),
    "grand_piano": ("Grand piano", "بيانو"),
    "lamp": ("Lamp", "مصباح"),
    "headphone": ("Headphones", "سماعة رأس"),
    "strawberry": ("Strawberry", "فراولة"),
    "sunflower": ("Sunflower", "عباد الشمس"),
}

# Distortion -> severity parameter per level (level 1 = mildest).
SEVERITIES = {
    "blur": [2.0, 4.0, 7.0, 12.0],  # gaussian sigma in px (optic neuritis)
    "contrast": [0.35, 0.18, 0.09, 0.04],  # contrast factor toward mid-gray
    "double": [6, 12, 22],  # ghost shift in px (diplopia / INO)
    "fog": [0.45, 0.7, 0.88],  # blend toward white (haze)
    "desaturation": [0.5, 0.2, 0.0],  # remaining saturation (dyschromatopsia)
}
GHOST_ALPHA = 0.5
FOG_BLUR_SIGMA = 1.5

# IRT mapping: b = b_min + (b_max - b_min) * (1 - ssim) / (1 - ssim_floor),
# clipped to [b_min, b_max].  Discrimination `a` per distortion; guessing c is
# fixed at 1 / NUM_OPTIONS.
IRT_PARAMS = {
    "b_min": -2.5,
    "b_max": 2.5,
    "ssim_floor": 0.55,
    "a_default": 1.5,
    "a_by_distortion": {
        "none": 1.5,
        "blur": 1.5,
        "contrast": 1.5,
        "double": 1.5,
        "fog": 1.5,
        "desaturation": 0.8,
    },
    "c": 1.0 / NUM_OPTIONS,
}


# --------------------------------------------------------------------------- #
# Source image selection and preprocessing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Candidate:
    """A source photo together with its quality score."""

    path: Path
    score: float


def _score_image(path: Path) -> float | None:
    """Return a quality score (higher is better) or None if unusable.

    Prefers large, colour photos whose edge content is concentrated in the
    central region (a proxy for a centered, un-cluttered object).
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    short_side = min(h, w)
    if short_side < MIN_SOURCE_SIDE:
        return None
    # Penalise near-grayscale photos (colour matters for desaturation items),
    # but do not exclude them: some classes (car_side) are grayscale only.
    is_gray = np.abs(img[..., 0].astype(np.int16) - img[..., 2].astype(np.int16)).mean() < 6
    colour_bonus = 0.0 if is_gray else 1.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    center = grad[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    centeredness = center.sum() / (grad.sum() + 1e-6)  # 0.25 if uniform
    squareness = min(h, w) / max(h, w)
    return float(min(short_side, 400) / 400 + 2.0 * centeredness + 0.5 * squareness + colour_bonus)


def select_sources(class_dir: Path, per_class: int, rng: np.random.Generator) -> list[Path]:
    """Deterministically pick `per_class` good images from one class folder."""
    scored = [(p, _score_image(p)) for p in sorted(class_dir.glob("*.jpg"))]
    ranked = sorted(
        (Candidate(p, s) for p, s in scored if s is not None),
        key=lambda c: (-c.score, c.path.name),
    )
    pool = ranked[: per_class * 3]
    if len(pool) < per_class:
        raise ValueError(f"{class_dir.name}: only {len(pool)} usable images")
    chosen = rng.choice(len(pool), size=per_class, replace=False)
    return [pool[i].path for i in sorted(chosen)]


def to_square(img: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Center-crop near-square photos, letterbox (mid-gray) elongated ones."""
    h, w = img.shape[:2]
    ratio = w / h
    if 0.75 <= ratio <= 1.33:
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        img = img[y0 : y0 + side, x0 : x0 + side]
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), MID_GRAY, dtype=np.uint8)
    y0, x0 = (size - new_h) // 2, (size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


# --------------------------------------------------------------------------- #
# Distortions (all take / return float32 BGR images in [0, 255])
# --------------------------------------------------------------------------- #
def apply_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur: reduced visual acuity, as in optic neuritis."""
    return cv2.GaussianBlur(img, (0, 0), sigma)


def apply_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale contrast toward mid-gray (low-contrast acuity)."""
    return MID_GRAY + factor * (img - MID_GRAY)


def apply_double(img: np.ndarray, shift_px: int) -> np.ndarray:
    """Alpha-blend a horizontally shifted ghost (diplopia / INO)."""
    pad = shift_px
    padded = cv2.copyMakeBorder(img, 0, 0, pad, 0, cv2.BORDER_REPLICATE)
    ghost = padded[:, : img.shape[1]]
    return (1 - GHOST_ALPHA) * img + GHOST_ALPHA * ghost


def apply_fog(img: np.ndarray, alpha: float) -> np.ndarray:
    """Slight blur plus blend toward white (haze / glare)."""
    blurred = cv2.GaussianBlur(img, (0, 0), FOG_BLUR_SIGMA)
    return (1 - alpha) * blurred + alpha * 255.0


def apply_desaturation(img: np.ndarray, saturation: float) -> np.ndarray:
    """Reduce colour saturation (dyschromatopsia); luminance is preserved."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None]
    return gray + saturation * (img - gray)


DISTORTIONS = {
    "blur": apply_blur,
    "contrast": apply_contrast,
    "double": apply_double,
    "fog": apply_fog,
    "desaturation": apply_desaturation,
}


# --------------------------------------------------------------------------- #
# Metric and IRT mapping
# --------------------------------------------------------------------------- #
def ssim_gray(ref_bgr: np.ndarray, test_bgr: np.ndarray) -> float:
    """Mean SSIM (11x11 gaussian window, sigma 1.5) on grayscale."""
    return _ssim_channel(_to_gray(ref_bgr), _to_gray(test_bgr))


def ssim_color(ref_bgr: np.ndarray, test_bgr: np.ndarray) -> float:
    """Mean of the per-channel (B, G, R) SSIM; also penalises colour loss.

    Grayscale SSIM is blind to desaturation, so this is the metric used for
    the IRT mapping.  It equals the grayscale SSIM closely for the other
    (luminance-driven) distortions.
    """
    return float(np.mean([_ssim_channel(ref_bgr[..., c].astype(np.float64), test_bgr[..., c].astype(np.float64)) for c in range(3)]))


def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(img_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)


def _ssim_channel(x: np.ndarray, y: np.ndarray) -> float:
    """SSIM of two float64 single-channel images."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def blur(a: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(a, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)

    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x**2
    var_y = blur(y * y) - mu_y**2
    cov = blur(x * y) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )
    return float(ssim_map.mean())


def difficulty_from_ssim(ssim: float, params: dict = IRT_PARAMS) -> float:
    """Monotone map from degradation (1 - SSIM) to IRT difficulty `b`."""
    span = params["b_max"] - params["b_min"]
    degradation = (1.0 - ssim) / (1.0 - params["ssim_floor"])
    b = params["b_min"] + span * degradation
    return float(np.clip(b, params["b_min"], params["b_max"]))


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def _variants() -> list[tuple[str, int | None]]:
    """All (distortion, level) pairs for one class; the control comes first."""
    variants: list[tuple[str, int | None]] = [("none", None)]
    for name, params in SEVERITIES.items():
        variants += [(name, level) for level in range(1, len(params) + 1)]
    return variants


def build(source: Path, out: Path, per_class: int, seed: int) -> dict:
    """Generate images and return the manifest dictionary."""
    rng = np.random.default_rng(seed)
    images_dir = out / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True)

    raw_items = []
    variants = _variants()
    for key in CLASSES:
        class_dir = source / key
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        sources = select_sources(class_dir, per_class, rng)
        offset = int(rng.integers(per_class))
        for i, (distortion, level) in enumerate(variants):
            path = sources[(i + offset) % per_class]
            clean = to_square(cv2.imread(str(path), cv2.IMREAD_COLOR)).astype(np.float32)
            if distortion == "none":
                degraded, param = clean, None
            else:
                param = SEVERITIES[distortion][level - 1]
                degraded = DISTORTIONS[distortion](clean, param)
            degraded = np.clip(np.rint(degraded), 0, 255).astype(np.uint8)
            ssim = ssim_color(clean, degraded)
            raw_items.append((key, distortion, level, param, ssim, degraded))

    # Shuffle so ids reveal neither class nor distortion.
    order = rng.permutation(len(raw_items))
    items = []
    for n, idx in enumerate(order, start=1):
        key, distortion, level, param, ssim, degraded = raw_items[idx]
        item_id = f"v{n:04d}"
        cv2.imwrite(str(images_dir / f"{item_id}.png"), degraded, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        items.append(
            {
                "id": item_id,
                "file": f"images/{item_id}.png",
                "class": key,
                "distortion": distortion,
                "severity_level": 0 if level is None else level,
                "severity_param": param,
                "ssim": round(ssim, 4),
                "b": round(difficulty_from_ssim(ssim), 3),
                "a": IRT_PARAMS["a_by_distortion"][distortion],
                "c": IRT_PARAMS["c"],
            }
        )

    manifest = {
        "dataset_version": DATASET_VERSION,
        "created_with": {
            "source": "Caltech-101 (CC BY 4.0, doi:10.22002/D1.20086)",
            "per_class": per_class,
            "seed": seed,
            "image_size": IMAGE_SIZE,
            "severities": SEVERITIES,
            "ghost_alpha": GHOST_ALPHA,
            "fog_blur_sigma": FOG_BLUR_SIGMA,
        },
        "num_options": NUM_OPTIONS,
        "classes": [
            {"key": k, "label_en": en, "label_ar": ar} for k, (en, ar) in CLASSES.items()
        ],
        "irt": {
            **IRT_PARAMS,
            "formula": "b = clip(b_min + (b_max - b_min) * (1 - ssim) / (1 - ssim_floor), b_min, b_max)",
        },
        "items": sorted(items, key=lambda it: it["id"]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


README = """# NeuroSense vision-test dataset

Generated by `scripts/build_vision_dataset.py` - do not edit by hand.

## Attribution

Source photos: **Caltech 101** - Fei-Fei Li, Marco Andreetto, Marc'Aurelio
Ranzato, Pietro Perona. CaltechDATA, 2022. Licensed CC BY 4.0.
doi:10.22002/D1.20086. Photos were cropped/resized and distorted; the
modified images are released under the same licence.

## Items

16 object classes x {1 clean control + graded distortions}. Image ids are
opaque (`v0001`...) so filenames do not reveal the answer.

| Distortion | Simulates | Levels (parameter) |
|---|---|---|
| `blur` | optic neuritis / reduced acuity | gaussian sigma 2, 4, 7, 12 px |
| `contrast` | reduced contrast sensitivity | contrast factor 0.35, 0.18, 0.09, 0.04 toward mid-gray |
| `double` | diplopia / INO | ghost shift 6, 12, 22 px, alpha 0.5 |
| `fog` | hazy vision | blend to white 0.45, 0.7, 0.88 + sigma 1.5 blur |
| `desaturation` | dyschromatopsia | saturation 0.5, 0.2, 0.0 |

## IRT mapping

Each item stores a 3PL parameter set (`a`, `b`, `c`). Degradation is measured
as SSIM (11x11 gaussian window, mean of the B/G/R channels) against the clean image, then

    b = clip(b_min + (b_max - b_min) * (1 - ssim) / (1 - ssim_floor), b_min, b_max)

with b_min = -2.5, b_max = 2.5, ssim_floor = 0.55, a = 1.5 (0.8 for
desaturation, because colour loss is only mildly harmful to object naming), and
c = 1/4 (four options). All values live in `manifest.json`.

**The norms are model-based**: normal observers are assumed to be N(0, 1) on
the theta scale by construction. They are not derived from measured data until
real normative responses (collected in `responses.jsonl`) are used to
recalibrate `b` (e.g. with a Rasch/3PL fit).

## Regenerate

    python scripts/build_vision_dataset.py --source <path>/101_ObjectCategories --out data/vision

Download Caltech-101 from https://data.caltech.edu/records/mzrjq-6wc02.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="101_ObjectCategories directory")
    parser.add_argument("--out", type=Path, default=Path("data/vision"))
    parser.add_argument("--per-class", type=int, default=4, help="source photos per class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = build(args.source, args.out, args.per_class, args.seed)
    (args.out / "README.md").write_text(README, encoding="utf-8")
    print(f"Wrote {len(manifest['items'])} items to {args.out}")


if __name__ == "__main__":
    main()
