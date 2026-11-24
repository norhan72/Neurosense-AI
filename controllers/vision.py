"""Vision test: object naming under MS-relevant image distortions.

The user names an object shown with a distortion (blur, low contrast, ghosting,
haze, desaturation) among four options.  Responses are scored with a 3PL item
response model (guessing fixed at 1 / num_options) and person ability (theta)
is estimated by EAP on a grid with an N(0, 1) prior.

IMPORTANT: item difficulties come from a documented mapping of an objective
degradation metric (SSIM, see ``scripts/build_vision_dataset.py``) and normal
observers are *assumed* to be N(0, 1) on the theta scale.  The percentile and
vision score are therefore model-based, not clinically validated norms, until
real normative data (collected in ``data/vision/responses.jsonl``) is used to
recalibrate the items.  This is a screening aid, not a diagnosis.
"""

import json
import random
import threading
import uuid
from datetime import datetime, timezone
from math import erf, sqrt
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "vision"
MANIFEST_PATH = DATA_DIR / "manifest.json"
RESPONSES_PATH = DATA_DIR / "responses.jsonl"

MIN_COUNT, MAX_COUNT, DEFAULT_COUNT = 3, 10, 6
MIN_ANSWERS, MAX_ANSWERS = 3, 10
THETA_GRID = np.linspace(-4.0, 4.0, 161)
LOG_PRIOR = -0.5 * THETA_GRID**2  # N(0, 1), unnormalised
TARGET_B_RANGE = (-1.5, 2.0)  # difficulty spread of the non-control items

SYMPTOMS = {
    "none": ("No distortion (control)", "بدون تشوه (عنصر تحكم)"),
    "blur": ("Blurred vision (optic neuritis / reduced acuity)", "ضبابية الرؤية (التهاب العصب البصري / ضعف حدة الإبصار)"),
    "contrast": ("Reduced contrast sensitivity", "ضعف الحساسية للتباين"),
    "double": ("Double vision (diplopia / INO)", "ازدواج الرؤية (الشفع / اعتلال العين البيني النووي)"),
    "fog": ("Hazy vision", "رؤية ضبابية (غشاوة)"),
    "desaturation": ("Colour vision loss (dyschromatopsia)", "فقدان رؤية الألوان"),
}

# Percentile lower bound -> (label_en, label_ar), checked from the top.
BANDS = (
    (25.0, "Normal", "طبيعي"),
    (10.0, "Borderline", "حدّي"),
    (3.0, "Mild impairment", "ضعف خفيف"),
    (0.0, "Significant impairment", "ضعف ملحوظ"),
)

NOTE_EN = (
    "This is a screening aid, not a diagnosis. Results are based on model-based "
    "norms; please consult a physician for any concerns."
)
NOTE_AR = (
    "هذا الاختبار أداة فحص أولي وليس تشخيصاً. النتائج مبنية على معايير نموذجية؛ "
    "يُرجى استشارة الطبيب عند وجود أي قلق."
)


class VisionDatasetUnavailable(RuntimeError):
    """Raised when the dataset manifest has not been generated yet."""


class VisionInputError(ValueError):
    """Raised for invalid client input (maps to HTTP 400)."""


# --------------------------------------------------------------------------- #
# Dataset access
# --------------------------------------------------------------------------- #
class Dataset:
    """In-memory view of ``manifest.json``."""

    def __init__(self, manifest: dict):
        self.version = manifest["dataset_version"]
        self.num_options = manifest["num_options"]
        self.classes = {c["key"]: c for c in manifest["classes"]}
        self.items = {it["id"]: it for it in manifest["items"]}
        self.items_by_distortion: dict[str, list[dict]] = {}
        for it in manifest["items"]:
            self.items_by_distortion.setdefault(it["distortion"], []).append(it)

    def option(self, class_key: str) -> dict:
        """Public (answer-free) representation of a class option."""
        c = self.classes[class_key]
        return {"key": c["key"], "label_en": c["label_en"], "label_ar": c["label_ar"]}


_dataset: Dataset | None = None
_dataset_lock = threading.Lock()
_responses_lock = threading.Lock()


def get_dataset() -> Dataset:
    """Load the manifest once (thread-safe); failures are not cached."""
    global _dataset
    if _dataset is None:
        with _dataset_lock:
            if _dataset is None:
                if not MANIFEST_PATH.is_file():
                    raise VisionDatasetUnavailable(
                        "Vision dataset not found. Run: python scripts/build_vision_dataset.py "
                        "--source <101_ObjectCategories dir> --out data/vision"
                    )
                _dataset = Dataset(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))
    return _dataset


def image_path(image_id: str) -> Path:
    """Resolve an image path from a manifest id (never from raw input)."""
    item = get_dataset().items.get(image_id)
    if item is None:
        raise KeyError(image_id)
    return DATA_DIR / item["file"]


# --------------------------------------------------------------------------- #
# Exercise selection
# --------------------------------------------------------------------------- #
def select_exercises(count: int = DEFAULT_COUNT, rng: random.Random | None = None) -> list[dict]:
    """Pick a balanced item set: one control, the rest spread over distortion
    types and difficulty, all with distinct classes.  Answers are never included."""
    if not MIN_COUNT <= count <= MAX_COUNT:
        raise VisionInputError(f"count must be between {MIN_COUNT} and {MAX_COUNT}")
    rng = rng or random.Random()
    ds = get_dataset()

    used_classes: set[str] = set()

    def pick(pool: list[dict], target_b: float) -> dict:
        free = [it for it in pool if it["class"] not in used_classes]
        best = min(abs(it["b"] - target_b) for it in free)
        # Random among near-equal difficulty for variety.
        chosen = rng.choice([it for it in free if abs(it["b"] - target_b) <= best + 0.15])
        used_classes.add(chosen["class"])
        return chosen

    chosen_items = [pick(ds.items_by_distortion["none"], -2.5)]
    types = [t for t in ds.items_by_distortion if t != "none"]
    type_cycle: list[str] = []
    while len(type_cycle) < count - 1:
        block = types[:]
        rng.shuffle(block)
        type_cycle += block
    n = count - 1
    lo, hi = TARGET_B_RANGE
    targets = [lo + (hi - lo) * i / max(n - 1, 1) for i in range(n)]
    rng.shuffle(targets)
    for dist, target in zip(type_cycle[:n], targets):
        chosen_items.append(pick(ds.items_by_distortion[dist], target))

    rng.shuffle(chosen_items)
    exercises = []
    for it in chosen_items:
        distractors = rng.sample([k for k in ds.classes if k != it["class"]], ds.num_options - 1)
        keys = [it["class"], *distractors]
        rng.shuffle(keys)
        exercises.append(
            {
                "image_id": it["id"],
                "image_url": f"/api/vision/images/{it['id']}",
                "options": [ds.option(k) for k in keys],
            }
        )
    return exercises


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def estimate_theta(items: list[dict], correct: list[bool]) -> tuple[float, float]:
    """EAP estimate and posterior SD of ability under the 3PL model."""
    log_post = LOG_PRIOR.copy()
    for it, ok in zip(items, correct):
        p = it["c"] + (1 - it["c"]) / (1 + np.exp(-it["a"] * (THETA_GRID - it["b"])))
        log_post += np.log(p if ok else 1 - p)
    post = np.exp(log_post - log_post.max())
    post /= post.sum()
    theta = float((THETA_GRID * post).sum())
    se = float(np.sqrt((post * (THETA_GRID - theta) ** 2).sum()))
    return theta, se


def _band(percentile: float) -> tuple[str, str]:
    for lower, en, ar in BANDS:
        if percentile >= lower:
            return en, ar
    return BANDS[-1][1], BANDS[-1][2]


def _breakdown(items: list[dict], correct: list[bool]) -> list[dict]:
    stats: dict[str, list[int]] = {}
    for it, ok in zip(items, correct):
        entry = stats.setdefault(it["distortion"], [0, 0])
        entry[0] += 1
        entry[1] += int(ok)
    return [
        {
            "distortion": dist,
            "count": n,
            "correct": k,
            "accuracy": round(k / n, 3),
            "symptom_en": SYMPTOMS[dist][0],
            "symptom_ar": SYMPTOMS[dist][1],
        }
        for dist, (n, k) in stats.items()
    ]


def evaluate(answers: list[dict]) -> dict:
    """Score a set of answers, persist the evaluation and return the result.

    `answers` entries: ``{"image_id", "answer", "response_time_ms"?}``.
    """
    ds = get_dataset()
    if not MIN_ANSWERS <= len(answers) <= MAX_ANSWERS:
        raise VisionInputError(f"answers must contain between {MIN_ANSWERS} and {MAX_ANSWERS} items")
    ids = [a["image_id"] for a in answers]
    if len(set(ids)) != len(ids):
        raise VisionInputError("duplicate image_id in answers")
    unknown = [i for i in ids if i not in ds.items]
    if unknown:
        raise VisionInputError(f"unknown image_id: {unknown[0]}")

    items = [ds.items[i] for i in ids]
    correct = [a["answer"] == it["class"] for a, it in zip(answers, items)]
    theta, se = estimate_theta(items, correct)
    percentile = round(_phi(theta) * 100, 1)
    label_en, label_ar = _band(percentile)
    num_correct = sum(correct)

    result = {
        "session_id": str(uuid.uuid4()),
        "dataset_version": ds.version,
        "theta": round(theta, 3),
        "theta_se": round(se, 3),
        "percentile": percentile,
        "vision_score": percentile,
        "label_en": label_en,
        "label_ar": label_ar,
        "accuracy": round(num_correct / len(items), 3),
        "num_correct": num_correct,
        "num_items": len(items),
        "breakdown": _breakdown(items, correct),
        "items": [
            {
                "image_id": it["id"],
                "answer": a["answer"],
                "correct_class": it["class"],
                "label_en": ds.classes[it["class"]]["label_en"],
                "label_ar": ds.classes[it["class"]]["label_ar"],
                "correct": ok,
            }
            for a, it, ok in zip(answers, items, correct)
        ],
        "note_en": NOTE_EN,
        "note_ar": NOTE_AR,
    }
    _persist(result, answers)
    return result


def _persist(result: dict, answers: list[dict]) -> None:
    """Append one JSON line per evaluation (kept for later recalibration)."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": result["session_id"],
        "dataset_version": result["dataset_version"],
        "items": [
            {
                "image_id": it["image_id"],
                "answer": it["answer"],
                "correct_class": it["correct_class"],
                "correct": it["correct"],
                "response_time_ms": a.get("response_time_ms"),
            }
            for it, a in zip(result["items"], answers)
        ],
        "result": {
            k: result[k]
            for k in ("theta", "theta_se", "percentile", "vision_score", "label_en", "accuracy")
        },
    }
    line = json.dumps(record, ensure_ascii=False)
    with _responses_lock:
        RESPONSES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RESPONSES_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
