"""Combine the five test results into one MS likelihood percentage.

Each test is first turned into a risk in [0, 1] (0 = typical, 1 = most impaired):

* survey:    symptom_score / 100           (share of symptom points)
* vision:    1 - vision_score / 100        (percentile vs normal observers)
* video:     1 - video_score / 100         (completion + response time score)
* force:     level low / medium / high  ->  0, 0.5, 1
* vibration: level low / medium / high  ->  0, 0.5, 1

The likelihood is the weighted mean of those risks, in percent. All tests have an equal
share for now (``WEIGHTS``); replace the weights (or the whole function) with a model
fitted on labelled patient data when it exists. Until then the percentage is a
heuristic, not a calibrated probability, and never a diagnosis.
"""

TESTS = ("survey", "vision", "video", "force", "vibration")
WEIGHTS = {test: 1.0 for test in TESTS}
LEVEL_RISK = {"low": 0.0, "medium": 0.5, "high": 1.0}

# Likelihood lower bound (percent) -> (label_en, label_ar), checked from the top.
BANDS = (
    (60.0, "High likelihood: please see a neurologist", "احتمال مرتفع: يُرجى مراجعة طبيب أعصاب"),
    (30.0, "Moderate likelihood: a medical follow-up is advised", "احتمال متوسط: يُنصح بالمتابعة الطبية"),
    (0.0, "Low likelihood", "احتمال منخفض"),
)
NOTE_EN = (
    "Screening estimate from equally weighted test results, not a calibrated probability "
    "or a diagnosis. Please consult a neurologist for any concerns."
)
NOTE_AR = (
    "تقدير أولي مبني على نتائج الاختبارات بأوزان متساوية، وليس احتمالاً مُعايَراً ولا تشخيصاً. "
    "يُرجى استشارة طبيب أعصاب عند وجود أي قلق."
)


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def risks_by_test(survey_score: float, vision_score: float, video_score: float,
                  force_level: str, vibration_level: str) -> dict[str, float]:
    """Per-test risk in [0, 1]. Scores are 0-100; levels are 'low' | 'medium' | 'high'."""
    return {
        "survey": _clip01(survey_score / 100.0),
        "vision": _clip01(1.0 - vision_score / 100.0),
        "video": _clip01(1.0 - video_score / 100.0),
        "force": LEVEL_RISK[force_level],
        "vibration": LEVEL_RISK[vibration_level],
    }


def combine(risks: dict[str, float]) -> dict:
    total_weight = sum(WEIGHTS[test] for test in TESTS)
    likelihood = 100.0 * sum(WEIGHTS[test] * risks[test] for test in TESTS) / total_weight
    label_en, label_ar = next((en, ar) for low, en, ar in BANDS if likelihood >= low)
    return {
        "ms_likelihood": round(likelihood, 1),
        "label_en": label_en,
        "label_ar": label_ar,
        "tests": [
            {"test": test, "risk": round(risks[test], 3), "weight": round(WEIGHTS[test] / total_weight, 3)}
            for test in TESTS
        ],
        "note_en": NOTE_EN,
        "note_ar": NOTE_AR,
    }
