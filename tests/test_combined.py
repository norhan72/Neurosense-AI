"""Unit tests for the combined MS likelihood and its API."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from controllers import combined  # noqa: E402
from main import app  # noqa: E402

TYPICAL = dict(survey_score=0, vision_score=100, video_score=100, force_level="low", vibration_level="low")
WORST = dict(survey_score=100, vision_score=0, video_score=0, force_level="high", vibration_level="high")


def likelihood(**kwargs) -> float:
    return combined.combine(combined.risks_by_test(**kwargs))["ms_likelihood"]


class WeightTests(unittest.TestCase):
    def test_weights_cover_all_tests_and_are_equal(self):
        self.assertEqual(set(combined.WEIGHTS), set(combined.TESTS))
        self.assertEqual(len(set(combined.WEIGHTS.values())), 1)

    def test_normalized_weights_sum_to_one(self):
        result = combined.combine(combined.risks_by_test(**TYPICAL))
        self.assertAlmostEqual(sum(t["weight"] for t in result["tests"]), 1.0, places=2)
        self.assertEqual([t["test"] for t in result["tests"]], list(combined.TESTS))


class RiskTests(unittest.TestCase):
    def test_typical_inputs_have_zero_risk(self):
        self.assertTrue(all(r == 0 for r in combined.risks_by_test(**TYPICAL).values()))

    def test_worst_inputs_have_full_risk(self):
        self.assertTrue(all(r == 1 for r in combined.risks_by_test(**WORST).values()))

    def test_scores_are_inverted_where_higher_is_better(self):
        risks = combined.risks_by_test(**{**TYPICAL, "survey_score": 40, "vision_score": 70, "video_score": 50})
        self.assertAlmostEqual(risks["survey"], 0.4)
        self.assertAlmostEqual(risks["vision"], 0.3)
        self.assertAlmostEqual(risks["video"], 0.5)

    def test_medium_level_is_half_risk(self):
        risks = combined.risks_by_test(**{**TYPICAL, "force_level": "medium", "vibration_level": "medium"})
        self.assertEqual((risks["force"], risks["vibration"]), (0.5, 0.5))

    def test_out_of_range_scores_are_clipped(self):
        risks = combined.risks_by_test(**{**TYPICAL, "survey_score": 150, "vision_score": 130})
        self.assertEqual((risks["survey"], risks["vision"]), (1.0, 0.0))


class CombineTests(unittest.TestCase):
    def test_all_typical_is_zero_percent(self):
        self.assertEqual(likelihood(**TYPICAL), 0.0)

    def test_all_worst_is_hundred_percent(self):
        self.assertEqual(likelihood(**WORST), 100.0)

    def test_known_mixed_example(self):
        # risks: 0.4 + 0.3 + 0.5 + 0.5 + 0.0 = 1.7 -> 1.7 / 5 = 34%
        result = combined.combine(combined.risks_by_test(
            survey_score=40, vision_score=70, video_score=50, force_level="medium", vibration_level="low",
        ))
        self.assertEqual(result["ms_likelihood"], 34.0)
        self.assertEqual(result["label_en"], combined.BANDS[1][1])

    def test_band_labels_at_boundaries(self):
        def label(value: float) -> str:
            return combined.combine(dict.fromkeys(combined.TESTS, value))["label_en"]

        low, moderate, high = (band[1] for band in reversed(combined.BANDS))
        self.assertEqual(label(0.29), low)
        self.assertEqual(label(0.3), moderate)
        self.assertEqual(label(0.59), moderate)
        self.assertEqual(label(0.6), high)
        self.assertEqual(label(1.0), high)

    def test_result_has_bilingual_labels_and_notes(self):
        result = combined.combine(combined.risks_by_test(**TYPICAL))
        for key in ("label_en", "label_ar", "note_en", "note_ar"):
            self.assertTrue(result[key])


class CombinedApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    @staticmethod
    def payload(**overrides) -> dict:
        body = {
            "survey": {"symptom_score": 40},
            "vision": {"vision_score": 70},
            "video": {"video_score": 50},
            "force": {"level": "medium"},
            "vibration": {"level": "low"},
        }
        body.update(overrides)
        return body

    def post(self, body: dict):
        return self.client.post("/api/results/combined", json=body)

    def test_ok(self):
        response = self.post(self.payload())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ms_likelihood"], 34.0)

    def test_extreme_inputs(self):
        best = self.payload(survey={"symptom_score": 0}, vision={"vision_score": 100},
                            video={"video_score": 100}, force={"level": "low"})
        worst = self.payload(survey={"symptom_score": 100}, vision={"vision_score": 0},
                             video={"video_score": 0}, force={"level": "high"}, vibration={"level": "high"})
        self.assertEqual(self.post(best).json()["ms_likelihood"], 0.0)
        self.assertEqual(self.post(worst).json()["ms_likelihood"], 100.0)

    def test_missing_test_is_422(self):
        for test in combined.TESTS:
            with self.subTest(missing=test):
                body = self.payload()
                del body[test]
                self.assertEqual(self.post(body).status_code, 422)

    def test_out_of_range_values_are_422(self):
        bad_bodies = [
            self.payload(survey={"symptom_score": 101}),
            self.payload(survey={"symptom_score": -1}),
            self.payload(vision={"vision_score": 100.5}),
            self.payload(video={"video_score": -0.1}),
            self.payload(force={"level": "extreme"}),
            self.payload(vibration={"level": ""}),
        ]
        for body in bad_bodies:
            with self.subTest(body=body):
                self.assertEqual(self.post(body).status_code, 422)


if __name__ == "__main__":
    unittest.main()
