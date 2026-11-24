"""Unit tests for the symptom survey scoring and its API."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from controllers import survey  # noqa: E402
from main import app  # noqa: E402


def section_of(result: dict, key: str) -> dict:
    return next(s for s in result["sections"] if s["key"] == key)


class EvaluateTests(unittest.TestCase):
    def test_no_answers_scores_zero_and_low(self):
        result = survey.evaluate({})
        self.assertEqual(result["symptom_score"], 0.0)
        self.assertEqual(result["points"], 0)
        self.assertEqual(result["level"], "low")
        self.assertTrue(all(s["level"] == "low" for s in result["sections"]))

    def test_explicit_no_answers_score_zero(self):
        self.assertEqual(survey.evaluate({"q1": False, "q1a": False})["points"], 0)

    def test_follow_up_without_main_question_scores_zero(self):
        result = survey.evaluate({"q1a": True, "q1b": True})
        self.assertEqual(result["points"], 0)

    def test_main_question_is_worth_two_points(self):
        self.assertEqual(survey.evaluate({"q1": True})["points"], 2)

    def test_each_follow_up_adds_one_point(self):
        self.assertEqual(survey.evaluate({"q1": True, "q1a": True})["points"], 3)
        self.assertEqual(survey.evaluate({"q1": True, "q1a": True, "q1b": True})["points"], 4)

    def test_points_add_up_across_sections(self):
        result = survey.evaluate({"q1": True, "q1a": True, "q3": True})
        self.assertEqual(section_of(result, "voice")["points"], 3)
        self.assertEqual(section_of(result, "movement")["points"], 2)
        self.assertEqual(result["points"], 5)

    def test_symptom_score_is_share_of_total_max(self):
        result = survey.evaluate({"q1": True, "q1a": True})
        self.assertEqual(result["max_points"], sum(s.max_score for s in survey.SECTIONS))
        self.assertEqual(result["symptom_score"], round(100 * 3 / result["max_points"], 1))

    def test_all_yes_is_high_overall(self):
        result = survey.evaluate({id_: True for id_ in survey._ALL_IDS})
        self.assertEqual(result["level"], "high")

    def test_section_level_boundaries(self):
        # voice has max_score 12: 25% = 3 points, 50% = 6 points.
        yes = ["q1", "q1a", "q1b", "q1c", "q1d"]

        def voice_level(count: int) -> str:
            result = survey.evaluate(dict.fromkeys(yes[:count], True))
            return section_of(result, "voice")["level"]

        self.assertEqual(voice_level(1), "low")  # 2 points
        self.assertEqual(voice_level(2), "medium")  # 3 points = exactly 25%
        self.assertEqual(voice_level(4), "medium")  # 5 points
        self.assertEqual(voice_level(5), "high")  # 6 points = exactly 50%

    def test_level_for_thresholds(self):
        self.assertEqual(survey.level_for(0, 10), "low")
        self.assertEqual(survey.level_for(2, 10), "low")
        self.assertEqual(survey.level_for(3, 10), "medium")
        self.assertEqual(survey.level_for(4, 10), "medium")
        self.assertEqual(survey.level_for(5, 10), "high")

    def test_unknown_ids_raise_and_are_listed(self):
        with self.assertRaises(survey.SurveyInputError) as ctx:
            survey.evaluate({"q1": True, "nope": True, "also_bad": False})
        self.assertIn("also_bad", str(ctx.exception))
        self.assertIn("nope", str(ctx.exception))


class QuestionsTests(unittest.TestCase):
    def setUp(self):
        self.data = survey.get_questions()
        self.questions = [q for s in self.data["sections"] for q in s["questions"]]
        self.items = [i for q in self.questions for i in q["follow_ups"]]

    def test_ids_are_unique(self):
        ids = [entry["id"] for entry in self.questions + self.items]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(survey._ALL_IDS))

    def test_every_question_has_both_languages(self):
        for entry in self.questions + self.items:
            self.assertTrue(entry["text_en"].strip(), entry["id"])
            self.assertTrue(entry["text_ar"].strip(), entry["id"])

    def test_titles_and_notes_exist(self):
        for section in self.data["sections"]:
            self.assertTrue(section["title_en"] and section["title_ar"])
        self.assertTrue(self.data["note_en"] and self.data["note_ar"])


class SurveyApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_get_questions(self):
        response = self.client.get("/api/survey/questions")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["sections"]), len(survey.SECTIONS))

    def test_evaluate_ok(self):
        response = self.client.post("/api/survey/evaluate", json={"answers": {"q1": True, "q1a": True}})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["points"], 3)

    def test_evaluate_empty_answers(self):
        response = self.client.post("/api/survey/evaluate", json={"answers": {}})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["level"], "low")

    def test_unknown_id_is_400(self):
        response = self.client.post("/api/survey/evaluate", json={"answers": {"zzz": True}})
        self.assertEqual(response.status_code, 400)
        self.assertIn("zzz", response.json()["detail"])

    def test_malformed_body_is_422(self):
        self.assertEqual(self.client.post("/api/survey/evaluate", json={}).status_code, 422)
        response = self.client.post("/api/survey/evaluate", json={"answers": {"q1": "maybe"}})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
