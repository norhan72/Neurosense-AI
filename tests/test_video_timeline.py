"""Unit tests for the pure timeline / scoring logic of the video analysis test.

Run from the project root:  python -m unittest discover -s tests -v
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controllers.video_analysis import (  # noqa: E402
    FEATURE_NAMES,
    FrameSeries,
    OnsetResult,
    action_score_series,
    detect_action_onset,
    detection_confidence,
    normalize_action,
    response_time_score,
    summarize_results,
)

FPS = 12.0


def timeline(duration_s: float = 5.0, fps: float = FPS) -> np.ndarray:
    return np.arange(0.0, duration_s, 1.0 / fps)


def step(times: np.ndarray, on_s: float, off_s: float = math.inf, level: float = 1.0) -> np.ndarray:
    return np.where((times >= on_s) & (times < off_s), level, 0.0)


class DetectOnsetTests(unittest.TestCase):
    def test_step_gives_response_time(self):
        t = timeline(fps=30.0)
        result = detect_action_onset(t, step(t, 1.2))
        self.assertTrue(result.detected)
        self.assertAlmostEqual(result.response_time_s, 1.2, delta=1 / 30)
        self.assertFalse(result.already_in_state)
        self.assertGreater(result.hold_duration_s, 3.5)

    def test_step_at_low_fps_is_interpolated(self):
        t = timeline()
        result = detect_action_onset(t, step(t, 1.2))
        self.assertAlmostEqual(result.response_time_s, 1.2, delta=1 / FPS)

    def test_short_blip_is_not_detected(self):
        t = timeline()
        result = detect_action_onset(t, step(t, 2.0, 2.25))  # a blink-length blip
        self.assertFalse(result.detected)
        self.assertIsNone(result.response_time_s)

    def test_single_frame_spike_is_removed(self):
        t = timeline()
        scores = np.zeros_like(t)
        scores[20] = 1.0
        self.assertEqual(detect_action_onset(t, scores).peak_score, 0.0)

    def test_already_in_state(self):
        t = timeline()
        result = detect_action_onset(t, np.ones_like(t))
        self.assertTrue(result.detected)
        self.assertTrue(result.already_in_state)
        self.assertEqual(result.response_time_s, 0.0)

    def test_instruction_offset_is_subtracted(self):
        t = timeline(6.0)
        result = detect_action_onset(t, step(t, 3.0), instruction_offset_s=2.0)
        self.assertAlmostEqual(result.onset_s, 3.0, delta=1 / FPS)
        self.assertAlmostEqual(result.response_time_s, 1.0, delta=1 / FPS)

    def test_hold_before_instruction_is_ignored(self):
        t = timeline(6.0)
        scores = step(t, 0.5, 1.5) + step(t, 3.5)
        result = detect_action_onset(t, scores, instruction_offset_s=2.0)
        self.assertAlmostEqual(result.response_time_s, 1.5, delta=1 / FPS)
        self.assertFalse(result.already_in_state)

    def test_hold_spanning_instruction_is_clamped_to_zero(self):
        t = timeline(6.0)
        result = detect_action_onset(t, step(t, 1.0), instruction_offset_s=2.0)
        self.assertTrue(result.already_in_state)
        self.assertEqual(result.response_time_s, 0.0)

    def test_short_dip_inside_hold_is_bridged(self):
        t = timeline()
        scores = step(t, 1.0)
        scores[(t >= 2.0) & (t < 2.2)] = 0.0  # e.g. a blink while looking left
        result = detect_action_onset(t, scores)
        self.assertGreater(result.hold_duration_s, 3.5)

    def test_short_detector_dropout_is_bridged_long_one_is_not(self):
        t = timeline()
        scores = step(t, 1.0)
        scores[(t >= 2.0) & (t < 2.2)] = np.nan
        self.assertGreater(detect_action_onset(t, scores).hold_duration_s, 3.5)

        scores = step(t, 1.0, 1.3)
        scores[(t >= 1.3) & (t < 5.0)] = np.nan  # face lost for good
        self.assertFalse(detect_action_onset(t, scores).detected)

    def test_hysteresis_keeps_hold_between_release_and_threshold(self):
        t = timeline()
        scores = step(t, 1.0, level=0.8)
        scores[t >= 1.5] = 0.4  # below threshold, above release
        result = detect_action_onset(t, scores)
        self.assertTrue(result.detected)
        self.assertGreater(result.hold_duration_s, 3.0)

    def test_hold_cut_by_video_end_needs_half_the_minimum(self):
        t = timeline(3.0)
        self.assertTrue(detect_action_onset(t, step(t, 2.7)).detected)
        self.assertFalse(detect_action_onset(t, step(t, 2.9)).detected)

    def test_empty_input(self):
        self.assertFalse(detect_action_onset(np.array([]), np.array([])).detected)


class ScoringTests(unittest.TestCase):
    def test_response_time_score(self):
        self.assertEqual(response_time_score(0.8), 1.0)
        self.assertEqual(response_time_score(1.5), 1.0)
        self.assertAlmostEqual(response_time_score(3.25), 0.5)
        self.assertEqual(response_time_score(5.0), 0.0)
        self.assertEqual(response_time_score(9.0), 0.0)

    def test_summary(self):
        results = [
            {"detected": True, "response_time_s": 1.0, "already_in_state": False},
            {"detected": True, "response_time_s": 3.25, "already_in_state": False},
            {"detected": True, "response_time_s": 0.0, "already_in_state": True},
            {"detected": False, "response_time_s": None},
            {"detected": False, "error": "video could not be decoded"},
        ]
        summary = summarize_results(results)
        self.assertEqual(summary["num_videos"], 5)
        self.assertEqual(summary["num_analyzed"], 4)
        self.assertEqual(summary["num_detected"], 3)
        self.assertEqual(summary["completion_rate"], 0.75)
        self.assertEqual(summary["mean_response_time_s"], 2.125)  # already-in-state excluded
        self.assertEqual(summary["video_score"], round(100 * (0.6 * 0.75 + 0.4 * 0.75), 1))
        self.assertEqual(summary["label_en"], "Mildly slowed responses")

    def test_summary_without_analysable_videos(self):
        summary = summarize_results([{"detected": False, "error": "x"}])
        self.assertIsNone(summary["video_score"])

    def test_confidence(self):
        strong = OnsetResult(detected=True, hold_mean_score=1.0)
        weak = OnsetResult(detected=True, hold_mean_score=0.55)
        self.assertEqual(detection_confidence(strong, coverage=1.0), 1.0)
        self.assertLess(detection_confidence(weak, coverage=1.0), 0.65)
        self.assertAlmostEqual(detection_confidence(strong, coverage=0.4), 0.5)
        clear_miss = OnsetResult(detected=False, peak_score=0.0)
        self.assertEqual(detection_confidence(clear_miss, coverage=1.0), 1.0)


class ActionTests(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(normalize_action("Look Right"), "look_right")
        self.assertEqual(normalize_action("  close   eyes "), "close_eyes")
        self.assertEqual(normalize_action("raise a hand"), "raise_hand")
        self.assertEqual(normalize_action("put hand down"), "lower_hand")
        self.assertEqual(normalize_action("look-straight"), "look_straight")
        self.assertIsNone(normalize_action("jump"))

    def _series(self, yaw: np.ndarray) -> FrameSeries:
        n = yaw.size
        features = {name: np.zeros(n) for name in FEATURE_NAMES}
        features["yaw"] = yaw
        features["raise_l"] = np.ones(n)
        return FrameSeries(times_s=timeline()[:n], features=features,
                           face_found=np.ones(n, bool), pose_found=np.ones(n, bool))

    def test_mirrored_flag_swaps_left_and_right(self):
        t = timeline()
        series = self._series(step(t, 1.0, level=25.0))  # head toward image right
        self.assertEqual(action_score_series("look_left", series).max(), 1.0)
        self.assertEqual(action_score_series("look_right", series).max(), 0.0)
        self.assertEqual(action_score_series("look_right", series, mirrored=True).max(), 1.0)
        self.assertEqual(action_score_series("raise_left_hand", series).max(), 1.0)
        self.assertEqual(action_score_series("raise_right_hand", series, mirrored=True).max(), 1.0)

    def test_baseline_removes_resting_head_turn(self):
        t = timeline()
        series = self._series(np.full(t.size, 8.0))  # turned 8 deg at rest, never moves
        self.assertEqual(action_score_series("look_left", series).max(), 0.0)


if __name__ == "__main__":
    unittest.main()
