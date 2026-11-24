"""Integration tests for the video analysis test on synthetic clips + the HTTP API.

Needs the MediaPipe models (downloaded on first use) and a frontal face photo:

    set NEUROSENSE_TEST_FACE=path/to/face.jpg      (PowerShell: $env:NEUROSENSE_TEST_FACE=...)
    python -m unittest discover -s tests -v

Set NEUROSENSE_TEST_DUMP=out.json to save the /analyze response for inspection.
A public-domain NASA portrait works well, e.g. "Neil Armstrong pose.jpg" on
Wikimedia Commons. The tests are skipped when the variable is not set.
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controllers import video_analysis as va  # noqa: E402

FACE_PHOTO = os.environ.get("NEUROSENSE_TEST_FACE")
# Clips are rendered at 25 fps and analysed at ~12 fps; the fake head turn ramps
# over 0.2 s, so the threshold crossing lands up to ~0.25 s after its start.
ONSET_TOLERANCE_S = 0.3


@unittest.skipUnless(FACE_PHOTO and Path(FACE_PHOTO).is_file(),
                     "set NEUROSENSE_TEST_FACE to a frontal face photo")
class SyntheticVideoTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.video_fixtures import build_clips

        cls._tmp = tempfile.TemporaryDirectory(prefix="neurosense_video_test_")
        cls.clips = build_clips(FACE_PHOTO, Path(cls._tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()


class PipelineTests(SyntheticVideoTestCase):
    def analyze(self, clip, action, offset=0.0, mirrored=False):
        started = time.perf_counter()
        result = va.analyze_video(str(self.clips[clip]), action, offset, mirrored)
        print(f"\n  {clip}/{action}{' (mirrored)' if mirrored else ''}: "
              f"detected={result['detected']} rt={result['response_time_s']} "
              f"conf={result['confidence']} in {time.perf_counter() - started:.2f}s", end="")
        self.assertIsNone(result["error"])
        self.assertEqual(result["face_detected_ratio"], 1.0)
        return result

    def test_static_face_states(self):
        for action in ("look_straight", "open_eyes"):
            result = self.analyze("static", action)
            self.assertTrue(result["detected"])
            self.assertTrue(result["already_in_state"])
            self.assertEqual(result["response_time_s"], 0.0)
            self.assertIsNotNone(result["note"])
        for action in ("close_eyes", "look_left", "look_right", "look_up", "look_down"):
            self.assertFalse(self.analyze("static", action)["detected"], action)

    def test_horizontal_flip_flips_yaw_sign(self):
        yaw = {}
        for clip in ("static", "static_flipped"):
            series, _ = va.extract_features_with_fallback(str(self.clips[clip]), False)
            yaw[clip] = float(np.nanmedian(series.features["yaw"]))
        self.assertLess(yaw["static"] * yaw["static_flipped"], 0, yaw)

    def test_head_turn_left(self):
        result = self.analyze("turn_left", "look_left")
        self.assertTrue(result["detected"])
        self.assertAlmostEqual(result["onset_s"], 1.5, delta=ONSET_TOLERANCE_S)
        self.assertFalse(self.analyze("turn_left", "look_right")["detected"])

    def test_head_turn_right(self):
        result = self.analyze("turn_right", "look_right")
        self.assertTrue(result["detected"])
        self.assertAlmostEqual(result["onset_s"], 2.0, delta=ONSET_TOLERANCE_S)
        self.assertFalse(self.analyze("turn_right", "look_left")["detected"])

    def test_mirrored_flag_swaps_sides(self):
        self.assertTrue(self.analyze("turn_left", "look_right", mirrored=True)["detected"])
        self.assertFalse(self.analyze("turn_left", "look_left", mirrored=True)["detected"])

    def test_instruction_offset(self):
        result = self.analyze("turn_left", "look_left", offset=1.0)
        self.assertAlmostEqual(result["response_time_s"], 0.5, delta=ONSET_TOLERANCE_S)

    def test_close_eyes(self):
        result = self.analyze("close_eyes", "close_eyes")
        self.assertTrue(result["detected"])
        self.assertAlmostEqual(result["response_time_s"], 1.0, delta=ONSET_TOLERANCE_S)

    def test_webm_decodes(self):
        result = self.analyze("turn_left_webm", "look_left")
        self.assertTrue(result["detected"])
        self.assertAlmostEqual(result["onset_s"], 1.5, delta=ONSET_TOLERANCE_S)

    def test_ffmpeg_fallback(self):
        real_extract = va.extract_features
        calls = []

        def opencv_fails_first(path, use_pose):
            calls.append(path)
            if len(calls) == 1:
                raise va.VideoDecodeError("simulated OpenCV failure")
            return real_extract(path, use_pose)

        with mock.patch.object(va, "extract_features", side_effect=opencv_fails_first):
            result = va.analyze_video(str(self.clips["turn_left_webm"]), "look_left")
        self.assertTrue(result["transcoded"])
        self.assertTrue(result["detected"])
        self.assertAlmostEqual(result["onset_s"], 1.5, delta=ONSET_TOLERANCE_S)

    def test_hand_states_on_still_photo(self):
        lowered = va.analyze_video(str(self.clips["static"]), "lower_hand")
        self.assertTrue(lowered["detected"])
        self.assertIsNotNone(lowered["pose_detected_ratio"])
        self.assertFalse(va.analyze_video(str(self.clips["static"]), "raise_hand")["detected"])

    def test_garbage_file_reports_error(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(os.urandom(4096))
        try:
            result = va.analyze_video(f.name, "look_left")
        finally:
            os.unlink(f.name)
        self.assertFalse(result["detected"])
        self.assertIn("decod", result["error"])


class ApiTests(SyntheticVideoTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from routes.video_analysis_routes import router

        app = FastAPI()
        app.include_router(router, prefix="/api/video")
        cls.client = TestClient(app)

    def upload(self, *names):
        return [("videos", (self.clips[n].name, self.clips[n].read_bytes(), "video/mp4"))
                for n in names]

    def test_actions(self):
        response = self.client.get("/api/video/actions")
        self.assertEqual(response.status_code, 200)
        actions = {a["action"]: a for a in response.json()["actions"]}
        self.assertIn("look_left", actions)
        self.assertTrue(actions["close_eyes"]["instruction_ar"])

    def test_analyze(self):
        response = self.client.post(
            "/api/video/analyze",
            files=self.upload("static", "turn_left_webm", "close_eyes", "static"),
            data={
                "actions": json.dumps(["Look Straight", "look left", "close eyes", "raise a hand"]),
                "instruction_offsets": "[0, 0.5, 0, 0]",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        if dump := os.environ.get("NEUROSENSE_TEST_DUMP"):  # optional: inspect a real response
            Path(dump).write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        results = body["results"]
        self.assertEqual([r["action"] for r in results],
                         ["look_straight", "look_left", "close_eyes", "raise_hand"])
        self.assertEqual([r["detected"] for r in results], [True, True, True, False])
        self.assertAlmostEqual(results[1]["response_time_s"], 1.0, delta=ONSET_TOLERANCE_S)
        summary = body["summary"]
        self.assertEqual(summary["num_detected"], 3)
        self.assertEqual(summary["completion_rate"], 0.75)
        self.assertIsNotNone(summary["video_score"])

    def test_bad_video_does_not_fail_request(self):
        files = self.upload("static") + [("videos", ("broken.webm", os.urandom(2048), "video/webm"))]
        response = self.client.post("/api/video/analyze", files=files,
                                    data={"actions": "look straight, look left"})
        self.assertEqual(response.status_code, 200, response.text)
        results = response.json()["results"]
        self.assertIsNone(results[0]["error"])
        self.assertIsNotNone(results[1]["error"])
        self.assertEqual(response.json()["summary"]["num_analyzed"], 1)

    def test_validation_errors(self):
        cases = {
            "count mismatch": dict(files=self.upload("static"), data={"actions": "look left, look right"}),
            "unsupported": dict(files=self.upload("static"), data={"actions": '["dance"]'}),
            "no videos": dict(data={"actions": "look left"}),
            "bad offsets": dict(files=self.upload("static"),
                                data={"actions": "look left", "instruction_offsets": "[1, 2]"}),
            "negative offset": dict(files=self.upload("static"),
                                    data={"actions": "look left", "instruction_offsets": "-1"}),
            "too many": dict(files=self.upload(*["static"] * 11), data={"actions": ",".join(["look left"] * 11)}),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                response = self.client.post("/api/video/analyze", **kwargs)
                self.assertEqual(response.status_code, 400, response.text)

    def test_too_large(self):
        with mock.patch("routes.video_analysis_routes.MAX_VIDEO_BYTES", 1024):
            response = self.client.post("/api/video/analyze", files=self.upload("static"),
                                        data={"actions": "look left"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("larger than", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
