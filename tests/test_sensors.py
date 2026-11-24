"""Unit tests for the force / vibration sensor measurements and their API."""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from controllers import sensors  # noqa: E402
from main import app  # noqa: E402


class FakeReader:
    """Returns fixed samples (or raises) instead of touching hardware."""

    def __init__(self, samples=None, error=None, simulated=False):
        self.samples = samples
        self.error = error
        self.simulated = simulated

    def read(self, spec):
        if self.error:
            raise self.error
        return list(self.samples)


def measure_value(sensor: str, samples: list[float]) -> dict:
    return sensors.measure(sensor, reader=FakeReader(samples))


class LevelTests(unittest.TestCase):
    def test_force_lower_is_worse(self):
        # high_at = 10, medium_at = 20 (inclusive boundaries)
        cases = [(5, "high"), (10, "high"), (10.1, "medium"), (20, "medium"), (20.1, "low"), (35, "low")]
        for value, level in cases:
            with self.subTest(value=value):
                self.assertEqual(measure_value("force", [value])["level"], level)

    def test_vibration_higher_is_worse(self):
        # medium_at = 15, high_at = 25 (inclusive boundaries)
        cases = [(5, "low"), (14.9, "low"), (15, "medium"), (24.9, "medium"), (25, "high"), (30, "high")]
        for value, level in cases:
            with self.subTest(value=value):
                self.assertEqual(measure_value("vibration", [value] * 3)["level"], level)


class SummaryTests(unittest.TestCase):
    def test_force_uses_peak(self):
        result = measure_value("force", [1.0, 30.0, 2.0])
        self.assertEqual(result["value"], 30.0)
        self.assertEqual(result["level"], "low")

    def test_vibration_uses_median_ignoring_outlier(self):
        result = measure_value("vibration", [10.0, 12.0, 100.0])
        self.assertEqual(result["value"], 12.0)
        self.assertEqual(result["level"], "low")

    def test_result_shape(self):
        result = measure_value("force", [12.345, 5.0])
        self.assertEqual(result["sensor"], "force")
        self.assertEqual(result["unit"], "kg")
        self.assertEqual(result["num_readings"], 2)
        self.assertEqual(result["value"], 12.35)
        self.assertEqual(result["level_label_en"], "Borderline")


class SimulatedFlagTests(unittest.TestCase):
    def test_simulated_reader_sets_flag_and_note(self):
        result = sensors.measure("force", reader=FakeReader([25.0], simulated=True))
        self.assertTrue(result["simulated"])
        self.assertIn(sensors.SIMULATED_NOTE_EN, result["note_en"])
        self.assertIn(sensors.SIMULATED_NOTE_AR, result["note_ar"])

    def test_real_reader_has_no_simulated_note(self):
        result = sensors.measure("force", reader=FakeReader([25.0], simulated=False))
        self.assertFalse(result["simulated"])
        self.assertEqual(result["note_en"], sensors.NOTE_EN)
        self.assertEqual(result["note_ar"], sensors.NOTE_AR)

    def test_simulated_reader_samples_are_plausible(self):
        with mock.patch.object(sensors, "SIMULATED_DELAY_S", 0):
            for spec in sensors.SENSORS.values():
                samples = sensors.SimulatedReader().read(spec)
                self.assertEqual(len(samples), int(spec.duration_s * spec.sample_rate_hz))
                self.assertTrue(all(s >= 0 for s in samples))

    def test_raspberry_pi_reader_is_unavailable(self):
        with self.assertRaises(sensors.SensorUnavailable):
            sensors.RaspberryPiReader().read(sensors.SENSORS["force"])


class ErrorTests(unittest.TestCase):
    def test_reader_exception_becomes_sensor_unavailable(self):
        with self.assertRaises(sensors.SensorUnavailable) as ctx:
            sensors.measure("force", reader=FakeReader(error=OSError("port closed")))
        self.assertIn("port closed", str(ctx.exception))

    def test_sensor_unavailable_passes_through(self):
        error = sensors.SensorUnavailable("not wired")
        with self.assertRaises(sensors.SensorUnavailable) as ctx:
            sensors.measure("force", reader=FakeReader(error=error))
        self.assertIs(ctx.exception, error)

    def test_empty_samples_are_unavailable(self):
        with self.assertRaises(sensors.SensorUnavailable):
            sensors.measure("vibration", reader=FakeReader([]))

    def test_busy_lock_raises_busy(self):
        with sensors._locks["force"]:
            with self.assertRaises(sensors.SensorBusy):
                sensors.measure("force", reader=FakeReader([10.0]))

    def test_lock_is_released_after_failure(self):
        with self.assertRaises(sensors.SensorUnavailable):
            sensors.measure("force", reader=FakeReader(error=RuntimeError("boom")))
        self.assertEqual(measure_value("force", [25.0])["level"], "low")

    def test_busy_is_per_sensor(self):
        with sensors._locks["force"]:
            self.assertEqual(measure_value("vibration", [5.0])["level"], "low")

    def test_unknown_sensor_raises_key_error(self):
        with self.assertRaises(KeyError):
            sensors.measure("balance", reader=FakeReader([1.0]))


class SensorApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def use_reader(self, reader):
        patcher = mock.patch.object(sensors, "_reader", reader)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_list_sensors(self):
        response = self.client.get("/api/sensors")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual({s["sensor"] for s in body["sensors"]}, {"force", "vibration"})
        self.assertIn("mode", body)

    def test_measure_ok(self):
        self.use_reader(FakeReader([30.0, 25.0]))
        response = self.client.post("/api/sensors/force/measure")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["value"], 30.0)
        self.assertEqual(response.json()["level"], "low")

    def test_measure_with_simulated_reader(self):
        self.use_reader(sensors.SimulatedReader())
        with mock.patch.object(sensors, "SIMULATED_DELAY_S", 0):
            for sensor in ("force", "vibration"):
                response = self.client.post(f"/api/sensors/{sensor}/measure")
                self.assertEqual(response.status_code, 200)
                body = response.json()
                self.assertTrue(body["simulated"])
                self.assertIn(body["level"], sensors.LEVELS)

    def test_reader_error_is_503(self):
        self.use_reader(FakeReader(error=OSError("no device")))
        response = self.client.post("/api/sensors/vibration/measure")
        self.assertEqual(response.status_code, 503)
        self.assertIn("no device", response.json()["detail"])

    def test_empty_samples_is_503(self):
        self.use_reader(FakeReader([]))
        self.assertEqual(self.client.post("/api/sensors/force/measure").status_code, 503)

    def test_busy_is_409(self):
        self.use_reader(FakeReader([10.0]))
        with sensors._locks["force"]:
            response = self.client.post("/api/sensors/force/measure")
        self.assertEqual(response.status_code, 409)

    def test_unknown_sensor_is_422(self):
        self.assertEqual(self.client.post("/api/sensors/balance/measure").status_code, 422)


if __name__ == "__main__":
    unittest.main()
