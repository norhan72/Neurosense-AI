"""Hardware sensor tests: force (pressure) and vibration.

A measurement asks a ``SensorReader`` for raw samples, reduces them to one value and
puts that value in an impairment level:

* ``low``    - typical result,
* ``medium`` - borderline,
* ``high``   - impaired.

The reader is chosen with the ``NEUROSENSE_SENSOR_MODE`` environment variable:

* ``simulated`` (default): random but plausible samples, so the whole app can be run
  and tested without hardware. Results are flagged ``simulated: true``.
* ``raspberry_pi``: the real sensors. ``RaspberryPiReader.read`` is the single place to
  implement once the Raspberry Pi code exists.

TODO: the units and thresholds below are placeholders. Calibrate them (and the
summary statistic) with the real sensors before relying on the levels.
"""

import os
import random
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

LEVELS = ("low", "medium", "high")
LEVEL_LABELS = {
    "low": ("Typical", "طبيعي"),
    "medium": ("Borderline", "حدّي"),
    "high": ("Impaired", "ضعف ملحوظ"),
}
NOTE_EN = "Sensor screening result, not a diagnosis. Thresholds are provisional until calibrated."
NOTE_AR = "نتيجة فحص بالمستشعر وليست تشخيصاً. الحدود مؤقتة إلى حين معايرتها."
SIMULATED_NOTE_EN = "Simulated reading: no sensor hardware is connected yet."
SIMULATED_NOTE_AR = "قراءة محاكاة: لم يتم توصيل المستشعر بعد."

SENSOR_MODE = os.environ.get("NEUROSENSE_SENSOR_MODE", "simulated").strip().lower()
SIMULATED_DELAY_S = 2.0  # pretend the measurement takes a moment, like the real one will


class SensorUnavailable(RuntimeError):
    """Raised when the sensor cannot be read (maps to HTTP 503)."""


class SensorBusy(RuntimeError):
    """Raised when a measurement on the same sensor is already running (maps to HTTP 409)."""


@dataclass(frozen=True)
class SensorSpec:
    key: str
    label_en: str
    label_ar: str
    unit: str
    duration_s: float
    sample_rate_hz: float
    summarize: Callable[[list[float]], float]
    higher_is_worse: bool
    medium_at: float  # value where the level becomes "medium"
    high_at: float  # value where the level becomes "high"
    simulated_range: tuple[float, float]  # spread of simulated values, covering all levels

    def level(self, value: float) -> str:
        if self.higher_is_worse:
            return "high" if value >= self.high_at else "medium" if value >= self.medium_at else "low"
        return "high" if value <= self.high_at else "medium" if value <= self.medium_at else "low"


SENSORS: dict[str, SensorSpec] = {
    # Peak grip/pressure force; weaker is worse.
    "force": SensorSpec(
        key="force",
        label_en="Grip force",
        label_ar="قوة القبضة",
        unit="kg",
        duration_s=5.0,
        sample_rate_hz=20.0,
        summarize=max,
        higher_is_worse=False,
        medium_at=20.0,
        high_at=10.0,
        simulated_range=(5.0, 40.0),
    ),
    # Vibration perception threshold; needing a stronger vibration to feel it is worse.
    "vibration": SensorSpec(
        key="vibration",
        label_en="Vibration sense",
        label_ar="الإحساس بالاهتزاز",
        unit="V",
        duration_s=5.0,
        sample_rate_hz=20.0,
        summarize=statistics.median,
        higher_is_worse=True,
        medium_at=15.0,
        high_at=25.0,
        simulated_range=(5.0, 35.0),
    ),
}


class SensorReader(Protocol):
    simulated: bool

    def read(self, spec: SensorSpec) -> list[float]:
        """Collect the raw samples of one measurement."""
        ...


class SimulatedReader:
    simulated = True

    def read(self, spec: SensorSpec) -> list[float]:
        time.sleep(SIMULATED_DELAY_S)
        true_value = random.uniform(*spec.simulated_range)
        n = int(spec.duration_s * spec.sample_rate_hz)
        noise = 0.05 * (spec.simulated_range[1] - spec.simulated_range[0])
        return [max(0.0, random.gauss(true_value, noise)) for _ in range(n)]


class RaspberryPiReader:
    simulated = False

    def read(self, spec: SensorSpec) -> list[float]:
        # TODO: talk to the Raspberry Pi here (e.g. over HTTP or serial), collect
        # `spec.duration_s` seconds of samples from sensor `spec.key` and return them.
        raise SensorUnavailable(f"Raspberry Pi reader for '{spec.key}' is not implemented yet")


def _make_reader() -> SensorReader:
    if SENSOR_MODE == "raspberry_pi":
        return RaspberryPiReader()
    if SENSOR_MODE != "simulated":
        raise ValueError(f"NEUROSENSE_SENSOR_MODE must be 'simulated' or 'raspberry_pi', got {SENSOR_MODE!r}")
    return SimulatedReader()


_reader = _make_reader()
_locks = {key: threading.Lock() for key in SENSORS}


def list_sensors() -> list[dict]:
    return [
        {"sensor": s.key, "label_en": s.label_en, "label_ar": s.label_ar, "unit": s.unit, "duration_s": s.duration_s}
        for s in SENSORS.values()
    ]


def measure(sensor: str, reader: SensorReader | None = None) -> dict:
    """Run one measurement. Raises KeyError for an unknown sensor."""
    spec = SENSORS[sensor]
    reader = reader or _reader
    lock = _locks[sensor]
    if not lock.acquire(blocking=False):
        raise SensorBusy(f"a {spec.label_en.lower()} measurement is already running")
    try:
        try:
            samples = reader.read(spec)
        except SensorUnavailable:
            raise
        except Exception as exc:  # hardware/IO errors surface as "unavailable"
            raise SensorUnavailable(f"could not read the {spec.key} sensor: {exc}") from exc
    finally:
        lock.release()
    if not samples:
        raise SensorUnavailable(f"the {spec.key} sensor returned no readings")

    value = float(spec.summarize(samples))
    level = spec.level(value)
    level_en, level_ar = LEVEL_LABELS[level]
    return {
        "sensor": spec.key,
        "label_en": spec.label_en,
        "label_ar": spec.label_ar,
        "level": level,
        "level_label_en": level_en,
        "level_label_ar": level_ar,
        "value": round(value, 2),
        "unit": spec.unit,
        "num_readings": len(samples),
        "duration_s": spec.duration_s,
        "simulated": reader.simulated,
        "note_en": " ".join([NOTE_EN, SIMULATED_NOTE_EN]) if reader.simulated else NOTE_EN,
        "note_ar": " ".join([NOTE_AR, SIMULATED_NOTE_AR]) if reader.simulated else NOTE_AR,
    }
