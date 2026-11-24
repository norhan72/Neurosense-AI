"""Hardware sensor test endpoints (mounted at /api/sensors)."""

from typing import Literal

from fastapi import APIRouter, HTTPException

from controllers import sensors

router = APIRouter()


@router.get("")
def list_sensors():
    return {"mode": sensors.SENSOR_MODE, "sensors": sensors.list_sensors()}


@router.post("/{sensor}/measure")
def measure(sensor: Literal["force", "vibration"]):
    """Take one measurement and return its impairment level (low / medium / high).

    Blocks until the measurement is done (a few seconds), so it runs in FastAPI's
    threadpool.
    """
    try:
        return sensors.measure(sensor)
    except sensors.SensorBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except sensors.SensorUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
