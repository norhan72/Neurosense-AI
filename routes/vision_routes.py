"""Vision test endpoints (mounted at /api/vision). Stateless."""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from controllers import vision

router = APIRouter()


class VisionAnswer(BaseModel):
    image_id: str
    answer: str
    response_time_ms: float | None = Field(default=None, ge=0)


class EvaluateRequest(BaseModel):
    answers: list[VisionAnswer]


def _unavailable(exc: vision.VisionDatasetUnavailable) -> HTTPException:
    return HTTPException(status_code=503, detail=str(exc))


@router.get("/exercises")
def exercises(count: int = Query(vision.DEFAULT_COUNT)):
    """Return a balanced set of distorted images with shuffled options."""
    try:
        selected = vision.select_exercises(count)
        dataset = vision.get_dataset()
    except vision.VisionDatasetUnavailable as exc:
        raise _unavailable(exc)
    except vision.VisionInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "dataset_version": dataset.version,
        "count": len(selected),
        "num_options": dataset.num_options,
        "exercises": selected,
    }


@router.get("/images/{image_id}")
def image(image_id: str):
    """Serve a dataset image; the id is validated against the manifest."""
    try:
        path = vision.image_path(image_id)
    except vision.VisionDatasetUnavailable as exc:
        raise _unavailable(exc)
    except KeyError:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@router.post("/evaluate")
def evaluate(body: EvaluateRequest):
    """Score the answers (3PL IRT, EAP ability) and persist the evaluation."""
    try:
        return vision.evaluate([a.model_dump() for a in body.answers])
    except vision.VisionDatasetUnavailable as exc:
        raise _unavailable(exc)
    except vision.VisionInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
