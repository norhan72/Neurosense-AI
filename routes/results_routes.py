"""Combined result endpoint (mounted at /api/results). Stateless."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from controllers import combined

router = APIRouter()

Level = Literal["low", "medium", "high"]


class SurveyResult(BaseModel):
    symptom_score: float = Field(ge=0, le=100)


class VisionResult(BaseModel):
    vision_score: float = Field(ge=0, le=100)


class VideoResult(BaseModel):
    video_score: float = Field(ge=0, le=100)


class SensorResult(BaseModel):
    level: Level


class CombinedRequest(BaseModel):
    """The relevant field of each test's own response."""

    survey: SurveyResult
    vision: VisionResult
    video: VideoResult
    force: SensorResult
    vibration: SensorResult


@router.post("/combined")
def combined_result(body: CombinedRequest):
    """MS likelihood (%) from the five test results, equally weighted."""
    risks = combined.risks_by_test(
        survey_score=body.survey.symptom_score,
        vision_score=body.vision.vision_score,
        video_score=body.video.video_score,
        force_level=body.force.level,
        vibration_level=body.vibration.level,
    )
    return combined.combine(risks)
