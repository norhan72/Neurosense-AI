"""Symptom survey endpoints (mounted at /api/survey). Stateless."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from controllers import survey

router = APIRouter()


class EvaluateRequest(BaseModel):
    # Question id -> answered "yes". Ids not sent count as "no".
    answers: dict[str, bool]


@router.get("/questions")
def questions():
    """The survey's sections and questions with bilingual texts."""
    return survey.get_questions()


@router.post("/evaluate")
def evaluate(body: EvaluateRequest):
    """Score the answers: overall symptom score (0-100) and a level per section."""
    try:
        return survey.evaluate(body.answers)
    except survey.SurveyInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
