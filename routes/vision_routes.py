from fastapi import APIRouter
from pydantic import BaseModel
from controllers.vision import VisionTest
import uuid

router = APIRouter()
sessions = {}


class VisionAnswer(BaseModel):
    session_id: str
    difficulty: float
    seen: bool


@router.get("/start")
def start():
    session_id = str(uuid.uuid4())
    test = VisionTest("data/vision")
    sessions[session_id] = test
    return test.next_step()


@router.post("/answer")
def answer(data: VisionAnswer):
    test = sessions[data.session_id]
    test.record_answer(data.difficulty, data.seen)
    out = test.next_step()

    if out["status"] == "finished":
        sessions.pop(data.session_id)
    return out
