from typing import Optional

from fastapi import APIRouter, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from controllers.speech import SpeechTest


router = APIRouter()
test = SpeechTest()


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data_bytes = await file.read()
    if not data_bytes:
        return JSONResponse({"error": "no audio bytes provided"}, status_code=400)

    try:
        # Feature extraction is CPU-heavy; keep it off the event loop.
        result = await run_in_threadpool(
            test.analyze, data_bytes, file.filename or "unknown.wav"
        )
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        return JSONResponse(
            content={"error": "analyze_failed", "detail": str(e)}, status_code=500
        )


@router.post("/upload_healthy")
async def upload_healthy(file: UploadFile = File(...), filename: Optional[str] = None):
    contents = await file.read()
    if not contents:
        return JSONResponse({"error": "no audio bytes provided"}, status_code=400)

    try:
        saved_as, feats = await run_in_threadpool(
            test.save_healthy_sample, contents, filename or file.filename
        )
        return JSONResponse(
            content={"status": "ok", "saved_as": saved_as, "features": feats},
            status_code=201,
        )
    except Exception as e:
        return JSONResponse(
            content={"error": "save_failed", "detail": str(e)}, status_code=500
        )
