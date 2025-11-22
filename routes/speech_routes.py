import os
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from controllers.speech import SpeechTest


router = APIRouter()
test = SpeechTest()


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data_bytes = await file.read()
    file_name = file.filename or "unknown.wav"
    if not data_bytes:
        return JSONResponse({"error": "no audio bytes provided"}, status_code=400)

    try:
        result = test.analyze(data_bytes, file_name)
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        return JSONResponse(
            content={"error": "analyze_failed", "detail": str(e)}, status_code=500
        )


@router.post("/upload_healthy")
async def upload_healthy(file: UploadFile = File(...), filename: str = None):
    try:
        contents = await file.read()
        fname = filename or file.filename or f"healthy_{np.random.randint(1e6)}.wav"
        feats = test.save_healthy_sample(contents, fname)
        return JSONResponse(
            content={"status": "ok", "saved_as": fname, "features": feats},
            status_code=201,
        )
    except Exception as e:
        return JSONResponse(
            content={"error": "save_failed", "detail": str(e)}, status_code=500
        )
