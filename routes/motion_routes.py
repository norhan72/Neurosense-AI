from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse
from controllers.motion import MotionTest


router = APIRouter()
test = MotionTest("data/motion")


@router.post("/analyze")
async def answer(request: Request):
    body = await request.json()
    # accept either { samples: [...] } or direct list
    samples = body.get("samples") if isinstance(body, dict) else body
    if samples is None:
        return JSONResponse({"error": "no samples provided"}, status_code=400)

    try:
        result = test.analyze_samples(samples)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
