from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from controllers.motion import MotionTest


router = APIRouter()
test = MotionTest()


@router.post("/analyze")
async def answer(request: Request):
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    # accept either { samples: [...] } or direct list
    samples = body.get("samples") if isinstance(body, dict) else body
    if not samples:
        return JSONResponse({"error": "no samples provided"}, status_code=400)

    try:
        return JSONResponse(test.analyze_samples(samples))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
