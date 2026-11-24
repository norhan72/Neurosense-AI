"""NeuroSense AI backend: MS screening APIs + the prebuilt React SPA.

Run with ``fastapi dev main.py``. The SPA build directory defaults to ``dist/``
and can be overridden with the ``NEUROSENSE_DIST_DIR`` environment variable.
"""

import glob
import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from routes.video_analysis_routes import router as video_router
from routes.vision_routes import router as vision_router

# Speech test is SUSPENDED: importing routes.speech_routes pulls in
# faster-whisper/torch, which fails to load on the deployment machine.
# Re-enable with:
#   from routes.speech_routes import router as speech_router
#   app.include_router(speech_router, prefix="/api/speech", tags=["Speech Test"])

logger = logging.getLogger("neurosense")

DIST_DIR = os.path.realpath(os.environ.get("NEUROSENSE_DIST_DIR", "dist"))
INDEX_HTML = os.path.join(DIST_DIR, "index.html")
ASSETS_DIR = os.path.join(DIST_DIR, "assets")
MOTION_DATA_DIR = os.path.join("data", "motion")

app = FastAPI(title="NeuroSense AI")

# Wildcard origins cannot be combined with credentials, so credentials stay off.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API routers -------------------------------------------------------------
app.include_router(vision_router, prefix="/api/vision", tags=["Vision Test"])
app.include_router(video_router, prefix="/api/video", tags=["Video Analysis"])

# Motion test is SUSPENDED but kept registered when its training data exists:
# MotionTest trains on data/motion/*.json at import time.
if glob.glob(os.path.join(MOTION_DATA_DIR, "*.json")):
    from routes.motion_routes import router as motion_router

    app.include_router(motion_router, prefix="/api/motion", tags=["Motion Test"])
else:
    logger.warning("No motion data in %s; /api/motion is disabled.", MOTION_DATA_DIR)


# --- SPA serving ---------------------------------------------------------------
if os.path.isdir(ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
else:
    logger.warning("Frontend build not found at %s; serving API only.", DIST_DIR)


def _index_response() -> FileResponse:
    if not os.path.isfile(INDEX_HTML):
        raise HTTPException(status_code=404, detail="frontend build not found")
    return FileResponse(INDEX_HTML)


@app.get("/", include_in_schema=False)
def serve_react():
    return _index_response()


@app.get("/{path:path}", include_in_schema=False)
def serve_spa(path: str):
    if path == "api" or path.startswith("api/"):
        raise HTTPException(status_code=404, detail="not found")

    # Resolve symlinks/.. and only serve files that really live inside the build.
    file_path = os.path.realpath(os.path.join(DIST_DIR, path))
    if file_path.startswith(DIST_DIR + os.sep) and os.path.isfile(file_path):
        return FileResponse(file_path)

    # Anything else is a client-side route: let the SPA handle it.
    return _index_response()
