import os
from fastapi import FastAPI
from routes.vision_routes import router as vision_router
from routes.motion_routes import router as motion_router
from routes.speech_routes import router as speech_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory="dist/assets"), name="assets")


# Register APIs
app.include_router(vision_router, prefix="/api/vision", tags=["Vision Test"])
app.include_router(motion_router, prefix="/api/motion", tags=["Motion Test"])
app.include_router(speech_router, prefix="/api/speech", tags=["Speech Test"])


@app.get("/")
async def serve_react():
    return FileResponse(os.path.join("dist", "index.html"))


@app.get("/{path:path}")
def read_react_app(path: str):
    file_path = f"dist/{path}"

    if os.path.exists(file_path) and not os.path.isdir(file_path):
        return FileResponse(file_path)

    # otherwise return index.html for SPA
    return FileResponse("dist/index.html")


# run using "fastapi dev main.py"
