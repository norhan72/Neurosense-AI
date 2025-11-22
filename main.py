from fastapi import FastAPI
from routes.vision_routes import router as vision_router
from routes.motion_routes import router as motion_router

# from routes.speech_routes import router as speech_router
# from routes.survey_routes import router as survey_router

app = FastAPI()

# Register APIs
app.include_router(vision_router, prefix="/vision", tags=["Vision Test"])
app.include_router(motion_router, prefix="/motion", tags=["Motion Test"])
# app.include_router(speech_router, prefix="/speech", tags=["Speech Test"])
# app.include_router(survey_router, prefix="/survey", tags=["Survey Test"])

# run using "fastapi dev main.py"
