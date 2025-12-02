"""FastAPI application for ASL gesture recognition"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import cv2
import numpy as np
import time
import base64
from io import BytesIO
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest

from .config import settings
from .hand_detector import HandDetector
from .gesture_classifier import GestureClassifier
from .translator import Translator

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI application for ASL hand gesture recognition and translation"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

prediction_counter = Counter('asl_predictions_total', 'Total predictions')
prediction_latency = Histogram('asl_prediction_latency_seconds', 'Prediction latency')

hand_detector = HandDetector()
gesture_classifier = GestureClassifier()
translator = Translator()


class PredictionRequest(BaseModel):
    image: str
    language: Optional[str] = "english"


class PredictionResponse(BaseModel):
    gesture: str
    translation: str
    confidence: float
    language: str
    processing_time_ms: float


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as f:
        return f.read()


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.app_version,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


@app.get("/info")
async def model_info():
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "supported_languages": translator.get_supported_languages(),
        "model_confidence_threshold": settings.classifier_confidence_threshold
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_gesture(request: PredictionRequest):
    start_time = time.time()
    
    try:
        image_data = base64.b64decode(request.image.split(',')[1] if ',' in request.image else request.image)
        image = Image.open(BytesIO(image_data))
        image_np = np.array(image)
        
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        annotated_image, hand_landmarks_list = hand_detector.detect_hands(image_np)
        
        if not hand_landmarks_list:
            raise HTTPException(status_code=400, detail="No hand detected in image")
        
        features = hand_detector.extract_features(hand_landmarks_list[0])
        # Note: Don't normalize here - the trained model's scaler handles normalization
        
        # Pass both features and image to classifier (MediaPipe recognizer uses image)
        gesture, confidence = gesture_classifier.predict(features, image=image_np)
        translation = translator.translate(gesture, request.language)
        
        processing_time = (time.time() - start_time) * 1000
        
        prediction_counter.inc()
        prediction_latency.observe(time.time() - start_time)
        
        return PredictionResponse(
            gesture=gesture,
            translation=translation,
            confidence=confidence,
            language=request.language,
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/metrics")
async def metrics():
    return generate_latest()


@app.on_event("startup")
async def startup_event():
    print(f"Starting {settings.app_name} v{settings.app_version}")


@app.on_event("shutdown")
async def shutdown_event():
    hand_detector.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=settings.debug)
