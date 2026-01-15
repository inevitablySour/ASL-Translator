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
from .multi_hand_body_detector import MultiHandBodyDetector
from .gesture_classifier import GestureClassifier
from .translator import Translator
from .temporal_feature_extractor import TemporalFeatureExtractor
from .temporal_gesture_classifier import TemporalGestureClassifier
from .gesture_buffer import GestureBuffer

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

hand_detector = HandDetector()  # For static mode
multi_hand_detector = MultiHandBodyDetector()  # For temporal mode
gesture_classifier = GestureClassifier()
translator = Translator()

# Temporal gesture recognition components
from pathlib import Path
temporal_model_path = Path(settings.models_dir) / "temporal_lstm.pth"
temporal_classifier = TemporalGestureClassifier(
    model_path=str(temporal_model_path) if temporal_model_path.exists() else None
)
temporal_extractor = TemporalFeatureExtractor()
gesture_buffer = GestureBuffer()


class PredictionRequest(BaseModel):
    image: str
    language: Optional[str] = "english"


class PredictionResponse(BaseModel):
    gesture: str
    translation: str
    confidence: float
    language: str
    processing_time_ms: float


class TemporalPredictionRequest(BaseModel):
    frames: List[str]  # List of base64-encoded images
    language: Optional[str] = "english"


class TemporalPredictionResponse(BaseModel):
    gesture: str
    translation: str
    confidence: float
    language: str
    processing_time_ms: float
    gesture_type: str  # "static" or "dynamic"
    frames_processed: int


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


@app.post("/predict/temporal", response_model=TemporalPredictionResponse)
async def predict_temporal_gesture(request: TemporalPredictionRequest):
    """
    Predict gesture from a sequence of frames (temporal/dynamic gesture detection)
    Analyzes hand movement over time for dynamic ASL signs like J, Z
    """
    start_time = time.time()
    
    try:
        if len(request.frames) < settings.temporal_window_size:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least {settings.temporal_window_size} frames, got {len(request.frames)}"
            )
        
        # Process frames and extract landmark sequences
        decode_start = time.time()
        sequence_features = []
        frames_to_process = request.frames[-settings.temporal_window_size:]
        
        # Decode images (skip every other frame for speed)
        decoded_images = []
        for i, frame_data in enumerate(frames_to_process):
            if i % 2 == 0:  # Process every other frame
                image_data = base64.b64decode(
                    frame_data.split(',')[1] if ',' in frame_data else frame_data
                )
                image = Image.open(BytesIO(image_data))
                image_np = np.array(image)
                
                if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                decoded_images.append(image_np)
        
        decode_time = (time.time() - decode_start) * 1000
        
        # Process frames for landmark detection
        detection_start = time.time()
        hands_detected = {'left': 0, 'right': 0, 'both': 0}
        for image_np in decoded_images:
            # Detect hands and body using multi-hand detector
            _, detection_results = multi_hand_detector.detect_hands_and_body(image_np)
            
            # Track hand detection
            if detection_results['left_hand']:
                hands_detected['left'] += 1
            if detection_results['right_hand']:
                hands_detected['right'] += 1
            if detection_results['left_hand'] and detection_results['right_hand']:
                hands_detected['both'] += 1
            
            # Extract features (151 features: left hand + right hand + body + spatial)
            features = multi_hand_detector.extract_features(detection_results)
            sequence_features.append(features)
            
            # Duplicate feature for skipped frame to maintain sequence length
            sequence_features.append(features)
        
        detection_time = (time.time() - detection_start) * 1000
        
        # Convert to numpy array: shape (window_size, 151)
        sequence = np.array(sequence_features[:settings.temporal_window_size], dtype=np.float32)
        
        # Predict using LSTM
        prediction_start = time.time()
        gesture, confidence = temporal_classifier.predict(sequence)
        prediction_time = (time.time() - prediction_start) * 1000
        
        # Translate gesture
        translation = translator.translate(gesture, request.language)
        
        # Determine gesture type based on spatial features
        # Check if gesture uses two hands (feature index 143)
        two_hand_usage = np.mean(sequence[:, 143])  # has_both_hands spatial feature
        gesture_type = "Two-Hand" if two_hand_usage > 0.3 else "Single-Hand"
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log performance metrics
        print(f"Temporal prediction timing: decode={decode_time:.1f}ms, detection={detection_time:.1f}ms, model={prediction_time:.1f}ms, total={processing_time:.1f}ms")
        print(f"Hands detected: left={hands_detected['left']}/{len(decoded_images)}, right={hands_detected['right']}/{len(decoded_images)}, both={hands_detected['both']}/{len(decoded_images)}")
        print(f"Predicted: {gesture} (confidence={confidence:.2f}, type={gesture_type})")
        
        prediction_counter.inc()
        prediction_latency.observe(time.time() - start_time)
        
        return TemporalPredictionResponse(
            gesture=gesture,
            translation=translation,
            confidence=confidence,
            language=request.language,
            processing_time_ms=processing_time,
            gesture_type=gesture_type,
            frames_processed=len(sequence_features)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Temporal prediction error: {str(e)}")


@app.get("/metrics")
async def metrics():
    return generate_latest()


@app.on_event("startup")
async def startup_event():
    print(f"Starting {settings.app_name} v{settings.app_version}")


@app.on_event("shutdown")
async def shutdown_event():
    hand_detector.close()
    multi_hand_detector.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=settings.debug)
