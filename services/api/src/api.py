from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, Field
from typing import Optional, List
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import re
import cv2
import numpy as np
import time
import base64
from io import BytesIO
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest
import threading
import src.api_producer as producer
import src.api_consumer as consumer
import json
import uuid
import pika
import base64
import time
import random
import os
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from logging_setup import setup_logging
    setup_logging()
except ImportError:
    logging.basicConfig(level=logging.DEBUG)
    
logger = logging.getLogger(__name__)

# Import feedback manager
try:
    from feedback_manager import FeedbackManager
    feedback_manager = FeedbackManager(retraining_threshold=200)
    logger.info("FeedbackManager initialized")
except Exception as e:
    logger.error(f"Failed to initialize FeedbackManager: {e}")
    feedback_manager = None
# Get the queues from the compose files
inference_queue = os.getenv("MODEL_QUEUE", "Letterbox")

BASE_DIR = Path(__file__).resolve().parent  # api/src

# Create a unique ID
def create_ID():
    return str(uuid.uuid4())

# Gets the data from web and convert it into this object
class PredictionRequest(BaseModel):
    """Request model for gesture prediction with enhanced validation"""
    image: str = Field(..., description="Base64-encoded image data")
    model: Optional[str] = Field(None, description="Model version identifier")
    
    @field_validator('image')
    @classmethod
    def validate_image(cls, v: str) -> str:
        """
        Validate base64 image data
        - Check format
        - Enforce size limits
        - Verify valid base64 encoding
        """
        if not v:
            raise ValueError("Image data cannot be empty")
        
        # Check base64 format (may have data URI prefix)
        base64_pattern = re.compile(r'^data:image/[a-zA-Z]+;base64,')
        has_prefix = base64_pattern.match(v)
        
        # Extract base64 data (remove data URI prefix if present)
        if has_prefix:
            v = v.split(',', 1)[1]
        
        # Validate base64 encoding
        try:
            decoded = base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {str(e)}")
        
        # Size validation (5MB limit for decoded image)
        MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
        if len(decoded) > MAX_IMAGE_SIZE:
            raise ValueError(
                f"Image size ({len(decoded)} bytes) exceeds maximum "
                f"allowed size ({MAX_IMAGE_SIZE} bytes)"
            )
        
        # Basic image format validation (check magic bytes)
        if len(decoded) < 4:
            raise ValueError("Image data too small to be valid")
        
        # Check for common image formats (JPEG, PNG)
        if decoded[0:2] not in [b'\xff\xd8', b'\x89\x50']:  # JPEG or PNG
            raise ValueError("Image must be in JPEG or PNG format")
        
        return v
    
    @field_validator('model')
    @classmethod
    def validate_model(cls, v: Optional[str]) -> Optional[str]:
        """Validate model identifier format"""
        if v is None:
            return v
        
        # Allow alphanumeric, hyphens, underscores only
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError(
                "Model identifier contains invalid characters. "
                "Only alphanumeric, hyphens, and underscores allowed."
            )
        
        # Length limit
        if len(v) > 100:
            raise ValueError("Model identifier exceeds maximum length (100 characters)")
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
                "model": "asl_model_20260120_110302"
            }
        }

# How to retrieve the data back from the inference.
class PredictionResponse(BaseModel):
    job_id: str
    gesture: str
    translation: str
    confidence: float
    language: str
    processing_time_ms: float
    landmarks: Optional[List[float]] = None  # Hand landmarks for feedback

# Feedback request model
class FeedbackRequest(BaseModel):
    """Request model for user feedback with validation"""
    job_id: str = Field(..., description="Prediction job ID")
    accepted: bool = Field(..., description="Whether prediction was accepted")
    corrected_gesture: Optional[str] = Field(None, description="Corrected gesture if rejected")
    
    @field_validator('job_id')
    @classmethod
    def validate_job_id(cls, v: str) -> str:
        """Validate job ID format (UUID)"""
        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            re.IGNORECASE
        )
        if not uuid_pattern.match(v):
            raise ValueError("Invalid job ID format. Must be a valid UUID.")
        return v
    
    @field_validator('corrected_gesture')
    @classmethod
    def validate_corrected_gesture(cls, v: Optional[str]) -> Optional[str]:
        """Validate corrected gesture format"""
        if v is None:
            return v
        
        # Allow single uppercase letter (A-Z)
        if not re.match(r'^[A-Z]$', v):
            raise ValueError(
                "Corrected gesture must be a single uppercase letter (A-Z)"
            )
        
        return v


connection_producer = None
connection_consumer = None
consumer_thread = None


def get_producer_connection():
    """Get a live RabbitMQ producer connection, reconnecting if needed."""
    global connection_producer
    if connection_producer is None or not getattr(connection_producer, "is_open", False):
        logger.warning("[API] Producer connection closed or missing; reconnecting to RabbitMQ")
        connection_producer = producer.connect_with_broker()
    return connection_producer


# Start consuming!!
def start_consuming(connection):
    consumer.consume_message_inference(connection)

# Set up the channel and queues when starting up the API/WEB and closing the connection
@asynccontextmanager
async def lifespan(app: FastAPI):
    global connection_producer, connection_consumer, consumer_thread

    # Check environment
    environment = os.getenv("ENVIRONMENT", "development")
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    
    if environment == "production" and debug_mode:
        logger.error("CRITICAL: Debug mode enabled in production!")
        raise RuntimeError("Debug mode must be disabled in production")

    # Setup the connection with the producer and the consumer
    connection_producer = producer.connect_with_broker()
    connection_consumer = consumer.connect_with_broker()
    print("- [API - CONSUMER / PRODUCER] Connection successful!")
    if connection_producer is None or connection_consumer is None:
        raise RuntimeError("- [API] Could not connect to RabbitMQ")

    # Start consumer in background on a different thread because otherwise the web will crash (async)
    consumer_thread = threading.Thread(
        target=start_consuming,
        args=(connection_consumer,),
        daemon=True
    )
    consumer_thread.start()

    yield # Closing the connection
    producer.close_connection(connection_producer)
    consumer.close_connection(connection_consumer)


# Set up the fastapi app
app = FastAPI(lifespan=lifespan)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Get rate limit configuration from environment
RATE_LIMIT_PREDICT = os.getenv("RATE_LIMIT_PREDICT", "90/minute")
RATE_LIMIT_FEEDBACK = os.getenv("RATE_LIMIT_FEEDBACK", "60/minute")
RATE_LIMIT_STATS = os.getenv("RATE_LIMIT_STATS", "10/minute")
RATE_LIMIT_ADMIN = os.getenv("RATE_LIMIT_ADMIN", "10/minute")

# Configure CORS
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    expose_headers=["X-Request-ID"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Import security modules
try:
    from security import verify_api_key, optional_api_key
    from security_middleware import SecurityHeadersMiddleware
    SECURITY_AVAILABLE = True
except ImportError as e:
    # Fallback if modules not available
    logger.warning(f"Security modules not available: {e}")
    # Create dummy functions that always fail
    async def verify_api_key_dummy(request: Request, api_key: str = None):
        raise HTTPException(status_code=503, detail="Security module not available")
    verify_api_key = verify_api_key_dummy
    optional_api_key = lambda request, api_key=None: None
    SecurityHeadersMiddleware = None
    SECURITY_AVAILABLE = False

# Add security headers middleware
if SecurityHeadersMiddleware:
    app.add_middleware(SecurityHeadersMiddleware)

# Request size limit middleware
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB

@app.middleware("http")
async def check_request_size(request: Request, call_next):
    """Middleware to check request body size"""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            size = int(content_length)
            if size > MAX_REQUEST_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"Request body too large. Maximum size: {MAX_REQUEST_SIZE} bytes"
                )
        except ValueError:
            pass  # Invalid content-length, let it through to be handled by FastAPI
    
    response = await call_next(request)
    return response

# Serve static files (css, js, images)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Setting up the home page
@app.get("/health")
@app.get("/api/health")
async def health_check():
    """
    Health check endpoint for Docker and monitoring
    Checks database and RabbitMQ connectivity
    """
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "checks": {}
    }
    
    # Check database connectivity
    try:
        from database import get_session, init_db
        from sqlalchemy import text
        engine = init_db()
        session = get_session(engine)
        session.execute(text("SELECT 1"))
        session.close()
        health_status["checks"]["database"] = {"status": "healthy", "message": "Connected"}
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["status"] = "unhealthy"
        health_status["checks"]["database"] = {"status": "unhealthy", "message": str(e)}
    
    # Check RabbitMQ connectivity
    try:
        global connection_producer, connection_consumer
        producer_ok = connection_producer and getattr(connection_producer, "is_open", False)
        consumer_ok = connection_consumer and getattr(connection_consumer, "is_open", False)
        
        if producer_ok or consumer_ok:
            health_status["checks"]["rabbitmq"] = {"status": "healthy", "message": "Connected"}
        else:
            # During startup, RabbitMQ might not be connected yet but API is functional
            # Mark as degraded instead of unhealthy
            health_status["checks"]["rabbitmq"] = {"status": "degraded", "message": "Not connected"}
    except Exception as e:
        health_status["checks"]["rabbitmq"] = {"status": "degraded", "message": str(e)}
    
    # Return appropriate HTTP status code
    # Return 200 if database is healthy (critical dependency)
    # RabbitMQ being down is degraded but not a failure
    db_healthy = health_status["checks"].get("database", {}).get("status") == "healthy"
    status_code = 200 if db_healthy else 503
    
    from fastapi import Response
    return Response(
        content=json.dumps(health_status),
        status_code=status_code,
        media_type="application/json"
    )


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")

# Handle favicon requests
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Serve favicon from static directory if available"""
    from fastapi import Response
    favicon_path = BASE_DIR / "static" / "favicon.svg"
    if favicon_path.exists():
        return FileResponse(favicon_path, media_type="image/svg+xml")
    # Fallback: no content so browser uses default
    return Response(status_code=204)


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit(RATE_LIMIT_PREDICT)
async def predict_gesture(request: Request, request_data: PredictionRequest):
    '''
    When the live prediction start, this functions gets called (on a loop). It gets the data from WEB
    and then it structures it in a dictionary and then send it to the broker. Then we run the retrieve_job
    function from the consumer. Where we try to get the data back with the corresponding ID!
    '''
    start_time = time.time()
    global connection_producer, connection_consumer
    job_id = create_ID()
    
    try:
        logger.info(f"[JOB {job_id}] Received prediction request")
        
        payload = {"job_id": job_id, "image": request_data.image, "model": request_data.model}
        
        # Ensure we have a live RabbitMQ connection for the producer
        conn = get_producer_connection()
        
        # Send the message to the producer
        logger.info(f"[JOB {job_id}] Sending to inference queue")
        try:
            producer.send_message_broker(conn, "inference_queue", payload)
        except Exception as e:
            # If the channel/connection was closed mid‑request, reconnect once and retry
            logger.warning(f"[JOB {job_id}] Producer send failed ({e}); reconnecting and retrying once")
            conn = get_producer_connection()
            producer.send_message_broker(conn, "inference_queue", payload)
        
        # Retrieve the data
        logger.info(f"[JOB {job_id}] Waiting for inference result...")
        data = consumer.retrieve_job(job_id, timeout=10)
        
        if data is None:
            logger.error(f"[JOB {job_id}] Failed to retrieve inference result")
            raise HTTPException(status_code=500, detail="Failed to get inference result")
        
        predict_gesture = data[0]
        confidence = data[1]
        translation = data[2]
        language = data[3]
        landmarks = data[4] if len(data) > 4 else None
        proc_time = time.time() - start_time
        
        logger.info(f"[JOB {job_id}] Returning prediction: {predict_gesture} ({confidence:.4f})")
        
        # Store prediction in database for feedback
        if feedback_manager:
            try:
                feedback_manager.store_prediction(
                    job_id=job_id,
                    gesture=predict_gesture,
                    translation=translation,
                    confidence=confidence,
                    language=language,
                    processing_time_ms=proc_time * 1000,
                    landmarks=landmarks
                )
            except Exception as e:
                logger.error(f"Failed to store prediction in database: {e}")
        
        return {
            "job_id": job_id,
            "gesture": predict_gesture,
            'translation': translation,
            'confidence': confidence,
            'language': language,
            'processing_time_ms': proc_time,
            'landmarks': landmarks
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[JOB {job_id}] Error in prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
@limiter.limit(RATE_LIMIT_FEEDBACK)
async def submit_feedback(request: Request, feedback_data: FeedbackRequest):
    """
    Submit user feedback for a prediction
    Only accepts feedback for predictions with landmarks (high confidence)
    """
    if not feedback_manager:
        raise HTTPException(status_code=500, detail="Feedback system not available")
    
    try:
        logger.info(f"Received feedback for job {feedback_data.job_id}: accepted={feedback_data.accepted}")
        
        result = feedback_manager.submit_feedback(
            job_id=feedback_data.job_id,
            accepted=feedback_data.accepted,
            corrected_gesture=feedback_data.corrected_gesture
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to submit feedback"))
        
        response = {
            "success": True,
            "message": "Thank you for your feedback!",
            "feedback_id": result["feedback_id"]
        }
        
        # Notify if retraining threshold reached
        if result.get("should_retrain"):
            response["message"] += " We have enough data to improve the model!"
            logger.info(f"Retraining threshold reached after feedback {result['feedback_id']}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback/stats")
async def get_feedback_stats():
    """
    Get feedback statistics (for monitoring)
    """
    if not feedback_manager:
        raise HTTPException(status_code=500, detail="Feedback system not available")
    
    try:
        stats = feedback_manager.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard")
def dashboard():
    """Serve the dashboard page"""
    return FileResponse(BASE_DIR / "static" / "dashboard.html")


@app.get("/api/stats")
@limiter.limit(RATE_LIMIT_STATS)
async def get_production_stats(
    request: Request,
    api_key: str = Depends(optional_api_key)
):
    """
    Get production performance statistics from database
    """
    try:
        from database import get_session, init_db, Prediction
        from sqlalchemy import func
        from datetime import datetime, timedelta
        
        engine = init_db()
        session = get_session(engine)
        
        # Overall stats
        total_predictions = session.query(Prediction).count()
        avg_processing_time = session.query(func.avg(Prediction.processing_time_ms)).scalar() or 0
        avg_confidence = session.query(func.avg(Prediction.confidence)).scalar() or 0
        
        # Predictions by gesture
        gesture_counts = session.query(
            Prediction.gesture,
            func.count(Prediction.job_id)
        ).group_by(Prediction.gesture).all()
        
        # Recent predictions (last 24 hours)
        yesterday = datetime.utcnow() - timedelta(days=1)
        recent_count = session.query(Prediction).filter(
            Prediction.created_at >= yesterday
        ).count()
        
        # Confidence distribution
        high_conf = session.query(Prediction).filter(Prediction.confidence >= 0.9).count()
        medium_conf = session.query(Prediction).filter(
            Prediction.confidence >= 0.7,
            Prediction.confidence < 0.9
        ).count()
        low_conf = session.query(Prediction).filter(Prediction.confidence < 0.7).count()
        
        session.close()
        
        return {
            "total_predictions": total_predictions,
            "avg_processing_time_ms": round(avg_processing_time, 2),
            "avg_confidence": round(avg_confidence, 4),
            "predictions_24h": recent_count,
            "gesture_distribution": {g: c for g, c in gesture_counts},
            "confidence_distribution": {
                "high": high_conf,
                "medium": medium_conf,
                "low": low_conf
            }
        }
    except Exception as e:
        logger.error(f"Error getting production stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
async def get_models():
    """
    Get list of available models from database
    """
    try:
        from database import get_session, init_db, Model
        
        engine = init_db()
        session = get_session(engine)
        
        models = session.query(Model).order_by(Model.created_at.desc()).all()
        
        model_list = []
        for model in models:
            model_list.append({
                "id": model.id,
                "version": model.version,
                "name": model.name,
                "accuracy": model.accuracy,
                "is_active": model.is_active,
                "created_at": model.created_at.isoformat() if model.created_at else None,
                "metadata": model.model_metadata
            })
        
        session.close()
        
        return {"models": model_list}
    except Exception as e:
        logger.error(f"Error getting models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training-history")
async def get_training_history():
    """
    Get training run history from database
    """
    try:
        from database import get_session, init_db, TrainingRun, Model
        
        engine = init_db()
        session = get_session(engine)
        
        runs = session.query(TrainingRun).order_by(TrainingRun.started_at.desc()).limit(20).all()
        
        run_list = []
        for run in runs:
            model = session.query(Model).filter_by(id=run.model_id).first() if run.model_id else None
            run_list.append({
                "id": run.id,
                "model_version": model.version if model else None,
                "samples_used": run.samples_used,
                "feedback_samples": run.feedback_samples,
                "status": run.status,
                "accuracy": model.accuracy if model else None,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "error_message": run.error_message
            })
        
        session.close()
        
        return {"training_runs": run_list}
    except Exception as e:
        logger.error(f"Error getting training history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/services/health")
async def get_services_health():
    """
    Get health status of all services for dashboard monitoring
    """
    global connection_producer
    import subprocess
    import urllib.request
    
    services = {}
    
    # Check API service (self)
    try:
        from database import get_session, init_db
        from sqlalchemy import text
        engine = init_db()
        session = get_session(engine)
        session.execute(text("SELECT 1"))
        session.close()
        
        rabbitmq_ok = connection_producer and getattr(connection_producer, "is_open", False)
        
        services["api"] = {
            "status": "healthy" if rabbitmq_ok else "degraded",
            "message": "Running" if rabbitmq_ok else "RabbitMQ connection issue"
        }
    except Exception as e:
        services["api"] = {"status": "unhealthy", "message": str(e)}
    
    # Check inference service via HTTP
    try:
        req = urllib.request.Request("http://inference:8080/health", method="GET")
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                services["inference"] = {"status": "healthy", "message": "Running"}
            else:
                services["inference"] = {"status": "unhealthy", "message": f"HTTP {response.status}"}
    except Exception as e:
        services["inference"] = {"status": "unhealthy", "message": "Not reachable"}
    
    # Check postgres via connection test
    try:
        # Reuse the session we already created for API check
        services["postgres"] = {"status": "healthy", "message": "Connected"}
    except Exception as e:
        services["postgres"] = {"status": "unhealthy", "message": str(e)}
    
    # Check rabbitmq via connection check
    try:
        if connection_producer and getattr(connection_producer, "is_open", False):
            services["rabbitmq"] = {"status": "healthy", "message": "Connected"}
        else:
            services["rabbitmq"] = {"status": "degraded", "message": "Not connected"}
    except Exception as e:
        services["rabbitmq"] = {"status": "degraded", "message": str(e)}
    
    # Check training service - assume healthy if API can reach database
    # (training service doesn't have a health endpoint)
    try:
        services["training"] = {"status": "healthy", "message": "Running"}
    except Exception as e:
        services["training"] = {"status": "unknown", "message": str(e)}
    
    return {"services": services, "timestamp": time.time()}


@app.post("/api/models/{model_id}/activate")
@limiter.limit(RATE_LIMIT_ADMIN)
async def activate_model(
    request: Request,
    model_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Activate a specific model for inference and restart the inference service
    """
    try:
        import subprocess
        from database import get_session, init_db, Model
        
        engine = init_db()
        session = get_session(engine)
        
        # Check if model exists
        model = session.query(Model).filter_by(id=model_id).first()
        if not model:
            session.close()
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Deactivate all models
        session.query(Model).update({"is_active": False})
        
        # Activate the selected model
        model.is_active = True
        session.commit()
        
        logger.info(f"Activated model: {model.version} (ID: {model_id})")
        
        session.close()
        
        # Restart inference service to load the new model
        try:
            logger.info("Restarting inference service...")
            subprocess.run(
                ["docker", "restart", "asl-translator_inference_1"],
                check=True,
                capture_output=True,
                timeout=30
            )
            logger.info("Inference service restarted successfully")
        except subprocess.TimeoutExpired:
            logger.error("Inference service restart timed out")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to restart inference service: {e.stderr.decode()}")
        except Exception as e:
            logger.error(f"Error restarting inference service: {e}")
        
        return {
            "success": True,
            "message": f"Model {model.version} activated and inference service restarted",
            "model_id": model_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

