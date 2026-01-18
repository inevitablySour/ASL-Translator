from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
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
    image: str
    language: Optional[str] = "english"

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
    job_id: str
    accepted: bool
    corrected_gesture: Optional[str] = None


connection_producer = None
connection_consumer = None
consumer_thread = None

# Start consuming!!
def start_consuming(connection):
    consumer.consume_message_inference(connection)

# Set up the channel and queues when starting up the API/WEB and closing the connection
@asynccontextmanager
async def lifespan(app: FastAPI):
    global connection_producer, connection_consumer, consumer_thread

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


# Serve static files (css, js, images)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Setting up the home page
@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/predict", response_model=PredictionResponse)
async def predict_gesture(request: PredictionRequest):
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
        
        payload = {"job_id": job_id, "image": request.image, "language": request.language}
        
        # Send the message to the producer
        logger.info(f"[JOB {job_id}] Sending to inference queue")
        producer.send_message_broker(connection_producer, "inference_queue", payload)
        
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
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for a prediction
    Only accepts feedback for predictions with landmarks (high confidence)
    """
    if not feedback_manager:
        raise HTTPException(status_code=500, detail="Feedback system not available")
    
    try:
        logger.info(f"Received feedback for job {request.job_id}: accepted={request.accepted}")
        
        result = feedback_manager.submit_feedback(
            job_id=request.job_id,
            accepted=request.accepted,
            corrected_gesture=request.corrected_gesture
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

