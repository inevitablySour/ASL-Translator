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
# Get the queues from the compose files
inference_queue = os.getenv("MODEL_QUEUE", "Letterbox")

BASE_DIR = Path(__file__).resolve().parent  # api/src

# Create a unique ID
def create_ID():
    return str(uuid.uuid4())

# Gets the data from web and convert it into this object
class PredictionRequest(BaseModel):
    image: str

# How to retrieve the data back from the inference.
class PredictionResponse(BaseModel):
    job_id: str
    gesture: str
    translation: str
    confidence: float
    language: str
    processing_time_ms: float


connection_producer = None
connection_consumer = None
consumer_thread = None

# Start consuming!! nom nom🍔
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
    start_time = time.time() # Sets time to measure the taken time to predict.
    global connection_producer; global connection_consumer # Get the global connections
    # Create a job ID for the request so that we get the right answer back, instead of just pulling the first from the stack
    job_id = create_ID()

    payload = {"job_id": job_id, "image": request.image}

    # Send the message to the producer, who sends it to the broker
    producer.send_message_broker(connection_producer, "inference_queue", payload)
    print("- [API] DECODED IMAGE - SENT TO PRODUCER")

    # Retrieve the data which needs to be in the PredictionResponse structure
    data = consumer.retrieve_job(job_id)
    print(f"- API   ::::: {data}")
    predict_gesture = data[0]
    confidence = data[1]
    proc_time = time.time() - start_time


    return {
        "job_id": job_id,
        "gesture": predict_gesture,
        'translation': "Yes",
        'confidence': confidence,
        'language': "little bitch",
        'processing_time_ms': time.time()-start_time
    }


