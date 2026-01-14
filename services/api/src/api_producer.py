from pydantic import BaseModel
from pathlib import Path
import json
import asyncio
import pika
import time
import random
import os

'''
This is the API PRODUCER file. When the API wants to send a message it can do so by accessing the functions in here.
'''


# Get the queues from the compose files (kinda like config.py)
inference_queue = os.getenv("MODEL_QUEUE", "Letterbox")

# Tries to make a connection with the broker over a number of retries
def connect_with_broker(retries=30, delay_s=2):
    # Gets the port from the OS(containers (set in compose), like config.py)
    host = os.getenv("RABBITMQ_HOST", "rabbitmq")
    port = int(os.getenv("RABBITMQ_PORT", 5672))

    # Sets the connection parameters
    connection_parameters = pika.ConnectionParameters(host=host, port=port)
    last_error = None

    # Tries to connect to the broker x amount of time
    for _ in range(retries):
        try:
            return pika.BlockingConnection(connection_parameters)
        except pika.exceptions.AMQPConnectionError as e:
            last_error = e
            time.sleep(delay_s)
    raise RuntimeError("- [API_PRODUCER] Couldn't connect to RabbitMQ") from last_error


# Sends message through a channel
def send_message_broker(connection, specific_queue, payload):
    '''
    connection: is send through the function to choose a specific channel if wanted.
    specific_queue: is the name of the queue to send the message to. (e.g. inference_queue or worker_queue)
    payload:  is the data/payload to send to the broker.
    '''

    # Setup a channel with the connection and declare a queue on the channel
    channel_apiproducer = connection.channel()
    channel_apiproducer.queue_declare(queue=specific_queue)

    # Publish a message to the queue onto the channel via the connection ;)
    channel_apiproducer.basic_publish(
        exchange="",
        routing_key=specific_queue,
        body=json.dumps(payload).encode("utf-8"), # Makes the dict. to json, then to bytes for RabbitMQ
        properties=pika.BasicProperties(content_type="api/prediction/json") # Gives additional metadata
    )
    print("- [API-PRODUCER] Data sent to broker!")


# To close the connection when the services end!!!!!!
def close_connection(connection):
    if connection and connection.is_open:
        connection.close()
        print("- [API] Connection closed!")