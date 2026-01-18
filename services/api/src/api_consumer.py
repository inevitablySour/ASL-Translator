import pika
import time
import json
import random
import os
import logging

logger = logging.getLogger(__name__)

'''
This is the API CONSUMER file. The API here listens for messages that it gets on the API queues.
'''

api_queue = os.getenv("API_QUEUE", "Letterbox")

# The dictionary with all the lists.
all_jobs = {}


def retrieve_job(job_id, timeout=10):
    '''
    This returns the data from the corresponding job-id (used in the API to get the data)
    Waits up to `timeout` seconds for the result
    '''
    logger.info(f"Waiting for job {job_id}...")
    start_time = time.time()
    retry_count = 0
    max_retries = int(timeout * 1000)  # 1ms sleep intervals
    
    try:
        while time.time() - start_time < timeout:
            if job_id in all_jobs:
                logger.info(f"Job {job_id} found after {retry_count} retries")
                result = all_jobs[job_id]
                del all_jobs[job_id]  # Clean up
                return result
            retry_count += 1
            time.sleep(0.001)
        
        logger.error(f"Job {job_id} not found after {timeout}s timeout. Available jobs: {list(all_jobs.keys())}")
        return None
    except Exception as e:
        logger.error(f"Error retrieving job {job_id}: {e}", exc_info=True)
        return None

def callback_function(ch, method, properties, body):
    '''
    When a message is received from the RabbitMQ server, this callback function is called. It loads the data
    into a dictionary. And then add it to the all_jobs dictionary. The API it self will pull from this dictionary to
    get its result with the corresponding job-id.
    '''
    try:
        data = json.loads(body.decode("utf-8"))
        job_id = data["job_id"]
        gesture = data["gesture"]
        confidence = data["confidence"]
        translation = data.get("translation", gesture)
        language = data.get("language", "english")
        logger.info(f'[API-CONSUMER] Received result for job {job_id}: gesture={gesture}, confidence={confidence}')
        all_jobs[job_id] = [gesture, confidence, translation, language]
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        logger.error(f"Error processing callback: {e}", exc_info=True)
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    
def connect_with_broker(retries=30, delay_s=2):
    # Gets the port from the OS(containers (set in compose))
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
    raise RuntimeError("- [API - CONSUMER] Couldn't connect to RabbitMQ") from last_error


# Starts the consumer
def consume_message_inference(connection):
    '''
    Makes a channel and then declare a queue. It listens continuously throughout the applications life to the
    specific queue.
    Then when it receives a message, the callback function is called (callback_function).
    TO-DO: based on the metadata a specific extraction should be made (that prediction is data['prediction'])
    '''
    channel_api_inference = connection.channel()
    channel_api_inference.queue_declare(queue=api_queue)
    channel_api_inference.basic_qos(prefetch_count=1) # Prefectch
    channel_api_inference.basic_consume(api_queue, on_message_callback=callback_function)
    print("- [API_INFERENCE] Waiting for messages!")
    channel_api_inference.start_consuming()


def close_connection(connection):
    if connection and connection.is_open:
        connection.close()
        logger.info("API Consumer connection closed")
        print("- [API] Connection closed!")
