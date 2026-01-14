import pika
import time
import json
import random
import os

'''
This is the API CONSUMER file. The API here listens for messages that it gets on the API queues.
'''

api_queue = os.getenv("API_QUEUE", "Letterbox")

# The dictionary with all the lists.
all_jobs = {}


def retrieve_job(job_id):
    '''
    This returns the data from the corresponding job-id (used in the API to get the data)
    '''
    try:
        for _ in range(100):
            if job_id in all_jobs.keys():
                return all_jobs[job_id]
            time.sleep(0.001)
    except:
        print(" - [API_CONSUMER] Job not existing.")

def callback_function(ch, method, properties, body):
    '''
    When a message is received from the RabbitMQ server, this callback function is called. It loads the data
    into a dictionary. And then add it to the all_jobs dictionary. The API it self will pull from this dictionary to
    get its result with the corresponding job-id.
    '''

    data = json.loads(body.decode("utf-8"))
    job_id = data["job_id"]
    gesture = data["gesture"]
    confidence = data["confidence"]
    print('- [API-CONSUMER] ', data )
    all_jobs.update({job_id: [gesture, confidence]})
    ch.basic_ack(delivery_tag=method.delivery_tag)

    
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
        print("- [API] Connection closed!")
