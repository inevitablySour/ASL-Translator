import pika
import time
import json
import os
import inference_worker as inference_worker

inference_queue = os.getenv("MODEL_QUEUE", "Letterbox")
api_queue = os.getenv("API_QUEUE", "Letterbox")
conn = None


def callback(ch, method, properties, body):
    api_data = json.loads(body.decode("utf-8"))
    job_id = api_data["job_id"]
    image = api_data["image"]
    inference_data = inference_worker.predict_gesture_from_base64(image)
    gesture = inference_data["gesture"]
    confidence = inference_data["confidence"]
    new_payload = {"job_id": job_id, "gesture": gesture, "confidence": confidence}
    print(new_payload)
    ch.queue_declare(queue=api_queue)
    ch.basic_publish(
        exchange="",
        routing_key=api_queue,
        body=json.dumps(new_payload).encode("utf-8"), # Makes the dict. to json, then to bytes for RabbitMQ
        properties=pika.BasicProperties(content_type="api/prediction/json")
    )
    print("- [INFERENCE - Prod.] Data sent to broker!")
    ch.basic_ack(delivery_tag = method.delivery_tag)


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
    raise RuntimeError("Couldn't connect to RabbitMQ") from last_error


def consume_message_inference(connection):
    global conn
    conn = connection
    channel_api_inference = connection.channel()
    channel_api_inference.queue_declare(queue=inference_queue)

    channel_api_inference.basic_qos(prefetch_count=1)
    channel_api_inference.basic_consume(inference_queue, on_message_callback=callback)
    print("- [INFERENCE - Cons.] Waiting for messages!")
    channel_api_inference.start_consuming()


connection_consumer = connect_with_broker()
consume_message_inference(connection_consumer)