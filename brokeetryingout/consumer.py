# consumer
import pika

def auto_message_received(ch, method, properties, body):
    print(f"Received new message: [{body}]")


connection_parameters = pika.ConnectionParameters('localhost')
connection = pika.BlockingConnection(connection_parameters)

channel = connection.channel()

channel.queue_declare(queue="Letterbox")

channel.basic_consume(queue="Letterbox", auto_ack=True,
                      on_message_callback=auto_message_received)

print("Starting consuming")
channel.start_consuming()
