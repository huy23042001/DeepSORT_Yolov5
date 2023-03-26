from datetime import datetime
import json
from kafka import KafkaProducer
from kafka.errors import KafkaError
from numpy import log
import time
import cv2
import base64
producer = KafkaProducer(bootstrap_servers=['10.0.2.196:9092','10.0.2.197:9093'])

def encode(self, frame):
    _, buff = cv2.imencode('.jpg', frame)
    b64 = base64.b64encode(buff).decode()
    send_time = datetime.now()
    global camera_id
    data = {
        'camera_id': camera_id,
        'data': b64,
        'time': send_time
    }
    return json.dumps(data).encode('utf-8')

global camera_id
camera_id = 0
camera = cv2.VideoCapture(camera_id)

while camera.grab():
    _, img = camera.retrieve()
    _, buff = cv2.imencode('.jpg', img)
    b64 = base64.b64encode(buff).decode()
    send_time = datetime.now()
    data = {
        'camera_id': camera_id,
        'data': b64,
        'time': send_time.timestamp()
    }
    producer.send('my-topic', json.dumps(data).encode('utf-8'))
    time.sleep(2)
    print("sending")

# # block until all async messages are sent
# producer.flush()

# # configure multiple retries
# producer = KafkaProducer(retries=5)