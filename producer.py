import json
from kafka import KafkaProducer
from kafka.errors import KafkaError
from numpy import log

producer = KafkaProducer(bootstrap_servers=['10.0.2.196:2181','10.0.2.197:2181'])

# produce asynchronously
for _ in range(1000):
    producer.send('my-topic', b'msg')


# block until all async messages are sent
producer.flush()

# configure multiple retries
producer = KafkaProducer(retries=5)