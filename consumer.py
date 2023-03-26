import argparse
import json
from kafka import KafkaConsumer
import cv2
import numpy as np
from pyspark.sql import SparkSession
import base64
from pyspark.sql.functions import *
from pyspark.sql.types import *
from tracking import VideoTracker

spark  =  SparkSession.builder.appName('video_tracking.com').getOrCreate()
spark.sparkContext.setLogLevel("WARN")
# To consume latest messages and auto-commit offsets
# consumer = KafkaConsumer('my-topic',
#                          group_id='my-group',
#                          bootstrap_servers=['localhost:9092'])
# for message in consumer:
#     # message value and key are raw bytes -- decode if necessary!
#     # e.g., for unicode: `message.value.decode('utf-8')`
#     print ("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
#                                           message.offset, message.key,
#                                           message.value))

# # consume earliest available messages, don't commit offsets
# KafkaConsumer(auto_offset_reset='earliest', enable_auto_commit=False)

# # consume json messages
# KafkaConsumer(value_deserializer=lambda m: json.loads(m.decode('ascii')))

# # consume msgpack
# KafkaConsumer(value_deserializer=msgpack.unpackb)

# # StopIteration if no message after 1sec
# KafkaConsumer(consumer_timeout_ms=1000)

# # Subscribe to a regex topic pattern
# consumer = KafkaConsumer()
# consumer.subscribe(pattern='^awesome.*')

# # Use multiple consumers in parallel w/ 0.9 kafka brokers
# # typically you would run each on a different server / process / CPU
# consumer1 = KafkaConsumer('my-topic',
#                           group_id='my-group',
#                           bootstrap_servers='my.server.com')
# consumer2 = KafkaConsumer('my-topic',
#                           group_id='my-group',
#                           bootstrap_servers='my.server.com')
class Consum(object):
    def __init__(self, args):
        self.args = args
        self.camera = cv2.VideoCapture(args.cam)
        self.track = VideoTracker(args=args)
        # df = spark.readStream\
        # .format("kafka")\
        # .option("kafka.bootstrap.servers", "10.0.2.196:9092,10.0.2.197:9093")\
        # .option("subscribe", "my-topic")\
        # .load()\
        # .select(col('value').cast('string').alias('data'))\
        # .writeStream\
        # .foreach(lambda x: print(x))\
        # .start().awaitTermination()

    def run(self):
        while self.camera.grab():
            _, frame = self.camera.retrieve()
            outputs,_,_ = self.track.image_track(frame)
            if len(outputs) > 0:
                print(outputs)
                frame = self.track.draw_box(frame,outputs)
            cv2.imshow("cam1", frame)

            if cv2.waitKey(1) == ord('q'):  # q to quit
                    cv2.destroyAllWindows()
                    break
            
        self.camera.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument("--frame_interval", type=int, default=2)
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    # YOLO-V5 parameters
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='model.pt path')
    parser.add_argument("--cam",type=int, default="0",help="set camera")
    args = parser.parse_args()
    print(args)

    consumer = Consum(args)
    consumer.run()