import json
from kafka import KafkaConsumer
import cv2
import numpy as np
from pyspark.sql import SparkSession

# To consume latest messages and auto-commit offsets
spark  =  SparkSession.builder.appName('video_tracking.com').getOrCreate()
spark.sparkContext.setLogLevel("WARN")
# consumer = KafkaConsumer('my-topic',
#                          group_id='my-group',
#                          bootstrap_servers=['10.0.2.196:9092','10.0.2.197:9093'])
# for message in consumer:
#     # message value and key are raw bytes -- decode if necessary!
#     # e.g., for unicode: `message.value.decode('utf-8')`
#     # print ("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
#     #                                       message.offset, message.key,
#     #                                       message.value))
#     json_converted = message.value.decode('utf8').replace('"',"'")
#     data = json.loads(json_converted)
#     print(json.dumps(data))
#     # data = message.value.data
#     # np_data = np.fromstring(data, np.unit8)
#     # img = cv2.imencode(np_data,cv2.IMREAD_UNCHANGED)
#     # cv2.imshow("images/ID_" + message.value.camera_id + "_" + str(message.value.time), img)
#     # cv2.imwrite()
#     # if cv2.waitKey(1) == ord('q'):  # q to quit
#     #                 cv2.destroyAllWindows()
#     #                 break
    
df = spark.readStream\
    .format("kafka")\
    .option("kafka.bootstrap.servers", "10.0.2.196:9092,10.0.2.197:9093")\
    .option("subscribe", "my-topic")\
    .load()\
    .writeStream\
    .foreach(lambda x: print(x))\
    .start().awaitTermination()

