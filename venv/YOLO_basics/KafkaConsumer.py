from confluent_kafka import Consumer, KafkaError
import numpy as np
import cv2

def display_frame(msg):
    frame = np.frombuffer(msg.value(), dtype=np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

def consume_video_frames(topic):
    c = Consumer({
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'mygroup',
        'auto.offset.reset': 'earliest'
    })

    c.subscribe([topic])

    while True:
        msg = c.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                # End of partition, ignore
                continue
            else:
                print(msg.error())
                break

        # Display the frame
        display_frame(msg)

    c.close()

if __name__ == '__main__':
    topic = 'frame-topic'
    consume_video_frames(topic)
