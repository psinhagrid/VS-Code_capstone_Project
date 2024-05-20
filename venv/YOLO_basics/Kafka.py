from confluent_kafka import Producer
import cv2
import time
import json

def delivery_report(err, msg):
    if err is not None:
        print("Message delivery failed:", err)
    else:
        print("Message delivered to", msg.topic(), "partition", msg.partition())

def produce_video_frames(video_file, topic):
    p = Producer({'bootstrap.servers': 'localhost:9092'})

    # Open the video file
    cap = cv2.VideoCapture(video_file)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to bytes
        _, buffer = cv2.imencode('.jpg', frame)
        data = buffer.tobytes()

        message = {
            'data': buffer,
            'timestamp': time.time()
        }
    

        # Convert dictionary to JSON string
        json_string = json.dumps(message)

        # Convert JSON string to bytes
        byte_data = json_string.encode('utf-8')


        # Produce the frame to Kafka topic
        p.produce(topic, value=byte_data, callback=delivery_report)
        frame_count += 1
        break

    print("Total frames sent:", frame_count)
    cap.release()
    p.flush()

if __name__ == '__main__':
    video_file = '/Users/psinha/Documents/capstone_project/venv/YOLO_basics/helmet4.mp4'
    topic = 'frame-topic'
    produce_video_frames(video_file, topic)


## Send frame functiom
# p = Producer({'bootstrap.servers': 'localhost:9092'})

# def sendFrame(frame, topic):
#     p.produce(topic, value=frame, callback=delivery_report)
