from confluent_kafka import Consumer, KafkaError
import time
import json

def consume_video_frames(topic):

    

    c = Consumer({
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'mygroup',
        'auto.offset.reset': 'earliest'
    })

    c.subscribe([topic])

    while True:
        end_time = time.time()
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
        print (msg.type())
        #json_string = msg.value.decode('utf-8')
    
        #print(json.loads(json_string))
        # Calculate time difference
        #print (msg)
        # frame_time = msg.value['timestamp']
        # # time_difference = end_time - frame_time # Convert timestamp to seconds
        # print(end_time)
        # print(frame_time)
        # print  ("\n\n")
        #print("Time taken to consume frame: {:.5f} seconds".format(end_time - frame_time))

    c.close()

if __name__ == '__main__':
    topic = 'frame-topic'
    consume_video_frames(topic)
