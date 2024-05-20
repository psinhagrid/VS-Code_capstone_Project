from PIL import Image
import cv2
import base64
from PIL import Image
import numpy as np
import json
from kafka import KafkaProducer


def compress_image_to_base64(image, quality=20):
    """
    Compress an image and encode it to Base64.
    
    Parameters:
        image (numpy array): The image to compress.
        quality (int): The quality of the compressed image (1-95).
    
    Returns:
        base64_string (str): The compressed image encoded in Base64.
    """

    # Convert the image from OpenCV format to PIL format
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Compress the image and encode it to Base64 directly
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    img_str = base64.b64encode(buffer).decode('utf-8')
  
    return img_str

def decode_image_from_json(json_file):
    """
    Read Base64 encoded image data from a JSON file and decode it.
    
    Parameters:
        json_file (str): Path to the JSON file.
    
    Returns:
        img (numpy array): Decoded image.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract Base64 encoded image data from JSON
    base64_string = data['image_encoded']
    
    # Decode Base64 string to bytes
    img_data = base64.b64decode(base64_string)
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(img_data, np.uint8)
    
    # Decode image array using OpenCV
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return img


# json_file = '/Users/psinha/Documents/capstone_project/venv/YOLO_basics/output_json/output_1.json'
# image = decode_image_from_json(json_file)
# cv2.imshow("Decoded Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#################################################################################################################


from kafka.admin import KafkaAdminClient, NewTopic

# Kafka broker configuration
bootstrap_servers = 'localhost:9092'

# Create an instance of KafkaAdminClient with the bootstrap servers
admin_client = KafkaAdminClient(bootstrap_servers=bootstrap_servers)

# Define the topic name and number of partitions
topic_name = 'your_topic_name'
num_partitions = 3  # Adjust the number of partitions as needed
replication_factor = 1  # Adjust the replication factor as needed

# Create a NewTopic object with the topic name, number of partitions, and replication factor
new_topic = NewTopic(name=topic_name, num_partitions=num_partitions, replication_factor=replication_factor)

# Create the topic
admin_client.create_topics(new_topics=[new_topic])

# Close the admin client
admin_client.close()

