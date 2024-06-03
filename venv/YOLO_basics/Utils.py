from PIL import Image
import cv2
import base64
from PIL import Image
import numpy as np
import json
from kafka import KafkaProducer
import math
import cvzone



def bounding_box(box,img, show_box_for_all):

    """    
        Function to calculate confidence and make box for all recognized object 
    
        Gives out co-ordinates of boxes, Put show_box_for_all == True to make a box for all detections
    
    """

    # Bounding Box
    x1,y1,x2,y2 = box.xyxy[0]  # Getting coorinates of bounding box
    x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)    # Making integer values to make boxes in next step
    w, h = x2-x1, y2-y1     # Calculating width and length 


    if show_box_for_all == True:
        #cvzone.cornerRect(img, (x1,y1,w,h))
        pass


    # Confidence Level Calculation
    conf = math.ceil((box.conf[0]*100))/100     # Rounding off the confidence levels
    return x1,y1,x2,y2,conf


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

def make_description(location, confidence: int, employee_id: str, violation_type: str):

    """

        Makes Description for the occuring fault, gives fault type, confidence, person location and employee ID in image.
        Returns a dictionary with the following informations.

    """

    if (location["x1"] < 1280/2):
        location_of_person = "left"
    else : 
        location_of_person = "right"

    
    confidence_percent = round(confidence * 100, 2)

    description_dict = {

        "line1": "There is a violation of type NO-Safety Vest identified,",
        "line2": f"We say this with {confidence_percent}% confidence.",
        "line3": f"The person who has the violation is present on the {location_of_person} half of the image",
        "line4": f"and is assigned an ID of {employee_id}."
    }

    return description_dict


def generate_json( description: str, event_type: str, timestamp: str, frame: str , 
                  location: dict[str, int], confidence: str, employee_id: str, violation_type: str, severity_level: str, 
                  metadata: dict[str, str], image_encoded, output_file: str,):
    """
    Generate a JSON file with the provided parameters and save it to the specified output file.
    
    Parameters:
        image_encoded(str): This is the frame image encoded in 64 bit format. 
        event_type (str): The type of event (e.g., "PPE Violation").
        timestamp (str): The timestamp of the event in ISO 8601 format.
        frame (str): The base64 encoded frame data.
        location (Dict[str, int]): A dictionary containing the coordinates of the location
            with keys "x1", "y1", "x2", and "y2".
        employee_id (str): The ID of the employee associated with the event.
        violation_type (str): The type of violation (e.g., "No Helmet").
        severity_level (str): The severity level of the violation (e.g., "high").
        metadata (Dict[str, str]): Additional metadata associated with the event,
            such as camera ID, location, and environmental conditions.
        output_file (str): The path to the output JSON file.
    """
    data = {
        "description": description,
        
        "event_type": event_type,
        "timestamp": timestamp,
        "frame": frame,
        "location": location,
        "confidence": confidence,
        "employee_id": employee_id,
        "violation_type": violation_type,
        "severity_level": severity_level,
        "metadata": metadata,
        "image_encoded": image_encoded,

    }
    
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)



# json_file = '/Users/psinha/Documents/capstone_project/venv/YOLO_basics/output_json/output_1.json'
# image = decode_image_from_json(json_file)
# cv2.imshow("Decoded Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#################################################################################################################

