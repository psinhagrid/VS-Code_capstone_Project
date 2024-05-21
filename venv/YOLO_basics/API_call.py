import os
from groq import Groq


import json
from typing import List, Dict

def Gen_AI_call ():
    # Instantiate the Groq client object with the API key
    client = Groq(api_key='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')

    completion = client.chat.completions.create(
        model="gemma-7b-it",
        messages=[
            {
                "role": "user",
                "content": "Can I generate a Json file using gemma 7b ? \n"
            },
            {
                "role": "assistant",
                "content": "**Guardian Titan** is the biggest creature in Subnautica."
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="")


# Gen_AI_call()




def generate_json( event_type: str, timestamp: str, frame: str , 
                  location: Dict[str, int], confidence: str, employee_id: str, violation_type: str, severity_level: str, 
                  metadata: Dict[str, str], output_file: str):
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
        #"image_encoded": image_encoded,
        "event_type": event_type,
        "timestamp": timestamp,
        "frame": frame,
        "location": location,
        "confidence": confidence,
        "employee_id": employee_id,
        "violation_type": violation_type,
        "severity_level": severity_level,
        "metadata": metadata,

    }
    
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)




# Example usage:
# generate_json(
#     event_type="PPE Violation",
#     timestamp="2024-07-01T14:23:45Z",
#     frame="base64_encoded_frame_data",
#     detected_items=[
#         {"item": "Helmet", "status": "missing", "confidence": 0.95},
#         {"item": "Gloves", "status": "worn", "confidence": 0.98},
#         {"item": "Safety Glasses", "status": "missing", "confidence": 0.90}
#     ],
#     location={"x1": 100, "y1": 150, "x2": 300, "y2": 450},
#     employee_id="EMP12345",
#     violation_type="No Helmet",
#     severity_level="high",
#     metadata={"camera_id": "CAM01", "location": "Warehouse Section A", "environmental_conditions": "Normal"},
#     output_file="venv/YOLO_basics/output_json/output.json"
# )
