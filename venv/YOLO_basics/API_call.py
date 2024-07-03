import os
from groq import Groq


import json
from typing import List, Dict

def Gen_AI_call ():

    # In the following image, what is the location of person not wearing safety vest? That person should have ID of 1 at the bottom of the box. Give that person's location with respect to the image and other people in the image. Is the person on the left or right of the image?
    
    # Instantiate the Groq client object with the API key
    client = Groq(api_key='')

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
