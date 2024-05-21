from ultralytics import YOLO
import cv2
import base64
from PIL import Image
import base64
import cvzone
import torch
import math
from sort import *
from API_call import *
from Utils import *

#class_names only set to ['Persons']

###############################################################################################


## Initialize the model 

#  model = YOLO('venv/YOLO-weights/yolov8l.pt')


model = YOLO('fine_tuned_weights.pt')


## Yolo class names 
className = [
    'Hardhat', 'Mask', 'NO_Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', "Safety Cone", 'Safety Vest', 
    'machinery', 'vehicle', 
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

className_finetuned = ['Hardhat', 'Mask', 'NO_Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 
                       'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']


# Frame Counter initialized
frame_number = 0

# Voilations dictionary initialized
voilation_dict = {}

# violators count:
violators_count = 0 

violator_ID = []
###############################################################################################

"""       UTILS         """




def raise_flag(img, event_type: str, timestamp: str, frame: str , 
                  location: Dict[str, int], confidence: str, employee_id: str, violation_type: str, severity_level: str, 
                  metadata: Dict[str, str], output_file: str):
    print ("Flag_raised")
    global violators_count
    violators_count += 1
    global violator_ID
    violator_ID.append(employee_id)
    image_encoded = compress_image_to_base64(img, quality=20)
    generate_json(image_encoded, event_type, timestamp, frame, 
                  location, confidence, employee_id, violation_type, severity_level, 
                  metadata, output_file)






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


def anomaly_detector(img, box, x1, y1, x2, y2, Id, currentClass, conf):

    global current_count
    global voilation_dict
    global frame_number

    output_file_base = "venv/YOLO_basics/output_json/output"

    if (currentClass in ('NO-Safety Vest', 'NO_Hardhat', 'NO-Mask')):

        global violation_count
        print (violators_count) 
        if (Id not in voilation_dict.keys() and Id != None):
            voilation_dict[Id] = [frame_number, 1, conf]

        else :

            if (voilation_dict[Id][1] == -999):
                #print ("\n\n PASSED")
                pass


            elif (voilation_dict[Id][0] in range (frame_number-10, frame_number+10)):
            
                print ("\nDICT_value : ", voilation_dict[Id][1])
                print ("\n")
                if (voilation_dict[Id][1] == 10):
                    voilation_dict[Id][1] = -999
                    #print ("\n\n ENTERED FRAME ")
                    #print (voilation_dict[1][1])
                    raise_flag(
                    img = img,
                    event_type="PPE Violation",
                    timestamp="2024-07-01T14:23:45Z", 
                    frame=frame_number, 
                    location={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    confidence = voilation_dict[Id][2],
                    employee_id=Id,
                    violation_type=currentClass,
                    severity_level="high",
                    metadata={"camera_id": "CAM01", "location": "Warehouse Section A", "environmental_conditions": "Normal"},
                    output_file = f"{output_file_base}_{violators_count+1}.json"
                
                    )

                else :
                    voilation_dict[Id][0] = frame_number
                    voilation_dict[Id][1] += 1
                    voilation_dict[Id][2] = max(voilation_dict[Id][2], conf)
                    #print ("\n Frame Number", frame_number)
                    #print (voilation_dict[1][1])

            elif (voilation_dict[Id][0] not in range (frame_number-10, frame_number+10)):
                #quit()
                voilation_dict[Id][0] = frame_number
                voilation_dict[Id][1] = 1
                voilation_dict[Id][2] = conf
                #print ("\n Frame Number", frame_number)
                

            else :

                print ("UNKNOWN VIOLATION")
                quit()
                
                



def object_ID(img, box, cls, result_tracker, current_Class, class_names, object_counter_requirement,conf):

    """
        Function will make a rectangle against selected class and will also display the ID for tracking
    
    """


    for results in result_tracker:
        x1, y1, x2, y2, Id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        

        cvzone.putTextRect(img, f"ID - {int(Id)}", (max(x1+w-10, 0), max(y1 -10, 0) ), 1.5, 2)

        x_center = x1+w//2
        y_center = y1+h//2

        #print ("\nID is : ",Id)

        if object_counter_requirement == True:
            anomaly_detector(img, box, x1, y1, x2, y2, Id, current_Class, conf)

            







def class_to_track(img, box, cls, detections, current_class, class_names, object_counter_requirement):

    """
        This function will make boxes and assign IDs to given classes. 
        It will assign to all IDs to all detected objects if class_names is empty

        Return Type : Detections, Center x coordinate, Center y coordinate
    
    """


    if (class_names == None or class_names == [] or current_class in class_names):
        
        global frame_number 
        frame_number += 1

        x1,y1,x2,y2,conf = bounding_box(box,img, show_box_for_all=True)
        
        current_array = np.array([x1,y1,x2,y2,conf])        
        detections = np.vstack((detections, current_array))     # Giving labels

        cvzone.putTextRect(img,f'{className[cls]} {conf}', (max(x1, 0), max(35, y1-10)), 2, 2)
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)     # Making rectangle

        resultTracker = tracker.update(detections)

        print ("Frame Number : ", frame_number)

        object_ID(img, box, cls, resultTracker, current_class, class_names, object_counter_requirement,conf)



    return detections









###############################################################################################

"""   Program to capture video from the device.     """

##    cv2.waitKey(0) VS cv2.waitKey(1) difference is 1 will continue execution after 1ms delay and 0 will wait till key given

address1 = 'venv/YOLO_basics/helmet.mp4'
address2 = 'venv/YOLO_basics/helmet2.mp4'
address3 = 'venv/YOLO_basics/helmet3.mp4'
address4 = 'venv/YOLO_basics/helmet4.mp4'

address = address3

# Available modes "LIVE" and "MP4"
video_mode = "MP4"


# Set if object counter is needed. Options are True and False
object_counter_requirement = True


# Set required class list
#class_names = ['Hardhat', 'Mask', 'NO_Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', "Safety Cone",'Safety Vest', 'machinery', 'vehicle']

class_names = [ 'NO-Safety Vest' ]





###############################################################################################

# Tracking
tracker = Sort(max_age=200, min_hits=30, iou_threshold=0.3)   # Used for tracking of cars

###############################################################################################


def main(address, video_mode, object_counter_requirement, class_names):


    if video_mode == "LIVE":
        #  For Live video capture
        cap = cv2.VideoCapture(0)
        cap.set(3,1280)     # Width of 1280
        cap.set(4,720)      # Length of 720


    elif video_mode == "MP4":
        #  For Video Processing mp4 format
        cap = cv2.VideoCapture(address)
        #mask = cv2.imread('venv/YOLO_basics/CAR_MASK.png')   # make mask from canva.com


    current_count = []



    while True:

        success, img = cap.read()

        if not success:
            print("No more Frames to capture")
            break  # Exit the loop if reading fails

        if video_mode == "MP4":
            imgRegion = img
            result = model(imgRegion,device = "mps" ,stream = True)    # Use mps and stream feature 

        else :
            result = model(img, device = "mps" ,stream = True)    # Use mps and stream feature )


        detections = np.empty((0,5))


        for r in result:

            boxes = r.boxes
            for box in boxes:
                
                x1,y1,x2,y2,conf = bounding_box(box,img, show_box_for_all=True)


                # Class Name Display
                cls = int(box.cls[0])
                #cls += 80
                currentClass = className[cls]
                #print ("\n\n current_count = ",currentClass)


                # Call detections with args detections, Current Class, Interested Classes
                detections = class_to_track(img, box, cls, detections, currentClass, class_names , object_counter_requirement)
                #call the ingestion api..
                
                


        cv2.imshow("Image", img)    # Show images
        torch.mps.empty_cache()
        cv2.waitKey(1)

        


    print (violators_count)
    print (violator_ID)


    """

        Performance Comparsion on GPU and CPU

    """

    ## For CPU
    ##     Speed: 1.4ms preprocess, 217.8ms inference, 0.5ms postprocess per image at shape (1, 3, 384, 640) p

    ## For GPU
    ##     Speed: 1.6ms preprocess, 36.3ms inference, 12.9ms postprocess per image at shape (1, 3, 384, 640)


main(address, video_mode, object_counter_requirement, class_names)
###############################################################################################
