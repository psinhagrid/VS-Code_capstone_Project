from ultralytics import YOLO
import cv2
import cvzone
import torch
import math
from sort import *


###############################################################################################


## Initialize the model 

model = YOLO('venv/YOLO-weights/yolov8l.pt')

#model = YOLO('/Users/psinha/Documents/capstone_project/runs/detect/train4/weights/best.pt')


## Yolo class names 
className = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush','Hardhat', 'Mask', 'NO_Hardhat', 'NO-Mask', 
    'NO-Safety Vest', 'Person', "Safety Cone", 'Safety Vest', 'machinery', 'vehicle'
]

className_finetuned = ['Hardhat', 'Mask', 'NO_Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 
                       "Safety Cone", 'Safety Vest', 'machinery', 'vehicle']
###############################################################################################

"""       UTILS         """



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
        cvzone.cornerRect(img, (x1,y1,w,h))


    # Confidence Level Calculation
    conf = math.ceil((box.conf[0]*100))/100     # Rounding off the confidence levels
    return x1,y1,x2,y2,conf


def object_counter(x_center, y_center, Id):

    print ("\n\nENTERED")
    global current_count
    # Making a line for detection 
    # Line limits
    limits = [400, 297, 673, 297]
    cv2.line(img,(limits[0],limits[1]), (limits[2], limits[3]), (0,0,255), 5)


    if limits[0] < x_center < limits[2] and limits[1] - 15 < y_center < limits[3] + 15:
        if current_count.count(Id) == 0:
            current_count.append(Id)
            

    cvzone.putTextRect(img, f'Count : {len(current_count)}', (50,50))


def object_ID(result_tracker, current_class, class_names, object_counter_requirement):

    """
        Function will make a rectangle against selected class and will also display the ID for tracking
    
    """

    if (class_names == None or class_names == [] or current_class in class_names):
        # Iterate over the tracked results
        for results in result_tracker:
            x1, y1, x2, y2, Id = results
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            
            # Class Name
            cls = int(box.cls[0])
            currentClass = className[cls]
            
            # Display the ID on the image
            cvzone.putTextRect(img, f"ID - {int(Id)}", (max(x1+w-10, 0), max(y1 - 10, 0)), 1.5, 2)

            x_center = x1+w//2
            y_center = y1+h//2

            if object_counter_requirement == True:
                object_counter(x_center, y_center, Id)







def class_to_track(detections, current_class, class_names, object_counter_requirement):

    """
        This function will make boxes and assign IDs to given classes. 
        It will assign to all IDs to all detected objects if class_names is empty

        Return Type : Detections, Center x coordinate, Center y coordinate
    
    """


    if (class_names == None or class_names == [] or current_class in class_names):

        x1,y1,x2,y2,conf = bounding_box(box,img, show_box_for_all=True)
        current_array = np.array([x1,y1,x2,y2,conf])        
        detections = np.vstack((detections, current_array))     # Giving labels

        cvzone.putTextRect(img,f'{className[cls]} {conf}', (max(x1, 0), max(35, y1-10)), 2, 2)
        #cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)     # Making rectangle

        resultTracker = tracker.update(detections)

        
        object_ID(resultTracker, current_class, class_names, object_counter_requirement)



    return detections









###############################################################################################

"""   Program to capture video from the device.     """

##    cv2.waitKey(0) VS cv2.waitKey(1) difference is 1 will cintinue execution after 1ms delay and 0 will wait till key given

address1 = '/Users/psinha/Documents/capstone_project/venv/YOLO_basics/cars.mp4'
address2 = '/Users/psinha/Documents/capstone_project/venv/YOLO_basics/helmet.mp4'

address = address1

# Available modes "LIVE" and "MP4"
video_mode = "MP4"


# Set if object counter is needed. Options are True and False
object_counter_requirement = True


# Set required class list
class_names = ['car']


if video_mode == "LIVE":
    #  For Live video capture
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)     # Width of 1280
    cap.set(4,720)      # Length of 720


elif video_mode == "MP4":
    #  For Video Processing mp4 format
    cap = cv2.VideoCapture(address)
    mask = cv2.imread('venv/YOLO_basics/CAR_MASK.png')   # make mask from canva.com


###############################################################################################

""" Initializing Tracker and Limits for the line for car counting  """

# Tracking
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)   # Used for tracking of cars



###############################################################################################

current_count = []

while True:

    success, img = cap.read()

    if video_mode == "MP4":
        imgRegion = cv2.bitwise_and(img,mask)       # Bitwise And of video and mask
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



            # Call detections with args detections, Current Class, Interested Classes
            detections = class_to_track(detections, currentClass, class_names , True)

            
              


    cv2.imshow("Image", img)    # Show images
    torch.mps.empty_cache()
    cv2.waitKey(1)


"""

    Performance Comparsion on GPU and CPU

"""

## For CPU
##     Speed: 1.4ms preprocess, 217.8ms inference, 0.5ms postprocess per image at shape (1, 3, 384, 640) 

## For GPU
##     Speed: 1.6ms preprocess, 36.3ms inference, 12.9ms postprocess per image at shape (1, 3, 384, 640)



###############################################################################################