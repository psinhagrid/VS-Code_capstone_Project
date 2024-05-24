from ultralyticsplus import YOLO, render_result
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
import numpy as np


# load model
model = YOLO('forklift.pt')

class_names = ['forklift' , 'person']

forklift_coordinates = {}

# Tracking Forklifts
tracker_forklift = Sort(max_age=200, min_hits=50, iou_threshold=0.5)

# Tracking People
tracker_person = Sort(max_age=200, min_hits=30, iou_threshold=0.3)

def person_proximity_alert(img, box, cls: int, detections_person, current_class: str, class_names: List):

    if current_class == "person":

        x1,y1,x2,y2,conf = bounding_box(box,img, show_box_for_all=True)

        current_array = np.array([x1,y1,x2,y2,conf])        
        detections_person = np.vstack((detections_person, current_array))     # Giving labels

        resultTracker_person = tracker_person.update(detections_person)

        cvzone.putTextRect(img,f'{class_names[cls]} {conf}', (max(x1, 0), max(35, y1-10)), 2, 2)
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)     # Making rectangle


        for results in resultTracker_person:
            x1, y1, x2, y2, Id = results
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            x_center, y_center = x1+w//2, y1+h//2 

            cvzone.putTextRect(img, f"ID - {int(Id)}", (max(x2 - w - 10, 0),max(y2 - 10, h) ), 1.5, 2)

    return detections_person






def fork_lift_tracker(img, box, cls: int, detections_forklift, current_class: str, class_names: List):

    global forklift_coordinates

    if current_class == "forklift":

        x1,y1,x2,y2,conf = bounding_box(box,img, show_box_for_all=True)
        
        current_array = np.array([x1,y1,x2,y2,conf])        
        detections_forklift = np.vstack((detections_forklift, current_array))     # Giving labels

        cvzone.putTextRect(img,f'{class_names[cls]} {conf}', (max(x1, 0), max(35, y1-10)), 2, 2)
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)     # Making rectangle

        resultTracker_forklift = tracker_forklift.update(detections_forklift)

        for results in resultTracker_forklift:
            x1, y1, x2, y2, Id = results
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            x_center, y_center = x1+w//2, y1+h//2 

            cvzone.putTextRect(img, f"ID - {int(Id)}", (max(x2 - w - 10, 0),max(y2 - 10, h) ), 1.5, 2)


            if Id not in forklift_coordinates.keys() and Id is not None: 
                
                forklift_coordinates[Id] = [x1, y1, x2, y2, w, h, x_center, y_center]
       
    
    return detections_forklift

def main(address: str, video_mode: str):

    if video_mode == "LIVE":
        #  For Live video capture
        cap = cv2.VideoCapture(0)
        cap.set(3,1280)     # Width of 128
        cap.set(4,720)      # Length of 720


    elif video_mode == "MP4":
        #  For Video Processing mp4 format
        cap = cv2.VideoCapture(address)
        #mask = cv2.imread('venv/YOLO_basics/CAR_MASK.png')   # make mask from canva.com


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


        detections_person = np.empty((0,5))
        detections_forklift = np.empty((0,5))
        


        for r in result:

            boxes = r.boxes
            for box in boxes:
                
                x1,y1,x2,y2,conf = bounding_box(box,img, show_box_for_all=True)
                w, h = x2-x1, y2-y1     # Calculating width and length 

                # Class Name Display
                cls = int(box.cls[0])
   
                currentClass = class_names[cls]
                print (cls)
                # print (currentClass)
                #print ("\n\n current_count = ",currentClass)

                cvzone.cornerRect(img, (x1,y1,w,h))
                # Call detections with args detections, Current Class, Interested Classes

                #detections_forklift = fork_lift_tracker(img, box, cls, detections_forklift, currentClass, class_names )

                if (currentClass == "person"):
                    detections_person = person_proximity_alert(img, box, cls, detections_forklift, currentClass, class_names )

                elif (currentClass == "forklift"):
                    detections_forklift = fork_lift_tracker(img, box, cls, detections_forklift, currentClass, class_names )
            
                
        
        cv2.imshow("Image", img)    # Show images
        torch.mps.empty_cache()
        cv2.waitKey(1)



main(address='venv/YOLO_basics/forklift2.mp4', video_mode="MP4")

for key, value in forklift_coordinates.items():
    print(f"{key}: {value}")