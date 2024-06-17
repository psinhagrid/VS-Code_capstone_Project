from ultralytics import YOLO
import cv2
import base64
from PIL import Image
import base64
import cvzone
import torch
from sort import *
from API_call import *
from Utils import *



###############################################################################################



model = YOLO('fine_tuned_weights.pt')


## Yolo class names 
className = [
    'Hardhat', 'Mask', 'NO_Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', "Safety Cone", 'Safety Vest', 
    'machinery', 'vehicle', 'person' ]



safety_vest_people_count = 0
no_safety_vest_people_count = 0
violation_score = 0

# Frame Counter initialized
frame_number = 0

# Voilations dictionary initialized
voilation_dict = {}

# violators count:
violators_count = 0 

# Global file path base 
output_file_base = "venv/YOLO_basics/output_json/output"

violator_ID = []
###############################################################################################

"""       UTILS         """

def raise_flag(category: str, img, event_type: str, timestamp: str, frame: str , 
                  location: Dict[str, int], confidence: int, employee_id: str, violation_type: str, violation_count:int, severity_level: str, 
                  metadata: Dict[str, str], output_file: str):
    
    print ("Flag_raised")
    global violators_count
    violators_count += 1

    global violator_ID
    violator_ID.append(employee_id)
    
    # Calculate width and height
    x1, y1, x2, y2 = location["x1"], location["y1"], location["x2"], location["y2"]
    w = x2 - x1
    h = y2 - y1

    # Draw the rectangle on the image
    img_copy = img.copy()  # Create a copy of the original image
    img_copy = cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255),5)     # Making rectangle


    #image_encoded = compress_image_to_base64(img_copy, quality=20)
    description = make_description(location, confidence, employee_id, violation_type)
    generate_json(category, description,  event_type, timestamp, frame, 
                  location, confidence, employee_id, violation_type, severity_level, 
                  metadata, output_file)
    






def anomaly_detector(img, box, x1: int, y1: int, x2: int, y2: int , Id: int, currentClass: str, conf: int):

    global current_count
    global voilation_dict
    global frame_number
    global output_file_base
    

    if (currentClass in ('NO-Safety Vest', 'NO_Hardhat', 'NO-Mask')):

        global violation_count
        print (violators_count) 
        if (Id not in voilation_dict.keys() and Id != None):
            voilation_dict[Id] = [frame_number, 1, conf]
            generate_json("Non-Alert", " ",  "PPE Violation", "2024-07-01T14:23:45Z", frame_number, {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, voilation_dict[Id][2], Id, currentClass, "high", {"camera_id": "CAM01", "location": "Warehouse Section A", "environmental_conditions": "Normal"},  f"{output_file_base}_{frame_number}.json")

        else :

            if (voilation_dict[Id][1] == -999):
                generate_json("Non-Alert", " ",  "PPE Violation", "2024-07-01T14:23:45Z", frame_number, {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, voilation_dict[Id][2], Id, currentClass, "high", {"camera_id": "CAM01", "location": "Warehouse Section A", "environmental_conditions": "Normal"},  f"{output_file_base}_{frame_number}.json")
                #print ("\n\n PASSED")
                


            elif (voilation_dict[Id][0] in range (frame_number-10, frame_number+10)):
            
                print ("\nDICT_value : ", voilation_dict[Id][1])
                print ("\n")
                if (voilation_dict[Id][1] == 50):
                    voilation_dict[Id][1] = -999
                    #print ("\n\n ENTERED FRAME ")
                    #print (voilation_dict[1][1])
                    raise_flag(
                    category = "Alert",
                    img = img,
                    event_type="PPE Violation",
                    timestamp="2024-07-01T14:23:45Z", 
                    frame=frame_number, 
                    location={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    confidence = voilation_dict[Id][2],
                    employee_id=Id,
                    violation_type=currentClass,
                    violation_count = violators_count,
                    severity_level="high",
                    metadata={"camera_id": "CAM01", "location": "Warehouse Section A", "environmental_conditions": "Normal"},
                    output_file = f"{output_file_base}_{frame_number}.json"
                    )
   

                else :
                    voilation_dict[Id][0] = frame_number
                    voilation_dict[Id][1] += 1
                    voilation_dict[Id][2] = max(voilation_dict[Id][2], conf)
                    generate_json("Non-Alert", "",  "PPE Violation", "2024-07-01T14:23:45Z", frame_number, {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, voilation_dict[Id][2], Id, currentClass, "high", {"camera_id": "CAM01", "location": "Warehouse Section A", "environmental_conditions": "Normal"},  f"{output_file_base}_{frame_number}.json")

                    #print ("\n Frame Number", frame_number)
                    #print (voilation_dict[1][1])

            elif (voilation_dict[Id][0] not in range (frame_number-10, frame_number+10)):
                #quit()
                voilation_dict[Id][0] = frame_number
                voilation_dict[Id][1] = 1
                voilation_dict[Id][2] = conf
                generate_json("Non-Alert", "",  "PPE Violation", "2024-07-01T14:23:45Z", frame_number, {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, voilation_dict[Id][2], Id, currentClass, "high", {"camera_id": "CAM01", "location": "Warehouse Section A", "environmental_conditions": "Normal"},  f"{output_file_base}_{frame_number}.json")
                
                #print ("\n Frame Number", frame_number)
                

            else :

                print ("UNKNOWN VIOLATION")
                quit()
                
        return f"{output_file_base}_{frame_number}.json" 



def object_ID(img, box, cls: int, result_tracker, current_Class: str, class_names: List ,conf: int):

    """
        Function will make a rectangle against selected class and will also display the ID for tracking
    
    """


    for results in result_tracker:
        x1, y1, x2, y2, Id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        

        cvzone.putTextRect(img, f"ID - {int(Id)}", (max(x2 - w - 10, 0),max(y2 - 10, h) ), 1.5, 2)

        x_center = x1+w//2
        y_center = y1+h//2


        output_json_path = anomaly_detector(img, box, x1, y1, x2, y2, Id, current_Class, conf)
        return output_json_path

            







def class_to_track(img, box, cls: int, detections, current_class: str, class_names: List ):

    """
        This function will make boxes and assign IDs to given classes. 
        It will assign to all IDs to all detected objects if class_names is empty

        Return Type : Detections, Center x coordinate, Center y coordinate
    
    """
    if (current_class == 'Safety Vest'):
        
        global safety_vest_people_count
        safety_vest_people_count += 1

    if (current_class == 'NO-Safety Vest'):
        global no_safety_vest_people_count
        no_safety_vest_people_count += 1


    if (current_class in [ 'NO-Safety Vest','NO_Hardhat' ]):
        


        x1,y1,x2,y2,conf = bounding_box(box,img, show_box_for_all=True)
        
        current_array = np.array([x1,y1,x2,y2,conf])        
        detections = np.vstack((detections, current_array))     # Giving labels

        cvzone.putTextRect(img,f'{className[cls]} {conf}', (max(x1, 0), max(35, y1-10)), 2, 2)
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)     # Making rectangle

        resultTracker = tracker.update(detections)

        print ("Frame Number : ", frame_number)
        #print (((safety_vest_people_count/(safety_vest_people_count+no_safety_vest_people_count)))*100)

        output_json_path = object_ID(img, box, cls, resultTracker, current_class, class_names ,conf)

    else :
        generate_json("Non-Alert", " ",  "PPE Violation", "2024-07-01T14:23:45Z", frame_number, {"x1": 0, "y1": 0, "x2": 0, "y2": 0}, None, None, None, "high", {"camera_id": "CAM01", "location": "Warehouse Section A", "environmental_conditions": "Normal"},  f"{output_file_base}_{violators_count}.json")
        output_json_path = f"{output_file_base}_{frame_number}.json"


    return detections,output_json_path









###############################################################################################

"""   Program to capture video from the device.     """

##    cv2.waitKey(0) VS cv2.waitKey(1) difference is 1 will continue execution after 1ms delay and 0 will wait till key given

address1 = 'venv/YOLO_basics/helmet.mp4'
# address2 = 'venv/YOLO_basics/helmet2.mp4'
# address3 = 'venv/YOLO_basics/helmet3.mp4'
# address4 = 'venv/YOLO_basics/helmet4.mp4'

address = address1

# Available modes "LIVE" and "MP4"
video_mode = "MP4"


# Set if object counter is needed. Options are True and False
object_counter_requirement = True


# Set required class list
#class_names = ['Hardhat', 'Mask', 'NO_Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', "Safety Cone",'Safety Vest', 'machinery', 'vehicle']

class_names = [ 'NO-Safety Vest','NO_Hardhat' ]





###############################################################################################

# Tracking
tracker = Sort(max_age=200, min_hits=50, iou_threshold=0.5)   # Used for tracking of cars

###############################################################################################

def main(address: str, video_mode: str, object_counter_requirement: bool, class_names: List):
    if video_mode != "PNG":
        print("Unsupported video mode. Only 'PNG' mode is supported.")
        return

    global frame_number
    frame_number += 1

    # For processing a single frame in PNG format
    img = cv2.imread(address)
    if img is None:
        print("Failed to read the image")
        return

    result = model(img, device="mps", stream=True)  # Use mps and stream feature
    detections = np.empty((0, 5))


    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2, conf = bounding_box(box, img, show_box_for_all=True)

            # Class Name Display
            cls = int(box.cls[0])
            currentClass = class_names[cls]

            # Call detections with args detections, Current Class, Interested Classes
            detections,output_json_path = class_to_track(img, box, cls, detections, currentClass, class_names)

    print(no_safety_vest_people_count)
    print(safety_vest_people_count)
    print("\nsafety_score")
    print(((safety_vest_people_count / (safety_vest_people_count + no_safety_vest_people_count)) * 100))

    cv2.imshow("Image", img)  # Show images
    torch.mps.empty_cache()
    cv2.waitKey(1)

    # Save the image with the detection box to a local folder
    output_image_path = f"venv/YOLO_basics/output_images/frame_{frame_number}.png"
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)  # Ensure the directory exists
    cv2.imwrite(output_image_path, img)  # Save the image

    print(violators_count)
    print(violator_ID)

    return output_image_path, output_json_path



def process_video(video_path: str, object_counter_requirement: bool, class_names: List):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    frame_number = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("No more frames to capture")
            break  # Exit the loop if reading fails

        # Save the current frame as a temporary PNG file
        temp_frame_path = "venv/output_frames/temp_frame.png"
        cv2.imwrite(temp_frame_path, frame)

        # Call the main function with the temporary PNG frame
        print (main(temp_frame_path, "PNG", object_counter_requirement, class_names))

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

# Example usage:
process_video(address, object_counter_requirement=True, class_names=className)
