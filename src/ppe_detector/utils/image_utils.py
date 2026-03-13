"""Image processing utilities"""

import cv2
import base64
import numpy as np
from PIL import Image
from typing import Tuple, Optional


class ImageProcessor:
    """Handles image processing operations"""
    
    @staticmethod
    def compress_image_to_base64(image: np.ndarray, quality: int = 20) -> str:
        """
        Compress an image and encode it to Base64
        
        Args:
            image: Image as numpy array
            quality: JPEG quality (1-95)
            
        Returns:
            Base64 encoded string
        """
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        _, buffer = cv2.imencode(
            '.jpg', 
            cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        return base64.b64encode(buffer).decode('utf-8')
    
    @staticmethod
    def decode_base64_to_image(base64_string: str) -> np.ndarray:
        """
        Decode Base64 string to image
        
        Args:
            base64_string: Base64 encoded image string
            
        Returns:
            Image as numpy array
        """
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    @staticmethod
    def draw_bounding_box(
        image: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        label: str = "",
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        draw_label: bool = True
    ) -> np.ndarray:
        """
        Draw bounding box on image
        
        Args:
            image: Image as numpy array
            x1, y1, x2, y2: Bounding box coordinates
            label: Text label for the box
            color: Box color in BGR
            thickness: Line thickness
            draw_label: Whether to draw label text
            
        Returns:
            Image with bounding box drawn
        """
        img_copy = image.copy()
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        
        if draw_label and label:
            font_scale = 0.6
            font_thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            cv2.rectangle(
                img_copy,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                color,
                -1
            )
            
            cv2.putText(
                img_copy,
                label,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
        
        return img_copy
    
    @staticmethod
    def get_bounding_box_from_detection(box) -> Tuple[int, int, int, int, float]:
        """
        Extract bounding box coordinates and confidence from YOLO detection
        
        Args:
            box: YOLO detection box object
            
        Returns:
            Tuple of (x1, y1, x2, y2, confidence)
        """
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = float(box.conf[0])
        return x1, y1, x2, y2, conf
    
    @staticmethod
    def calculate_position(x1: int, x2: int, image_width: int = 1280) -> str:
        """
        Calculate position of bounding box in image (left/right/center)
        
        Args:
            x1: Left x coordinate
            x2: Right x coordinate
            image_width: Total image width
            
        Returns:
            Position string: 'left', 'center', or 'right'
        """
        center_x = (x1 + x2) / 2
        
        if center_x < image_width / 3:
            return "left"
        elif center_x > 2 * image_width / 3:
            return "right"
        else:
            return "center"
