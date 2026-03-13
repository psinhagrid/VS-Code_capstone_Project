"""PPE Detection module using YOLOv8"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
import cvzone
import torch
from ultralytics import YOLO

from ..utils.logger import Logger
from ..utils.image_utils import ImageProcessor
from .sort_tracker import Sort
from .tracker import ViolationTracker


class PPEDetector:
    """Personal Protective Equipment Detection System"""
    
    def __init__(
        self,
        weights_path: str,
        class_names: List[str],
        violation_classes: List[str],
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        tracker_config: Optional[Dict] = None,
        logger: Optional[Logger] = None
    ):
        """
        Initialize PPE Detector
        
        Args:
            weights_path: Path to YOLO model weights
            class_names: List of all class names
            violation_classes: List of violation class names
            device: Device to run inference on (cpu, cuda, mps)
            confidence_threshold: Minimum confidence for detections
            tracker_config: Configuration for object tracker
            logger: Logger instance
        """
        self.logger = logger or Logger()
        self.class_names = class_names
        self.violation_classes = violation_classes
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        self.logger.info(f"Loading YOLO model from {weights_path}")
        weights_path = Path(weights_path)
        
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        
        self.model = YOLO(str(weights_path))
        self.logger.info(f"Model loaded successfully on device: {device}")
        
        tracker_config = tracker_config or {}
        self.sort_tracker = Sort(
            max_age=tracker_config.get('max_age', 200),
            min_hits=tracker_config.get('min_hits', 50),
            iou_threshold=tracker_config.get('iou_threshold', 0.5)
        )
        
        self.violation_tracker = ViolationTracker(
            violation_frame_threshold=tracker_config.get('violation_frame_threshold', 50),
            frame_window=tracker_config.get('frame_window', 10),
            logger=self.logger
        )
        
        self.image_processor = ImageProcessor()
        
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'violations_detected': 0,
            'alerts_raised': 0
        }
    
    def detect(
        self,
        image: np.ndarray,
        frame_number: int,
        draw_boxes: bool = True
    ) -> Tuple[np.ndarray, List[Dict], List[Dict]]:
        """
        Detect PPE violations in an image
        
        Args:
            image: Input image as numpy array
            frame_number: Current frame number
            draw_boxes: Whether to draw bounding boxes on image
            
        Returns:
            Tuple of (annotated_image, detections, violation_events)
        """
        self.stats['frames_processed'] += 1
        
        results = self.model(image, device=self.device, stream=True, verbose=False)
        
        detections = np.empty((0, 5))
        all_detections = []
        violation_events = []
        
        annotated_image = image.copy()
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2, conf = self.image_processor.get_bounding_box_from_detection(box)
                
                if conf < self.confidence_threshold:
                    continue
                
                cls_id = int(box.cls[0])
                
                if cls_id >= len(self.class_names):
                    self.logger.warning(f"Class ID {cls_id} out of range")
                    continue
                
                class_name = self.class_names[cls_id]
                
                detection = {
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                    'frame_number': frame_number
                }
                all_detections.append(detection)
                self.stats['total_detections'] += 1
                
                if class_name in self.violation_classes:
                    self.stats['violations_detected'] += 1
                    
                    current_array = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, current_array))
                    
                    if draw_boxes:
                        color = (0, 0, 255)
                        label = f"{class_name} {conf:.2f}"
                        annotated_image = self.image_processor.draw_bounding_box(
                            annotated_image, x1, y1, x2, y2, label, color, thickness=3
                        )
        
        if len(detections) > 0:
            tracked_objects = self.sort_tracker.update(detections)
            
            for track in tracked_objects:
                x1, y1, x2, y2, track_id = track
                x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
                
                if draw_boxes:
                    cvzone.putTextRect(
                        annotated_image,
                        f"ID - {track_id}",
                        (max(x2 - (x2 - x1) - 10, 0), max(y2 - 10, y2 - y1)),
                        scale=1.5,
                        thickness=2
                    )
                
                for detection in all_detections:
                    bbox = detection['bbox']
                    if (abs(bbox['x1'] - x1) < 10 and abs(bbox['y1'] - y1) < 10):
                        
                        violation_event = self.violation_tracker.update_violation(
                            employee_id=track_id,
                            violation_type=detection['class_name'],
                            frame_number=frame_number,
                            confidence=detection['confidence'],
                            location=bbox
                        )
                        
                        if violation_event:
                            violation_events.append(violation_event)
                            self.stats['alerts_raised'] += 1
                            
                            if draw_boxes:
                                annotated_image = self.image_processor.draw_bounding_box(
                                    annotated_image,
                                    bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'],
                                    f"ALERT: {detection['class_name']}",
                                    color=(0, 0, 255),
                                    thickness=5
                                )
                        break
        
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        
        return annotated_image, all_detections, violation_events
    
    def get_stats(self) -> Dict:
        """Get detection statistics"""
        return {
            **self.stats,
            'violation_tracker_stats': self.violation_tracker.get_violation_stats()
        }
    
    def reset_stats(self):
        """Reset all statistics"""
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'violations_detected': 0,
            'alerts_raised': 0
        }
        self.violation_tracker.reset()
        self.logger.info("Statistics reset")
