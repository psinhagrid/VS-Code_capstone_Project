"""Video processing service"""

import cv2
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import numpy as np

from ..core.detector import PPEDetector
from ..utils.logger import Logger
from ..utils.json_utils import JSONGenerator


class VideoProcessor:
    """Processes video files for PPE detection"""
    
    def __init__(
        self,
        detector: PPEDetector,
        output_config: Dict[str, str],
        logger: Optional[Logger] = None
    ):
        """
        Initialize video processor
        
        Args:
            detector: PPEDetector instance
            output_config: Output configuration (directories for JSON, images, etc.)
            logger: Logger instance
        """
        self.detector = detector
        self.output_config = output_config
        self.logger = logger or Logger()
        self.json_generator = JSONGenerator()
        
        self._setup_output_directories()
    
    def _setup_output_directories(self):
        """Create output directories if they don't exist"""
        for key, path in self.output_config.items():
            if key.endswith('_dir'):
                Path(path).mkdir(parents=True, exist_ok=True)
    
    def process_video(
        self,
        video_path: str,
        display_video: bool = True,
        save_frames: bool = True,
        save_output_video: bool = False,
        output_video_path: Optional[str] = None,
        frame_skip: int = 1,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Process video file for PPE violations
        
        Args:
            video_path: Path to input video file
            display_video: Whether to display video while processing
            save_frames: Whether to save annotated frames
            save_output_video: Whether to save output video
            output_video_path: Path for output video (if save_output_video=True)
            frame_skip: Process every nth frame (1 = all frames)
            max_frames: Maximum number of frames to process
            progress_callback: Callback function(current_frame, total_frames)
            
        Returns:
            Dictionary with processing results
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.logger.info(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.logger.info(
            f"Video info: {total_frames} frames, {fps} FPS, {width}x{height}"
        )
        
        video_writer = None
        if save_output_video:
            if output_video_path is None:
                output_video_path = Path(self.output_config['video_dir']) / f"output_{video_path.stem}.mp4"
            else:
                output_video_path = Path(output_video_path)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_video_path),
                fourcc,
                fps,
                (width, height)
            )
            self.logger.info(f"Saving output video to: {output_video_path}")
        
        frame_number = 0
        processed_frames = 0
        all_violation_events = []
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    self.logger.info("No more frames to read")
                    break
                
                frame_number += 1
                
                if frame_skip > 1 and frame_number % frame_skip != 0:
                    continue
                
                if max_frames and processed_frames >= max_frames:
                    self.logger.info(f"Reached max frames limit: {max_frames}")
                    break
                
                processed_frames += 1
                
                annotated_frame, detections, violation_events = self.detector.detect(
                    frame,
                    frame_number=frame_number,
                    draw_boxes=True
                )
                
                for event in violation_events:
                    all_violation_events.append(event)
                    
                    json_file = Path(self.output_config['json_dir']) / f"alert_frame_{frame_number}.json"
                    self.json_generator.save_json(event.to_dict(), str(json_file))
                    
                    self.logger.warning(
                        f"Alert saved: Frame {frame_number}, "
                        f"ID {event.employee_id}, Type {event.violation_type}"
                    )
                
                if len(detections) > 0:
                    json_file = Path(self.output_config['json_dir']) / f"frame_{frame_number}.json"
                    frame_data = {
                        'frame_number': frame_number,
                        'detections': detections,
                        'violation_count': len([d for d in detections if d['class_name'] in self.detector.violation_classes])
                    }
                    self.json_generator.save_json(frame_data, str(json_file))
                
                if save_frames:
                    frame_path = Path(self.output_config['image_dir']) / f"frame_{frame_number:06d}.jpg"
                    cv2.imwrite(str(frame_path), annotated_frame)
                
                if video_writer:
                    video_writer.write(annotated_frame)
                
                if display_video:
                    cv2.imshow('PPE Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.logger.info("Processing interrupted by user")
                        break
                
                if progress_callback:
                    progress_callback(processed_frames, total_frames // frame_skip)
                
                if processed_frames % 100 == 0:
                    stats = self.detector.get_stats()
                    self.logger.info(
                        f"Progress: {processed_frames} frames, "
                        f"{stats['violations_detected']} violations, "
                        f"{stats['alerts_raised']} alerts"
                    )
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            if display_video:
                cv2.destroyAllWindows()
        
        final_stats = self.detector.get_stats()
        
        results = {
            'video_path': str(video_path),
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'alerts_raised': len(all_violation_events),
            'statistics': final_stats,
            'violation_events': [event.to_dict() for event in all_violation_events]
        }
        
        summary_file = Path(self.output_config['json_dir']) / f"summary_{video_path.stem}.json"
        self.json_generator.save_json(results, str(summary_file))
        
        self.logger.info(
            f"Processing complete: {processed_frames} frames processed, "
            f"{len(all_violation_events)} alerts raised"
        )
        
        return results
    
    def process_image(
        self,
        image_path: str,
        save_output: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single image
        
        Args:
            image_path: Path to input image
            save_output: Whether to save annotated image
            
        Returns:
            Dictionary with detection results
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        self.logger.info(f"Processing image: {image_path}")
        
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        
        annotated_image, detections, violation_events = self.detector.detect(
            image,
            frame_number=1,
            draw_boxes=True
        )
        
        if save_output:
            output_path = Path(self.output_config['image_dir']) / f"output_{image_path.name}"
            cv2.imwrite(str(output_path), annotated_image)
            self.logger.info(f"Saved annotated image to: {output_path}")
        
        results = {
            'image_path': str(image_path),
            'detections': detections,
            'violation_events': [event.to_dict() for event in violation_events]
        }
        
        return results
