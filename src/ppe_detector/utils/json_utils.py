"""JSON generation and handling utilities"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class JSONGenerator:
    """Handles JSON file generation for violations and events"""
    
    @staticmethod
    def create_violation_description(
        location: Dict[str, int],
        confidence: float,
        employee_id: int,
        violation_type: str,
        image_width: int = 1280
    ) -> Dict[str, str]:
        """
        Create human-readable description of violation
        
        Args:
            location: Bounding box coordinates
            confidence: Detection confidence
            employee_id: Tracked person ID
            violation_type: Type of violation
            image_width: Image width for position calculation
            
        Returns:
            Dictionary with description lines
        """
        x1 = location.get("x1", 0)
        x2 = location.get("x2", 0)
        center_x = (x1 + x2) / 2
        
        position = "left" if center_x < image_width / 2 else "right"
        confidence_percent = round(confidence * 100, 2)
        
        return {
            "line1": f"A {violation_type} violation has been identified.",
            "line2": f"Detection confidence: {confidence_percent}%",
            "line3": f"Person located on the {position} side of the frame.",
            "line4": f"Assigned tracking ID: {employee_id}"
        }
    
    @staticmethod
    def generate_event_json(
        category: str,
        event_type: str,
        frame_number: int,
        location: Dict[str, int],
        confidence: Optional[float] = None,
        employee_id: Optional[int] = None,
        violation_type: Optional[str] = None,
        severity_level: str = "medium",
        metadata: Optional[Dict[str, str]] = None,
        description: Optional[Dict[str, str]] = None,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate event JSON data
        
        Args:
            category: Event category (Alert, Non-Alert)
            event_type: Type of event
            frame_number: Video frame number
            location: Bounding box coordinates
            confidence: Detection confidence
            employee_id: Tracked person ID
            violation_type: Type of violation
            severity_level: Severity (low, medium, high)
            metadata: Additional metadata
            description: Human-readable description
            output_file: Path to save JSON file
            
        Returns:
            Event data dictionary
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        data = {
            "category": category,
            "event_type": event_type,
            "timestamp": timestamp,
            "frame": frame_number,
            "location": location,
            "severity_level": severity_level,
        }
        
        if confidence is not None:
            data["confidence"] = round(confidence, 4)
        
        if employee_id is not None:
            data["employee_id"] = int(employee_id)
        
        if violation_type:
            data["violation_type"] = violation_type
        
        if description:
            data["description"] = description
        
        if metadata:
            data["metadata"] = metadata
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)
        
        return data
    
    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """
        Load JSON file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            JSON data as dictionary
        """
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_json(data: Dict[str, Any], file_path: str):
        """
        Save data to JSON file
        
        Args:
            data: Data to save
            file_path: Output file path
        """
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
