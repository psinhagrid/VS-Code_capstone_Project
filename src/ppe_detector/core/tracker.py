"""Violation tracking module"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from ..models.violation import Violation, ViolationEvent
from ..utils.logger import Logger


class ViolationTracker:
    """Tracks PPE violations across video frames"""
    
    def __init__(
        self,
        violation_frame_threshold: int = 50,
        frame_window: int = 10,
        logger: Optional[Logger] = None
    ):
        """
        Initialize violation tracker
        
        Args:
            violation_frame_threshold: Number of frames before raising alert
            frame_window: Frame tolerance window for tracking
            logger: Logger instance
        """
        self.violation_frame_threshold = violation_frame_threshold
        self.frame_window = frame_window
        self.violations: Dict[int, Violation] = {}
        self.alerted_violations: set = set()
        self.logger = logger or Logger()
        
    def update_violation(
        self,
        employee_id: int,
        violation_type: str,
        frame_number: int,
        confidence: float,
        location: Dict[str, int]
    ) -> Optional[ViolationEvent]:
        """
        Update or create violation record
        
        Args:
            employee_id: Tracked person ID
            violation_type: Type of violation detected
            frame_number: Current frame number
            confidence: Detection confidence
            location: Bounding box coordinates
            
        Returns:
            ViolationEvent if alert should be raised, None otherwise
        """
        violation_id = employee_id
        
        if violation_id not in self.violations:
            self.violations[violation_id] = Violation(
                employee_id=employee_id,
                violation_type=violation_type,
                frame_number=frame_number,
                confidence=confidence,
                location=location,
                first_detected_frame=frame_number,
                violation_count=1,
                max_confidence=confidence
            )
            self.logger.debug(
                f"New violation detected: ID={employee_id}, "
                f"Type={violation_type}, Frame={frame_number}"
            )
            return None
        
        violation = self.violations[violation_id]
        
        if violation.violation_count == -999:
            return None
        
        if violation.is_persistent(frame_number, self.frame_window):
            violation.update(frame_number, confidence)
            violation.location = location
            
            if (violation.should_alert(self.violation_frame_threshold) and 
                violation_id not in self.alerted_violations):
                
                violation.violation_count = -999
                self.alerted_violations.add(violation_id)
                
                self.logger.warning(
                    f"ALERT: Persistent violation detected - "
                    f"ID={employee_id}, Type={violation_type}, "
                    f"Frames={violation.violation_count}"
                )
                
                return ViolationEvent.from_violation(violation, category="Alert")
        else:
            violation.frame_number = frame_number
            violation.violation_count = 1
            violation.confidence = confidence
            violation.location = location
        
        return None
    
    def get_violation_stats(self) -> Dict:
        """Get statistics about tracked violations"""
        active_violations = sum(
            1 for v in self.violations.values() 
            if v.violation_count != -999
        )
        
        return {
            "total_tracked": len(self.violations),
            "active_violations": active_violations,
            "alerted_violations": len(self.alerted_violations),
            "violation_types": self._count_violation_types()
        }
    
    def _count_violation_types(self) -> Dict[str, int]:
        """Count violations by type"""
        counts = {}
        for violation in self.violations.values():
            violation_type = violation.violation_type
            counts[violation_type] = counts.get(violation_type, 0) + 1
        return counts
    
    def reset(self):
        """Reset tracker state"""
        self.violations.clear()
        self.alerted_violations.clear()
        self.logger.info("Violation tracker reset")
