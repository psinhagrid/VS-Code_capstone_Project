"""Data models for violations and events"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime


@dataclass
class Violation:
    """Represents a PPE violation detection"""
    
    employee_id: int
    violation_type: str
    frame_number: int
    confidence: float
    location: Dict[str, int]
    first_detected_frame: int
    violation_count: int = 1
    max_confidence: float = 0.0
    
    def __post_init__(self):
        """Initialize max confidence"""
        if self.max_confidence == 0.0:
            self.max_confidence = self.confidence
    
    def update(self, frame_number: int, confidence: float):
        """Update violation with new detection"""
        self.frame_number = frame_number
        self.violation_count += 1
        self.max_confidence = max(self.max_confidence, confidence)
        self.confidence = confidence
    
    def is_persistent(self, current_frame: int, frame_window: int = 10) -> bool:
        """Check if violation is still active within frame window"""
        return self.frame_number in range(
            current_frame - frame_window,
            current_frame + frame_window
        )
    
    def should_alert(self, threshold: int = 50) -> bool:
        """Check if violation count exceeds alert threshold"""
        return self.violation_count >= threshold


@dataclass
class ViolationEvent:
    """Represents a violation event for logging/alerting"""
    
    category: str
    event_type: str
    timestamp: str
    frame_number: int
    location: Dict[str, int]
    severity_level: str
    confidence: Optional[float] = None
    employee_id: Optional[int] = None
    violation_type: Optional[str] = None
    description: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, str]] = None
    
    @classmethod
    def from_violation(
        cls,
        violation: Violation,
        category: str = "Alert",
        metadata: Optional[Dict[str, str]] = None
    ) -> "ViolationEvent":
        """Create ViolationEvent from Violation object"""
        return cls(
            category=category,
            event_type="PPE Violation",
            timestamp=datetime.utcnow().isoformat() + "Z",
            frame_number=violation.frame_number,
            location=violation.location,
            confidence=violation.max_confidence,
            employee_id=violation.employee_id,
            violation_type=violation.violation_type,
            severity_level="high" if violation.violation_type in ["NO_Hardhat", "NO-Mask"] else "medium",
            metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = {
            "category": self.category,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "frame": self.frame_number,
            "location": self.location,
            "severity_level": self.severity_level,
        }
        
        if self.confidence is not None:
            data["confidence"] = round(self.confidence, 4)
        
        if self.employee_id is not None:
            data["employee_id"] = self.employee_id
        
        if self.violation_type:
            data["violation_type"] = self.violation_type
        
        if self.description:
            data["description"] = self.description
        
        if self.metadata:
            data["metadata"] = self.metadata
        
        return data
