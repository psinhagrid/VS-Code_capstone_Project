"""Tests for PPE Detector"""

import pytest
import numpy as np
from pathlib import Path


class TestPPEDetector:
    """Test PPE Detector functionality"""
    
    def test_detector_initialization(self):
        """Test detector can be initialized"""
        # This is a placeholder test
        # Actual tests would require model weights
        assert True
    
    def test_image_processing(self):
        """Test image processing utilities"""
        from src.ppe_detector.utils.image_utils import ImageProcessor
        
        # Create a dummy image
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Test base64 encoding
        processor = ImageProcessor()
        encoded = processor.compress_image_to_base64(dummy_image, quality=50)
        
        assert isinstance(encoded, str)
        assert len(encoded) > 0
        
        # Test decoding
        decoded = processor.decode_base64_to_image(encoded)
        assert decoded.shape == dummy_image.shape


class TestConfigLoader:
    """Test configuration loading"""
    
    def test_config_loading(self):
        """Test config can be loaded"""
        from src.ppe_detector.utils.config_loader import ConfigLoader
        
        # This will fail if config.yaml doesn't exist
        config = ConfigLoader()
        assert config.get_all() is not None
        assert 'model' in config.get_all()


class TestJSONGenerator:
    """Test JSON generation"""
    
    def test_json_generation(self):
        """Test JSON can be generated"""
        from src.ppe_detector.utils.json_utils import JSONGenerator
        
        generator = JSONGenerator()
        
        description = generator.create_violation_description(
            location={"x1": 100, "y1": 150, "x2": 300, "y2": 450},
            confidence=0.95,
            employee_id=5,
            violation_type="NO-Safety Vest",
            image_width=1280
        )
        
        assert isinstance(description, dict)
        assert 'line1' in description
        assert 'violation' in description['line1'].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
