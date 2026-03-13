"""Example: Consume video frames from Kafka and process them"""

import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ppe_detector.utils.config_loader import ConfigLoader
from src.ppe_detector.utils.logger import get_logger
from src.ppe_detector.services.kafka_service import KafkaConsumerService
from src.ppe_detector.core.detector import PPEDetector


def main():
    """Kafka consumer example"""
    
    config_loader = ConfigLoader()
    config = config_loader.get_all()
    logger = get_logger("kafka_consumer_example", config)
    
    kafka_config = config_loader.kafka_config
    
    if not kafka_config.get('enabled', False):
        logger.warning("Kafka is not enabled in configuration")
        response = input("Do you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    logger.info("Initializing PPE Detector")
    
    project_root = Path(__file__).parent.parent
    weights_path = project_root / config_loader.get('model.weights_path')
    
    detector = PPEDetector(
        weights_path=str(weights_path),
        class_names=config_loader.get('classes.all_classes'),
        violation_classes=config_loader.get('classes.violation_classes'),
        device=config_loader.get('model.device', 'cpu'),
        confidence_threshold=config_loader.get('model.confidence_threshold', 0.5),
        tracker_config=config_loader.tracking_config,
        logger=logger
    )
    
    logger.info("Initializing Kafka Consumer")
    
    consumer = KafkaConsumerService(
        bootstrap_servers=kafka_config.get('bootstrap_servers', 'localhost:9092'),
        topic_name=kafka_config.get('topic_name', 'ppe-violations'),
        group_id=kafka_config.get('group_id', 'ppe-detector-group'),
        config=kafka_config.get('consumer', {}),
        logger=logger
    )
    
    frames_processed = 0
    
    def frame_callback(frame, frame_number, metadata):
        """Process each frame received from Kafka"""
        nonlocal frames_processed
        
        annotated_frame, detections, violation_events = detector.detect(
            frame,
            frame_number=frame_number,
            draw_boxes=True
        )
        
        for event in violation_events:
            logger.warning(
                f"VIOLATION ALERT: Frame {frame_number}, "
                f"ID {event.employee_id}, Type {event.violation_type}"
            )
        
        cv2.imshow('PPE Detection - Kafka Stream', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            consumer.stop()
        
        frames_processed += 1
        
        if frames_processed % 10 == 0:
            stats = detector.get_stats()
            logger.info(
                f"Processed {frames_processed} frames, "
                f"{stats['violations_detected']} violations detected"
            )
    
    try:
        logger.info("Starting frame consumption from Kafka")
        consumer.consume_frames(frame_callback, timeout=1.0)
    
    except KeyboardInterrupt:
        logger.info("Consumer interrupted by user")
    
    finally:
        cv2.destroyAllWindows()
        consumer.close()
        logger.info(f"Total frames processed: {frames_processed}")
        logger.info(f"Final statistics: {detector.get_stats()}")


if __name__ == "__main__":
    main()
