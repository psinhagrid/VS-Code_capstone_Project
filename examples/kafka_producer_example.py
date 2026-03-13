"""Example: Stream video frames to Kafka"""

import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ppe_detector.utils.config_loader import ConfigLoader
from src.ppe_detector.utils.logger import get_logger
from src.ppe_detector.services.kafka_service import KafkaProducerService


def main():
    """Kafka producer example"""
    
    config_loader = ConfigLoader()
    config = config_loader.get_all()
    logger = get_logger("kafka_producer_example", config)
    
    kafka_config = config_loader.kafka_config
    
    if not kafka_config.get('enabled', False):
        logger.warning("Kafka is not enabled in configuration")
        response = input("Do you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    logger.info("Initializing Kafka Producer")
    
    producer = KafkaProducerService(
        bootstrap_servers=kafka_config.get('bootstrap_servers', 'localhost:9092'),
        topic_name=kafka_config.get('topic_name', 'ppe-violations'),
        config=kafka_config.get('producer', {}),
        logger=logger
    )
    
    video_path = input("Enter path to video file: ").strip()
    
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return
    
    frame_number = 0
    sent_count = 0
    
    try:
        logger.info(f"Streaming video to Kafka topic: {producer.topic_name}")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                logger.info("End of video reached")
                break
            
            frame_number += 1
            
            if frame_number % 5 != 0:
                continue
            
            metadata = {
                'source': video_path,
                'frame_number': frame_number
            }
            
            if producer.send_frame(frame, frame_number, metadata):
                sent_count += 1
                
                if sent_count % 10 == 0:
                    logger.info(f"Sent {sent_count} frames to Kafka")
    
    except KeyboardInterrupt:
        logger.info("Streaming interrupted by user")
    
    finally:
        cap.release()
        producer.close()
        logger.info(f"Total frames sent: {sent_count}")


if __name__ == "__main__":
    main()
