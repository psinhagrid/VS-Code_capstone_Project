"""Kafka integration service for streaming video frames and violations"""

import json
import time
from typing import Optional, Callable, Dict, Any
import cv2
import numpy as np

try:
    from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

from ..utils.logger import Logger


class KafkaProducerService:
    """Kafka producer for sending video frames and violation alerts"""
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic_name: str = "ppe-violations",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[Logger] = None
    ):
        """
        Initialize Kafka producer
        
        Args:
            bootstrap_servers: Kafka server address
            topic_name: Kafka topic name
            config: Additional Kafka configuration
            logger: Logger instance
        """
        if not KAFKA_AVAILABLE:
            raise ImportError(
                "confluent-kafka is not installed. "
                "Install it with: pip install confluent-kafka"
            )
        
        self.logger = logger or Logger()
        self.topic_name = topic_name
        
        producer_config = {
            'bootstrap.servers': bootstrap_servers,
            'compression.type': 'gzip',
            'batch.size': 16384,
        }
        
        if config:
            producer_config.update(config)
        
        self.producer = Producer(producer_config)
        self.logger.info(f"Kafka producer initialized: {bootstrap_servers}, topic: {topic_name}")
    
    def _delivery_callback(self, err, msg):
        """Callback for message delivery confirmation"""
        if err is not None:
            self.logger.error(f"Message delivery failed: {err}")
        else:
            self.logger.debug(
                f"Message delivered to {msg.topic()} "
                f"partition {msg.partition()} offset {msg.offset()}"
            )
    
    def send_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Send video frame to Kafka
        
        Args:
            frame: Video frame as numpy array
            frame_number: Frame number
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            frame_bytes = buffer.tobytes()
            
            message = {
                'frame_number': frame_number,
                'timestamp': time.time(),
                'frame_data': frame_bytes.hex(),
                'metadata': metadata or {}
            }
            
            json_message = json.dumps(message).encode('utf-8')
            
            self.producer.produce(
                self.topic_name,
                value=json_message,
                callback=self._delivery_callback
            )
            
            self.producer.poll(0)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send frame: {e}")
            return False
    
    def send_violation_event(
        self,
        event_data: Dict[str, Any]
    ) -> bool:
        """
        Send violation event to Kafka
        
        Args:
            event_data: Violation event dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            message = {
                'event_type': 'violation',
                'timestamp': time.time(),
                'data': event_data
            }
            
            json_message = json.dumps(message).encode('utf-8')
            
            self.producer.produce(
                self.topic_name,
                value=json_message,
                callback=self._delivery_callback
            )
            
            self.producer.poll(0)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send violation event: {e}")
            return False
    
    def flush(self, timeout: float = 10.0):
        """
        Flush pending messages
        
        Args:
            timeout: Maximum time to wait (seconds)
        """
        remaining = self.producer.flush(timeout)
        if remaining > 0:
            self.logger.warning(f"{remaining} messages were not delivered")
    
    def close(self):
        """Close producer and flush remaining messages"""
        self.logger.info("Closing Kafka producer")
        self.flush()


class KafkaConsumerService:
    """Kafka consumer for receiving video frames and processing them"""
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic_name: str = "ppe-violations",
        group_id: str = "ppe-detector-group",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[Logger] = None
    ):
        """
        Initialize Kafka consumer
        
        Args:
            bootstrap_servers: Kafka server address
            topic_name: Kafka topic name
            group_id: Consumer group ID
            config: Additional Kafka configuration
            logger: Logger instance
        """
        if not KAFKA_AVAILABLE:
            raise ImportError(
                "confluent-kafka is not installed. "
                "Install it with: pip install confluent-kafka"
            )
        
        self.logger = logger or Logger()
        self.topic_name = topic_name
        self.running = False
        
        consumer_config = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True,
        }
        
        if config:
            consumer_config.update(config)
        
        self.consumer = Consumer(consumer_config)
        self.consumer.subscribe([topic_name])
        self.logger.info(
            f"Kafka consumer initialized: {bootstrap_servers}, "
            f"topic: {topic_name}, group: {group_id}"
        )
    
    def consume_frames(
        self,
        frame_callback: Callable[[np.ndarray, int, Dict], None],
        timeout: float = 1.0,
        max_messages: Optional[int] = None
    ):
        """
        Consume video frames from Kafka
        
        Args:
            frame_callback: Callback function(frame, frame_number, metadata)
            timeout: Poll timeout in seconds
            max_messages: Maximum number of messages to consume
        """
        self.running = True
        messages_consumed = 0
        
        self.logger.info("Starting frame consumption")
        
        try:
            while self.running:
                msg = self.consumer.poll(timeout=timeout)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        raise KafkaException(msg.error())
                
                try:
                    message_data = json.loads(msg.value().decode('utf-8'))
                    
                    frame_number = message_data.get('frame_number', 0)
                    frame_hex = message_data.get('frame_data', '')
                    metadata = message_data.get('metadata', {})
                    
                    frame_bytes = bytes.fromhex(frame_hex)
                    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        frame_callback(frame, frame_number, metadata)
                        messages_consumed += 1
                        
                        if max_messages and messages_consumed >= max_messages:
                            self.logger.info(f"Consumed {messages_consumed} messages, stopping")
                            break
                
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
        
        except KeyboardInterrupt:
            self.logger.info("Consumer interrupted by user")
        
        finally:
            self.logger.info(f"Total messages consumed: {messages_consumed}")
    
    def consume_violation_events(
        self,
        event_callback: Callable[[Dict], None],
        timeout: float = 1.0,
        max_messages: Optional[int] = None
    ):
        """
        Consume violation events from Kafka
        
        Args:
            event_callback: Callback function(event_data)
            timeout: Poll timeout in seconds
            max_messages: Maximum number of messages to consume
        """
        self.running = True
        messages_consumed = 0
        
        self.logger.info("Starting violation event consumption")
        
        try:
            while self.running:
                msg = self.consumer.poll(timeout=timeout)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        raise KafkaException(msg.error())
                
                try:
                    message_data = json.loads(msg.value().decode('utf-8'))
                    
                    if message_data.get('event_type') == 'violation':
                        event_data = message_data.get('data', {})
                        event_callback(event_data)
                        messages_consumed += 1
                        
                        if max_messages and messages_consumed >= max_messages:
                            self.logger.info(f"Consumed {messages_consumed} events, stopping")
                            break
                
                except Exception as e:
                    self.logger.error(f"Error processing violation event: {e}")
        
        except KeyboardInterrupt:
            self.logger.info("Consumer interrupted by user")
        
        finally:
            self.logger.info(f"Total events consumed: {messages_consumed}")
    
    def stop(self):
        """Stop consuming messages"""
        self.running = False
    
    def close(self):
        """Close consumer connection"""
        self.logger.info("Closing Kafka consumer")
        self.consumer.close()
