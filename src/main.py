"""Main entry point for PPE Detection System"""

import argparse
from pathlib import Path
import sys

from ppe_detector.utils.config_loader import ConfigLoader
from ppe_detector.utils.logger import get_logger
from ppe_detector.core.detector import PPEDetector
from ppe_detector.services.video_processor import VideoProcessor


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Personal Protective Equipment (PPE) Violation Detection System"
    )
    
    parser.add_argument(
        '--video',
        type=str,
        help='Path to input video file'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to input image file'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Path to YOLO weights file (overrides config)'
    )
    
    parser.add_argument(
        '--display',
        action='store_true',
        help='Display video/image during processing'
    )
    
    parser.add_argument(
        '--no-save-frames',
        action='store_true',
        help='Do not save individual frames'
    )
    
    parser.add_argument(
        '--save-video',
        action='store_true',
        help='Save output video'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for processed video'
    )
    
    parser.add_argument(
        '--frame-skip',
        type=int,
        default=1,
        help='Process every nth frame (default: 1 = all frames)'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum number of frames to process'
    )
    
    args = parser.parse_args()
    
    if not args.video and not args.image:
        parser.error("Either --video or --image must be specified")
    
    config_loader = ConfigLoader(args.config)
    config = config_loader.get_all()
    
    logger = get_logger("ppe_detector_main", config)
    logger.info("=" * 60)
    logger.info("PPE Detection System Starting")
    logger.info("=" * 60)
    
    weights_path = args.weights or config_loader.get('model.weights_path')
    
    project_root = Path(__file__).parent.parent
    weights_path = project_root / weights_path
    
    if not weights_path.exists():
        logger.error(f"Model weights not found: {weights_path}")
        sys.exit(1)
    
    logger.info("Initializing PPE Detector")
    
    detector = PPEDetector(
        weights_path=str(weights_path),
        class_names=config_loader.get('classes.all_classes'),
        violation_classes=config_loader.get('classes.violation_classes'),
        device=config_loader.get('model.device', 'cpu'),
        confidence_threshold=config_loader.get('model.confidence_threshold', 0.5),
        tracker_config=config_loader.tracking_config,
        logger=logger
    )
    
    output_config = config_loader.output_config
    for key, path in output_config.items():
        if key.endswith('_dir'):
            output_config[key] = str(project_root / path)
    
    video_processor = VideoProcessor(
        detector=detector,
        output_config=output_config,
        logger=logger
    )
    
    try:
        if args.video:
            logger.info(f"Processing video: {args.video}")
            
            results = video_processor.process_video(
                video_path=args.video,
                display_video=args.display or config_loader.get('video.display_video', True),
                save_frames=not args.no_save_frames,
                save_output_video=args.save_video,
                output_video_path=args.output,
                frame_skip=args.frame_skip,
                max_frames=args.max_frames
            )
            
            logger.info("=" * 60)
            logger.info("Processing Complete")
            logger.info(f"Total frames: {results['total_frames']}")
            logger.info(f"Processed frames: {results['processed_frames']}")
            logger.info(f"Alerts raised: {results['alerts_raised']}")
            logger.info(f"Statistics: {results['statistics']}")
            logger.info("=" * 60)
        
        elif args.image:
            logger.info(f"Processing image: {args.image}")
            
            results = video_processor.process_image(
                image_path=args.image,
                save_output=True
            )
            
            logger.info("=" * 60)
            logger.info("Processing Complete")
            logger.info(f"Detections: {len(results['detections'])}")
            logger.info(f"Violation events: {len(results['violation_events'])}")
            logger.info("=" * 60)
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    
    except Exception as e:
        logger.exception(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
