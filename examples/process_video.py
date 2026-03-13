"""Example: Process video file for PPE violations"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ppe_detector.utils.config_loader import ConfigLoader
from src.ppe_detector.utils.logger import get_logger
from src.ppe_detector.core.detector import PPEDetector
from src.ppe_detector.services.video_processor import VideoProcessor


def main():
    """Process video example"""
    
    config_loader = ConfigLoader()
    config = config_loader.get_all()
    logger = get_logger("video_processing_example", config)
    
    logger.info("Initializing PPE Detection System")
    
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
    
    output_config = config_loader.output_config
    for key, path in output_config.items():
        if key.endswith('_dir'):
            output_config[key] = str(project_root / path)
    
    processor = VideoProcessor(
        detector=detector,
        output_config=output_config,
        logger=logger
    )
    
    video_path = input("Enter path to video file: ").strip()
    
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        return
    
    logger.info(f"Processing: {video_path}")
    
    results = processor.process_video(
        video_path=video_path,
        display_video=True,
        save_frames=True,
        save_output_video=False,
        frame_skip=1
    )
    
    print("\n" + "=" * 60)
    print("PROCESSING RESULTS")
    print("=" * 60)
    print(f"Video: {results['video_path']}")
    print(f"Total Frames: {results['total_frames']}")
    print(f"Processed Frames: {results['processed_frames']}")
    print(f"Alerts Raised: {results['alerts_raised']}")
    print(f"\nStatistics:")
    for key, value in results['statistics'].items():
        print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
