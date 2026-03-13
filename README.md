# PPE Detection System

PPE (Personal Protective Equipment) violation detection using YOLOv8 for safety monitoring in industrial environments.

## Features

- Detect missing safety equipment (hardhats, masks, safety vests) in video streams
- Multi-object tracking with SORT algorithm
- Violation alerts with JSON reporting
- Kafka integration for streaming
- YAML configuration

## Installation

```bash
pip install -r requirements.txt
```

**Prerequisites**: Python 3.8+, PyTorch, OpenCV. Place `fine_tuned_weights.pt` in project root.

## Usage

```bash
# Process video
python src/main.py --video path/to/video.mp4 --display

# Process image
python src/main.py --image path/to/image.jpg

# Options: --frame-skip N, --max-frames N, --save-video, --no-save-frames
```

## Configuration

Edit `config/config.yaml` for model path, device (cpu/cuda/mps), tracking parameters, and output directories.

## Output

- **data/output/images/** - Annotated frames
- **data/output/json/** - Detection reports and alerts
- **logs/** - Log files

## Project Structure

```
├── config/config.yaml
├── src/
│   ├── main.py
│   └── ppe_detector/
│       ├── core/        # Detector, tracker
│       ├── services/    # Video processor, Kafka
│       ├── models/
│       └── utils/
├── examples/
└── data/input/, data/output/
```

## Troubleshooting

- **Weights not found**: Ensure `fine_tuned_weights.pt` is in project root
- **CUDA/MPS unavailable**: Set `device: cpu` in config.yaml
- **Out of memory**: Use `--frame-skip 3` or `--no-save-frames`
