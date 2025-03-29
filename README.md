# YOLOPv2-Studio

A user-friendly studio environment for YOLOPv2 that enables easy model selection, media upload, and real-time camera processing. Makes advanced computer vision accessible with an intuitive interface for object detection, lane detection, and drivable area segmentation tasks.

## Overview

YOLOPv2-Studio provides a web-based interface for the YOLOPv2 model, which performs:
- Object detection (vehicles, pedestrians, etc.)
- Lane line detection
- Drivable area segmentation

All within a single unified model, optimized for autonomous driving scenarios.

## Setup

### Prerequisites
- Python 3.6+ 
- pip

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/YOLOPv2-Studio.git
   cd YOLOPv2-Studio
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv studio-env
   
   # On Windows:
   studio-env\Scripts\activate
   
   # On macOS/Linux:
   source studio-env/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Download YOLOPv2 Model

Before running the application, you need to download the YOLOPv2 model:

```bash
mkdir -p data/weights
wget https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt -P data/weights/
```

Or download it manually from [this link](https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt) and place it in the `data/weights/` directory.

## Running the Application

### Web Interface (Recommended)

```bash
cd app
python app.py
```

This will start the Flask server on http://0.0.0.0:8080. Open this in your browser (or http://localhost:8080 on your local machine).

### Command Line Demo

For processing individual images from the command line:

```bash
python demo.py --source data/example.jpg
```

#### Demo Options:
- `--source`: Path to image or video file
- `--weights`: Path to model weights (default: data/weights/yolopv2.pt)
- `--img-size`: Input image size (default: 640)
- `--conf-thres`: Confidence threshold (default: 0.3)
- `--iou-thres`: IOU threshold for NMS (default: 0.45)
- `--device`: Specify device ('0' for GPU or 'cpu')

## Features

The web interface provides:

1. Multiple input methods:
   - Upload your own images
   - Use demo images from the data/demo directory
   - Use webcam for real-time detection

2. Customizable detection options:
   - Toggle object detection, drivable area, and lane detection
   - Adjust confidence and IoU thresholds

## Example Usage

1. Start the web server
2. Upload or select an image
3. Toggle detection options
4. Click "Process Image"
5. View the results with detected objects, drivable areas, and lane lines

## License

[Specify your license here]
