# YOLOPv2-Studio

A user-friendly studio environment for YOLOPv2 that enables easy model selection, media upload, and real-time camera processing. This tool makes advanced computer vision accessible with an intuitive interface for object detection, lane detection, and drivable area segmentation.

## Overview

YOLOPv2-Studio provides a web-based interface for the YOLOPv2 model, which performs:
- **Object Detection** (vehicles, pedestrians, etc.)
- **Lane Line Detection**
- **Drivable Area Segmentation**

All within a single unified model optimized for autonomous driving scenarios.

## Features

The web interface provides:

1. **Multiple Input Methods:**
   - Upload your own images
   - Use the example image from the `data/demo/` directory
   - Use a webcam for real-time detection

2. **Customizable Detection Options:**
   - Toggle object detection, drivable area, and lane detection
   - Adjust confidence and IoU thresholds
   - Real-time FPS counter for performance monitoring

## Setup

### Prerequisites
- Python 3.6+
- pip

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/usamakhangt4/YOLOPv2-Studio.git
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

Before running the application, download the YOLOPv2 model:

```bash
mkdir -p data/weights
wget https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt -P data/weights/
```

Or manually download it from [this link](https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt) and place it in the `data/weights/` directory.

## Running the Application

### Web Interface (Recommended)

```bash
cd app
python app.py
```

This will start the Flask server at `http://0.0.0.0:8080`. Open it in your browser (`http://localhost:8080` on your local machine).

## Example Usage

1. Start the web server.
2. Select an input method (upload an image, select a demo image, or use a camera).
3. Adjust detection options using toggles and sliders.
4. For camera input, use the Start/Stop buttons to control processing.
5. View results with detected objects, drivable areas, and lane lines.

## Project Structure

```
YOLOPv2-Studio/
├── app/                 # Web application
│   ├── app.py           # Flask application main file
│   ├── static/          # Static assets
│   │   ├── css/         # CSS stylesheets
│   │   ├── js/          # JavaScript files
│   │   ├── images/      # App images and icons
│   └── templates/       # HTML templates
│       └── index.html   # Main application page
├── data/                # Data directory
│   ├── demo/            # Demo images and videos
│   │   └── example.jpg  # Example image for testing
│   └── weights/         # Model weights directory
│       └── yolopv2.pt   # YOLOPv2 model weights (downloaded separately)
├── utils/               # Utility functions
│   └── utils.py         # General utility functions
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
└── README.md            # Project documentation
```

### Key Components

- **`app/app.py`**: Main Flask application serving the web interface and handling image processing requests.
- **`app/static/js/main.js`**: Handles frontend functionality including file uploads, camera input, and result display.
- **`app/templates/index.html`**: HTML template for the web interface.
- **`demo.py`**: Standalone script for processing images without the web interface.
- **`utils/utils.py`**: Utility functions for image processing, logging, and other helper functions.

## License

Based on [YOLOPv2](https://github.com/CAIC-AD/YOLOPv2) developed by CAIC-AD.

MIT License - © 2025 Muhammad Usama Bin Akhtar Khan.

