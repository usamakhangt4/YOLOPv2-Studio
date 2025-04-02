import os
import sys
import time
import platform
import urllib.request
from pathlib import Path
import cv2
import torch
import base64
import json
import numpy as np
from io import BytesIO
import logging
import threading
import tempfile
from functools import wraps
from flask import Flask, render_template, request, jsonify, send_from_directory, session
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add the project root to the path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Import YOLOPv2 utilities
from utils.utils import (
    time_synchronized, select_device, increment_path,
    scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model,
    driving_area_mask, lane_line_mask, plot_one_box, show_seg_result,
    AverageMeter
)

# Configuration
class Config:
    """Application configuration"""
    # Model settings
    WEIGHTS_URL = "https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt"
    WEIGHTS_PATH = os.path.join(ROOT_DIR, 'weights', 'yolopv2.pt')
    MODEL_IMAGE_SIZE = 640
    CONF_THRESHOLD = 0.3
    IOU_THRESHOLD = 0.45
    
    # Server settings
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 8080))
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    THREADED = True
    
    # Security settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # Performance settings
    MAX_IMAGE_DIMENSION = 1280  # Maximum dimension for processing
    JPEG_QUALITY = 85  # Quality for JPEG encoding

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24).hex())
CORS(app)  # Enable CORS for all routes

# Thread-local storage for model state
class ModelState(threading.local):
    def __init__(self):
        self.model = None
        self.device = None
        
model_state = ModelState()

# Decorators
def requires_model(f):
    """Decorator to ensure model is loaded before endpoint is called"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if model_state.model is None:
            try:
                suggested_device = suggest_device()
                load_model(device_str=suggested_device)
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                return jsonify({'error': 'Failed to initialize model'}), 500
        return f(*args, **kwargs)
    return decorated_function

def validate_file_upload(f):
    """Decorator to validate file uploads"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
            
        return f(*args, **kwargs)
    return decorated_function

# Helper functions
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def get_system_info():
    """Get system information"""
    info = {
        'os': platform.system(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'current_device': str(model_state.device) if model_state.device else None,
        'model_loaded': model_state.model is not None,
        'available_devices': [],
        'server_config': {
            'host': Config.HOST,
            'port': Config.PORT,
            'debug': Config.DEBUG,
            'threaded': Config.THREADED
        }
    }
    
    # Get available devices
    if torch.cuda.is_available():
        info['available_devices'].extend([f'cuda:{i}' for i in range(torch.cuda.device_count())])
    if platform.system() == 'Darwin' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info['available_devices'].append('mps')
    info['available_devices'].append('cpu')
    
    return info

def download_weights(weights_path):
    """Download model weights if they don't exist"""
    if not os.path.exists(weights_path):
        logger.info(f"Downloading model weights from {Config.WEIGHTS_URL}...")
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            
            # Download to a temporary file first to prevent incomplete downloads
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                urllib.request.urlretrieve(Config.WEIGHTS_URL, temp_file.name)
                
                # Verify download size
                if os.path.getsize(temp_file.name) < 1000:  # Arbitrary small size check
                    os.unlink(temp_file.name)
                    logger.error("Downloaded file is too small, likely an error page")
                    return False
                    
                # Move to final location
                import shutil
                shutil.move(temp_file.name, weights_path)
            
            logger.info(f"Model weights downloaded successfully to {weights_path}!")
            return True
        except Exception as e:
            logger.error(f"Error downloading model weights: {str(e)}")
            return False
    return True

def suggest_device():
    """Suggest the best device based on system capabilities"""
    system_info = get_system_info()
    
    if system_info['os'] == 'Darwin':  # macOS
        if torch.backends.mps.is_available():
            return 'mps'  # Use Metal Performance Shaders on Apple Silicon
        return 'cpu'
    elif system_info['cuda_available']:
        return '0'  # Use CUDA if available
    return 'cpu'

def unload_model():
    """Properly unload the current model and free resources"""
    if model_state.model is not None:
        # Move model to CPU first to free GPU/MPS memory
        if hasattr(model_state.model, 'to') and model_state.device and model_state.device.type != 'cpu':
            model_state.model.to('cpu')
            
        # Delete model and clear device cache if applicable
        model_state.model = None
        if torch.cuda.is_available() and model_state.device and model_state.device.type == 'cuda':
            with torch.cuda.device(model_state.device):
                torch.cuda.empty_cache()
        elif model_state.device and model_state.device.type == 'mps':
            # MPS has no direct empty_cache equivalent, use garbage collection
            import gc
            gc.collect()
                
        model_state.device = None
        logger.info("Model unloaded and resources freed")

def load_model(weights_path=None, device_str=''):
    """Load YOLOPv2 model"""
    # Unload existing model if any
    unload_model()
    
    # Default weights path if not specified
    if weights_path is None:
        weights_path = Config.WEIGHTS_PATH
    
    # Check if weights file exists
    if not os.path.exists(weights_path):
        logger.warning(f"Model weights not found at {weights_path}")
        if not download_weights(weights_path):
            logger.error(f"Failed to download model weights")
            return False
    
    # Select device
    try:
        model_state.device = select_device(device_str)
        logger.info(f"Using device: {model_state.device}")
        
        # Load model with specified device - use torch.jit.load instead of torch.load
        model_state.model = torch.jit.load(weights_path, map_location=model_state.device)
        
        # Check if model loaded correctly
        if model_state.model is None:
            raise RuntimeError("Model loaded as None")
            
        # Performance optimization: convert model to FP16 if using CUDA
        if model_state.device.type == 'cuda':
            logger.info("Using half precision for CUDA device")
            model_state.model = model_state.model.half()
            
            # Performance optimization: set CUDA optimization flags
            torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
            if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on cudnn
            
            # Warmup model with a dummy input
            logger.info("Warming up model...")
            input_size = Config.MODEL_IMAGE_SIZE
            dummy_input = torch.zeros(1, 3, input_size, input_size).to(model_state.device).type_as(next(model_state.model.parameters()))
            with torch.no_grad():
                for _ in range(2):  # Run twice for warmup
                    _ = model_state.model(dummy_input)
            torch.cuda.synchronize()
            logger.info("Warmup complete")
        
        # Set model to evaluation mode
        model_state.model.to(model_state.device).eval()
        
        # Verify model has parameters
        try:
            param = next(model_state.model.parameters())
            logger.info(f"Model loaded with parameters, dtype={param.dtype}, device={param.device}")
        except StopIteration:
            logger.warning("Model has no parameters!")
        
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        return False

def resize_if_needed(img, max_dimension=None):
    """Resize image if it exceeds maximum dimension"""
    if max_dimension is None:
        max_dimension = Config.MAX_IMAGE_DIMENSION
        
    height, width = img.shape[:2]
    
    if max(height, width) > max_dimension:
        # Calculate new dimensions while maintaining aspect ratio
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")
    
    return img

def process_image(image_bytes, conf_thres=None, iou_thres=None, 
                 show_objects=True, show_drivable=True, show_lanes=True):
    """Process an image with YOLOPv2 model"""
    # Use default thresholds if not provided
    if conf_thres is None:
        conf_thres = Config.CONF_THRESHOLD
    if iou_thres is None:
        iou_thres = Config.IOU_THRESHOLD
    
    try:
        # Check if model is loaded
        if model_state.model is None:
            raise RuntimeError("Model not loaded")
        
        # Convert image from bytes to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        img0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img0 is None:
            raise ValueError("Failed to decode image")
        
        # Resize image if too large
        img0 = resize_if_needed(img0)
        logger.debug(f"Original image shape: {img0.shape}")
        
        # Preprocess image
        img, ratio, (dw, dh) = letterbox(img0, Config.MODEL_IMAGE_SIZE, stride=32)
        
        # Convert
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(model_state.device)
        img = img.float() / 255.0
        
        # Convert to half precision if model is in half precision
        model_param = next(model_state.model.parameters(), None)
        if model_param is not None and model_state.device.type == 'cuda' and model_param.dtype == torch.float16:
            img = img.half()
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference with timing
        t1 = time_synchronized()
        
        # Use CUDA streams for better GPU utilization
        if model_state.device and model_state.device.type == 'cuda':
            torch.cuda.synchronize()
        
        with torch.no_grad():
            [pred, anchor_grid], seg, ll = model_state.model(img)
        
        # Sync CUDA operations for accurate timing
        if model_state.device and model_state.device.type == 'cuda':
            torch.cuda.synchronize()
        
        t2 = time_synchronized()
        inference_time = t2 - t1
        logger.debug(f"Inference complete in {inference_time:.3f}s")
        
        # Post-processing: move tensors to CPU for post-processing
        pred_cpu = pred.detach().cpu() if isinstance(pred, torch.Tensor) else pred
        anchor_grid_cpu = anchor_grid.detach().cpu() if isinstance(anchor_grid, torch.Tensor) else anchor_grid
        
        pred = split_for_trace_model(pred_cpu, anchor_grid_cpu)
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        
        # Get masks
        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)
        
        # Convert masks to numpy if they are PyTorch tensors
        if isinstance(da_seg_mask, torch.Tensor):
            da_seg_mask = da_seg_mask.squeeze().cpu().numpy()
        else:
            da_seg_mask = np.squeeze(da_seg_mask)
            
        if isinstance(ll_seg_mask, torch.Tensor):
            ll_seg_mask = ll_seg_mask.squeeze().cpu().numpy()
        else:
            ll_seg_mask = np.squeeze(ll_seg_mask)
        
        # Convert to uint8 (0-255) for OpenCV
        da_seg_mask = (da_seg_mask * 255).astype(np.uint8)
        ll_seg_mask = (ll_seg_mask * 255).astype(np.uint8)
        
        # Resize masks to match original image size
        if da_seg_mask.shape[:2] != img0.shape[:2]:
            da_seg_mask = cv2.resize(da_seg_mask, (img0.shape[1], img0.shape[0]), interpolation=cv2.INTER_LINEAR)
        if ll_seg_mask.shape[:2] != img0.shape[:2]:
            ll_seg_mask = cv2.resize(ll_seg_mask, (img0.shape[1], img0.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Convert back to binary masks (0-1)
        da_seg_mask = da_seg_mask > 127
        ll_seg_mask = ll_seg_mask > 127
        
        # Process detections
        results = []
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to img0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                
                # Collect detection results
                for *xyxy, conf, cls in reversed(det):
                    results.append({
                        'xyxy': [float(x) for x in xyxy],
                        'conf': float(conf),
                        'cls': int(cls)
                    })
        
        # Create visualization
        im0 = img0.copy()
        
        # Apply object detection boxes if enabled
        if show_objects and len(pred[0]):
            for *xyxy, conf, cls in reversed(pred[0]):
                label = f'{conf:.2f}'
                plot_one_box(xyxy, im0, label=label, line_thickness=3)
        
        # Create modified masks based on display options
        display_masks = []
        if show_drivable:
            display_masks.append(da_seg_mask)
        else:
            display_masks.append(np.zeros_like(da_seg_mask))
            
        if show_lanes:
            display_masks.append(ll_seg_mask)
        else:
            display_masks.append(np.zeros_like(ll_seg_mask))
        
        # Apply segmentation mask
        im0 = show_seg_result(im0, tuple(display_masks), is_demo=True, return_img=True)
        
        # Optimize image for web display
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY]
        
        # Convert OpenCV image to base64 for web display
        _, buffer = cv2.imencode('.jpg', im0, encode_params)
        if buffer is None:
            raise ValueError("Failed to encode output image")
            
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Clean up tensors to avoid memory leaks
        if model_state.device and model_state.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif model_state.device and model_state.device.type == 'mps':
            # There's no direct equivalent to empty_cache for MPS,
            # but we can help garbage collection by deleting variables
            import gc
            gc.collect()
        
        return {
            'image': img_base64,
            'inference_time': float(inference_time),
            'detections': results
        }
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}", exc_info=True)
        raise

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resize and pad image while meeting stride-multiple constraints"""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

def parse_display_options(request_data, is_json=True):
    """Parse display options from request data (works with both JSON and form data)"""
    if is_json:
        # JSON data
        show_objects_val = request_data.get('show_objects', 'true')
        show_drivable_val = request_data.get('show_drivable', 'true')
        show_lanes_val = request_data.get('show_lanes', 'true')
        conf_thres = float(request_data.get('conf_threshold', Config.CONF_THRESHOLD))
        iou_thres = float(request_data.get('iou_threshold', Config.IOU_THRESHOLD))
    else:
        # Form data
        show_objects_val = request_data.get('show_objects', 'true')
        show_drivable_val = request_data.get('show_drivable', 'true')
        show_lanes_val = request_data.get('show_lanes', 'true')
        
        try:
            conf_thres = float(request_data.get('conf_threshold', Config.CONF_THRESHOLD))
            iou_thres = float(request_data.get('iou_threshold', Config.IOU_THRESHOLD))
        except ValueError:
            conf_thres = Config.CONF_THRESHOLD
            iou_thres = Config.IOU_THRESHOLD
    
    # Convert string values to boolean - more explicitly handling all possible values
    if isinstance(show_objects_val, str):
        show_objects = show_objects_val.lower() in ['true', '1', 't', 'yes', 'y', 'on']
    else:
        show_objects = bool(show_objects_val)
        
    if isinstance(show_drivable_val, str):
        show_drivable = show_drivable_val.lower() in ['true', '1', 't', 'yes', 'y', 'on']
    else:
        show_drivable = bool(show_drivable_val)
        
    if isinstance(show_lanes_val, str):
        show_lanes = show_lanes_val.lower() in ['true', '1', 't', 'yes', 'y', 'on']
    else:
        show_lanes = bool(show_lanes_val)
    
    return {
        'show_objects': show_objects,
        'show_drivable': show_drivable,
        'show_lanes': show_lanes,
        'conf_thres': conf_thres,
        'iou_thres': iou_thres
    }

# Routes
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
@requires_model
@validate_file_upload
def process():
    """Process an uploaded image"""
    try:
        # Read image file
        file = request.files['file']
        img_bytes = file.read()
        
        # Get display options from form data
        options = parse_display_options(request.form, is_json=False)
        
        # Log for debugging
        logger.info(f"Processing uploaded image with options: {options}")
        
        # Process image with model
        result = process_image(
            img_bytes, 
            conf_thres=options['conf_thres'],
            iou_thres=options['iou_thres'],
            show_objects=options['show_objects'],
            show_drivable=options['show_drivable'],
            show_lanes=options['show_lanes']
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/demo/<path:filename>')
def demo_image(filename):
    """Get demo image"""
    demo_dir = os.path.join(ROOT_DIR, 'data', 'demo')
    # Security: sanitize filename
    filename = secure_filename(os.path.basename(filename))
    try:
        return send_from_directory(demo_dir, filename)
    except Exception as e:
        logger.error(f"Error serving demo image: {str(e)}")
        return jsonify({'error': f'Error serving demo image: {str(e)}'}), 500

@app.route('/detect-demo', methods=['POST'])
@requires_model
def detect_demo():
    """Process a demo image"""
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image specified'}), 400
        
        # Security: sanitize filename
        image_name = secure_filename(os.path.basename(data['image']))
        
        # Get absolute path to demo directory
        demo_dir = os.path.join(ROOT_DIR, 'data', 'demo')
        img_path = os.path.join(demo_dir, image_name)
        
        logger.debug(f"Looking for demo image at: {img_path}")
        
        # Check if file exists
        if not os.path.exists(img_path):
            logger.error(f"Demo image not found at: {img_path}")
            return jsonify({'error': f'Demo image not found: {image_name}'}), 404
        
        # Read image file
        try:
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
        except Exception as e:
            logger.error(f"Failed to read demo image: {str(e)}")
            return jsonify({'error': f'Error reading demo image: {str(e)}'}), 500
        
        # Get display options
        options = parse_display_options(data, is_json=True)
        
        logger.debug(f"Processing demo image with settings: {options}")
        
        # Process image with model
        result = process_image(
            img_bytes,
            conf_thres=options['conf_thres'],
            iou_thres=options['iou_thres'],
            show_objects=options['show_objects'],
            show_drivable=options['show_drivable'],
            show_lanes=options['show_lanes']
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Unexpected error in detect_demo: {str(e)}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/camera', methods=['POST'])
@requires_model
def camera():
    """Process a camera frame"""
    try:
        if 'image' not in request.json:
            return jsonify({'error': 'No image data'}), 400
        
        # Get base64 image from request
        img_data = request.json['image']
        # Handle if the image is a data URL (starts with data:image)
        if ',' in img_data:
            img_data = img_data.split(',')[1]
        
        img_bytes = base64.b64decode(img_data)
        
        # Get display options
        options = parse_display_options(request.json, is_json=True)
        
        # Log for debugging (less verbose for camera to avoid console spam)
        logger.debug(f"Processing camera frame with options: {options}")
        
        # Process image with model
        result = process_image(
            img_bytes,
            conf_thres=options['conf_thres'],
            iou_thres=options['iou_thres'],
            show_objects=options['show_objects'],
            show_drivable=options['show_drivable'],
            show_lanes=options['show_lanes']
        )
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing camera frame: {str(e)}")
        return jsonify({'error': f'Error processing camera frame: {str(e)}'}), 500

@app.route('/system-info', methods=['GET'])
def system_info_route():
    """Get system information endpoint"""
    return jsonify(get_system_info())

@app.route('/set-device', methods=['POST'])
def set_device():
    """Change the device being used"""
    try:
        device_str = request.json.get('device', 'cpu')
        
        # Validate device input
        system_info = get_system_info()
        if device_str not in system_info['available_devices'] and device_str != 'cpu':
            return jsonify({'error': f'Invalid device: {device_str}. Available devices: {system_info["available_devices"]}'}), 400
        
        # Load model on new device
        success = load_model(device_str=device_str)
        
        if success:
            device_info = str(next(model_state.model.parameters()).device)
            return jsonify({'success': True, 'device': device_info})
        else:
            return jsonify({'error': 'Failed to load model on specified device'}), 500
    except Exception as e:
        logger.error(f"Error setting device: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': f'File too large. Maximum size is {Config.MAX_CONTENT_LENGTH/1024/1024}MB'}), 413

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all unhandled exceptions"""
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({'error': str(e)}), 500

def main():
    """Main function to start the application"""
    try:
        # Print system information
        system_info = get_system_info()
        logger.info("\nSystem Information:")
        logger.info(f"Operating System: {system_info['os']}")
        logger.info(f"Python Version: {system_info['python_version']}")
        logger.info(f"PyTorch Version: {system_info['torch_version']}")
        logger.info(f"CUDA Available: {system_info['cuda_available']}")
        if system_info['cuda_available']:
            logger.info(f"CUDA Version: {system_info['cuda_version']}")
        
        # Suggest best device for this system
        suggested_device = suggest_device()
        logger.info(f"\nSuggested device for this system: {suggested_device}")
        
        # Load model before starting the app
        logger.info('\nLoading YOLOPv2 model...')
        success = load_model(device_str=suggested_device)
        
        if not success:
            raise RuntimeError("Model failed to load")
        
        logger.info('Model loaded successfully!')
        logger.info(f"Model device: {next(model_state.model.parameters()).device}")
        
        # Run Flask app with appropriate settings for the platform
        use_threading = system_info['os'] != 'Windows' or suggested_device == 'cpu'
        logger.info(f"\nStarting server with threading={'enabled' if use_threading else 'disabled'}")
        
        app.run(
            host=Config.HOST,
            port=Config.PORT,
            debug=Config.DEBUG,
            threaded=use_threading
        )
    except Exception as e:
        logger.error(f"\nError starting application: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main() 