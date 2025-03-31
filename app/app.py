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
from flask import Flask, render_template, request, jsonify, send_from_directory

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

app = Flask(__name__)

# Global variables
model = None
device = None
WEIGHTS_URL = "https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt"

def get_system_info():
    """Get system information"""
    info = {
        'os': platform.system(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'current_device': str(device),
        'available_devices': [],
        'model_loaded': model is not None,
        'server_config': {
            'host': '0.0.0.0',
            'port': 8080,
            'debug': False,
            'threaded': True
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
        print(f"Downloading model weights from {WEIGHTS_URL}...")
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            
            # Download the file
            urllib.request.urlretrieve(WEIGHTS_URL, weights_path)
            
            # Verify download was successful
            if os.path.exists(weights_path) and os.path.getsize(weights_path) > 0:
                print(f"Model weights downloaded successfully to {weights_path}!")
                return True
            else:
                print("Downloaded file is empty or not found.")
                return False
        except Exception as e:
            print(f"Error downloading model weights: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    return True

def suggest_device():
    """Suggest the best device based on system capabilities"""
    system_info = get_system_info()
    
    if system_info['os'] == 'Darwin':  # macOS
        if torch.backends.mps.is_available():
            return 'mps'  # Use Metal Performance Shaders if available
        return 'cpu'
    elif system_info['cuda_available']:
        return '0'  # Use CUDA if available
    return 'cpu'

def load_model(weights_path=None, device_str=''):
    """Load YOLOPv2 model"""
    global device  # Make sure to modify the global device variable
    
    # Default weights path if not specified
    if weights_path is None:
        weights_path = os.path.join(ROOT_DIR, 'weights', 'yolopv2.pt')
    
    # Check if weights file exists
    if not os.path.exists(weights_path):
        print(f"[WARN] Model weights not found at {weights_path}")
        if not download_weights(weights_path):
            print(f"[ERROR] Failed to download model weights")
            return None
    
    # Select device
    device = select_device(device_str)
    print(f"Using device: {device}")
    
    try:
        # First, check if the file exists
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Model weights file not found at {weights_path}")
            
        # Load model with specified device - use torch.jit.load instead of torch.load
        # This is because YOLOPv2 models are TorchScript models
        model = torch.jit.load(weights_path, map_location=device)
        
        # Check if model loaded correctly
        if model is None:
            raise RuntimeError("Model loaded as None")
            
        # Performance optimization: convert model to FP16 if using CUDA
        if device.type == 'cuda':
            print("Using half precision for CUDA device")
            model = model.half()
            
            # Performance optimization: set CUDA optimization flags
            torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
            if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on cudnn
            
            # Warmup model with a dummy input
            print("Warming up model...")
            input_size = 640
            dummy_input = torch.zeros(1, 3, input_size, input_size).to(device).type_as(next(model.parameters()))
            with torch.no_grad():
                for _ in range(2):  # Run twice for warmup
                    _ = model(dummy_input)
            torch.cuda.synchronize()
            print("Warmup complete")
        
        # Set model to evaluation mode
        model.to(device).eval()
        
        # Verify model has parameters
        try:
            param = next(model.parameters())
            print(f"Model loaded with parameters, dtype={param.dtype}, device={param.device}")
        except StopIteration:
            print("WARNING: Model has no parameters!")
            
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_image(image_bytes, img_size=640, conf_thres=0.3, iou_thres=0.45, 
                 show_objects=True, show_drivable=True, show_lanes=True):
    """Process an image with YOLOPv2 model"""
    global model, device
    
    try:
        # Check if model is loaded
        if model is None:
            raise RuntimeError("Model not loaded")
            
        # Check if device is set
        if device is None:
            device = select_device('')
            print(f"[DEBUG] Device was None, setting to: {device}")
            
        # Convert image from bytes to OpenCV format
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img0 is None:
                raise ValueError("Failed to decode image")
                
            # Performance optimization: reduce image size if it's too large
            height, width = img0.shape[:2]
            max_dimension = 1280  # Maximum dimension for processing
            
            if max(height, width) > max_dimension:
                # Calculate new dimensions while maintaining aspect ratio
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                
                img0 = cv2.resize(img0, (new_width, new_height), interpolation=cv2.INTER_AREA)
                print(f"[OPTIMIZE] Resized image from {width}x{height} to {new_width}x{new_height}")
            
            print(f"[DEBUG] Original image shape: {img0.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to decode image: {str(e)}")
            raise ValueError(f"Failed to decode image: {str(e)}")
        
        # Preprocess image
        try:
            # Resize and pad image
            img, ratio, (dw, dh) = letterbox(img0, img_size, stride=32)
            
            # Convert
            img = img.transpose(2, 0, 1)  # HWC to CHW
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.float() / 255.0
            
            # Check model parameters
            model_param = next(model.parameters(), None)
            if model_param is None:
                print("[WARNING] Model has no parameters. Using float precision.")
            # Convert to half precision if model is in half precision
            elif device.type == 'cuda' and model_param.dtype == torch.float16:
                img = img.half()
                print(f"[DEBUG] Using half precision for image: {img.dtype}")
            
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            print(f"[DEBUG] Preprocessed image shape: {img.shape}, dtype: {img.dtype}")
        except Exception as e:
            print(f"[ERROR] Failed to preprocess image: {str(e)}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Failed to preprocess image: {str(e)}")
        
        # Inference
        try:
            t1 = time_synchronized()
            # Use CUDA streams for better GPU utilization
            if device is not None and device.type == 'cuda':
                torch.cuda.synchronize()
                
            with torch.no_grad():
                [pred, anchor_grid], seg, ll = model(img)
                
            # Sync CUDA operations for accurate timing
            if device is not None and device.type == 'cuda':
                torch.cuda.synchronize()
                
            t2 = time_synchronized()
            
            print(f"[DEBUG] Inference complete in {t2-t1:.3f}s")
        except Exception as e:
            print(f"[ERROR] Model inference failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Model inference failed: {str(e)}")
        
        # Post-processing
        try:
            # Performance optimization: move tensors to CPU for post-processing
            pred_cpu = pred.detach().cpu() if isinstance(pred, torch.Tensor) else pred
            anchor_grid_cpu = anchor_grid.detach().cpu() if isinstance(anchor_grid, torch.Tensor) else anchor_grid
            
            pred = split_for_trace_model(pred_cpu, anchor_grid_cpu)
            pred = non_max_suppression(pred, conf_thres, iou_thres)
            
            # Get masks
            da_seg_mask = driving_area_mask(seg)  # [1, h, w]
            ll_seg_mask = lane_line_mask(ll)      # [1, h, w]
            
            # Convert masks to numpy if they are PyTorch tensors
            if isinstance(da_seg_mask, torch.Tensor):
                da_seg_mask = da_seg_mask.squeeze().cpu().numpy()  # [h, w]
            else:
                da_seg_mask = np.squeeze(da_seg_mask)  # Already numpy, just squeeze
                
            if isinstance(ll_seg_mask, torch.Tensor):
                ll_seg_mask = ll_seg_mask.squeeze().cpu().numpy()  # [h, w]
            else:
                ll_seg_mask = np.squeeze(ll_seg_mask)  # Already numpy, just squeeze
            
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
            
            print(f"[DEBUG] Mask shapes - DA: {da_seg_mask.shape}, LL: {ll_seg_mask.shape}")
            
            # Process detections
            results = []
            for i, det in enumerate(pred):
                if len(det):
                    # Rescale boxes from img_size to img0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                    
                    # Draw boxes and append to results
                    for *xyxy, conf, cls in reversed(det):
                        results.append({
                            'xyxy': [float(x) for x in xyxy],
                            'conf': float(conf),
                            'cls': int(cls)
                        })
            
            print(f"[DEBUG] Found {len(results)} detections")
        except Exception as e:
            print(f"[ERROR] Post-processing failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Post-processing failed: {str(e)}")
        
        # Create visualization
        try:
            im0 = img0.copy()
            
            # Apply object detection boxes if enabled
            if show_objects and len(pred[0]):
                for *xyxy, conf, cls in reversed(pred[0]):
                    plot_one_box(xyxy, im0, line_thickness=3)
            
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
            encode_quality = 85  # Lower quality for faster transmission
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, encode_quality]
            
            # Convert OpenCV image to base64 for web display
            _, buffer = cv2.imencode('.jpg', im0, encode_params)
            if buffer is None:
                raise ValueError("Failed to encode output image")
                
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                'image': img_base64,
                'inference_time': float(t2 - t1),
                'detections': results
            }
        except Exception as e:
            print(f"[ERROR] Visualization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Visualization failed: {str(e)}")
            
    except Exception as e:
        print(f"[ERROR] Error in process_image: {str(e)}")
        import traceback
        traceback.print_exc()
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

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    """Process an uploaded image"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Read image file
        img_bytes = file.read()
        
        # Get display options from form data if available
        show_objects_val = request.form.get('show_objects', 'true')
        show_drivable_val = request.form.get('show_drivable', 'true')
        show_lanes_val = request.form.get('show_lanes', 'true')
        
        # Convert string values to boolean - more explicitly handling all possible values
        show_objects = show_objects_val.lower() in ['true', '1', 't', 'yes', 'y', 'on']
        show_drivable = show_drivable_val.lower() in ['true', '1', 't', 'yes', 'y', 'on']
        show_lanes = show_lanes_val.lower() in ['true', '1', 't', 'yes', 'y', 'on']
        
        # Get threshold values
        try:
            conf_thres = float(request.form.get('conf_threshold', 0.3))
            iou_thres = float(request.form.get('iou_threshold', 0.45))
        except ValueError:
            conf_thres = 0.3
            iou_thres = 0.45
        
        # Log for debugging
        print(f"[PROCESS] Processing uploaded image with options:")
        print(f"  - show_objects: {show_objects} (raw: '{show_objects_val}')")
        print(f"  - show_drivable: {show_drivable} (raw: '{show_drivable_val}')")
        print(f"  - show_lanes: {show_lanes} (raw: '{show_lanes_val}')")
        print(f"  - conf_threshold: {conf_thres}")
        print(f"  - iou_threshold: {iou_thres}")
        
        # Process image with model
        try:
            result = process_image(
                img_bytes, 
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                show_objects=show_objects,
                show_drivable=show_drivable,
                show_lanes=show_lanes
            )
            return jsonify(result)
        except Exception as e:
            print(f"[ERROR] Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
            
    except Exception as e:
        print(f"[ERROR] Unexpected error in process route: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/demo/<path:filename>')
def demo_image(filename):
    """Get demo image"""
    demo_dir = os.path.join(ROOT_DIR, 'data', 'demo')
    try:
        return send_from_directory(demo_dir, filename)
    except Exception as e:
        return jsonify({'error': f'Error serving demo image: {str(e)}'}), 500

@app.route('/detect-demo', methods=['POST'])
def detect_demo():
    """Process a demo image"""
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image specified'}), 400
        
        # Get absolute path to demo directory
        demo_dir = os.path.join(ROOT_DIR, 'data', 'demo')
        img_path = os.path.join(demo_dir, data['image'])
        
        print(f"[DEBUG] Looking for demo image at: {img_path}")
        
        # Check if file exists
        if not os.path.exists(img_path):
            print(f"[ERROR] Demo image not found at: {img_path}")
            return jsonify({'error': f'Demo image not found: {data["image"]}'}), 404
        
        # Read image file
        try:
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
        except Exception as e:
            print(f"[ERROR] Failed to read demo image: {str(e)}")
            return jsonify({'error': f'Error reading demo image: {str(e)}'}), 500
        
        # Get display options
        show_objects = bool(data.get('show_objects', True))
        show_drivable = bool(data.get('show_drivable', True))
        show_lanes = bool(data.get('show_lanes', True))
        
        # Get threshold values with error handling
        try:
            conf_thres = float(data.get('conf_threshold', 0.3))
            iou_thres = float(data.get('iou_threshold', 0.45))
        except (ValueError, TypeError) as e:
            print(f"[ERROR] Invalid threshold values: {str(e)}")
            conf_thres = 0.3
            iou_thres = 0.45
        
        print(f"[DEBUG] Processing demo image with settings:")
        print(f"  - Image path: {img_path}")
        print(f"  - Image size: {len(img_bytes)} bytes")
        print(f"  - show_objects: {show_objects}")
        print(f"  - show_drivable: {show_drivable}")
        print(f"  - show_lanes: {show_lanes}")
        print(f"  - conf_threshold: {conf_thres}")
        print(f"  - iou_threshold: {iou_thres}")
        
        # Process image with model
        try:
            result = process_image(
                img_bytes,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                show_objects=show_objects,
                show_drivable=show_drivable,
                show_lanes=show_lanes
            )
            return jsonify(result)
        except Exception as e:
            print(f"[ERROR] Failed to process image: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
            
    except Exception as e:
        print(f"[ERROR] Unexpected error in detect_demo: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/camera', methods=['POST'])
def camera():
    """Process a camera frame"""
    if 'image' not in request.json:
        return jsonify({'error': 'No image data'}), 400
    
    # Get base64 image from request
    img_data = request.json['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    
    # Get display options
    show_objects_val = request.json.get('show_objects', True)
    show_drivable_val = request.json.get('show_drivable', True)
    show_lanes_val = request.json.get('show_lanes', True)
    
    # Convert to boolean properly
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
    
    # Get threshold values with error handling
    try:
        conf_thres = float(request.json.get('conf_threshold', 0.3))
        iou_thres = float(request.json.get('iou_threshold', 0.45))
    except (ValueError, TypeError):
        conf_thres = 0.3
        iou_thres = 0.45
    
    # Log for debugging (less verbose for camera to avoid console spam)
    print(f"[CAMERA] Processing camera frame with options: obj={show_objects} (raw: {repr(show_objects_val)}), driv={show_drivable} (raw: {repr(show_drivable_val)}), lanes={show_lanes} (raw: {repr(show_lanes_val)}), conf={conf_thres}, iou={iou_thres}")
    
    # Process image with model
    result = process_image(
        img_bytes,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        show_objects=show_objects,
        show_drivable=show_drivable,
        show_lanes=show_lanes
    )
    
    return jsonify(result)

@app.route('/system-info', methods=['GET'])
def system_info_route():
    """Get system information endpoint"""
    return jsonify(get_system_info())

@app.route('/set-device', methods=['POST'])
def set_device():
    """Change the device being used"""
    try:
        device_str = request.json.get('device', 'cpu')
        global model
        model = load_model(device_str=device_str)
        return jsonify({'success': True, 'device': str(next(model.parameters()).device)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all unhandled exceptions"""
    print(f"[ERROR] Unhandled exception: {str(e)}")
    import traceback
    traceback.print_exc()
    return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # Print system information
        system_info = get_system_info()
        print("\nSystem Information:")
        print(f"Operating System: {system_info['os']}")
        print(f"Python Version: {system_info['python_version']}")
        print(f"PyTorch Version: {system_info['torch_version']}")
        print(f"CUDA Available: {system_info['cuda_available']}")
        if system_info['cuda_available']:
            print(f"CUDA Version: {system_info['cuda_version']}")
        
        # Suggest best device for this system
        suggested_device = suggest_device()
        print(f"\nSuggested device for this system: {suggested_device}")
        
        # Load model before starting the app
        print('\nLoading YOLOPv2 model...')
        model = load_model(device_str=suggested_device)
        
        if model is None:
            raise RuntimeError("Model failed to load")
            
        print('Model loaded successfully!')
        print(f"Model device: {next(model.parameters()).device}")
        
        # Run Flask app with appropriate settings for the platform
        use_threading = system_info['os'] != 'Windows' or suggested_device == 'cpu'
        print(f"\nStarting server with threading={'enabled' if use_threading else 'disabled'}")
        
        app.run(
            host='0.0.0.0',
            port=8080,
            debug=True,
            threaded=use_threading
        )
    except Exception as e:
        print(f"\nError starting application: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 