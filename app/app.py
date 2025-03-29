import os
import sys
import time
from pathlib import Path
import cv2
import torch
import base64
import json
import numpy as np
from io import BytesIO
from flask import Flask, render_template, request, jsonify, send_from_directory

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import YOLOPv2 utilities
from utils.utils import (
    time_synchronized, select_device, increment_path,
    scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model,
    driving_area_mask, lane_line_mask, plot_one_box, show_seg_result,
    AverageMeter
)

app = Flask(__name__)

# Global variables for model
model = None
device = None

def load_model(weights_path='../data/weights/yolopv2.pt', device_str='cpu'):
    """Load YOLOPv2 model"""
    global model, device
    
    # Select device
    device = select_device(device_str)
    map_location = torch.device('cpu') if device.type == 'cpu' else device
    
    # Load model
    model = torch.jit.load(weights_path, map_location=map_location)
    model = model.to(device)
    
    # Half precision
    half = device.type != 'cpu' and device.type != 'mps'
    if half:
        model.half()
        
    model.eval()
    
    # Warmup
    if device.type != 'cpu':
        model(torch.zeros(1, 3, 640, 640).to(device).type_as(next(model.parameters())))
    
    return model

def process_image(image_bytes, img_size=640, conf_thres=0.3, iou_thres=0.45, 
                 show_objects=True, show_drivable=True, show_lanes=True):
    """Process an image with YOLOPv2 model"""
    global model, device
    
    # Convert image from bytes to OpenCV format
    nparr = np.frombuffer(image_bytes, np.uint8)
    img0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Preprocess image
    img = letterbox(img0, img_size, stride=32)[0]
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.ascontiguousarray(img)
    
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # Inference
    t1 = time_synchronized()
    [pred, anchor_grid], seg, ll = model(img)
    t2 = time_synchronized()
    
    # Post-processing
    pred = split_for_trace_model(pred, anchor_grid)
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    
    # Get masks
    da_seg_mask = driving_area_mask(seg)
    ll_seg_mask = lane_line_mask(ll)
    
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
    
    # Create visualization
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
        # Create empty mask with same shape
        display_masks.append(np.zeros_like(da_seg_mask))
        
    if show_lanes:
        display_masks.append(ll_seg_mask)
    else:
        # Create empty mask with same shape
        display_masks.append(np.zeros_like(ll_seg_mask))
    
    # Apply segmentation mask
    im0 = show_seg_result(im0, tuple(display_masks), is_demo=True, return_img=True)
    
    # Convert OpenCV image to base64 for web display
    _, buffer = cv2.imencode('.jpg', im0)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        'image': img_base64,
        'inference_time': float(t2 - t1),
        'detections': results
    }

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
    result = process_image(
        img_bytes, 
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        show_objects=show_objects,
        show_drivable=show_drivable,
        show_lanes=show_lanes
    )
    
    return jsonify(result)

@app.route('/demo/<path:filename>')
def demo_image(filename):
    """Get demo image"""
    return send_from_directory('../data/demo', filename)

@app.route('/detect-demo', methods=['POST'])
def detect_demo():
    """Process a demo image"""
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image specified'}), 400
    
    # Get image path
    img_path = os.path.join('../data/demo', data['image'])
    
    # Read image file
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
    
    # Get display options
    show_objects_val = data.get('show_objects', True)
    show_drivable_val = data.get('show_drivable', True)
    show_lanes_val = data.get('show_lanes', True)
    
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
        conf_thres = float(data.get('conf_threshold', 0.3))
        iou_thres = float(data.get('iou_threshold', 0.45))
    except (ValueError, TypeError):
        conf_thres = 0.3
        iou_thres = 0.45
    
    # Log for debugging
    print(f"[DETECT-DEMO] Processing image '{data['image']}' with options:")
    print(f"  - show_objects: {show_objects} (raw: {repr(show_objects_val)})")
    print(f"  - show_drivable: {show_drivable} (raw: {repr(show_drivable_val)})")
    print(f"  - show_lanes: {show_lanes} (raw: {repr(show_lanes_val)})")
    print(f"  - conf_threshold: {conf_thres}")
    print(f"  - iou_threshold: {iou_thres}")
    
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

if __name__ == '__main__':
    # Load model before starting the app
    print('Loading YOLOPv2 model...')
    load_model()
    print('Model loaded successfully!')
    
    # Run Flask app
    app.run(host='0.0.0.0', port=8080, debug=True) 