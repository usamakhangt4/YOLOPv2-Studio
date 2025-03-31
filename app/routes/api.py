"""
API routes for YOLOPv2-Studio application.
"""
import os
import base64
import traceback
from flask import Blueprint, request, jsonify, send_from_directory

# Import app modules
from app.config import MODEL_SETTINGS, DEMO_SETTINGS
from app.models import model_manager
from app.utils_app import image_processor

# Create blueprint
api_bp = Blueprint('api', __name__)

@api_bp.route('/system-info', methods=['GET'])
def system_info():
    """Get system information endpoint"""
    return jsonify(model_manager.get_system_info())

@api_bp.route('/set-device', methods=['POST'])
def set_device():
    """Change the device being used"""
    try:
        data = request.json
        if not data or 'device' not in data:
            return jsonify({'error': 'No device specified'}), 400
            
        device_str = data['device']
        model = model_manager.load_model(device_str=device_str)
        
        if model is None:
            return jsonify({'error': 'Failed to load model on specified device'}), 500
            
        return jsonify({
            'success': True, 
            'device': str(model_manager.device),
            'model_loaded': model is not None
        })
    except Exception as e:
        print(f"[ERROR] Error setting device: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@api_bp.route('/process', methods=['POST'])
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
        
        # Get display options from form data
        show_objects_val = request.form.get('show_objects', 'true')
        show_drivable_val = request.form.get('show_drivable', 'true')
        show_lanes_val = request.form.get('show_lanes', 'true')
        
        # Convert string values to boolean
        show_objects = show_objects_val.lower() in ['true', '1', 't', 'yes', 'y', 'on']
        show_drivable = show_drivable_val.lower() in ['true', '1', 't', 'yes', 'y', 'on']
        show_lanes = show_lanes_val.lower() in ['true', '1', 't', 'yes', 'y', 'on']
        
        # Get threshold values
        try:
            conf_thres = float(request.form.get('conf_threshold', MODEL_SETTINGS['DEFAULT_CONF_THRES']))
            iou_thres = float(request.form.get('iou_threshold', MODEL_SETTINGS['DEFAULT_IOU_THRES']))
        except ValueError:
            conf_thres = MODEL_SETTINGS['DEFAULT_CONF_THRES']
            iou_thres = MODEL_SETTINGS['DEFAULT_IOU_THRES']
        
        # Log for debugging
        print(f"[PROCESS] Processing uploaded image with options:")
        print(f"  - show_objects: {show_objects}")
        print(f"  - show_drivable: {show_drivable}")
        print(f"  - show_lanes: {show_lanes}")
        print(f"  - conf_threshold: {conf_thres}")
        print(f"  - iou_threshold: {iou_thres}")
        
        # Process image with model
        try:
            result = image_processor.process_image(
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
            traceback.print_exc()
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
            
    except Exception as e:
        print(f"[ERROR] Unexpected error in process route: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@api_bp.route('/demo/<path:filename>')
def demo_image(filename):
    """Get demo image"""
    demo_dir = DEMO_SETTINGS['DEMO_DIR']
    try:
        return send_from_directory(demo_dir, filename)
    except Exception as e:
        return jsonify({'error': f'Error serving demo image: {str(e)}'}), 500

@api_bp.route('/detect-demo', methods=['POST'])
def detect_demo():
    """Process a demo image"""
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image specified'}), 400
        
        # Get absolute path to demo directory
        demo_dir = DEMO_SETTINGS['DEMO_DIR']
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
            conf_thres = float(data.get('conf_threshold', MODEL_SETTINGS['DEFAULT_CONF_THRES']))
            iou_thres = float(data.get('iou_threshold', MODEL_SETTINGS['DEFAULT_IOU_THRES']))
        except (ValueError, TypeError) as e:
            print(f"[ERROR] Invalid threshold values: {str(e)}")
            conf_thres = MODEL_SETTINGS['DEFAULT_CONF_THRES']
            iou_thres = MODEL_SETTINGS['DEFAULT_IOU_THRES']
        
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
            result = image_processor.process_image(
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
            traceback.print_exc()
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
            
    except Exception as e:
        print(f"[ERROR] Unexpected error in detect_demo: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@api_bp.route('/camera', methods=['POST'])
def camera():
    """Process a camera frame"""
    try:
        if 'image' not in request.json:
            return jsonify({'error': 'No image data'}), 400
        
        # Get base64 image from request
        img_data = request.json['image'].split(',')[1] if ',' in request.json['image'] else request.json['image']
        img_bytes = base64.b64decode(img_data)
        
        # Get display options
        show_objects = bool(request.json.get('show_objects', True))
        show_drivable = bool(request.json.get('show_drivable', True))
        show_lanes = bool(request.json.get('show_lanes', True))
        
        # Get threshold values with error handling
        try:
            conf_thres = float(request.json.get('conf_threshold', MODEL_SETTINGS['DEFAULT_CONF_THRES']))
            iou_thres = float(request.json.get('iou_threshold', MODEL_SETTINGS['DEFAULT_IOU_THRES']))
        except (ValueError, TypeError):
            conf_thres = MODEL_SETTINGS['DEFAULT_CONF_THRES']
            iou_thres = MODEL_SETTINGS['DEFAULT_IOU_THRES']
        
        # Process image with model (reduced logging for camera frames to avoid spam)
        try:
            result = image_processor.process_image(
                img_bytes,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                show_objects=show_objects,
                show_drivable=show_drivable,
                show_lanes=show_lanes
            )
            return jsonify(result)
        except Exception as e:
            print(f"[ERROR] Camera processing error: {str(e)}")
            return jsonify({'error': f'Error processing camera frame: {str(e)}'}), 500
            
    except Exception as e:
        print(f"[ERROR] Unexpected error in camera route: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500 