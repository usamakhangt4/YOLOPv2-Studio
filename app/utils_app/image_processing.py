"""
Image processing utilities for YOLOPv2.
Handles preprocessing, postprocessing, and visualization.
"""
import os
import sys
import time
import base64
import traceback
from io import BytesIO

import cv2
import torch
import numpy as np

# Add the project root to the path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

# Import YOLOPv2 utilities
from utils.utils import (
    time_synchronized, scale_coords, non_max_suppression, split_for_trace_model,
    driving_area_mask, lane_line_mask, plot_one_box, show_seg_result, letterbox
)

# Import app config and model
from app.config import MODEL_SETTINGS, PERFORMANCE
from app.models import model_manager

class ImageProcessor:
    """
    Handles image processing operations including preprocessing, inference, and visualization.
    """
    def __init__(self):
        """Initialize the image processor."""
        self.img_size = MODEL_SETTINGS['IMG_SIZE']
        self.max_dimension = PERFORMANCE['MAX_IMAGE_DIMENSION']
        self.jpeg_quality = PERFORMANCE['JPEG_QUALITY']
        
    def process_image(self, image_bytes, conf_thres=None, iou_thres=None, 
                     show_objects=True, show_drivable=True, show_lanes=True):
        """
        Process an image with YOLOPv2 model.
        
        Args:
            image_bytes: The input image as bytes
            conf_thres: Confidence threshold for detections
            iou_thres: IoU threshold for NMS
            show_objects: Whether to show object detections
            show_drivable: Whether to show drivable area
            show_lanes: Whether to show lane lines
            
        Returns:
            Dictionary with processed image and metadata
        """
        # Set default thresholds if not provided
        if conf_thres is None:
            conf_thres = MODEL_SETTINGS['DEFAULT_CONF_THRES']
        if iou_thres is None:
            iou_thres = MODEL_SETTINGS['DEFAULT_IOU_THRES']
            
        # Get model and device
        model = model_manager.model
        device = model_manager.device
        
        try:
            # Check if model is loaded
            if model is None:
                raise RuntimeError("Model not loaded")
                
            # Check if device is set
            if device is None:
                raise RuntimeError("Device not set")
                
            # Decode and preprocess image
            img0, decoded_img = self._decode_image(image_bytes)
            img = self._preprocess_image(decoded_img)
            
            # Run inference
            t1 = time_synchronized()
            # Synchronize before inference for accurate timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            with torch.no_grad():
                [pred, anchor_grid], seg, ll = model(img)
                
            # Synchronize after inference for accurate timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            t2 = time_synchronized()
            
            # Postprocess results
            result = self._postprocess_results(
                pred, anchor_grid, seg, ll, img, img0, 
                conf_thres, iou_thres, show_objects, show_drivable, show_lanes
            )
            
            # Add inference time
            result['inference_time'] = float(t2 - t1)
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Error in process_image: {str(e)}")
            traceback.print_exc()
            raise RuntimeError(f"Error processing image: {str(e)}")
            
    def _decode_image(self, image_bytes):
        """
        Decode image bytes to OpenCV format.
        
        Args:
            image_bytes: Image as bytes
            
        Returns:
            Tuple of (original image, possibly resized image)
        """
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img0 is None:
                raise ValueError("Failed to decode image")
                
            # Store original for later
            original_img = img0.copy()
            
            # Performance optimization: resize large images
            height, width = img0.shape[:2]
            
            if max(height, width) > self.max_dimension:
                # Calculate new dimensions while maintaining aspect ratio
                if width > height:
                    new_width = self.max_dimension
                    new_height = int(height * (self.max_dimension / width))
                else:
                    new_height = self.max_dimension
                    new_width = int(width * (self.max_dimension / height))
                
                img0 = cv2.resize(img0, (new_width, new_height), interpolation=cv2.INTER_AREA)
                print(f"[OPTIMIZE] Resized image from {width}x{height} to {new_width}x{new_height}")
            
            print(f"[DEBUG] Original image shape: {img0.shape}")
            return original_img, img0
            
        except Exception as e:
            print(f"[ERROR] Failed to decode image: {str(e)}")
            traceback.print_exc()
            raise ValueError(f"Failed to decode image: {str(e)}")
        
    def _preprocess_image(self, img0):
        """
        Preprocess image for model input.
        
        Args:
            img0: The decoded image
            
        Returns:
            Tensor ready for model inference
        """
        try:
            # Resize and pad image
            img, ratio, (dw, dh) = letterbox(img0, self.img_size, stride=32)
            
            # Convert to model input format
            img = img.transpose(2, 0, 1)  # HWC to CHW
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(model_manager.device)
            img = img.float() / 255.0
            
            # Convert to half precision if appropriate
            if model_manager.device.type == 'cuda':
                model_param = next(model_manager.model.parameters(), None)
                if model_param is not None and model_param.dtype == torch.float16:
                    img = img.half()
            
            # Add batch dimension if needed
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            print(f"[DEBUG] Preprocessed image shape: {img.shape}, dtype: {img.dtype}")
            return img
            
        except Exception as e:
            print(f"[ERROR] Failed to preprocess image: {str(e)}")
            traceback.print_exc()
            raise ValueError(f"Failed to preprocess image: {str(e)}")
    
    def _postprocess_results(self, pred, anchor_grid, seg, ll, img, img0, 
                           conf_thres, iou_thres, show_objects, show_drivable, show_lanes):
        """
        Postprocess model outputs.
        
        Args:
            Various model outputs and options
            
        Returns:
            Dictionary with processed results
        """
        try:
            # Move tensors to CPU for processing if they're on GPU
            if isinstance(pred, torch.Tensor) and pred.is_cuda:
                pred = pred.detach().cpu()
            if isinstance(anchor_grid, torch.Tensor) and anchor_grid.is_cuda:
                anchor_grid = anchor_grid.detach().cpu()
            
            # NMS and detection processing
            pred = split_for_trace_model(pred, anchor_grid)
            pred = non_max_suppression(pred, conf_thres, iou_thres)
            
            # Get segmentation masks
            da_seg_mask = driving_area_mask(seg)
            ll_seg_mask = lane_line_mask(ll)
            
            # Convert masks to numpy arrays
            if isinstance(da_seg_mask, torch.Tensor):
                da_seg_mask = da_seg_mask.squeeze().cpu().numpy()
            else:
                da_seg_mask = np.squeeze(da_seg_mask)
                
            if isinstance(ll_seg_mask, torch.Tensor):
                ll_seg_mask = ll_seg_mask.squeeze().cpu().numpy()
            else:
                ll_seg_mask = np.squeeze(ll_seg_mask)
            
            # Convert masks to uint8 for OpenCV
            da_seg_mask = (da_seg_mask * 255).astype(np.uint8)
            ll_seg_mask = (ll_seg_mask * 255).astype(np.uint8)
            
            # Resize masks to match original image size
            if da_seg_mask.shape[:2] != img0.shape[:2]:
                da_seg_mask = cv2.resize(da_seg_mask, (img0.shape[1], img0.shape[0]), interpolation=cv2.INTER_LINEAR)
            if ll_seg_mask.shape[:2] != img0.shape[:2]:
                ll_seg_mask = cv2.resize(ll_seg_mask, (img0.shape[1], img0.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            # Convert to binary masks
            da_seg_mask = da_seg_mask > 127
            ll_seg_mask = ll_seg_mask > 127
            
            # Process detections
            results = []
            for i, det in enumerate(pred):
                if len(det):
                    # Rescale boxes to original image size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                    
                    # Store detections
                    for *xyxy, conf, cls in reversed(det):
                        results.append({
                            'xyxy': [float(x) for x in xyxy],
                            'conf': float(conf),
                            'cls': int(cls)
                        })
            
            # Create visualization
            visualized_img = self._create_visualization(
                img0, pred[0] if len(pred) > 0 else None, 
                da_seg_mask, ll_seg_mask, show_objects, show_drivable, show_lanes
            )
            
            # Encode image as base64
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            _, buffer = cv2.imencode('.jpg', visualized_img, encode_params)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                'image': img_base64,
                'detections': results,
                'detection_count': len(results)
            }
            
        except Exception as e:
            print(f"[ERROR] Post-processing failed: {str(e)}")
            traceback.print_exc()
            raise RuntimeError(f"Post-processing failed: {str(e)}")
    
    def _create_visualization(self, img0, detections, da_seg_mask, ll_seg_mask, 
                             show_objects, show_drivable, show_lanes):
        """
        Create visualization of the detection and segmentation results.
        
        Args:
            Various detection results and options
            
        Returns:
            Visualized image
        """
        try:
            # Start with a copy of the original image
            im0 = img0.copy()
            
            # Draw bounding boxes for object detections
            if show_objects and detections is not None and len(detections):
                for *xyxy, conf, cls in reversed(detections):
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
            
            # Apply segmentation masks
            im0 = show_seg_result(im0, tuple(display_masks), is_demo=True, return_img=True)
            
            return im0
            
        except Exception as e:
            print(f"[ERROR] Visualization failed: {str(e)}")
            traceback.print_exc()
            raise RuntimeError(f"Visualization failed: {str(e)}")

# Create global instance
image_processor = ImageProcessor() 