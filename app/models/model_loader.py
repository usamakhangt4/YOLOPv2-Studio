"""
Model loader module for YOLOPv2.
Handles loading, downloading, and initializing the model.
"""
import os
import sys
import time
import platform
import urllib.request
import traceback
from pathlib import Path

import torch
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import YOLOPv2 utilities
from utils.utils import select_device, time_synchronized

# Import app config
from app.config import MODEL_SETTINGS, PERFORMANCE, PLATFORM, IS_WINDOWS, IS_MACOS

class ModelManager:
    """
    Manages the YOLO model including loading, device selection, and inference.
    """
    def __init__(self):
        """Initialize the model manager."""
        self.model = None
        self.device = None
        self.weights_path = MODEL_SETTINGS['WEIGHTS_PATH']
        self.weights_url = MODEL_SETTINGS['WEIGHTS_URL']
        self.half_precision = PERFORMANCE['ENABLE_HALF_PRECISION']
        self.enable_tf32 = PERFORMANCE['ENABLE_TF32']
        self.warmup_cycles = PERFORMANCE['WARMUP_CYCLES']
        
    def download_weights(self):
        """Download model weights if they don't exist."""
        if not os.path.exists(self.weights_path):
            print(f"Downloading model weights from {self.weights_url}...")
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
                
                # Download the file
                urllib.request.urlretrieve(self.weights_url, self.weights_path)
                
                # Verify download was successful
                if os.path.exists(self.weights_path) and os.path.getsize(self.weights_path) > 0:
                    print(f"Model weights downloaded successfully to {self.weights_path}!")
                    return True
                else:
                    print("Downloaded file is empty or not found.")
                    return False
            except Exception as e:
                print(f"Error downloading model weights: {str(e)}")
                traceback.print_exc()
                return False
        return True
    
    def suggest_device(self):
        """Suggest the best device based on system capabilities."""
        # Check for CUDA first
        if torch.cuda.is_available():
            return '0'  # Use first CUDA device
            
        # Check for MPS (Apple Silicon) second
        if IS_MACOS and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'  # Use Apple Metal
            
        # Default to CPU
        return 'cpu'
    
    def load_model(self, device_str=None):
        """
        Load the YOLOPv2 model.
        
        Args:
            device_str: Device to use ('0', 'cpu', 'mps', etc.)
            
        Returns:
            The loaded model, or None if loading failed
        """
        # Use suggested device if none specified
        if device_str is None:
            device_str = self.suggest_device()
        
        # Ensure weights exist
        if not os.path.exists(self.weights_path):
            if not self.download_weights():
                print(f"Failed to download model weights to {self.weights_path}")
                return None
        
        # Select device
        self.device = select_device(device_str)
        print(f"Using device: {self.device}")
        
        try:
            # Check if file exists
            if not os.path.isfile(self.weights_path):
                raise FileNotFoundError(f"Model weights file not found at {self.weights_path}")
                
            # Load model with specified device - use torch.jit.load for TorchScript models
            model = torch.jit.load(self.weights_path, map_location=self.device)
            
            # Check if model loaded correctly
            if model is None:
                raise RuntimeError("Model loaded as None")
                
            # Configure model based on device capabilities
            self._configure_model_for_device(model)
            
            # Set model to evaluation mode
            model.to(self.device).eval()
            
            # Verify model has parameters
            try:
                param = next(model.parameters())
                print(f"Model loaded with parameters, dtype={param.dtype}, device={param.device}")
            except StopIteration:
                print("WARNING: Model has no parameters!")
            
            # Set the global model reference
            self.model = model
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            return None
            
    def _configure_model_for_device(self, model):
        """
        Configure model based on the device capabilities.
        
        Args:
            model: The loaded model to configure
        """
        if self.device.type == 'cuda':
            if self.half_precision:
                print("Using half precision for CUDA device")
                model = model.half()
                
            # Performance optimization: set CUDA optimization flags
            torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
            
            if self.enable_tf32:
                if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                    torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True
                    
            # Warmup model with a dummy input
            self._warmup_model(model)
            
        elif self.device.type == 'mps':
            print("Using MPS (Metal Performance Shaders) device")
            # MPS-specific optimizations could be added here
    
    def _warmup_model(self, model):
        """
        Warm up the model with a dummy input to initialize GPU kernels.
        
        Args:
            model: The loaded model to warm up
        """
        input_size = MODEL_SETTINGS['IMG_SIZE']
        print(f"Warming up model with {self.warmup_cycles} cycles...")
        
        try:
            dummy_input = torch.zeros(1, 3, input_size, input_size).to(self.device)
            if self.device.type == 'cuda' and next(model.parameters()).dtype == torch.float16:
                dummy_input = dummy_input.half()
                
            with torch.no_grad():
                for _ in range(self.warmup_cycles):
                    _ = model(dummy_input)
                    
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
                
            print("Warmup complete")
        except Exception as e:
            print(f"Warning: Model warmup failed: {e}")
            
    def get_system_info(self):
        """
        Get system and model information.
        
        Returns:
            Dictionary with system information
        """
        info = {
            'os': platform.system(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'current_device': str(self.device) if self.device else 'not set',
            'available_devices': [],
            'model_loaded': self.model is not None,
        }
        
        # Get available devices
        if torch.cuda.is_available():
            info['available_devices'].extend([f'cuda:{i}' for i in range(torch.cuda.device_count())])
        if platform.system() == 'Darwin' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info['available_devices'].append('mps')
        info['available_devices'].append('cpu')
        
        return info

# Create a global instance of the model manager
model_manager = ModelManager() 