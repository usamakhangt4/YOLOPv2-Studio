"""
Configuration settings for YOLOPv2-Studio application.
This file centralizes all configuration parameters that might need adjustment.
"""
import os
import platform
from pathlib import Path

# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Model settings
MODEL_SETTINGS = {
    'WEIGHTS_URL': "https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt",
    'WEIGHTS_PATH': os.path.join(ROOT_DIR, 'weights', 'yolopv2.pt'),
    'IMG_SIZE': 640,
    'DEFAULT_CONF_THRES': 0.3,
    'DEFAULT_IOU_THRES': 0.45,
}

# Demo image settings
DEMO_SETTINGS = {
    'DEMO_DIR': os.path.join(ROOT_DIR, 'data', 'demo'),
}

# Server settings
SERVER_SETTINGS = {
    'HOST': '0.0.0.0',
    'PORT': 8080,
    'DEBUG': False,  # Set to False in production
    'THREADED': True,  # May need to be False with certain ML frameworks
}

# Platform-specific optimizations
PLATFORM = platform.system()
IS_WINDOWS = PLATFORM == 'Windows'
IS_MACOS = PLATFORM == 'Darwin'
IS_LINUX = PLATFORM == 'Linux'

# Performance settings
PERFORMANCE = {
    'MAX_IMAGE_DIMENSION': 1280,  # Resize images larger than this
    'JPEG_QUALITY': 85,  # JPEG encoding quality for responses
    'ENABLE_HALF_PRECISION': True,  # Use FP16 on supported devices
    'ENABLE_TF32': True,  # Use TF32 on supported NVIDIA GPUs
    'WARMUP_CYCLES': 2,  # Number of warmup cycles for the model
}

# Logging
LOG_LEVEL = 'INFO'  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Function to get platform-specific settings
def get_platform_specific_settings():
    if IS_WINDOWS:
        return {
            'THREADED': False,  # Better CUDA performance on Windows with threading disabled
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
        }
    elif IS_MACOS:
        return {
            'ENABLE_MPS': True,  # Enable Metal Performance Shaders on macOS
        }
    else:  # Linux
        return {
            'THREADED': True,
        }

# Update settings with platform-specific configurations
PLATFORM_SETTINGS = get_platform_specific_settings() 