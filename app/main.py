"""
Main application for YOLOPv2-Studio.
This file initializes the Flask app, registers blueprints, and handles startup.
"""
import os
import sys
import logging
import traceback

from flask import Flask, jsonify

# Add the project root to the path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Import configuration
from app.config import SERVER_SETTINGS, PLATFORM_SETTINGS, LOG_LEVEL

# Import routes
from app.routes import api_bp, web_bp

# Import model manager
from app.models import model_manager

# Configure logging
logging_level = getattr(logging, LOG_LEVEL)
logging.basicConfig(
    level=logging_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(ROOT_DIR, 'app.log'))
    ]
)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Register blueprints
    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp)
    
    # Error handlers
    @app.errorhandler(Exception)
    def handle_exception(e):
        """Handle all unhandled exceptions"""
        logger.error(f"Unhandled exception: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
        
    return app

def init_model():
    """Initialize the model"""
    try:
        logger.info("Loading YOLOPv2 model...")
        device_str = model_manager.suggest_device()
        logger.info(f"Suggested device for this system: {device_str}")
        
        model = model_manager.load_model(device_str=device_str)
        if model is None:
            logger.error("Failed to load model")
            return False
            
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == '__main__':
    try:
        # Print system information
        logger.info("Starting YOLOPv2-Studio...")
        
        # Get system info
        system_info = model_manager.get_system_info()
        logger.info("\nSystem Information:")
        logger.info(f"Operating System: {system_info['os']}")
        logger.info(f"Python Version: {system_info['python_version']}")
        logger.info(f"PyTorch Version: {system_info['torch_version']}")
        logger.info(f"CUDA Available: {system_info['cuda_available']}")
        if system_info['cuda_available'] and system_info['cuda_version']:
            logger.info(f"CUDA Version: {system_info['cuda_version']}")
        
        # Initialize model
        if not init_model():
            logger.error("Failed to initialize model. Exiting.")
            sys.exit(1)
        
        # Create Flask app
        app = create_app()
        
        # Get server settings
        host = SERVER_SETTINGS['HOST']
        port = SERVER_SETTINGS['PORT']
        debug = SERVER_SETTINGS['DEBUG']
        # Get threading setting from platform-specific settings
        threaded = PLATFORM_SETTINGS.get('THREADED', SERVER_SETTINGS['THREADED'])
        
        logger.info(f"\nStarting server on {host}:{port} with threading={'enabled' if threaded else 'disabled'}")
        
        # Run the app
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=threaded
        )
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
        traceback.print_exc()
        sys.exit(1) 