"""
Web routes for YOLOPv2-Studio application.
"""
from flask import Blueprint, render_template

# Create blueprint
web_bp = Blueprint('web', __name__)

@web_bp.route('/')
def index():
    """Render the main application page"""
    return render_template('index.html') 