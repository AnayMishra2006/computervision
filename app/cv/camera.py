"""
Camera handler module.
"""
import cv2
from app.config import CAMERA_SOURCE

def get_camera():
    """Initialize and return camera stream."""
    
    cap = cv2.VideoCapture(CAMERA_SOURCE)

    # Safety check: ensure camera opened successfully
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check CAMERA_SOURCE.")

    return cap