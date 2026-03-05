"""
Camera handler module.

Supports:
  - Local webcam  : CAMERA_SOURCE = 0
  - RTSP CCTV     : CAMERA_SOURCE = "rtsp://user:pass@192.168.1.100:554/stream"
  - HTTP MJPEG    : CAMERA_SOURCE = "http://192.168.1.100/video"
  - Video file    : CAMERA_SOURCE = "recording.mp4"
"""
import cv2
from app.config import CAMERA_SOURCE


def get_camera() -> cv2.VideoCapture:
    """
    Initialise and return a VideoCapture object.

    For RTSP streams a short timeout is set so the system does not hang
    indefinitely if the camera is unreachable.
    """
    # CAP_FFMPEG gives better RTSP / network stream support than the default back-end
    cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_FFMPEG)

    # Safety check: ensure camera opened successfully
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera source '{CAMERA_SOURCE}'. "
            "Check CAMERA_SOURCE in config.py or the CAMERA_SOURCE env var."
        )

    return cap