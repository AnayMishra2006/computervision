"""
Central configuration file for Watt-Watch.

Every setting can be overridden by an environment variable of the same name.
Example:  CAMERA_SOURCE=rtsp://192.168.1.5/stream1 python run.py
"""
import os

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

# 0 = default webcam; use an RTSP URL string for IP/CCTV cameras
# e.g. "rtsp://admin:pass@192.168.1.100:554/stream1"
CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", "0")
# Convert "0" → int so OpenCV accepts it as a device index
if CAMERA_SOURCE.isdigit():
    CAMERA_SOURCE = int(CAMERA_SOURCE)

# CCTV-style resolution
FRAME_WIDTH  = int(os.environ.get("FRAME_WIDTH",  640))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", 480))

# JPEG compression quality for the internal frame bus (0-100)
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", 50))

# ---------------------------------------------------------------------------
# YOLO / Detection
# ---------------------------------------------------------------------------

# Minimum confidence to count a detection (lower → catch far-away people)
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.25))

# YOLO model variant: yolov8n (fastest) | yolov8s | yolov8m (balanced) | yolov8l/x (best)
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8m.pt")

# Image size fed to YOLO — larger = better long-distance detection, slower
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", 640))

# ---------------------------------------------------------------------------
# Appliance / Brightness detection
# ---------------------------------------------------------------------------

# Fraction of frame height treated as the "ceiling / light ROI"
LIGHT_ROI_FRACTION = float(os.environ.get("LIGHT_ROI_FRACTION", 0.30))

# Pixel brightness threshold for "light is ON" (0-255)
BRIGHTNESS_THRESHOLD = int(os.environ.get("BRIGHTNESS_THRESHOLD", 220))

# Minimum number of above-threshold pixels to declare a light ON
BRIGHT_PIXEL_MIN = int(os.environ.get("BRIGHT_PIXEL_MIN", 300))

# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------

# Minimum face size (pixels) for Haar-cascade — smaller = catch far-away faces
# but also increases false positives; tune per camera height
FACE_MIN_SIZE = int(os.environ.get("FACE_MIN_SIZE", 25))

# ---------------------------------------------------------------------------
# Logic Engine
# ---------------------------------------------------------------------------

# Seconds that the "empty + appliance ON" condition must persist before alert
WASTE_DELAY_SECONDS = int(os.environ.get("WASTE_DELAY_SECONDS", 5))

# Watts assumed per room (used for energy-saved calculation)
ROOM_WATTAGE = float(os.environ.get("ROOM_WATTAGE", 500.0))

# ---------------------------------------------------------------------------
# MQTT  (talks to the ESP32)
# ---------------------------------------------------------------------------

MQTT_BROKER   = os.environ.get("MQTT_BROKER", "localhost")
MQTT_PORT     = int(os.environ.get("MQTT_PORT", 1883))
MQTT_USERNAME = os.environ.get("MQTT_USERNAME", "")  # leave blank if no auth
MQTT_PASSWORD = os.environ.get("MQTT_PASSWORD", "")

# Room / device identifiers that appear in MQTT topics
ROOM_ID = os.environ.get("ROOM_ID", "room101")

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", 8000))