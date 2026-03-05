"""
Central configuration file for Watt-Watch.
Every tunable knob lives here so you never have to dig through source files.
"""
import os

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
# 0 = default webcam; pass a RTSP/HTTP URL string for a real CCTV feed
_cam_src = os.getenv("CAMERA_SOURCE", "0")
try:
    # Numeric value → local webcam index (0, 1, 2 …)
    CAMERA_SOURCE = int(_cam_src)
except ValueError:
    # Non-numeric value → RTSP / HTTP URL or video file path
    CAMERA_SOURCE = _cam_src

# Resolution fed into the model (affects both speed and accuracy)
FRAME_WIDTH  = int(os.getenv("FRAME_WIDTH",  640))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", 480))

# JPEG quality used to compress frames before processing (0-100)
# Lower = faster network streaming, slightly less accurate detection
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", 50))

# ---------------------------------------------------------------------------
# YOLO people-detection
# ---------------------------------------------------------------------------
# Model weights: n=nano(fast) | s=small | m=medium | l=large | x=extra-large(accurate)
# For a hackathon demo on a laptop, "n" or "s" is fine.
# For real CCTV/long-distance, bump to "m" or "l".
YOLO_MODEL_SIZE = os.getenv("YOLO_MODEL_SIZE", "n")

# Minimum confidence to count a bounding box as a person (0.0 – 1.0)
# Lower value = catches more people (better recall), but more false positives
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.3))

# Object-tracking algorithm: "bytetrack" (recommended) or "botsort"
# Tracking prevents the same person being counted twice across frames.
TRACKER = os.getenv("TRACKER", "bytetrack")

# ---------------------------------------------------------------------------
# Logic engine
# ---------------------------------------------------------------------------
# Room must be empty AND appliance ON for this many seconds before triggering
WASTE_DELAY_SECONDS = int(os.getenv("WASTE_DELAY_SECONDS", 5))

# ---------------------------------------------------------------------------
# Appliance / light detection
# ---------------------------------------------------------------------------
# Fraction of the frame height used as the "ceiling ROI" for light detection
CEILING_ROI_FRACTION = float(os.getenv("CEILING_ROI_FRACTION", 0.30))

# Number of bright pixels (above threshold 230) required to call lights "ON"
BRIGHT_PIXEL_THRESHOLD = int(os.getenv("BRIGHT_PIXEL_THRESHOLD", 500))

# ---------------------------------------------------------------------------
# Energy savings estimation
# ---------------------------------------------------------------------------
# Combined wattage of all appliances the system can cut (projector + AC + lights)
APPLIANCE_POWER_WATTS = float(os.getenv("APPLIANCE_POWER_WATTS", 500.0))

# Local electricity rate in ₹ (or $) per kWh
ENERGY_RATE_PER_KWH = float(os.getenv("ENERGY_RATE_PER_KWH", 8.0))

# ---------------------------------------------------------------------------
# MQTT (for ESP32 communication)
# ---------------------------------------------------------------------------
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT   = int(os.getenv("MQTT_PORT", 1883))

# Room identifier used in MQTT topics  e.g. wattwatch/room101/lights/cmd
MQTT_ROOM = os.getenv("MQTT_ROOM", "room101")

# Client ID shown in your MQTT broker logs
MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "wattwatch-cv-module")

# Keepalive interval in seconds
MQTT_KEEPALIVE = int(os.getenv("MQTT_KEEPALIVE", 60))

# ---------------------------------------------------------------------------
# API server
# ---------------------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Origins allowed to call this API (your friend's website domain goes here).
# "*" means any origin — fine for a hackathon demo.
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# ---------------------------------------------------------------------------
# Runtime flags
# ---------------------------------------------------------------------------
# Set HEADLESS=1 to disable the cv2.imshow() window (e.g. on a server with no display)
HEADLESS = os.getenv("HEADLESS", "0") == "1"