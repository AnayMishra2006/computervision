"""
Central configuration file for Watt-Watch.

All tunable parameters live here so you never have to dig through
individual modules to change a threshold or broker address.
"""

# ---------------------------------------------------------------------------
# Camera / video source
# ---------------------------------------------------------------------------
# Can be:
#   0          – default webcam (demo / hackathon mode)
#   "rtsp://…" – real CCTV RTSP stream
#   "/path/to/video.mp4" – recorded file for offline testing
CAMERA_SOURCE = 0

# Output resolution (resize every frame to this before processing)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# JPEG compression quality sent through the API stream (0-100)
# Lower = less bandwidth, higher latency tolerance
JPEG_QUALITY = 50

# ---------------------------------------------------------------------------
# YOLO people detection
# ---------------------------------------------------------------------------
# Minimum confidence to count a detection as a person
CONFIDENCE_THRESHOLD = 0.3

# Input image size fed to YOLO (larger = more accurate for distant subjects)
# Must be a multiple of 32.  Use 1280 for large CCTV rooms.
YOLO_IMGSZ = 640

# Use half-precision (FP16) on CUDA GPU for faster inference
# Set False if running on CPU-only hardware
YOLO_HALF = False

# ---------------------------------------------------------------------------
# Appliance / lighting detection
# ---------------------------------------------------------------------------
# Fraction of frame height that counts as the "ceiling ROI"
# (where lights are expected to be)
LIGHT_ROI_FRACTION = 0.30

# Pixel brightness (0-255) above which a pixel is counted as "lit"
LIGHT_PIXEL_THRESHOLD = 230

# Minimum number of bright pixels in the ROI to declare lights ON
LIGHT_MIN_BRIGHT_PIXELS = 500

# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------
# How many consecutive seconds the room must be empty-with-appliance-on
# before a WASTE alert is raised (debounce to avoid flicker)
WASTE_DELAY_SECONDS = 5

# ---------------------------------------------------------------------------
# MQTT – ESP32 broker settings
# ---------------------------------------------------------------------------
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# Topics used to command devices on the ESP32
MQTT_TOPIC_LIGHTS = "wattwatch/room101/lights/cmd"
MQTT_TOPIC_FAN    = "wattwatch/room101/fan/cmd"
MQTT_TOPIC_WASTE  = "wattwatch/room101/waste"

# ---------------------------------------------------------------------------
# Energy savings estimation
# ---------------------------------------------------------------------------
# Assumed wattage of a typical classroom light fixture (Watts)
LIGHT_WATTAGE_W = 60.0

# Assumed wattage of a typical ceiling fan (Watts)
FAN_WATTAGE_W = 75.0