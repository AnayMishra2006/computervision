"""
Watt-Watch application entry point.

Starts two long-running components in parallel:

  1. **CV Pipeline thread** — captures frames, runs YOLO, updates the
     shared state store, and publishes MQTT commands to the ESP32.

  2. **FastAPI server** — serves the REST API (GET /status, /metrics,
     /rooms) and the anonymised MJPEG video stream (GET /stream).

Usage
-----
    python run.py

Environment variables (all optional, see app/config.py for defaults):
    CAMERA_SOURCE   — 0 for webcam, or an RTSP URL for IP cameras
    YOLO_MODEL      — yolov8n.pt | yolov8s.pt | yolov8m.pt | yolov8l.pt
    MQTT_BROKER     — hostname of the MQTT broker (default: localhost)
    ROOM_ID         — room identifier used in MQTT topics (default: room101)
    API_PORT        — port the FastAPI server listens on (default: 8000)
"""
import threading
import logging
import uvicorn
from app.main   import main as cv_main
from app.config import API_HOST, API_PORT

# Configure root logger so every module's logger produces output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _start_cv_pipeline():
    """Run the CV pipeline — crashes are logged but don't kill the API."""
    try:
        cv_main()
    except Exception as exc:
        logger.exception("CV pipeline crashed: %s", exc)


if __name__ == "__main__":
    logger.info("Starting Watt-Watch CV pipeline thread …")
    cv_thread = threading.Thread(target=_start_cv_pipeline, daemon=True, name="cv-pipeline")
    cv_thread.start()

    logger.info("Starting Watt-Watch FastAPI server on %s:%d …", API_HOST, API_PORT)
    uvicorn.run(
        "app.api.server:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,     # reload=True would restart the whole process (breaks CV thread)
        log_level="info",
    )
