# run.py
#
# Entry point for the Watt-Watch application.
#
# Architecture:
#   ┌──────────────────────────────────────┐
#   │  Thread 1: CV loop  (app/main.py)    │  ← reads camera, runs YOLO, writes shared state
#   │  Thread 2: FastAPI  (app/api/server) │  ← serves HTTP API + MJPEG stream
#   └──────────────────────────────────────┘
#
# Your friend's website calls the FastAPI endpoints.
# The ESP32 subscribes to MQTT topics published by the CV loop.
#
# Usage:
#   python run.py
#
# Environment variables (all optional, see app/config.py for defaults):
#   CAMERA_SOURCE, HEADLESS, MQTT_BROKER, MQTT_PORT, MQTT_ROOM,
#   YOLO_MODEL_SIZE, CONFIDENCE_THRESHOLD, API_HOST, API_PORT, CORS_ORIGINS

import threading
import uvicorn

from app.main import main as cv_main
from app.config import API_HOST, API_PORT


def start_cv_thread():
    """Run the computer-vision pipeline in a daemon thread."""
    try:
        cv_main()
    except Exception as exc:
        # Log and let the main process continue so the API stays up
        print(f"❌ CV loop crashed: {exc}")


if __name__ == "__main__":
    # Start the CV loop in a background daemon thread.
    # Daemon=True means it will be killed automatically when the main process exits.
    cv_thread = threading.Thread(target=start_cv_thread, daemon=True, name="cv-loop")
    cv_thread.start()
    print(f"🚀 Watt-Watch API starting on http://{API_HOST}:{API_PORT}")
    print(f"   Docs: http://{API_HOST}:{API_PORT}/docs")

    # Run FastAPI in the main thread (blocking call)
    uvicorn.run(
        "app.api.server:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,          # reload=True breaks the CV thread
        log_level="warning",   # keep console clean; CV loop already prints status
    )