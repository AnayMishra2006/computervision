"""
Entry point for Watt-Watch.

Starts two concurrent tasks:
  1. CV processing loop  (daemon thread – captures camera, runs YOLO, MQTT)
  2. FastAPI HTTP server (main thread – serves /status, /metrics, /stream)

Usage
-----
    python run.py

The server listens on http://0.0.0.0:8000
Your website friend can then call:
  GET /status  →  JSON snapshot of the current room state
  GET /metrics →  JSON detection performance metrics
  GET /stream  →  MJPEG live stream  (<img src="http://host:8000/stream" />)
  GET /frame   →  Single base64-encoded JPEG frame
"""

import threading
import uvicorn

from app.main import main as cv_main


def _start_cv_loop():
    """Run the CV processing loop in a background daemon thread."""
    try:
        cv_main()
    except Exception as exc:
        # Log but don't crash the whole process if the camera fails
        print(f"❌ CV loop exited with error: {exc}")


if __name__ == "__main__":
    # Start CV loop as a daemon so it stops automatically when the main
    # process (uvicorn) exits.
    cv_thread = threading.Thread(target=_start_cv_loop, daemon=True, name="cv-loop")
    cv_thread.start()

    print("🚀 Starting Watt-Watch API server on http://0.0.0.0:8000")
    uvicorn.run(
        "app.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # reload=True would spin up a second process and break threading
    )