# app/cv/frame_store.py
"""
Thread-safe shared frame buffer.

The CV processing loop writes the latest anonymised JPEG bytes here.
The FastAPI streaming endpoint reads from here to serve the live feed
to the website without needing direct access to the camera.
"""

import threading

# The most-recent JPEG-encoded frame (bytes), or None before first frame
latest_frame: bytes | None = None

# Lock to prevent simultaneous read+write from the CV thread and API thread
_lock = threading.Lock()


def set_frame(jpeg_bytes: bytes) -> None:
    """Store the latest processed frame (called by the CV loop)."""
    global latest_frame
    with _lock:
        latest_frame = jpeg_bytes


def get_frame() -> bytes | None:
    """Retrieve the latest processed frame (called by the API endpoint)."""
    with _lock:
        return latest_frame
