# app/api/state.py
#
# Shared mutable state between the CV loop (writer) and the FastAPI server (reader).
# A threading.Lock ensures that partial updates are never read by the API.

import threading

_lock = threading.Lock()

# ----- Live detection snapshot -----
latest_state: dict = {
    "people": 0,
    "faces": 0,
    "occupied": False,
    "appliance_on": False,
    "waste_detected": False,
    "brightness": 0,
    "latency": 0.0,
}

# ----- Performance metrics -----
latest_metrics: dict = {
    "precision": 0.0,
    "recall": 0.0,
    "f1_score": 0.0,
    "false_trigger_rate": 0.0,
}

# ----- Cumulative energy savings -----
# Updated by the CV loop every second a waste condition is active
energy_state: dict = {
    "total_waste_seconds": 0,     # seconds the room was wasting energy
    "energy_saved_kwh": 0.0,      # kilowatt-hours saved by acting on waste alerts
    "cost_saved": 0.0,            # money saved (₹ / $) based on local rate
}

# ----- Shared MJPEG frame (bytes, JPEG-encoded) -----
# The CV loop writes here; the /stream endpoint reads from here
latest_frame: bytes = b""


def update_state(data: dict):
    """Thread-safe update of the detection snapshot."""
    with _lock:
        latest_state.update(data)


def update_metrics(data: dict):
    """Thread-safe update of performance metrics."""
    with _lock:
        latest_metrics.update(data)


def update_energy(data: dict):
    """Thread-safe update of energy savings."""
    with _lock:
        energy_state.update(data)


def set_frame(frame_bytes: bytes):
    """Thread-safe update of the latest JPEG frame for streaming."""
    global latest_frame
    with _lock:
        latest_frame = frame_bytes


def get_frame() -> bytes:
    """Thread-safe read of the latest JPEG frame."""
    with _lock:
        return latest_frame