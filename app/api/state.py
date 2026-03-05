"""
Shared in-process state store.

The CV pipeline (main.py) writes to this store from its own thread;
the FastAPI request handlers read from it.  A threading.Lock protects
each dict so reads/writes never see a partially-updated snapshot.

A second dict — ``latest_frame`` — holds the most recent JPEG-encoded
anonymised frame for the MJPEG /stream endpoint.
"""
import threading

# ------------------------------------------------------------------
# Locks
# ------------------------------------------------------------------
_state_lock   = threading.Lock()
_metrics_lock = threading.Lock()
_frame_lock   = threading.Lock()

# ------------------------------------------------------------------
# Default values
# ------------------------------------------------------------------
latest_state: dict = {
    "room_id":       "room101",
    "people":        0,
    "faces":         0,
    "occupied":      False,
    "appliance_on":  False,
    "waste_detected": False,
    "brightness":    0,
    "latency":       0.0,
    "energy_saved_wh": 0.0,
}

latest_metrics: dict = {
    "precision":         0.0,
    "recall":            0.0,
    "f1_score":          0.0,
    "false_trigger_rate": 0.0,
}

# Raw JPEG bytes of the most recent anonymised frame (None until first frame)
latest_frame: dict = {"jpeg": None}


# ------------------------------------------------------------------
# Thread-safe updaters
# ------------------------------------------------------------------

def update_state(data: dict):
    with _state_lock:
        latest_state.update(data)


def update_metrics(data: dict):
    with _metrics_lock:
        latest_metrics.update(data)


def update_frame(jpeg_bytes: bytes):
    with _frame_lock:
        latest_frame["jpeg"] = jpeg_bytes


def get_frame() -> bytes | None:
    with _frame_lock:
        return latest_frame["jpeg"]


def get_state() -> dict:
    """Return a thread-safe snapshot of the current room state."""
    with _state_lock:
        return dict(latest_state)


def get_metrics() -> dict:
    """Return a thread-safe snapshot of the current metrics."""
    with _metrics_lock:
        return dict(latest_metrics)
