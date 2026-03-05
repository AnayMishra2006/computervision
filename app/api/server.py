"""
FastAPI application — Watt-Watch REST API.

Endpoints
---------
GET  /               → health check
GET  /status         → live room state (people, appliance, waste flag, …)
GET  /metrics        → detection-quality metrics (precision, recall, F1)
GET  /stream         → MJPEG video stream of the anonymised camera feed
GET  /rooms          → list of active rooms (extensible for multi-room)

CORS is fully open so the frontend team can call this API from any origin
without running into browser security blocks during the hackathon demo.
"""
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api import state as st
from app.api.schemas import StatusResponse, MetricsResponse, HealthResponse

# ---------------------------------------------------------------------------
# App creation
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Watt-Watch API",
    version="2.0",
    description=(
        "Campus Energy Auditor — Computer Vision backend. "
        "Call /status for live room state, /stream for anonymised MJPEG feed."
    ),
)

# Allow any origin so the frontend dev can point their website at this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse, tags=["Health"])
def root():
    """Health check — confirms the API is alive."""
    return {"status": "ok", "version": "2.0"}


@app.get("/status", response_model=StatusResponse, tags=["Detection"])
def get_status():
    """
    Return the latest room-state snapshot produced by the CV pipeline.
    Safe to poll every second from the frontend dashboard.
    """
    return st.get_state()


@app.get("/metrics", response_model=MetricsResponse, tags=["Detection"])
def get_metrics():
    """Return self-evaluated detection metrics (precision, recall, F1, FTR)."""
    return st.get_metrics()


@app.get("/rooms", tags=["Rooms"])
def list_rooms():
    """
    Return a list of monitored rooms with their current status.
    Currently single-room; extend by adding more WasteDetector instances.
    """
    snap = st.get_state()
    return {
        "rooms": [
            {
                "room_id":        snap["room_id"],
                "occupied":       snap["occupied"],
                "waste_detected": snap["waste_detected"],
                "appliance_on":   snap["appliance_on"],
                "energy_saved_wh": snap["energy_saved_wh"],
            }
        ]
    }


@app.get("/stream", tags=["Video"])
def video_stream():
    """
    MJPEG stream of the live, privacy-anonymised camera feed.

    Use this URL directly in an <img> tag:
        <img src="http://<host>:8000/stream" />

    The stream blocks until the pipeline produces its first frame, then
    delivers one JPEG per processed frame (~10-25 fps depending on hardware).
    """
    def _generate():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            jpeg = st.get_frame()
            if jpeg is None:
                # Pipeline not started yet — wait briefly
                time.sleep(0.05)
                continue
            yield boundary + jpeg + b"\r\n"

    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
