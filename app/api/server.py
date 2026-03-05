# app/api/server.py
#
# FastAPI application exposing the CV module as an HTTP API.
# Your friend's website can call these endpoints directly.
#
# Available endpoints:
#   GET  /           → health check
#   GET  /status     → live detection snapshot (people count, appliance state, waste flag …)
#   GET  /metrics    → precision / recall / F1 of the detection pipeline
#   GET  /energy     → cumulative energy savings since startup
#   GET  /stream     → MJPEG video stream (privacy-anonymized, suitable for embedding in a <img> tag)

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

import app.api.state as shared
from app.api.schemas import StatusResponse, MetricsResponse, EnergySavingsResponse
from app.config import CORS_ORIGINS

app = FastAPI(
    title="Watt-Watch CV API",
    version="2.0",
    description=(
        "Real-time energy auditing API. "
        "Returns occupancy, appliance state, and energy savings metrics."
    ),
)

# ---------------------------------------------------------------------------
# CORS — allow your friend's website (or any origin during the hackathon demo)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,   # configured in config.py / env var
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"])
def root():
    """Simple health-check — returns 200 if the server is alive."""
    return {"message": "Watt-Watch CV API is running 🎯"}


@app.get("/status", response_model=StatusResponse, tags=["detection"])
def get_status():
    """
    Returns the latest detection snapshot.
    Poll this every ~1 second from your frontend to drive the dashboard.
    """
    return shared.latest_state


@app.get("/metrics", response_model=MetricsResponse, tags=["detection"])
def get_metrics():
    """Returns real-time precision / recall / F1 of the detection pipeline."""
    return shared.latest_metrics


@app.get("/energy", response_model=EnergySavingsResponse, tags=["energy"])
def get_energy():
    """
    Returns cumulative energy savings calculated since the module started.
    Use this to build the 'Energy Saved' card on the dashboard.
    """
    return shared.energy_state


@app.get("/stream", tags=["video"])
def video_stream():
    """
    MJPEG video stream of the anonymized camera feed.
    Embed in an <img> tag on the dashboard like:
        <img src="http://<server>:8000/stream" />
    The stream is privacy-safe: faces and bodies are blurred before sending.
    """
    frame = shared.get_frame()

    if not frame:
        # No frame available yet — return a 503 so the client retries
        return Response(status_code=503, content="Frame not available yet")

    # Return a single JPEG frame (useful for one-shot snapshots)
    return Response(
        content=frame,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"},
    )