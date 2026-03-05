# app/api/server.py
"""
FastAPI server – the public interface that your website friend will call.

Endpoints
---------
GET  /           – health check
GET  /status     – latest room status (JSON)
GET  /metrics    – running detection metrics (JSON)
GET  /stream     – MJPEG live video stream (for embedding in an <img> tag)
GET  /frame      – latest single frame as base64-encoded JPEG (JSON)

CORS
----
All origins are allowed so the front-end can run on any port/domain
during development.  Tighten this in production by listing only your
website's domain in `allow_origins`.
"""

import base64
import time
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.api.state import latest_state, latest_metrics
from app.api.schemas import StatusResponse, MetricsResponse
from app.cv import frame_store

app = FastAPI(
    title="Watt-Watch CV API",
    description=(
        "Real-time energy audit API powered by computer vision. "
        "Detects occupancy and appliance state; triggers IoT actions via ESP32."
    ),
    version="2.0",
)

# ------------------------------------------------------------------
# CORS – required so a browser on a different port can call this API
# ------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # ← tighten to your website domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    """Basic health-check endpoint."""
    return {"message": "Watt-Watch CV API is running ✅"}


@app.get("/status", response_model=StatusResponse, tags=["Detection"])
def get_status():
    """
    Return the most recent room status snapshot.

    Your website can poll this every second (or use WebSockets for
    push) to update the dashboard in real time.
    """
    return latest_state


@app.get("/metrics", response_model=MetricsResponse, tags=["Detection"])
def get_metrics():
    """
    Return running precision / recall / F1 metrics for the waste detector.
    Useful for a 'model health' panel on the dashboard.
    """
    return latest_metrics


@app.get("/stream", tags=["Video"])
def video_stream():
    """
    MJPEG live video stream – embed in your website with:

        <img src="http://<server>:8000/stream" />

    Each frame is the anonymised (blurred) output, so no raw faces are
    ever exposed to the front-end.
    """
    def _generate():
        while True:
            jpeg = frame_store.get_frame()
            if jpeg is not None:
                # MJPEG multipart format understood by browsers
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
            else:
                # No frame yet – wait briefly before retrying
                time.sleep(0.05)

    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/frame", tags=["Video"])
def single_frame():
    """
    Return the latest processed frame as a base64-encoded JPEG string.

    Useful for dashboard thumbnails or one-off snapshots without setting
    up a continuous stream.

    Response JSON:
        { "frame": "<base64 string>", "content_type": "image/jpeg" }
    """
    jpeg = frame_store.get_frame()
    if jpeg is None:
        return Response(content='{"error": "No frame available yet"}',
                        media_type="application/json", status_code=503)

    encoded = base64.b64encode(jpeg).decode("utf-8")
    return {"frame": encoded, "content_type": "image/jpeg"}