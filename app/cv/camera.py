"""
Camera handler module.

Supports three source types so the same code works for:
  - Webcam during the hackathon demo  (CAMERA_SOURCE = 0)
  - Real CCTV RTSP stream             (CAMERA_SOURCE = "rtsp://...")
  - Recorded video for offline tests  (CAMERA_SOURCE = "/path/file.mp4")
"""

import cv2
from app.config import CAMERA_SOURCE


def get_camera() -> cv2.VideoCapture:
    """
    Initialise and return a VideoCapture object.

    For RTSP streams we disable OpenCV's internal buffer (set to 1 frame)
    so we always read the *latest* frame rather than a queued stale one –
    this keeps end-to-end latency well under the 3-second target.
    """
    cap = cv2.VideoCapture(CAMERA_SOURCE)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera source: {CAMERA_SOURCE!r}. "
            "Check CAMERA_SOURCE in app/config.py."
        )

    # For RTSP streams, reduce the internal frame buffer to 1 so we always
    # process the freshest frame (avoids built-up latency over time).
    if isinstance(CAMERA_SOURCE, str) and CAMERA_SOURCE.startswith("rtsp"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap