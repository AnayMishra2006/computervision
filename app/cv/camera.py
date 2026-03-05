"""
Camera handler module.

Supports:
  - USB webcam  (integer device index, e.g. 0)
  - IP / CCTV streams via RTSP URL string
  - Auto-reconnect on dropped frames
"""
import cv2
import time
import logging
from app.config import CAMERA_SOURCE, FRAME_WIDTH, FRAME_HEIGHT

logger = logging.getLogger(__name__)


class Camera:
    """
    Thin wrapper around cv2.VideoCapture that handles reconnections
    automatically. This is crucial for RTSP streams that drop occasionally.
    """

    def __init__(self, source=CAMERA_SOURCE, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        self.source = source
        self.width  = width
        self.height = height
        self._cap   = None
        self._connect()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self):
        """Open (or reopen) the capture device."""
        if self._cap is not None:
            self._cap.release()

        self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera source: {self.source!r}. "
                "Check CAMERA_SOURCE in your environment or config.py."
            )

        # Set resolution hint (may be ignored by some RTSP streams)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Reduce internal buffer for lower latency (1 = most recent frame only)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        logger.info("Camera opened: source=%r  %dx%d", self.source, self.width, self.height)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self):
        """
        Read one frame.  On failure, attempt a single reconnect after a
        short delay before giving up.  Returns (success: bool, frame).
        """
        ret, frame = self._cap.read()

        if not ret:
            logger.warning("Frame read failed — attempting reconnect …")
            time.sleep(1.0)
            try:
                self._connect()
                ret, frame = self._cap.read()
            except RuntimeError as exc:
                logger.error("Reconnect failed: %s", exc)

        return ret, frame

    def release(self):
        """Release the underlying VideoCapture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def get_camera() -> Camera:
    """Factory function — returns a ready-to-use Camera instance."""
    return Camera()
