"""
Appliance-detection module.

Detects whether energy-consuming appliances (ceiling lights, projector
screens, monitors) appear to be ON by analysing pixel brightness.

Strategy
--------
1. **Ceiling-light ROI** — examine the top fraction of the frame for
   very-bright pixels that indicate fluorescent / LED ceiling panels.
2. **Screen / monitor ROI** — scan the lower two-thirds for large,
   uniformly-bright rectangular regions (active displays).
3. Return a single boolean ``appliance_on`` plus a raw brightness value
   the API can expose for debugging.

This approach uses only OpenCV (no extra model download) and runs fast
enough that it doesn't meaningfully add to per-frame latency.
"""
import cv2
import numpy as np
from app.config import (
    LIGHT_ROI_FRACTION,
    BRIGHTNESS_THRESHOLD,
    BRIGHT_PIXEL_MIN,
)


def detect_appliance(frame):
    """
    Determine whether any appliance is ON.

    Parameters
    ----------
    frame : np.ndarray  (BGR, H×W×3)

    Returns
    -------
    appliance_on : bool
    brightness   : float  — mean brightness of the ceiling ROI (0-255)
    """
    height, width = frame.shape[:2]

    # ------------------------------------------------------------------
    # 1. Ceiling-light detection
    # ------------------------------------------------------------------
    ceiling_bottom = int(height * LIGHT_ROI_FRACTION)
    ceiling_roi    = frame[0:ceiling_bottom, :]
    gray_ceiling   = cv2.cvtColor(ceiling_roi, cv2.COLOR_BGR2GRAY)

    # Count pixels brighter than threshold
    _, ceiling_mask   = cv2.threshold(gray_ceiling, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
    ceiling_bright_px = cv2.countNonZero(ceiling_mask)

    ceiling_on  = ceiling_bright_px >= BRIGHT_PIXEL_MIN
    brightness  = float(gray_ceiling.mean())

    # ------------------------------------------------------------------
    # 2. Screen / monitor detection (lower 70 % of frame)
    # ------------------------------------------------------------------
    screen_roi  = frame[ceiling_bottom:, :]
    gray_screen = cv2.cvtColor(screen_roi, cv2.COLOR_BGR2GRAY)

    # A screen tends to be a large bright rectangle.
    # Use adaptive threshold + contour area to find it.
    blurred     = cv2.GaussianBlur(gray_screen, (5, 5), 0)
    _, s_thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(s_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    screen_on = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # A projector screen / monitor is large — at least 5 % of the ROI
        if area > 0.05 * screen_roi.shape[0] * screen_roi.shape[1]:
            screen_on = True
            break

    appliance_on = ceiling_on or screen_on
    return appliance_on, brightness
