# app/cv/appliance.py
"""
Appliance / artificial-lighting detector.

Strategy – ceiling ROI + bright-spot counting
---------------------------------------------
We look only at the upper portion of the frame (configurable via
LIGHT_ROI_FRACTION in config.py) where ceiling light fixtures live.
Counting pixels above a high-brightness threshold lets us distinguish
artificial electric lights from natural daylight, which tends to be
diffuse rather than producing sharp, very-bright spots.

Returns
-------
appliance_on : bool   – True when lights appear to be ON
brightness   : float  – Mean greyscale brightness of the ROI (for display)
"""

import cv2
from app.config import (
    LIGHT_ROI_FRACTION,
    LIGHT_PIXEL_THRESHOLD,
    LIGHT_MIN_BRIGHT_PIXELS,
)


def detect_appliance(frame):
    """
    Analyse the ceiling region of *frame* for active artificial lighting.

    Parameters
    ----------
    frame : np.ndarray  BGR image (H x W x 3)

    Returns
    -------
    appliance_on : bool
    brightness   : float  (mean greyscale of the ceiling ROI)
    """
    height, width = frame.shape[:2]

    # Crop to the top LIGHT_ROI_FRACTION of the image (ceiling area)
    roi_bottom = int(height * LIGHT_ROI_FRACTION)
    roi = frame[0:roi_bottom, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Threshold to isolate very bright spots (bulbs / tube lights)
    _, thresh = cv2.threshold(gray, LIGHT_PIXEL_THRESHOLD, 255, cv2.THRESH_BINARY)
    bright_pixels = cv2.countNonZero(thresh)

    # Lights are ON if the bright-pixel count exceeds the minimum
    appliance_on = bright_pixels > LIGHT_MIN_BRIGHT_PIXELS

    # Average brightness of the ROI – useful for display / debugging
    brightness = float(gray.mean())

    return appliance_on, brightness