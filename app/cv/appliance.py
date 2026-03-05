# app/cv/appliance.py
#
# Appliance / artificial-light detection.
#
# Strategy:
#   Ceiling lights in a typical classroom sit in the upper 30 % of the frame.
#   We analyse that "ceiling ROI" for bright spots (pixel value > 230 on the
#   grey-scale image).  If enough bright pixels exist, we call lights "ON".
#
#   This avoids false positives from sunlight streaming through windows because
#   window glare appears at the sides / bottom of the frame, not the ceiling.

import cv2
from app.config import CEILING_ROI_FRACTION, BRIGHT_PIXEL_THRESHOLD


def detect_appliance(frame):
    """
    Detect whether artificial ceiling lights (or a projector screen) are ON.

    Parameters
    ----------
    frame : np.ndarray  (BGR)

    Returns
    -------
    appliance_on : bool  — True if lights/projector appear to be ON
    brightness   : float — mean brightness of the ceiling ROI (for display)
    """
    height, width, _ = frame.shape

    # Crop the ceiling region of the frame
    roi = frame[0 : int(height * CEILING_ROI_FRACTION), :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Count pixels brighter than threshold — these are light sources / reflections
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    bright_pixels = cv2.countNonZero(thresh)

    # Configured threshold: enough bright pixels means lights are ON
    appliance_on = bright_pixels > BRIGHT_PIXEL_THRESHOLD

    # Mean brightness is shown on the overlay for debugging / tuning
    brightness = float(gray.mean())

    return appliance_on, brightness