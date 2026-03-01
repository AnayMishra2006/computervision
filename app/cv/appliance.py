# app/cv/appliance.py
import cv2

def detect_appliance(frame):
    """
    Detect artificial lighting using ceiling ROI + bright spot detection.
    This avoids sunlight false positives.
    """

    height, width, _ = frame.shape

    # Assume ceiling lights are in upper 30% of frame
    roi = frame[0:int(height * 0.3), :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Detect bright spots (bulbs/tubes)
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    bright_pixels = cv2.countNonZero(thresh)

    # Artificial light ON if many bright pixels
    appliance_on = bright_pixels > 500

    # Average brightness for display
    brightness = gray.mean()

    return appliance_on, brightness