# app/cv/face_detector.py
"""
Haar-cascade face detector – secondary occupancy signal.

Why a second detector?
-----------------------
YOLO counts *whole-body* detections and can miss people who are partially
occluded (e.g. sitting at a desk with their torso hidden).  The Haar face
detector catches faces from a head-on angle even when the body is hidden,
giving us a "belt-and-braces" approach to avoiding false negatives (the
sleeping-student problem mentioned in the problem statement).

If EITHER the body detector OR the face detector fires → room is occupied.
"""

import cv2

# Load the pre-trained frontal-face cascade that ships with OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces(frame):
    """
    Detect frontal faces in *frame*.

    Parameters
    ----------
    frame : np.ndarray  BGR image

    Returns
    -------
    face_count : int
    faces      : list of (x, y, w, h) tuples (may be empty ndarray)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,   # pyramid step – smaller = slower but catches more scales
        minNeighbors=5,    # higher = fewer false positives
        minSize=(30, 30),  # ignore tiny detections (noise / reflections)
    )

    return len(faces), faces