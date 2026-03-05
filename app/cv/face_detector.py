# app/cv/face_detector.py
#
# Face detection using OpenCV's Haar Cascade.
# Used as a secondary occupancy signal — if a face is visible, the room is occupied
# even if the YOLO body detector missed the person (e.g. only their head is in frame).

import cv2

# Load the pre-trained cascade once at import time
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces(frame):
    """
    Detect frontal faces in *frame*.

    Returns
    -------
    face_count : int
    face_boxes : list of (x, y, w, h)
        Bounding rectangles suitable for pixelation by privacy.pixelate_faces().
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    # detectMultiScale returns a numpy array or an empty tuple — normalise to list
    face_boxes = list(faces) if len(faces) > 0 else []

    return len(face_boxes), face_boxes