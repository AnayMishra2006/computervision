"""
Face-detection module using OpenCV's Haar-cascade classifier.

Used as a secondary occupancy signal (catches seated / stationary people
that YOLO may miss) and as the face-localisation step inside the
privacy/anonymisation pipeline.
"""
import cv2
from app.config import FACE_MIN_SIZE

# Load the pre-trained model once — loading inside the function would
# re-read the XML file on every call which is unnecessarily slow.
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces(frame):
    """
    Detect frontal faces in *frame*.

    Parameters
    ----------
    frame : np.ndarray  (BGR, H×W×3)

    Returns
    -------
    face_count : int
    faces      : list/array of (x, y, w, h) rectangles
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # equalizeHist improves detection in variable / low lighting
    gray  = cv2.equalizeHist(gray)

    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,                      # image scale between pyramid scans
        minNeighbors=5,                        # higher = fewer false positives
        minSize=(FACE_MIN_SIZE, FACE_MIN_SIZE), # smallest face to detect
    )

    # detectMultiScale returns an empty tuple when nothing is found
    if len(faces) == 0:
        return 0, []

    return len(faces), faces