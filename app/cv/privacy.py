"""
Privacy / anonymisation module.

Implements a two-layer "Ghost Mode":
  1. **Body blur** — Gaussian blur every bounding box returned by YOLO so
     that individual body features are completely indistinguishable.
  2. **Face blur** — additional heavy Gaussian blur over any detected face
     bounding box (from the Haar-cascade face detector) to guarantee that
     even partial detections are covered.

No raw face or body data is ever stored or forwarded — only the blurred
composite frame is placed in the shared frame buffer for the MJPEG stream.
"""
import cv2
from app.cv.face_detector import detect_faces


def anonymise_frame(frame, person_boxes):
    """
    Apply both body-level and face-level blur to the frame.

    Parameters
    ----------
    frame        : np.ndarray  (BGR, H×W×3)  — will be modified in-place
    person_boxes : list of [x1, y1, x2, y2]  — YOLO person bounding boxes

    Returns
    -------
    frame : np.ndarray  — same array, now anonymised
    """
    frame = _blur_regions(frame, person_boxes, kernel=(61, 61), sigma=40)

    # Detect faces on the already-blurred frame so we don't re-identify
    # anyone through partial YOLO misses (e.g. a face peeking around a door)
    _, face_boxes = detect_faces(frame)
    frame = _blur_face_boxes(frame, face_boxes, kernel=(45, 45), sigma=30)

    return frame


# ---------------------------------------------------------------------------
# Kept for backwards compatibility — main.py previously called blur_people
# ---------------------------------------------------------------------------

def blur_people(frame, boxes):
    """Alias: blur only person bounding boxes (no face detection pass)."""
    return _blur_regions(frame, boxes, kernel=(51, 51), sigma=30)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _blur_regions(frame, boxes, kernel=(51, 51), sigma=30):
    """Gaussian-blur each [x1,y1,x2,y2] region."""
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        # Guard against out-of-bounds slicing
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        region = frame[y1:y2, x1:x2]
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, kernel, sigma)
    return frame


def _blur_face_boxes(frame, faces, kernel=(45, 45), sigma=30):
    """Blur face rectangles returned by the Haar-cascade detector."""
    for (fx, fy, fw, fh) in faces:
        fx, fy = max(0, fx), max(0, fy)
        fx2 = min(frame.shape[1], fx + fw)
        fy2 = min(frame.shape[0], fy + fh)
        if fx2 <= fx or fy2 <= fy:
            continue
        region = frame[fy:fy2, fx:fx2]
        frame[fy:fy2, fx:fx2] = cv2.GaussianBlur(region, kernel, sigma)
    return frame