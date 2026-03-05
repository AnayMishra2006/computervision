"""
People-detection module using YOLOv8.

Design choices for CCTV / long-distance accuracy:
  - Uses YOLOv8m (medium) by default — good accuracy / speed trade-off.
    Switch YOLO_MODEL=yolov8l.pt or yolov8x.pt for even better recall at
    the cost of ~2× inference time.
  - CONFIDENCE_THRESHOLD defaults to 0.25 (lower than typical 0.5) so that
    small, far-away people are not missed.
  - YOLO_IMGSZ defaults to 640; set to 1280 for high-resolution CCTV feeds.
  - NMS (Non-Maximum Suppression) is handled internally by Ultralytics.
"""
import logging
from ultralytics import YOLO
from app.config import CONFIDENCE_THRESHOLD, YOLO_MODEL, YOLO_IMGSZ

logger = logging.getLogger(__name__)

# Load the model once at import time so every call reuses the same weights.
# The first run will download the model file automatically if not cached.
_model = YOLO(YOLO_MODEL)

# COCO class index for "person"
_PERSON_CLASS = 0


def detect_people(frame):
    """
    Detect all people in *frame* and return their count + bounding boxes.

    Parameters
    ----------
    frame : np.ndarray  (BGR, H×W×3)

    Returns
    -------
    person_count : int
    boxes        : list of [x1, y1, x2, y2]  (float pixel coords)
    """
    results = _model(frame, imgsz=YOLO_IMGSZ, verbose=False)

    boxes        = []
    person_count = 0

    for r in results:
        for box in r.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])

            # Only keep "person" detections above threshold
            if cls == _PERSON_CLASS and conf >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append([x1, y1, x2, y2])
                person_count += 1

    logger.debug("Detected %d person(s)", person_count)
    return person_count, boxes
