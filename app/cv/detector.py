# app/cv/detector.py
#
# People detection using YOLOv8 + ByteTrack object tracking.
#
# Why tracking?
#   A raw per-frame detector counts the same person multiple times if they
#   overlap with another detection box.  ByteTrack assigns a persistent ID to
#   each person across frames, so we count unique IDs instead of raw boxes.
#   This is critical for CCTV / long-distance scenarios where people partially
#   leave and re-enter the field of view.

from ultralytics import YOLO
from app.config import CONFIDENCE_THRESHOLD, YOLO_MODEL_SIZE, TRACKER

# Load once at import time — YOLO initialisation is expensive
# Model filename: e.g. YOLO_MODEL_SIZE="n" → "yolov8n.pt"
_model = YOLO(f"yolov8{YOLO_MODEL_SIZE}.pt")


def detect_people(frame):
    """
    Detect and track people in *frame*.

    Returns
    -------
    person_count : int
        Number of unique people currently visible.
    boxes : list of [x1, y1, x2, y2]
        Bounding boxes for each detected person (used for blurring).
    """
    # track() instead of predict() enables ByteTrack/BoTSORT across frames.
    # persist=True keeps tracker state between calls so IDs are consistent.
    results = _model.track(
        frame,
        imgsz=640,
        conf=CONFIDENCE_THRESHOLD,
        classes=[0],        # class 0 = person in COCO
        tracker=f"{TRACKER}.yaml",
        persist=True,
        verbose=False,
    )

    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            boxes.append([x1, y1, x2, y2])

    person_count = len(boxes)
    return person_count, boxes