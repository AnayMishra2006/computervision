# app/cv/detector.py
"""
People detector powered by YOLOv8.

Why YOLOv8m?
  - 'medium' variant gives the best trade-off between accuracy and speed
    for campus CCTV scenarios where people can be small and far away.
  - We use a configurable `imgsz` (default 640, can be set to 1280 in
    config for large rooms) so distant subjects are resolved properly.
  - Half-precision (FP16) is optional for GPU hardware to cut inference
    time roughly in half without meaningful accuracy loss.
"""

from ultralytics import YOLO
from app.config import CONFIDENCE_THRESHOLD, YOLO_IMGSZ, YOLO_HALF

# Load model once at import time so every call reuses the same weights.
# YOLOv8m = medium model; swap to "yolov8l.pt" for even higher accuracy
# if your hardware can handle it.
model = YOLO("yolov8m.pt")


def detect_people(frame):
    """
    Detect humans in *frame* and return their count and bounding boxes.

    Parameters
    ----------
    frame : np.ndarray  BGR image (H x W x 3)

    Returns
    -------
    person_count : int
    boxes        : list of [x1, y1, x2, y2] (float pixel coords)
    """
    # Run inference.
    # verbose=False suppresses per-frame console spam.
    # half=YOLO_HALF enables FP16 on CUDA GPUs for faster throughput.
    results = model(frame, imgsz=YOLO_IMGSZ, verbose=False, half=YOLO_HALF)

    boxes = []
    person_count = 0

    for r in results:
        for box in r.boxes:
            # COCO class 0 = "person"
            cls = int(box.cls[0])

            if cls == 0 and float(box.conf[0]) >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append([x1, y1, x2, y2])
                person_count += 1

    return person_count, boxes