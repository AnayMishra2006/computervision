# app/cv/detector.py

from ultralytics import YOLO
from app.config import CONFIDENCE_THRESHOLD

# Load model
model = YOLO("yolov8m.pt")  # using medium model


def detect_people(frame):
    """
    Detect only people in the frame.
    """

    results = model(frame, imgsz=640, verbose=False)

    boxes = []
    person_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])

            if cls == 0 and box.conf[0] > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = box.xyxy[0]
                boxes.append([x1, y1, x2, y2])
                person_count += 1

    return person_count, boxes