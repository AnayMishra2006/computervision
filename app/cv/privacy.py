# app/cv/privacy.py
import cv2

def blur_people(frame, boxes):
    """
    Blur detected people using bounding boxes.
    """

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        # Extract person region
        person_region = frame[y1:y2, x1:x2]

        # Apply blur
        blurred = cv2.GaussianBlur(person_region, (51, 51), 30)

        # Replace region
        frame[y1:y2, x1:x2] = blurred

    return frame