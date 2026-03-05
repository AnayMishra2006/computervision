# app/cv/privacy.py
#
# Privacy-first anonymisation layer.
#
# Why two-pass blur?
#   1. We first blur the entire person bounding box (body region) so that
#      clothing / body shape cannot be used to identify someone.
#   2. We then apply a heavier pixelation pass specifically over face regions
#      (detected by the face detector) to guarantee facial privacy.
#
# The result is an "ethical surveillance" feed:
#   ✅ Appliance state visible
#   ✅ Occupancy count visible
#   ✗  No identifiable personal data

import cv2


def blur_people(frame, boxes):
    """
    Blur each person's body region using a strong Gaussian kernel.

    Parameters
    ----------
    frame : np.ndarray  (BGR)
    boxes : list of [x1, y1, x2, y2]

    Returns
    -------
    Anonymised frame (same shape as input).
    """
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        # Guard against out-of-bounds crops (can happen at frame edges)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])

        if x2 <= x1 or y2 <= y1:
            continue

        person_region = frame[y1:y2, x1:x2]

        # Kernel size must be odd; 51×51 is intentionally heavy
        blurred = cv2.GaussianBlur(person_region, (51, 51), 30)
        frame[y1:y2, x1:x2] = blurred

    return frame


def pixelate_faces(frame, face_boxes):
    """
    Apply pixelation to each detected face for an extra layer of anonymisation.
    Pixelation is more visually obvious than blur — it makes it immediately
    clear to any viewer that the system is privacy-aware.

    Parameters
    ----------
    frame     : np.ndarray  (BGR)
    face_boxes: list of (x, y, w, h)  — output from the face detector

    Returns
    -------
    Anonymised frame.
    """
    for (x, y, w, h) in face_boxes:
        x, y = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])

        if x2 <= x or y2 <= y:
            continue

        face_region = frame[y:y2, x:x2]

        # Shrink to a tiny block, then scale back up → pixelation effect
        small = cv2.resize(face_region, (10, 10), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (x2 - x, y2 - y), interpolation=cv2.INTER_NEAREST)
        frame[y:y2, x:x2] = pixelated

    return frame