# app/cv/privacy.py
"""
Privacy layer – anonymise detected people before any frame is stored
or streamed to the dashboard.

Why this matters
----------------
The problem statement explicitly requires "Ethical Surveillance": admins
should see *aggregate* occupancy data, not identifiable individuals.  We
achieve this by replacing every person's bounding-box region with a
heavy Gaussian blur, making faces and clothing unrecognisable while
keeping the outline/silhouette visible enough for the logic engine to
work correctly.

Future enhancement: swap the blur for MediaPipe pose-skeleton overlay
so only stick-figure skeletons appear – zero identifiable information.
"""

import cv2


def blur_people(frame, boxes):
    """
    Apply a strong Gaussian blur over every detected person region.

    Parameters
    ----------
    frame : np.ndarray   BGR image (modified in-place and returned)
    boxes : list         Each element is [x1, y1, x2, y2] (float or int)

    Returns
    -------
    frame : np.ndarray   Anonymised frame
    """
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        # Guard against out-of-bound crops (can happen at frame edges)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            continue  # skip degenerate boxes

        person_region = frame[y1:y2, x1:x2]

        # kernel (51,51) + sigma 30 → heavy blur, unrecognisable face
        blurred = cv2.GaussianBlur(person_region, (51, 51), 30)

        frame[y1:y2, x1:x2] = blurred

    return frame