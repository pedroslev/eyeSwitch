"""Debug view of what the face tracker sees: the camera image with landmarks and live numbers."""
import cv2
import numpy as np

CORNERS = (33, 133, 263, 362)               # eye corners
IRISES = (468, 473)                         # iris centers
IRIS_RING = (469, 470, 471, 472, 474, 475, 476, 477)


def _text(img, line, y):
    cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def render(frame, landmarks, lines, width=480):
    """BGR image, mirrored like a mirror, with the face mesh, eyes and irises drawn on it."""
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (width, max(1, int(h * width / w))))
    ih, iw = img.shape[:2]
    if landmarks:
        pts = np.array([(p.x * iw, p.y * ih) for p in landmarks], np.int32)
        for x, y in pts[:468]:
            cv2.circle(img, (int(x), int(y)), 1, (190, 190, 190), -1)
        for i in IRIS_RING:
            cv2.circle(img, tuple(int(v) for v in pts[i]), 1, (255, 255, 0), -1)
        for i in CORNERS:
            cv2.circle(img, tuple(int(v) for v in pts[i]), 3, (0, 255, 0), -1)
        for i in IRISES:
            cv2.circle(img, tuple(int(v) for v in pts[i]), 3, (0, 0, 255), -1)
    img = cv2.flip(img, 1)                  # text is drawn after the flip so it stays readable
    for n, line in enumerate(lines):
        _text(img, line, 22 + 22 * n)
    return img
