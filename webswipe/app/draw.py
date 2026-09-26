# drawing helpers; draw_landmarks adapted from kinivi/hand-gesture-recognition-mediapipe (Apache-2.0)
import cv2 as cv
import numpy as np

BONES = [(2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14), (14, 15),
         (15, 16), (17, 18), (18, 19), (19, 20), (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)]
TIPS = {4, 8, 12, 16, 20}


def draw_landmarks(img, px):
    pts = [tuple(int(v) for v in p) for p in px]
    for a, b in BONES:
        cv.line(img, pts[a], pts[b], (0, 0, 0), 6)
        cv.line(img, pts[a], pts[b], (255, 255, 255), 2)
    for i, p in enumerate(pts):
        r = 8 if i in TIPS else 5
        cv.circle(img, p, r, (255, 255, 255), -1)
        cv.circle(img, p, r, (0, 0, 0), 1)
    return img


def draw_trace(img, pts, color=(0, 200, 255)):
    if len(pts) > 1:
        cv.polylines(img, [np.asarray(pts, np.int32)], False, color, 3, cv.LINE_AA)


def text(img, s, org, scale=0.6, color=(255, 255, 255), thick=1):
    cv.putText(img, s, org, cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv.LINE_AA)
    cv.putText(img, s, org, cv.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv.LINE_AA)
