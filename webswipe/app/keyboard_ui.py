# on-screen QWERTY overlay and image <-> keyboard frame conversion
import cv2 as cv
import numpy as np

from webswipe.decoder.features import LETTERS, Keyboard


class KeyboardOverlay:
    def __init__(self, frame_w, frame_h, width_frac=0.8, top_frac=0.55):
        self.pitch = frame_w * width_frac / 10.33   # keyboard is ~10.33 key pitches wide
        self.x0 = frame_w * (1 - width_frac) / 2 + self.pitch / 2
        self.y0 = frame_h * top_frac
        # decoder keyboard: same geometry, y up, origin at the 'q' key centre
        self.kb = Keyboard.from_pitch(self.pitch, self.pitch)
        self.key_px = {c: self.to_img(self.kb.centers[i]) for i, c in enumerate(LETTERS)}

    def to_kb(self, p):
        p = np.asarray(p, float)
        return np.stack([p[..., 0] - self.x0, -(p[..., 1] - self.y0)], -1)

    def to_img(self, q):
        q = np.asarray(q, float)
        return np.stack([q[..., 0] + self.x0, -q[..., 1] + self.y0], -1)

    def key_at(self, p):
        d = {c: np.linalg.norm(np.asarray(p) - xy) for c, xy in self.key_px.items()}
        c = min(d, key=d.get)
        return c if d[c] < self.pitch * 0.6 else None

    def draw(self, img, highlight=None):
        half = self.pitch * 0.45
        over = img.copy()
        for c, (x, y) in self.key_px.items():
            p1, p2 = (int(x - half), int(y - half)), (int(x + half), int(y + half))
            cv.rectangle(over, p1, p2, (80, 200, 255) if c == highlight else (40, 40, 40), -1)
        cv.addWeighted(over, 0.45, img, 0.55, 0, img)
        for c, (x, y) in self.key_px.items():
            cv.putText(img, c.upper(), (int(x - 7), int(y + 7)), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 255), 1, cv.LINE_AA)
        return img
