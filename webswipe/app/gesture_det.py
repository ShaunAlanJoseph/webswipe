# Appearance-based hand state from the fine-tuned HaGRID YOLOv10n detector (ONNX, CPU).
# Replaces the landmark-distance pinch rule when the app runs with --pinch detector.
# Detections are assigned to MediaPipe hands by box overlap; the hand keeps MediaPipe's landmarks for the cursor.
from collections import deque

import numpy as np

# class order of the fine-tuned model (tools/label_frames.py CLASSES); ok = pinch
CLASSES = ["ok", "palm", "fist", "no_gesture"]


class GestureDetector:
    def __init__(self, path, imgsz=320, threads=2, conf=0.05):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.imgsz, self.conf = imgsz, conf
        n_out = self.sess.get_outputs()[0].shape[1]
        if n_out != 4 + len(CLASSES):
            raise ValueError(f"{path}: expected {4 + len(CLASSES)} outputs per anchor (fine-tuned model), got {n_out}")

    def __call__(self, frame_bgr):
        """Returns [(class_name, score, (x0, y0, x1, y1) in pixels)], best box per class region, highest first."""
        import cv2 as cv
        h, w = frame_bgr.shape[:2]
        r = self.imgsz / max(h, w)
        nh, nw = round(h * r), round(w * r)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv.resize(frame_bgr, (nw, nh))
        x = (canvas[:, :, ::-1].transpose(2, 0, 1)[None] / 255.0).astype("float32")
        out = self.sess.run(None, {self.inp: x})[0][0].T  # anchors x (cx, cy, w, h, class scores...)
        scores = out[:, 4:]
        best = scores.max(1)
        keep = np.flatnonzero(best >= self.conf)
        keep = keep[np.argsort(-best[keep])][:50]
        dets = []
        for i in keep:
            cx, cy, bw, bh = out[i, :4] / r
            box = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
            if all(_iou(box, d[2]) < 0.5 for d in dets):  # simple NMS across classes
                dets.append((CLASSES[int(scores[i].argmax())], float(best[i]), box))
        return dets


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def hand_box(px, margin=0.12):
    lo, hi = px.min(0), px.max(0)
    pad = (hi - lo) * margin
    return (*(lo - pad), *(hi + pad))


def assign(dets, hands):
    """Per MediaPipe hand: {class_name: best score} from the detections overlapping its landmark box."""
    out = []
    for h in hands:
        hb = hand_box(h.px)
        sc = {c: 0.0 for c in CLASSES}
        for c, s, box in dets:
            if _iou(box, hb) >= 0.3 and s > sc[c]:
                sc[c] = s
        out.append(sc)
    return out


class StateFilter:
    """Hysteresis on one class score: on when `votes` of the last `window` frames reach `on`, off below `off`.

    votes=1 starts on a single frame; 2 of 3 ignores one-frame flickers at the cost of about one frame of delay.
    """

    def __init__(self, cls="ok", on=0.5, off=0.3, votes=1, window=3):
        self.cls, self.on_thr, self.off_thr, self.votes = cls, on, off, votes
        self.hits = deque(maxlen=window)
        self.on = False

    def update(self, scores):
        s = scores.get(self.cls, 0.0) if scores else 0.0
        self.hits.append(s >= self.on_thr)
        self.on = s > self.off_thr if self.on else sum(self.hits) >= self.votes
        return self.on, s

    def started(self, scores, thr):
        """Vote only (no hysteresis): has `cls` reached `thr` in `votes` of the last frames, this one included?"""
        self.hits.append((scores or {}).get(self.cls, 0.0) >= thr)
        return sum(self.hits) >= self.votes

    def reset_state(self):
        self.on = False
        self.hits.clear()
