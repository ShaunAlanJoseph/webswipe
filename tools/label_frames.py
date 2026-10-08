# Box the hands in recorded frames with MediaPipe and attach the segment's pose label.
#   python tools/label_frames.py --session s1
# Output: data/frames/<session>/labels.jsonl, one line per kept frame:
#   {"file": "pinch_0/000012.jpg", "seg": "pinch", "round": 0, "boxes": [[cls, cx, cy, w, h], ...],
#    "filled": false, "bright": 131.2, "ratio": 0.31}
# ratio = thumb-index distance / palm length of the first (posed) hand, for the landmark-rule baseline.
# Classes follow the released HaGRID names so the pretrained class weights carry over:
#   ok = pinch, palm = open hand, fist, no_gesture = any other hand.
import argparse
import json
from pathlib import Path

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

CLASSES = ["ok", "palm", "fist", "no_gesture"]
SEG_CLASS = {"pinch": "ok", "open": "palm", "palm": "palm", "fist": "fist", "none": "no_gesture"}
# in the mirrored frame the user's right hand is on the right; open/pinch use it, fist/palm use the left
SEG_SIDE = {"pinch": "right", "open": "right", "palm": "left", "fist": "left"}


def box(lm, margin=0.12):
    xy = np.array([[p.x, p.y] for p in lm])
    lo, hi = xy.min(0), xy.max(0)
    pad = (hi - lo) * margin
    lo, hi = np.clip(lo - pad, 0, 1), np.clip(hi + pad, 0, 1)
    return [*((lo + hi) / 2).round(5).tolist(), *(hi - lo).round(5).tolist()]


def pinch_ratio(lm, w, h):
    """Thumb-tip to index-tip distance over palm length (wrist to middle knuckle): the app's landmark rule."""
    px = np.array([[p.x * w, p.y * h] for p in lm])
    return round(float(np.linalg.norm(px[4] - px[8]) / max(np.linalg.norm(px[0] - px[9]), 1e-6)), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--model", default="models/hand_landmarker.task")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--trim", type=int, default=15, help="drop this many frames at each end of a segment")
    ap.add_argument("--max-gap", type=int, default=10, help="interpolate the box across gaps up to this many frames")
    a = ap.parse_args()

    root = Path(a.frames) / a.session
    kept, stats = [], {}
    for seg_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        seg, rnd = seg_dir.name.rsplit("_", 1)
        st = stats.setdefault(seg, {"frames": 0, "kept": 0, "filled": 0, "no_hand": 0, "two_hands": 0})
        files = sorted(seg_dir.glob("*.jpg"))
        # track through the segment as video: tracking finds the hand in frames a single-image detector misses
        lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=a.model), running_mode=vision.RunningMode.VIDEO,
            num_hands=2, min_hand_detection_confidence=0.5, min_tracking_confidence=0.5))
        per, bright, ratios = [], [], []
        for i, f in enumerate(files):
            img = cv.imread(str(f))
            r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                             data=cv.cvtColor(img, cv.COLOR_BGR2RGB)), i * 33)
            hands = [(box(h), pinch_ratio(h, img.shape[1], img.shape[0])) for h in r.hand_landmarks]
            if seg != "none":  # posed hand first: it is the one on its side of the frame
                hands.sort(key=lambda hb: hb[0][0], reverse=SEG_SIDE[seg] == "right")
            per.append([hb[0] for hb in hands])
            ratios.append(hands[0][1] if hands else None)
            bright.append(round(float(cv.cvtColor(img, cv.COLOR_BGR2GRAY).mean()), 1))
        lm.close()
        filled = set()
        if seg != "none":  # short gaps: the hand moves smoothly, so interpolate the posed hand's box
            have = [i for i, bx in enumerate(per) if bx]
            for lo, hi in zip(have, have[1:]):
                if 1 < hi - lo <= a.max_gap + 1:
                    for i in range(lo + 1, hi):
                        t = (i - lo) / (hi - lo)
                        per[i] = [[round(u + (v - u) * t, 5) for u, v in zip(per[lo][0], per[hi][0])]]
                        filled.add(i)
        for i in range(a.trim, len(files) - a.trim, a.every):
            boxes = per[i]
            st["frames"] += 1
            if seg == "none":
                out = [[CLASSES.index("no_gesture"), *bx] for bx in boxes]  # any hand here is a non-gesture
            elif not boxes:
                st["no_hand"] += 1
                continue  # posed hand not found and gap too long: cannot box it
            else:
                st["two_hands"] += len(boxes) > 1
                out = [[CLASSES.index(SEG_CLASS[seg]), *boxes[0]]]
                out += [[CLASSES.index("no_gesture"), *bx] for bx in boxes[1:]]
            st["kept"] += 1
            st["filled"] += i in filled
            kept.append({"file": f"{seg_dir.name}/{files[i].name}", "seg": seg, "round": int(rnd), "boxes": out,
                         "filled": i in filled, "bright": bright[i],
                         "ratio": None if i in filled else ratios[i]})
    (root / "labels.jsonl").write_text("".join(json.dumps(k) + "\n" for k in kept))
    print(f"{a.session}: kept {len(kept)} frames -> {root / 'labels.jsonl'}")
    for seg, st in stats.items():
        print(f"  {seg:6s} sampled {st['frames']:4d}  kept {st['kept']:4d}  (box interpolated {st['filled']:3d})"
              f"  hand not found {st['no_hand']:3d}  two hands {st['two_hands']:3d}")


if __name__ == "__main__":
    main()
