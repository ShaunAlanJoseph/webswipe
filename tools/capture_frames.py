# Record labelled webcam frames for the hand-state detector: one pose per segment, prompted on screen.
#   python tools/capture_frames.py --session s1
# Frames are saved mirrored (as the app sees them) to data/frames/<session>/<label>_<round>/NNNNNN.jpg.
# Keys: q = quit, n = skip this segment, r = redo this segment.
import argparse
import json
import shutil
import time
from pathlib import Path

import cv2 as cv

SEGMENTS = [
    ("open", "RIGHT hand: open and relaxed. Move it slowly around the whole keyboard area."),
    ("pinch", "RIGHT hand: PINCH thumb + index and HOLD. Move around as if swiping words."),
    ("fist", "LEFT hand: closed FIST facing the camera. Move it around a little."),
    ("palm", "LEFT hand: flat open PALM facing the camera, fingers spread. Move it around a little."),
    ("none", "NO gesture: hands down or doing anything else (touch face, reach for something)."),
]


def text(img, s, y, scale=0.6, color=(255, 255, 255)):
    cv.putText(img, s, (12, y), cv.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv.LINE_AA)


def bar(img, y0, y1, alpha=0.65):
    """Darken a horizontal band so the text on it stays readable over any background."""
    img[y0:y1] = (img[y0:y1] * (1 - alpha)).astype(img.dtype)


def wrap(s, n=58):
    out, line = [], ""
    for w in s.split():
        if len(line) + len(w) + 1 > n:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    return out + [line]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="name for this recording, e.g. s1_daylight")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=20, help="recorded seconds per segment")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--ready", type=float, default=5, help="get-ready seconds before each segment")
    ap.add_argument("--out", default="data/frames")
    a = ap.parse_args()

    root = Path(a.out) / a.session
    root.mkdir(parents=True, exist_ok=True)
    cap = cv.VideoCapture(a.camera)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.read()[0]:
        raise SystemExit("cannot read from camera")

    plan = [(lab, tip, r) for r in range(a.rounds) for lab, tip in SEGMENTS]
    i, total = 0, 0
    while i < len(plan):
        lab, tip, r = plan[i]
        d = root / f"{lab}_{r}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        t0, n, action = time.time(), 0, None
        while action is None:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv.flip(frame, 1)
            el = time.time() - t0
            rec = el >= a.ready
            if rec and el >= a.ready + a.seconds:
                action = "done"
                break
            if rec:
                cv.imwrite(str(d / f"{n:06d}.jpg"), frame, [cv.IMWRITE_JPEG_QUALITY, 95])
                n += 1
            view = frame.copy()
            lines = wrap(tip)
            bar(view, 0, 44 + 24 * len(lines))
            bar(view, 452, 480)
            head = f"[{i + 1}/{len(plan)}] {lab.upper()}"
            if rec:
                text(view, f"{head}   REC {a.ready + a.seconds - el:4.1f}s", 28, 0.7, (80, 80, 255))
            else:
                text(view, f"{head}   get ready {a.ready - el:3.1f}s", 28, 0.7, (0, 255, 255))
            for k, line in enumerate(lines):
                text(view, line, 56 + 24 * k)
            text(view, "q quit   n skip   r redo", 471, 0.5)
            cv.imshow("capture", view)
            key = cv.waitKey(1) & 0xFF
            action = {ord("q"): "quit", ord("n"): "skip", ord("r"): "redo"}.get(key)
        if action == "quit":
            shutil.rmtree(d, ignore_errors=True)
            break
        if action == "redo":
            continue
        if action == "skip":
            shutil.rmtree(d, ignore_errors=True)
        else:
            total += n
            print(f"{d.name}: {n} frames ({n / a.seconds:.1f} fps)", flush=True)
        i += 1
    cap.release()
    cv.destroyAllWindows()
    (root / "meta.json").write_text(json.dumps(
        {"session": a.session, "seconds": a.seconds, "rounds": a.rounds, "mirrored": True,
         "labels": [s[0] for s in SEGMENTS], "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    print(f"saved {total} frames under {root}")


if __name__ == "__main__":
    main()
