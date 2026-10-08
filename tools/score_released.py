# How good is the released HaGRID YOLOv10n gesture detector on our own webcam frames, before any fine-tuning?
#   python tools/score_released.py --session s1
# Per recorded label: how often a hand is found, which HaGRID class it is called, and whether that counts as right.
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2 as cv
import numpy as np
import onnxruntime as ort

NAMES = ["grabbing", "grip", "holy", "point", "call", "three3", "timeout", "xsign", "hand_heart", "hand_heart2",
         "little_finger", "middle_finger", "take_picture", "dislike", "fist", "four", "like", "mute", "ok", "one",
         "palm", "peace", "peace_inverted", "rock", "stop", "stop_inverted", "three", "three2", "two_up",
         "two_up_inverted", "three_gun", "thumb_index", "thumb_index2", "no_gesture"]
# HaGRID classes accepted as correct for each of our labels (generous to the released model)
ACCEPT = {"pinch": {"ok", "grip"}, "fist": {"fist"}, "palm": {"palm", "stop", "four"},
          "open": {"no_gesture", "palm", "stop", "four"}}


def infer(sess, frame, size):
    h, w = frame.shape[:2]
    r = size / max(h, w)
    nh, nw = round(h * r), round(w * r)
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[:nh, :nw] = cv.resize(frame, (nw, nh))
    x = (canvas[:, :, ::-1].transpose(2, 0, 1)[None] / 255.0).astype("float32")
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0].T  # anchors x (4 box + 34 class scores)
    scores = out[:, 4:]
    best = scores.max(1)
    return best, scores.argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--model-dir", default="models/hagrid")
    ap.add_argument("--sizes", type=int, nargs="*", default=[640, 320])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--every", type=int, default=3, help="score every n-th frame")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    root = Path(a.frames) / a.session
    segs = sorted(p for p in root.iterdir() if p.is_dir())
    if not segs:
        raise SystemExit(f"no segments under {root}")
    res = {}
    for size in a.sizes:
        so = ort.SessionOptions()
        so.intra_op_num_threads = a.threads
        sess = ort.InferenceSession(f"{a.model_dir}/yolov10n_gestures_{size}.onnx", so,
                                    providers=["CPUExecutionProvider"])
        stats = {}
        for seg in segs:
            lab = seg.name.rsplit("_", 1)[0]
            st = stats.setdefault(lab, {"n": 0, "correct": 0, "any_gesture": 0, "called": Counter()})
            for f in sorted(seg.glob("*.jpg"))[::a.every]:
                best, cls = infer(sess, cv.imread(str(f)), size)
                keep = best >= a.conf
                # the frame's answer = most confident detection that is not "no_gesture", else no_gesture/nothing
                names = [(float(best[i]), NAMES[int(cls[i])]) for i in np.flatnonzero(keep)]
                gest = sorted((s, n) for s, n in names if n != "no_gesture")
                top = gest[-1][1] if gest else ("no_gesture" if names else "nothing")
                st["n"] += 1
                st["called"][top] += 1
                st["any_gesture"] += bool(gest)
                if lab == "none":
                    st["correct"] += not gest
                elif lab == "open":
                    st["correct"] += top in ACCEPT["open"] or top == "nothing"
                else:
                    st["correct"] += top in ACCEPT.get(lab, set())
        res[size] = stats
        print(f"\n=== released model at {size} px (confidence >= {a.conf}) ===")
        print(f"{'label':7s} {'frames':>6s} {'correct':>8s}   most common answers")
        for lab, st in stats.items():
            common = ", ".join(f"{k} {v / st['n']:.0%}" for k, v in st["called"].most_common(4))
            print(f"{lab:7s} {st['n']:6d} {st['correct'] / max(st['n'], 1):8.1%}   {common}")
    out = Path(a.out) if a.out else root / "released_score.json"
    out.write_text(json.dumps({str(s): {k: {**v, "called": dict(v["called"])} for k, v in d.items()}
                               for s, d in res.items()}, indent=1))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
