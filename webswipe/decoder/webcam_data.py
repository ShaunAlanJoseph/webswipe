# convert webcam recordings (data/recordings/*.jsonl from the app's record mode) into decoder features (D3)
import argparse
import glob
import json

import numpy as np

from .features import LETTERS, Keyboard, featurize, resample_time


def load_recordings(pattern):
    recs = []
    for path in sorted(glob.glob(pattern)):
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            word = r["target"].lower()
            if not word or any(ch not in LETTERS for ch in word):
                continue
            tr = np.asarray(r["trace_kb"], float)  # (T, 3): x, y in keyboard frame (y up), t in ms
            recs.append({"subject": r["participant"], "word": word, "coords": tr[:, :2], "t_ms": tr[:, 2],
                         "pitch": float(r["keyboard"]["pitch"])})
    return recs


def build(pattern, out, resample_hz=0.0):
    recs = load_recordings(pattern)
    X, keep = [], []
    for i, r in enumerate(recs):
        c = resample_time(r["coords"], r["t_ms"], resample_hz) if resample_hz else r["coords"]
        f = featurize(c, Keyboard.from_pitch(r["pitch"], r["pitch"]))  # same geometry as the overlay
        if f is not None:
            X.append(f)
            keep.append(i)
    if not X:
        raise SystemExit(f"no usable gestures in {pattern}")
    recs = [recs[i] for i in keep]
    np.savez_compressed(out, X=np.stack(X), words=np.array([r["word"] for r in recs]),
                        subject=np.array([r["subject"] for r in recs]),
                        candidate=np.zeros(len(recs), int))
    by = {s: sum(r["subject"] == s for r in recs) for s in sorted({r["subject"] for r in recs})}
    print(f"{len(recs)} gestures, {len({r['word'] for r in recs})} unique words, per participant {by} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", default="data/recordings/*.jsonl")
    ap.add_argument("--out", default="data/webcam_features.npz")
    ap.add_argument("--resample-hz", type=float, default=0.0, help="put traces on a uniform clock first")
    a = ap.parse_args()
    build(a.recordings, a.out, a.resample_hz)
