# sanity report for webcam recordings (D3): run after each recording session
import argparse
import glob
import json
from collections import defaultdict

import numpy as np

MIN_POINTS, MIN_MS, MAX_MS = 6, 150, 5000


def word_stats(r):
    tr = np.asarray(r["trace_kb"], float)
    dur = float(tr[-1, 2] - tr[0, 2]) if len(tr) > 1 else 0.0
    # frames between pinch start and the end of the saved window where no hand was detected
    frames = r.get("frames", [])[r.get("pinch_start_frame", 0):]
    lost = sum(1 for f in frames if not f["hands"])
    problems = []
    if len(tr) < MIN_POINTS:
        problems.append(f"{len(tr)} points")
    if dur < MIN_MS:
        problems.append(f"{dur:.0f} ms")
    if dur > MAX_MS:
        problems.append(f"long {dur / 1000:.1f} s")
    if lost:
        problems.append(f"hand lost {lost} frames")
    return {"points": len(tr), "ms": dur, "fps": (len(tr) - 1) / dur * 1000 if dur > 0 else 0.0,
            "problems": problems}


def report(pattern, show=5):
    by = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        for i, line in enumerate(open(path)):
            if line.strip():
                r = json.loads(line)
                by[r["participant"]].append((path, i + 1, r, word_stats(r)))
    if not by:
        print(f"no recordings match {pattern}")
        return {}
    out = {}
    for p, rows in sorted(by.items()):
        st = [s for *_, s in rows]
        saved = [r["saved"] for _, _, r, _ in rows]
        flagged = [(path, ln, r["target"], s["problems"]) for path, ln, r, s in rows if s["problems"]]
        out[p] = {
            "words": len(rows),
            "unique_words": len({r["target"] for _, _, r, _ in rows}),
            "phrases": len({r.get("phrase_index", r.get("phrase")) for _, _, r, _ in rows}),
            "minutes": (max(saved) - min(saved)) / 60,
            "median_points": float(np.median([s["points"] for s in st])),
            "median_word_ms": float(np.median([s["ms"] for s in st])),
            "median_fps": float(np.median([s["fps"] for s in st])),
            "flagged": len(flagged),
        }
        o = out[p]
        print(f"{p}: {o['words']} words ({o['unique_words']} unique, {o['phrases']} phrases) in "
              f"{o['minutes']:.0f} min | median {o['median_points']:.0f} points, {o['median_word_ms']:.0f} ms, "
              f"{o['median_fps']:.1f} fps | flagged {o['flagged']}")
        for path, ln, w, prob in flagged[:show]:
            print(f"    {path}:{ln} '{w}': {', '.join(prob)}")
        if len(flagged) > show:
            print(f"    ... {len(flagged) - show} more")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", default="data/recordings/*.jsonl")
    ap.add_argument("--show", type=int, default=5, help="flagged gestures to list per participant")
    a = ap.parse_args()
    report(a.recordings, a.show)
