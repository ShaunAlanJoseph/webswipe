# Replay recorded segments through the app headless and report the share of frames with pinch ON per segment.
#   PYTHONPATH=. python tools/replay_frames.py data/frames/s3/pinch_1,data/frames/s3/none_1 --pinch contact
# play recorded segments through the app instead of the webcam; prints pinch-on share per segment
import glob, sys, runpy, time, collections
import cv2 as cv
segs = sys.argv[1].split(",")
files = [(s.split("/")[-1].rsplit("_", 1)[0], f) for s in segs for f in sorted(glob.glob(f"{s}/*.jpg"))]
state = {"i": 0}
class Cap:
    def __init__(s, *a): pass
    def set(s, *a): pass
    def read(s):
        i = state["i"]
        if i >= len(files): return False, None
        state["i"] += 1
        return True, cv.flip(cv.imread(files[i][1]), 1)  # app flips again -> recorded (mirrored) view
    def release(s): pass
cv.VideoCapture = Cap
import webswipe.app.main as m
orig = m.text
on_count = collections.Counter(); n_count = collections.Counter()
def spy(img, s, org, *a, **k):
    if s.startswith("FPS"):
        seg = files[state["i"] - 1][0]
        n_count[seg] += 1; on_count[seg] += "pinch ON" in s
    return orig(img, s, org, *a, **k)
m.text = spy
sys.argv = ["app", "--headless"] + sys.argv[2:]
t = time.time(); m.main(); el = time.time() - t
print(f"{state['i']} frames in {el:.1f}s = {state['i']/el:.1f} fps (whole pipeline, CPU)")
for k in n_count: print(f"  {k:6s} pinch ON in {on_count[k]/n_count[k]:.0%} of {n_count[k]} frames")
