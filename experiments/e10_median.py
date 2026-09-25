# E10 again with median filter
import json
import numpy as np
import run_experiments_2 as X

def med5(x):
    p = np.pad(x, 2, mode="edge")
    return np.median(np.stack([p[i:i + len(x)] for i in range(5)]), axis=0)

_orig = X.pinch_session
def filtered_session(dcm, light, frames=900):
    d, s, l = _orig(dcm, light, frames)
    return med5(d), s, l
X.pinch_session = filtered_session
r = X.e10()
import shutil; shutil.move(X.OUT / "e10_pinch_detection.png", X.OUT / "e10b_pinch_detection_median.png")
(X.OUT / "results_e10b.json").write_text(json.dumps(r, indent=2))
