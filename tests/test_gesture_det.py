import numpy as np

from webswipe.app.gesture_det import StateFilter, assign
from webswipe.app.pinch import PinchDetector


class H:
    def __init__(self, x0, y0, x1, y1):
        self.px = np.array([[x0, y0], [x1, y1], [(x0 + x1) / 2, (y0 + y1) / 2]], float)


def test_assign_matches_boxes_to_hands():
    left, right = H(10, 10, 110, 110), H(300, 10, 400, 110)
    dets = [("ok", 0.9, (300, 10, 400, 110)), ("fist", 0.8, (12, 8, 108, 112)), ("palm", 0.4, (305, 15, 395, 105))]
    sl, sr = assign(dets, [left, right])
    assert sl["fist"] == 0.8 and sl["ok"] == 0.0
    assert sr["ok"] == 0.9 and sr["palm"] == 0.4 and sr["fist"] == 0.0


def test_state_filter_hysteresis():
    f = StateFilter("ok", on=0.5, off=0.3)
    seq = [0.2, 0.45, 0.6, 0.4, 0.35, 0.25, 0.4]
    assert [f.update({"ok": s})[0] for s in seq] == [False, False, True, True, True, False, False]
    assert f.update(None)[0] is False


def test_pinch_threshold_cannot_drift_onto_open_hand():
    # open hand held in view for a long time: two "open" modes (0.7 and 1.1), no pinch at all
    d = PinchDetector()
    rng = np.random.default_rng(0)
    for i in range(600):
        r = 0.7 if (i // 40) % 2 else 1.1
        tip = np.array([r + rng.normal(0, 0.02), 0.0])
        px = np.zeros((21, 2)); px[9] = [0, 1.0]; px[4] = [0, 0]; px[8] = tip
        on, _ = d.update(px)
    assert d.thr <= 0.4 and not on


def test_state_filter_two_of_three_ignores_single_frame():
    f = StateFilter("ok", on=0.5, off=0.3, votes=2)
    assert [f.update({"ok": s})[0] for s in [0.9, 0.1, 0.1, 0.9, 0.9, 0.4]] == [False, False, False, False, True, True]
    g = StateFilter("ok", votes=2)
    assert [g.started({"ok": s}, 0.5) for s in [0.9, 0.1, 0.9, 0.1, 0.1]] == [False, False, True, False, False]
