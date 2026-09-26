import json

import numpy as np
import torch

from webswipe.app.keyboard_ui import KeyboardOverlay
from webswipe.app.pinch import PinchDetector
from webswipe.app.recorder import Recorder
from webswipe.decoder.features import LETTERS, N_FEATURES, Keyboard, featurize
from webswipe.decoder.live import LiveDecoder
from webswipe.decoder.model import ShenDecoder


def word_trace(ui, word, per_seg=8):
    pts = [ui.key_px[c] for c in word]
    path = [pts[0]]
    for p, q in zip(pts[:-1], pts[1:]):
        path += [p + (q - p) * t for t in np.linspace(0, 1, per_seg)[1:]]
    return np.array(path)


def test_overlay_roundtrip():
    ui = KeyboardOverlay(640, 480)
    q = np.array([[12.0, -30.0], [100.0, 5.0]])
    assert np.allclose(ui.to_kb(ui.to_img(q)), q)
    for i, c in enumerate(LETTERS):
        assert np.allclose(ui.to_kb(ui.key_px[c]), ui.kb.centers[i])
    assert ui.key_at(ui.key_px["g"]) == "g"


def test_features_shape():
    f = featurize(np.random.default_rng(0).normal(size=(30, 2)) * 50, Keyboard.shen(169))
    assert f.shape == (50, N_FEATURES) and np.isfinite(f).all()
    assert featurize(np.zeros((3, 2)), Keyboard.shen(169)) is None


def test_pinch_detector_events():
    rng = np.random.default_rng(0)
    det = PinchDetector()
    hand = np.zeros((21, 2))
    hand[9] = [0, -100]  # hand scale 100 px
    states = []
    for t in range(600):
        pinched = (t // 40) % 2 == 1
        d = (10 if pinched else 45) + rng.normal(0, 2)
        hand[4], hand[8] = [0, 0], [d, 0]
        states.append(det.update(hand)[0])
    s = np.array(states, int)
    onsets = int(((s[1:] - s[:-1]) == 1).sum())
    assert 6 <= onsets <= 8  # 7 pinches after the first open period
    assert 0.1 < det.thr < 0.45


def test_decoder_and_recorder(tmp_path):
    ckpt = tmp_path / "rand.pt"
    torch.save({"model": ShenDecoder().state_dict()}, ckpt)
    ui = KeyboardOverlay(640, 480)
    dec = LiveDecoder(str(ckpt), ["hello", "world", "help", "shell", "yellow"])
    tr = word_trace(ui, "hello")
    trace_kb = [[*ui.to_kb(p).tolist(), 33 * i] for i, p in enumerate(tr)]
    cands, ms = dec(trace_kb, ui.kb)
    assert len(cands) == 4 and set(cands) <= set(dec.lex.words) and ms > 0

    rec = Recorder("t", out_dir=tmp_path)
    rec.frame(0, [])
    rec.start()
    rec.frame(33, [])
    rec.save("hello", trace_kb, {"pitch": ui.pitch})
    line = json.loads(open(rec.path).read().splitlines()[0])
    assert line["target"] == "hello" and len(line["frames"]) == 2 and line["pinch_start_frame"] == 1
