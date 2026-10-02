import numpy as np

from webswipe.baselines.shark2 import Shark2, resample_arc
from webswipe.decoder.augment import Degrade, random_webcam
from webswipe.decoder.features import LETTERS, Keyboard

WORDS = ["hello", "world", "help", "shell", "yellow", "quiet", "zebra"]


def ideal(word, kb, hz=50, sec=1.0):
    idx = [LETTERS.index(c) for c in word]
    c = resample_arc(kb.centers[idx], int(hz * sec))
    return c, np.arange(len(c)) * 1000 / hz


def test_degrade_rate_and_noise():
    kb, rng = Keyboard.shen(169), np.random.default_rng(0)
    c, t = ideal("hello", kb)
    assert np.allclose(Degrade()(c, t, kb.key_w, rng), c)
    assert abs(len(Degrade(fps=12)(c, t, kb.key_w, rng)) - 12) <= 1
    n = Degrade(jitter=0.2)(c, t, kb.key_w, rng) - c
    assert 0.15 < n.std() / kb.key_w < 0.25
    short = Degrade(trunc_ms=100)(c, t, kb.key_w, rng)
    assert len(c) - 10 <= len(short) <= len(c)
    for _ in range(50):
        assert len(random_webcam(rng)(c, t, kb.key_w, rng)) >= 2


def test_shark2_recognises_ideal_traces():
    kb = Keyboard.shen(169)
    for norm in ["bbox", "rms"]:
        dec = Shark2(WORDS, norm=norm, smooth=True)
        for w in WORDS:
            c, _ = ideal(w, kb)
            assert dec.topk(c, kb, "k", 0.5, 0.25, k=1) == [w]


def test_live_decoder_prior_and_resample(tmp_path):
    import torch
    from webswipe.decoder.live import LiveDecoder
    from webswipe.decoder.model import ShenDecoder
    ckpt = tmp_path / "rand.pt"
    torch.save({"model": ShenDecoder().state_dict()}, ckpt)
    kb = Keyboard.shen(169)
    c, t = ideal("hello", kb, hz=12)
    dec = LiveDecoder(str(ckpt), WORDS, prior_weight=1.0, resample_hz=50)
    cands, ms = dec(np.column_stack([c, t]), kb)
    assert len(cands) == 4 and set(cands) <= set(WORDS) and ms > 0


def test_live_shark2_on_overlay_trace():
    from webswipe.app.keyboard_ui import KeyboardOverlay
    from webswipe.decoder.live import LiveShark2
    ui = KeyboardOverlay(640, 480)
    lex = WORDS + ["hell", "jello", "word", "quit", "zebras"]
    rng = np.random.default_rng(0)
    for prior in [0.0, 0.25]:
        dec = LiveShark2(lex, prior_weight=prior)
        for w in WORDS:
            c, t = ideal(w, ui.kb, hz=12)  # webcam rate, overlay geometry
            c = c + rng.normal(0, 0.1 * ui.kb.key_w, c.shape)
            cands, ms = dec(np.column_stack([c, t]), ui.kb)
            assert len(cands) == 4 and ms < 100
            # without a prior the shape decides; with it, a frequent neighbour may win (help > hello)
            # but the intended word must stay among the candidates the user can pick
            assert cands[0] == w if prior == 0 else w in cands


def test_recordings_to_features(tmp_path):
    from webswipe.app.keyboard_ui import KeyboardOverlay
    from webswipe.app.recorder import Recorder
    from webswipe.decoder.webcam_data import build
    ui = KeyboardOverlay(640, 480)
    rec = Recorder("p1", out_dir=tmp_path)
    for w in ["hello", "world", "it's"]:  # the apostrophe word is skipped
        c, t = ideal(w.replace("'", ""), ui.kb, hz=12)
        rec.start()
        rec.save(w, np.column_stack([c, t]).tolist(), {"pitch": ui.pitch, "x0": ui.x0, "y0": ui.y0})
    out = tmp_path / "f.npz"
    build(str(tmp_path / "*.jsonl"), out, resample_hz=50)
    d = np.load(out)
    assert d["X"].shape == (2, 50, 39) and list(d["words"]) == ["hello", "world"]
    assert list(d["subject"]) == ["p1", "p1"]


def test_one_euro_uses_timestamps():
    from webswipe.app.pinch import OneEuro
    f = OneEuro()
    for i in range(5):
        f([float(i), 0.0], t=i / 12)
    assert abs(f.freq - 12) < 1e-6
