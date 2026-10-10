import json

import numpy as np

from webswipe.app.controls import Candidates, OtherHandControl, SwipeSegmenter
from webswipe.app.tap import EventLog, TapTyper
from webswipe.decoder.features import LETTERS, Keyboard

FRAME = 83  # ms per frame at ~12 fps


def run_segmenter(frames, **kw):
    """frames: list of (hand_seen, pinched); returns events and number of recorded points."""
    seg, events, points = SwipeSegmenter(**kw), [], 0
    for i, (seen, pinched) in enumerate(frames):
        ev, keep = seg.update(seen, pinched, i * FRAME)
        if ev:
            events.append(ev)
        points += keep
    return events, points


def test_swipe_survives_short_dropout():
    frames = [(True, False)] * 2 + [(True, True)] * 5 + [(False, False)] * 2 + [(True, True)] * 5 + [(True, False)]
    events, points = run_segmenter(frames)
    assert events == ["start", "end"] and points == 10  # one gesture, lost frames add no points


def test_long_dropout_ends_swipe():
    frames = [(True, True)] * 5 + [(False, False)] * 6 + [(True, True)] * 3 + [(True, False)]
    events, _ = run_segmenter(frames)
    assert events == ["start", "end", "start", "end"]


def test_no_events_without_hand():
    assert run_segmenter([(False, False)] * 10) == ([], 0)


def run_other(pattern):
    ctl, out = OtherHandControl(hold_ms=600), []
    for i, p in enumerate(pattern):
        a = ctl.update(p, i * FRAME)
        if a:
            out.append(a)
    return out


def test_other_hand_short_and_long_pinch():
    assert run_other([False, True, True, False]) == ["next"]          # ~170 ms
    assert run_other([False] + [True] * 10 + [False]) == ["delete"]   # ~830 ms, fires once while held
    assert run_other([True, False, False, True, False]) == ["next", "next"]


def test_candidates_swap_and_delete():
    w = Candidates()
    w.commit(["help", "hello", "jello", "hell"])
    w.next()
    assert w.typed == ["hello"]
    w.choose(3)
    assert w.typed == ["hell"]
    w.commit(["world", "word"])
    w.delete()
    assert w.typed == ["hell"] and w.cands == []
    w.next()  # nothing to swap after a delete
    assert w.typed == ["hell"]


def test_tap_typer_raw_autocorrect_backspace():
    kb = Keyboard.from_pitch(50, 50)
    key = {c: kb.centers[i] for i, c in enumerate(LETTERS)}
    t = TapTyper(kb, lexicon=["hello", "world", "help"])
    for c in "hello":
        assert t.tap(key[c]) == c
    assert t.space() == ("hello", "hello")
    # noisy taps: 'w' lands on 'e', still snapped to the only 5-letter word starting nearby
    for c, off in zip("world", [(30, 0), (0, 0), (0, 0), (0, 0), (0, 0)]):  # keys are 50 apart
        t.tap(key[c] + np.array(off))
    assert t.space() == ("eorld", "world")
    t.tap(key["x"])
    t.backspace()
    t.backspace()  # nothing pending: removes the last word
    assert t.text == "hello" and t.space() is None
    raw = TapTyper(kb)  # no lexicon: raw letters
    for c in "xyz":
        raw.tap(key[c])
    assert raw.space() == ("xyz", "xyz")


def test_event_log(tmp_path):
    log = EventLog("p1", "tap", out_dir=tmp_path)
    log(10, "tap", key="a")
    log(20, "space", raw="a", word="a")
    log.close()
    lines = [json.loads(x) for x in open(log.path)]
    assert [x["event"] for x in lines] == ["tap", "space"] and lines[0]["t"] == 10


def _hand():
    from types import SimpleNamespace
    return SimpleNamespace(side="Right", score=0.9, px=np.zeros((21, 2)), world=np.zeros((21, 3)))


def test_recording_checker_and_resume(tmp_path):
    import importlib.util
    from webswipe.app.main import resume_phrase
    from webswipe.app.recorder import Recorder
    spec = importlib.util.spec_from_file_location("check_recordings", "tools/check_recordings.py")
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)

    rec = Recorder("p1", out_dir=tmp_path)
    good = [[i * 10.0, 0.0, i * 83.0] for i in range(10)]  # 10 points, 750 ms
    rec.start()
    for i in range(10):
        rec.frame(i * 83, [_hand()])
    rec.save("hello", good, {"pitch": 50}, {"phrase_index": 3})
    rec.start()
    rec.frame(0, [_hand()])
    rec.frame(83, [])  # hand lost mid-swipe
    rec.save("hi", good[:3], {"pitch": 50}, {"phrase_index": 4})

    out = chk.report(str(tmp_path / "*.jsonl"))
    assert out["p1"]["words"] == 2 and out["p1"]["phrases"] == 2 and out["p1"]["flagged"] == 1
    assert abs(out["p1"]["median_fps"] - 12) < 1
    assert resume_phrase("p1", out_dir=tmp_path) == 5
    assert resume_phrase("nobody", out_dir=tmp_path) == 0


def test_brief_pinch_loss_does_not_finish_word():
    seg = SwipeSegmenter(release_ms=120)
    assert seg.update(True, True, 0) == ("start", True)
    assert seg.update(True, False, 50) == (None, False)
    assert seg.update(True, False, 100) == (None, False)
    assert seg.update(True, True, 150) == (None, True)
    assert seg.update(True, False, 200) == (None, False)
    assert seg.update(True, False, 320) == ("end", False)
    assert seg.update(True, False, 400) == (None, False)


def test_tracking_loss_still_ends_delayed_release():
    seg = SwipeSegmenter(max_gap_ms=200, release_ms=120)
    seg.update(True, True, 0)
    seg.update(True, False, 50)
    assert seg.update(False, False, 100) == (None, False)
    assert seg.update(False, False, 301) == ("end", False)
    assert seg.update(True, True, 350) == ("start", True)


def test_other_hand_hold_survives_short_false_reading():
    from webswipe.app.controls import ReleaseDelay
    delay, ctl = ReleaseDelay(120), OtherHandControl(600)
    actions = []
    for t in range(0, 1001, 50):
        detected = t < 750 and t != 300
        action = ctl.update(delay.update(detected, t), t)
        if action:
            actions.append(action)
    assert actions == ["delete"]
    assert not delay.on


def test_other_hand_short_pinch_fires_once_after_release():
    from webswipe.app.controls import ReleaseDelay
    delay, ctl = ReleaseDelay(120), OtherHandControl(600)
    actions = []
    for t in range(0, 501, 50):
        action = ctl.update(delay.update(t < 150, t), t)
        if action:
            actions.append(action)
    assert actions == ["next"]


def test_other_hand_rejects_single_frame_phantom_before_release_delay():
    from webswipe.app.controls import ReleaseDelay
    for frame_ms in (33, 100):
        delay, ctl = ReleaseDelay(80, start_votes=2), OtherHandControl()
        actions = []
        for i, detected in enumerate([False, True, False, False, False, False]):
            action = ctl.update(delay.update(detected, i * frame_ms), i * frame_ms)
            if action:
                actions.append(action)
        assert actions == []
        assert not delay.on


def test_confirmed_other_hand_short_and_held_controls_with_dropout():
    from webswipe.app.controls import ReleaseDelay
    for frame_ms in (33, 100):
        for duration, expected in ((300, "next"), (1000, "delete")):
            delay, ctl = ReleaseDelay(80, start_votes=2), OtherHandControl()
            actions = []
            for t in range(0, duration + 400, frame_ms):
                detected = t < duration and not (frame_ms == 33 and t == 198)
                action = ctl.update(delay.update(detected, t), t)
                if action:
                    actions.append(action)
            assert actions == [expected]


def test_contact_confirmation_rejects_fist_and_unconfirmed_near_contact():
    from webswipe.app.controls import ContactAppearanceGate, ReleaseDelay
    gate, delay, ctl = ContactAppearanceGate(), ReleaseDelay(80, start_votes=2), OtherHandControl()
    events = []
    for t in range(0, 1200, 33):
        on = gate.update(t < 900, {"fist": .95}, active=delay.on)
        action = ctl.update(delay.update(on, t), t)
        if action:
            events.append(action)
    assert events == []


def test_contact_confirmation_survives_classifier_flicker_and_resets():
    from webswipe.app.controls import ContactAppearanceGate, ReleaseDelay
    gate, delay, ctl = ContactAppearanceGate(), ReleaseDelay(80, start_votes=2), OtherHandControl()
    events = []
    for t in range(0, 1000, 33):
        scores = {"ok": .7} if t == 0 else {"no_gesture": .8}
        on = gate.update(t < 700, scores, active=delay.on)
        action = ctl.update(delay.update(on, t), t)
        if action:
            events.append(action)
    assert events == ["delete"]
    assert not gate.update(True, {"fist": .9}, active=False)


def test_contact_confirmation_short_pinch_and_brief_tracking_loss():
    from webswipe.app.controls import ContactAppearanceGate, ReleaseDelay
    gate, delay, ctl = ContactAppearanceGate(), ReleaseDelay(80, start_votes=2), OtherHandControl()
    events = []
    for t in range(0, 600, 33):
        on = gate.update(t < 250 and t != 132, {"ok": .6} if t == 0 else {}, active=delay.on)
        action = ctl.update(delay.update(on, t), t)
        if action:
            events.append(action)
    assert events == ["next"]
