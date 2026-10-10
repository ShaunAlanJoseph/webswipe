import json
from pathlib import Path
from webswipe.app.pinch import DetectorRelease
from webswipe.app.controls import SwipeSegmenter


def test_sticky_release_ends_after_sustained_open_palm():
    release=DetectorRelease()
    seg=SwipeSegmenter(release_ms=120)
    seg.update(True,True,0)
    events=[]
    for t in range(50,650,50):
        r=.14 if t<200 else .29
        scores={'ok':.92} if t<200 else {'ok':0.,'palm':.94}
        on=release.update(True,r,scores,t)
        event,_=seg.update(True,on,t)
        if event:events.append((event,t))
    assert events==[('end',500)]
    # The landmark contact flag remains sticky, but no duplicate start fires.
    assert release.blocked and not release.update(True,.29,{'palm':.94},650)
    # A genuine new, tight pinch can start even if the measured open gap never reached 0.30.
    assert release.update(True,.14,{'ok':.9},700)


def test_short_detector_error_does_not_end_swipe():
    release=DetectorRelease()
    for t in (0,50,100):
        assert release.update(True,.14,{'ok':.9},t)
    assert release.update(True,.26,{'palm':.9},150)
    assert release.update(True,.26,{'palm':.9},200)
    assert release.update(True,.2,{'ok':.8},250)
    assert release.update(True,.27,{'palm':.9},300)
    assert release.update(True,.27,{'palm':.9},350)
    assert not release.blocked


def test_low_confidence_or_no_gap_cannot_force_release():
    release=DetectorRelease()
    for t in range(0,1000,50):
        assert release.update(True,.14,{'palm':.99},t)
        assert release.update(True,.27,{'palm':.4},t+1)
        assert release.update(True,.27,{'palm':.99,'ok':.7},t+2)


def test_real_sticky_is_trace_would_release(tmp_path):
    p=Path('data/guided/20261010_000050_d4d0ed/attempt_199aab9c77/frames.jsonl')
    if not p.exists():
        return
    frames=[json.loads(line) for line in p.read_text().splitlines()]
    release=DetectorRelease()
    seg=SwipeSegmenter(release_ms=120)
    events=[]
    for f in frames:
        on=release.update(f['contact'],f['ratio'],f['detector_scores'],f['t_ms'])
        event,_=seg.update(bool(f['hands']),on,f['t_ms'])
        if event: events.append((event,f['frame_index']))
    assert events[0][0]=='start'
    assert len([e for e in events if e[0]=='end'])==1
    assert events[-1][1]<len(frames)-1


def test_sustained_detector_disagreement_with_small_gap_preserves_swipe():
    # A real "his" gesture briefly looked like palm with r around 0.24.
    release=DetectorRelease()
    for t in range(0,400,50):
        assert release.update(True,.24,{'ok':0.,'palm':.95},t)
    assert not release.blocked


def test_contact_releases_open_fingers_despite_confident_pinch_detector():
    import numpy as np
    from webswipe.app.pinch import ContactPinch
    pinch=ContactPinch()
    seg=SwipeSegmenter(release_ms=80)
    px=np.zeros((21,2));px[9]=[0,100]
    events=[]
    # Ratios around 0.28 stayed stuck in the user's "good" recording.
    for i,r in enumerate([.15]*5+[.28]*6):
        px[8]=[r*100,0]
        on,_=pinch.update(px)
        event,_=seg.update(True,on,i*50)
        if event:events.append((event,i))
    assert events==[('start',0),('end',8)]


def test_contact_preserves_valid_swipe_with_quarter_palm_gap():
    import numpy as np
    from webswipe.app.pinch import ContactPinch
    pinch=ContactPinch()
    px=np.zeros((21,2));px[9]=[0,100]
    for r in [.15]*3+[.255]*8+[.15]*3:
        px[8]=[r*100,0]
        assert pinch.update(px)[0]
