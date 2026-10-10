from pathlib import Path
from types import SimpleNamespace
from webswipe.app.controls import SwipeStartBuffer
from webswipe.app.runtime import configure_decoder


def test_delayed_detector_keeps_contact_prefix():
    b = SwipeStartBuffer(700)
    for t in range(0, 651, 50):
        b.update((500-t/10, 290), t, True, .15, {'ok': 0})
    b.update((430,290),700,True,.15,{'ok':.9})
    points=b.take_before(700)
    assert len(points)==14 and points[0]==(500.,290.,0)
    assert all(p[2]<700 for p in points)
    assert b.take_before(701)==[]


def test_near_contact_needs_confident_detector_and_hover_is_excluded():
    b=SwipeStartBuffer()
    b.update((10,10),0,False,.8,{'ok':.9})
    b.update((11,10),50,False,.3,{'ok':.1})
    assert b.take_before(100)==[]
    b.update((12,10),100,False,.35,{'ok':.9})
    b.update((13,10),150,True,.15,{'ok':.9})
    assert b.take_before(150)==[(12.,10.,100)]


def test_recovery_does_not_cross_gaps_or_include_active_swipe():
    b=SwipeStartBuffer()
    b.update((10,10),0,True,.1)
    b.update(None,50,False,None)
    b.update((12,10),100,True,.1)
    assert b.take_before(150)==[(12.,10.,100)]
    b.update((10,10),200,True,.1)
    b.update((11,10),250,True,.1,active=True)
    assert b.take_before(300)==[]


def test_recovery_window_and_disable():
    for limit in [100,0]:
        b=SwipeStartBuffer(limit)
        for t in [0,50,100,150]: b.update((1,2),t,True,.1)
        pts=b.take_before(150)
        assert [p[2] for p in pts]==([50,100] if limit else [])


def args(**kw):
    d=dict(mode='guided',lexicon=None,decoder=None,ckpt=None,prior_weight=None,resample_hz=None)
    return SimpleNamespace(**(d|kw))


def test_auto_decoder_uses_local_model_and_records_effective_defaults(tmp_path):
    (tmp_path/'checkpoints').mkdir();(tmp_path/'checkpoints/d1_loso0_aug_last.pt').touch()
    (tmp_path/'data').mkdir();(tmp_path/'data/lexicon_10k.txt').touch()
    a=configure_decoder(args(),tmp_path)
    assert a.decoder=='neural' and Path(a.ckpt).is_file()
    assert a.resample_hz==50 and a.prior_weight==0
    assert a.lexicon.endswith('lexicon_10k.txt')
    a=configure_decoder(args(decoder='shark2',prior_weight=0,resample_hz=0),tmp_path)
    assert a.decoder=='shark2' and a.ckpt is None
    assert a.prior_weight==0 and a.resample_hz==0


def test_auto_decoder_without_checkpoint_uses_shark2(tmp_path):
    a=configure_decoder(args(),tmp_path)
    assert a.decoder=='shark2' and a.prior_weight==.25
