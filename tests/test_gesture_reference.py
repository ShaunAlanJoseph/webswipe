import sys
from types import SimpleNamespace
import pytest
from webswipe.app.gesture_reference import checked_scores
from webswipe.app.gesture_det import StateFilter


def test_confident_open_reference_ends_a_latched_pinch():
    state=StateFilter('ok',.5,.3,2)
    state.update({'ok':.9});assert state.update({'ok':.9})[0]
    primary={'ok':.95,'fist':0.,'no_gesture':0.}
    assert not state.update(checked_scores(primary,{'no_gesture':.8}))[0]
    assert primary['ok']==.95  # The inference cache must remain reusable.


def test_reference_fist_cannot_create_a_pinch():
    state=StateFilter('ok',.5,.3,2)
    for _ in range(4):
        assert not state.update(checked_scores({'ok':0.,'fist':.9},{'fist':.99}))[0]


def test_missing_reference_preserves_primary_evidence():
    assert checked_scores({'ok':.9},None)['ok']==.9
    assert checked_scores(None,{'ok':.99}).get('ok',0)==0


@pytest.mark.parametrize('label',['palm','no_gesture'])
def test_reference_veto_threshold(label):
    assert checked_scores({'ok':.9},{label:.5})['ok']==0
    assert checked_scores({'ok':.9},{label:.49})['ok']==.9


@pytest.mark.parametrize('arguments',[
    ['--gesture-reference-model','ref.onnx'],
    ['--gesture-reference-model','ref.onnx','--pinch','detector'],
])
def test_reference_cli_rejects_incompatible_input(monkeypatch,arguments):
    monkeypatch.setitem(sys.modules,'webswipe.app.tracking',SimpleNamespace(HandTracker=None))
    from webswipe.app.main import get_args
    monkeypatch.setattr(sys,'argv',['app',*arguments])
    with pytest.raises(SystemExit) as error:get_args()
    assert error.value.code==2
