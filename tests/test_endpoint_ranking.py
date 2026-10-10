import numpy as np
import torch
from webswipe.decoder.features import Keyboard, LETTERS, endpoint_cost
from webswipe.decoder.live import LiveDecoder
from webswipe.decoder.decode import Lexicon


def test_endpoint_cost_uses_geometry_not_target_and_is_scale_invariant():
    kb=Keyboard.from_pitch(20,20)
    trace=np.array([[*kb.centers[LETTERS.index('h')],0],[*kb.centers[LETTERS.index('o')],100]])
    cost=endpoint_cost(trace,kb,['hell','hello'])
    assert cost[1]==0 and cost[0]>0
    kb2=Keyboard.from_pitch(40,40)
    trace2=trace.copy();trace2[:,:2]*=2
    np.testing.assert_allclose(cost,endpoint_cost(trace2,kb2,['hell','hello']))
    assert endpoint_cost([[10000,10000],[10001,10001]],kb,['hello'])[0]==12


def test_optional_ranking_can_use_last_key_without_changing_default():
    kb=Keyboard.from_pitch(20,20)
    h,o=kb.centers[LETTERS.index('h')],kb.centers[LETTERS.index('o')]
    trace=np.array([[*(h+(o-h)*i/5),i*100] for i in range(6)])
    d=LiveDecoder.__new__(LiveDecoder)
    d.model=lambda f:torch.zeros(1,len(f[0]),29)
    d.resample_hz=50;d.prior_weight=0;d.endpoint_weight=0
    class FakeLex:
        words=['hell','hello']
        def scores(self,*args):return torch.tensor([.4,0.])
        def topk(self,*args):return ['hell','hello']
    d.lex=FakeLex()
    assert d(trace,kb)[0][0]=='hell'
    d.endpoint_weight=1
    assert d(trace,kb)[0][0]=='hello'


def test_impossible_ctc_target_is_not_given_a_perfect_likelihood():
    lex=Lexicon(['a','aaaa'])
    scores=lex.scores(torch.zeros(5,29))
    assert torch.isfinite(scores[0])
    assert torch.isneginf(scores[1])
    assert lex.topk(torch.zeros(5,29),1)==['a']
