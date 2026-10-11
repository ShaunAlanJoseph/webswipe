from webswipe.app.left_pinch import FusedLeftPinch
from webswipe.app.controls import OtherHandControl

def actions(samples):
    gate=FusedLeftPinch();ctl=OtherHandControl();out=[]
    for t,contact,gap,score in samples:
        event=ctl.update(gate.update(contact,gap,score,t),t)
        if event:out.append(event)
    return out

def test_brief_pinch_precedes_smoothed_contact():
    assert actions([(0,False,.5,{'no_gesture':.95}),
                    (100,False,.22,{'ok':.91}),
                    (200,False,.23,{'ok':.9}),
                    (300,False,.5,{'no_gesture':.95}),
                    (400,False,.5,{'no_gesture':.95})]) == ['next']

def test_visual_release_overrides_phantom_contact_without_delete_or_retrigger():
    samples=[(0,False,.5,{'no_gesture':.95}),(100,True,.1,{'ok':.91})]
    samples += [(t,True,.1,{'no_gesture':.95}) for t in range(200,1200,100)]
    assert actions(samples)==['next']

def test_weaker_but_real_hold_survives_classifier_dropout():
    samples=[(0,False,.5,{}),(100,True,.15,{'ok':.6}),(200,True,.15,{'ok':.6})]
    samples += [(t,True,.15,{'no_gesture':.75}) for t in range(300,1200,100)]
    samples += [(1200,False,.5,{'no_gesture':.95}),(1300,False,.5,{'no_gesture':.95})]
    assert actions(samples)==['delete']

def test_strong_hold_still_deletes_once():
    samples=[(0,False,.5,{})]+[(t,True,.12,{'ok':.93}) for t in range(100,1200,100)]
    samples += [(1200,False,.5,{}),(1300,False,.5,{})]
    assert actions(samples)==['delete']

def test_geometry_alone_never_starts_a_command():
    assert actions([(t,True,.1,{'no_gesture':.95}) for t in range(0,1500,100)])==[]

def test_model_alone_with_separated_fingers_never_starts():
    assert actions([(t,False,.5,{'ok':.95}) for t in range(0,1500,100)])==[]

def test_release_timers_do_not_borrow_each_others_duration():
    g=FusedLeftPinch();g.update(False,.5,{},0);assert g.update(True,.1,{'ok':.95},100)
    assert g.update(True,.1,{'no_gesture':.95},200)
    assert g.update(True,.1,{'no_gesture':.95},300)
    # One geometry-release observation must not inherit 100 ms of visual release.
    assert g.update(False,.5,{'fist':.9},310)
    assert not g.update(False,.5,{'fist':.9},400)

def test_fast_episode_cannot_immediately_rearm_from_same_closure():
    g=FusedLeftPinch();g.update(False,.5,{},0);assert g.update(False,.22,{'ok':.95},100)
    assert g.update(False,.5,{},200);assert not g.update(False,.5,{},300)
    assert not g.update(False,.17,{'ok':.95},350)

def test_missing_hand_ends_episode():
    assert actions([(0,False,.5,{}),(100,True,.1,{'ok':.95}),
                    (200,False,None,{}),(300,False,None,{})])==['next']
