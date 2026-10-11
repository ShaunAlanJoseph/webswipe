from webswipe.app.left_pinch import FusedLeftPinch
from webswipe.app.controls import OtherHandControl

def commands(samples):
 g=FusedLeftPinch();ctl=OtherHandControl();return [a for t,c,r,s in samples if (a:=ctl.update(g.update(c,r,s,t),t))]

def test_moderate_confidence_brief_closure_with_two_observations():
 assert commands([(0,False,.344,{'no_gesture':.86}),(60,False,.225,{'ok':.785}),(120,False,.206,{'ok':.734}),(180,False,.4,{'no_gesture':.9}),(280,False,.4,{'no_gesture':.9})])==['next']

def test_closure_from_relaxed_nearby_fingers_does_not_require_wide_open_pose():
 assert commands([(0,False,.293,{'no_gesture':.88}),(70,False,.187,{'ok':.87}),(140,False,.206,{'ok':.826}),(210,False,.275,{'no_gesture':.9}),(310,False,.275,{'no_gesture':.9})])==['next']

def test_static_near_pinch_classifier_change_is_not_motion():
 assert commands([(0,False,.25,{'no_gesture':.9}),(70,False,.24,{'ok':.77}),(140,False,.24,{'ok':.77}),(210,False,.25,{'no_gesture':.9}),(310,False,.25,{'no_gesture':.9})])==[]

def test_one_moderate_frame_cannot_trigger():
 assert commands([(0,False,.4,{'no_gesture':.9}),(70,False,.23,{'ok':.77}),(140,False,.4,{'no_gesture':.9}),(240,False,.4,{'no_gesture':.9})])==[]

def test_stale_or_interrupted_evidence_cannot_accumulate():
 assert commands([(0,False,.34,{'no_gesture':.9}),(60,False,.23,{'ok':.77}),(230,False,.23,{'ok':.77}),(290,False,.23,{'ok':.77}),(390,False,.4,{'no_gesture':.9}),(490,False,.4,{'no_gesture':.9})])==[]

def test_unconfirmed_open_pose_cannot_enable_rescue():
 assert commands([(0,False,.34,{}),(60,False,.23,{'ok':.77}),(130,False,.23,{'ok':.77}),(230,False,.4,{}),(330,False,.4,{})])==[]
