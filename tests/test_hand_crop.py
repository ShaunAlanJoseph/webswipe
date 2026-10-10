import numpy as np
from types import SimpleNamespace
from webswipe.app.hand_crop import crop_bounds, detect_hands


def test_crop_clips_to_frame_and_rejects_tiny_or_missing_region():
    px=np.array([[600.,400.],[680.,500.]])
    x0,y0,x1,y1=crop_bounds(px,(480,640,3))
    assert 0<=x0<x1<=640 and 0<=y0<y1<=480
    assert crop_bounds(np.ones((21,2)),(480,640,3)) is None
    assert crop_bounds(px+1000,(480,640,3)) is None


def test_crop_detections_are_mapped_back_to_image_coordinates():
    frame=np.zeros((480,640,3),np.uint8)
    hand=SimpleNamespace(px=np.array([[200.,200.],[300.,300.]]))
    x0,y0,x1,y1=crop_bounds(hand.px,frame.shape)
    seen=[]
    def detector(crop):
        seen.append(crop.shape)
        return [('ok',.9,(10.,20.,50.,70.))]
    result=detect_hands(detector,frame,[hand])
    assert seen==[(y1-y0,x1-x0,3)]
    assert result==[('ok',.9,(10+x0,20+y0,50+x0,70+y0))]
    assert detect_hands(detector,frame,[])==[]
