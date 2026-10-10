"""Hand crop geometry shared by the remote experiment and optional app inference."""
import numpy as np


def crop_bounds(px, shape, padding=1.35):
    h,w=shape[:2]
    lo,hi=np.min(px,axis=0),np.max(px,axis=0)
    center=(lo+hi)/2
    # Match the training annotation's 12% margin, then its 1.35 crop expansion.
    side=float(np.max(hi-lo))*1.24*padding
    x0=max(0,int(np.floor(center[0]-side/2)))
    y0=max(0,int(np.floor(center[1]-side/2)))
    x1=min(w,int(np.ceil(center[0]+side/2)))
    y1=min(h,int(np.ceil(center[1]+side/2)))
    return (x0,y0,x1,y1) if min(x1-x0,y1-y0)>=20 else None


def detect_hands(detector,frame,hands):
    detections=[]
    for hand in hands:
        box=crop_bounds(hand.px,frame.shape)
        if box is None:continue
        x0,y0,x1,y1=box
        for cls,score,(a,b,c,d) in detector(frame[y0:y1,x0:x1]):
            detections.append((cls,score,(a+x0,b+y0,c+x0,d+y0)))
    return detections
