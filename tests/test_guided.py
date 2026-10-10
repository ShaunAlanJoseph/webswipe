import json
from types import SimpleNamespace
import cv2 as cv
import numpy as np
from webswipe.app.guided import Attempt, Guided
from webswipe.app.keyboard_ui import KeyboardOverlay


def test_video_frames_and_feedback_survive_broken_swipes(tmp_path):
    a = Attempt(tmp_path, 'hello', 0, 'tester', dict(pitch=20, x0=100, y0=100))
    for i in range(5):
        a.frame(np.full((48,64,3), i*30, np.uint8), dict(t_ms=i*70, event='end' if i==2 else None))
    a.segment([[i,i,i*70] for i in range(4)])
    a.segment([[i,i,300+i*70] for i in range(4)])
    a.finish(lambda tr,kb: (['hell','hello'], 2.0), None)
    a.feedback('tracking_failure')
    cap=cv.VideoCapture(str(a.path/'camera.avi'))
    count=0
    while cap.read()[0]: count+=1
    cap.release()
    assert count==5
    frames=[json.loads(l) for l in (a.path/'frames.jsonl').read_text().splitlines()]
    assert [f['frame_index'] for f in frames]==list(range(5))
    assert [f['t_ms'] for f in frames]==[0,70,140,210,280]
    saved=json.loads((a.path/'attempt.json').read_text())
    assert len(saved['segments'])==2 and saved['segments'][0]['candidates']==['hell','hello']
    assert not saved['training_eligible']


def test_retry_kept_and_only_clean_accepted_exported(tmp_path):
    args=SimpleNamespace(words=None, guided_out=str(tmp_path), participant='tester',width=640,height=480)
    g=Guided(args, KeyboardOverlay(640,480), lambda tr,kb: (['hell'],1))
    g.start(); g.attempt.segment([[i,i,i*60] for i in range(5)])
    g.finish([],False); first=g.attempt.path; g.feedback('user_retry')
    assert g.index==0 and json.loads((first/'attempt.json').read_text())['status']=='user_retry'
    g.start(); g.attempt.segment([[i,i,i*60] for i in range(5)])
    g.finish([],False); g.feedback('accepted')
    assert g.index==1
    rows=(g.root/'accepted.jsonl').read_text().splitlines()
    assert len(rows)==1 and json.loads(rows[0])['target']=='hello'
    g.start();g.finish([[i,i,i*60] for i in range(5)],True);g.feedback('accepted')
    assert not g.attempt.data['training_eligible']
    assert len((g.root/'accepted.jsonl').read_text().splitlines())==1


def test_close_preserves_unfinished_attempt(tmp_path):
    args=SimpleNamespace(words=None,guided_out=str(tmp_path),participant='p',width=640,height=480)
    g=Guided(args,KeyboardOverlay(640,480),None)
    g.start();g.close();g.close()
    assert json.loads((g.attempt.path/'attempt.json').read_text())['status']=='aborted'


def test_main_guided_records_across_two_segments(tmp_path,monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "webswipe.app.tracking", SimpleNamespace(HandTracker=None))
    from webswipe.app import main as m
    from webswipe.decoder import live
    monkeypatch.setattr(sys,'argv',['app','--mode','guided','--decoder','shark2','--guided-out',str(tmp_path)])
    # One setup frame, then ready/start, two swipes separated by sustained release, finish/accept/quit.
    pattern=[False]+[True]*5+[False]*4+[True]*5+[False]*4+[False]*3
    class Cap:
        def set(self,*a): pass
        def read(self): return True,np.zeros((480,640,3),np.uint8)
        def release(self): pass
    class Tracker:
        i=-1
        def __call__(self,frame):
            self.i+=1
            px=np.zeros((21,2));px[:,0]=350+self.i;px[:,1]=300
            return [SimpleNamespace(side='Right',score=.9,px=px,world=np.zeros((21,3)))],self.i*60
    class Pinch:
        thr=.2
        i=-1
        def __init__(self,*a): pass
        def reset_state(self): pass
        def update(self,px):
            self.i+=1
            return pattern[min(self.i,len(pattern)-1)],.1
    keys=iter([32]+[-1]*17+[32,ord('a'),ord('q')])
    monkeypatch.setattr(m.cv,'VideoCapture',lambda *a:Cap())
    for name in ['namedWindow','resizeWindow','setMouseCallback','imshow','destroyAllWindows']:
        monkeypatch.setattr(m.cv,name,lambda *a:None)
    monkeypatch.setattr(m.cv,'getWindowProperty',lambda *a:1)
    monkeypatch.setattr(m.cv,'waitKey',lambda *a:next(keys))
    monkeypatch.setattr(m,'HandTracker',Tracker)
    monkeypatch.setattr(m,'ContactPinch',Pinch)
    monkeypatch.setattr(live,'load_lexicon',lambda *a:['hello','hell'])
    monkeypatch.setattr(live,'LiveShark2',lambda *a:lambda tr,kb:(['hell','hello'],1.))
    m.main()
    paths=list(tmp_path.glob('*/attempt_*/attempt.json'))
    assert len(paths)==1
    meta=json.loads(paths[0].read_text())
    assert meta['status']=='accepted' and len(meta['segments'])==2
    assert meta['frame_count']==18 and not meta['training_eligible']
    frames=[json.loads(l) for l in (paths[0].parent/'frames.jsonl').read_text().splitlines()]
    assert sum(f['event']=='end' for f in frames)==2
    assert all('hands' in f and 'cursor' in f for f in frames)
