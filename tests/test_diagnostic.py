import json
from types import SimpleNamespace
import numpy as np
from webswipe.app.diagnostic import calibration_phase,export_word,analyze
from webswipe.app.guided import Attempt,write_json


def test_phase_sequence_and_keyboard_export(tmp_path):
    assert [calibration_phase(t)[1] for t in [0,1.9,2,3.9,4,6]]==[False,False,True,True,False,False]
    a=Attempt(tmp_path,'hello',0,'p',{'pitch':20})
    a.segment([[i,i,i*50] for i in range(5)],'keyboard_release');a.close();a.feedback('accepted')
    export_word(a,tmp_path)
    assert a.data['training_eligible']
    assert json.loads((tmp_path/'accepted.jsonl').read_text())['boundary_source']=='keyboard'
    a.data['missing_while_held']=1;export_word(a,tmp_path)
    assert not a.data['training_eligible']
    assert len((tmp_path/'accepted.jsonl').read_text().splitlines())==1


def test_analysis_uses_only_accepted_trials_and_separates_repetitions(tmp_path):
    write_json(tmp_path/'session.json',{'mode':'calibration'})
    for rep in [0,1]:
        a=Attempt(tmp_path,'Right left',0,'p',{})
        a.close();a.data.update(status='accepted',repetition=rep);write_json(a.path/'attempt.json',a.data)
        rows=[]
        for t in np.arange(0,6,.1):
            _,expected,phase=calibration_phase(t)
            rows.append(dict(ratio=.12 if expected else .35,expected_pinch=expected,phase_elapsed=phase))
        (a.path/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=analyze(tmp_path)
    assert report['training_trials']==report['heldout_trials']==1
    assert report['proposed_heldout']['balanced_accuracy']==1
    assert report['applied'] is False


def test_space_release_finishes_word_even_when_pinch_never_releases(tmp_path,monkeypatch):
    import sys
    monkeypatch.setenv('SDL_VIDEODRIVER','dummy');monkeypatch.setenv('SDL_AUDIODRIVER','dummy')
    import pygame as pg
    from webswipe.app import diagnostic as m, runtime, gesture_det
    from webswipe.decoder import live
    class Tracker:
        i=0
        def __call__(self,frame):
            self.i+=1
            px=np.zeros((21,2));px[:]=[400+self.i,300];px[9]=[400+self.i,400];px[8]=[410+self.i,300]
            return [SimpleNamespace(side='Right',score=.9,px=px,world=np.zeros((21,3)))],self.i*100
    monkeypatch.setitem(sys.modules,'webswipe.app.tracking',SimpleNamespace(HandTracker=Tracker))
    monkeypatch.setattr(gesture_det,'GestureDetector',lambda *a:lambda f:[])
    monkeypatch.setattr(runtime,'configure_decoder',lambda a:SimpleNamespace(decoder='shark2',ckpt=None,lexicon='unused',prior_weight=0,resample_hz=50))
    monkeypatch.setattr(live,'load_lexicon',lambda p:['hello'])
    monkeypatch.setattr(live,'LiveShark2',lambda *a:lambda tr,kb:(['hello'],1))
    class Cap:
        def set(self,*a):pass
        def read(self):return True,np.zeros((480,640,3),np.uint8)
        def release(self):pass
    monkeypatch.setattr(m.cv,'VideoCapture',lambda *a:Cap())
    events=[[pg.event.Event(pg.KEYDOWN,key=pg.K_s)], [pg.event.Event(pg.KEYDOWN,key=pg.K_SPACE)]]
    events += [[]]*5+[[pg.event.Event(pg.KEYUP,key=pg.K_SPACE)]]+[[]]*8
    events += [[pg.event.Event(pg.KEYDOWN,key=pg.K_a)],[pg.event.Event(pg.KEYDOWN,key=pg.K_q)]]
    it=iter(events);monkeypatch.setattr(pg.event,'get',lambda:next(it))
    monkeypatch.setattr(sys,'argv',['diagnostic','--out',str(tmp_path)])
    m.main()
    p=next(tmp_path.glob('*/attempt_*/attempt.json'));d=json.loads(p.read_text())
    assert d['status']=='accepted' and d['training_eligible']
    assert len(d['segments'])==1 and len(d['segments'][0]['trace_kb'])==6
    fs=[json.loads(l) for l in (p.parent/'frames.jsonl').read_text().splitlines()]
    assert all(f['pinched'] for f in fs)
    end=next(f for f in fs if f['event']=='end')
    assert end['key_held'] is False
    assert fs[-1]['t_ms']-end['t_ms']>=700
