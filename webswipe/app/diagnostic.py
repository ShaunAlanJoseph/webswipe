"""Independent word boundaries and prompted pinch transitions, with synchronized video."""
import argparse
import json
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from .guided import Attempt, write_json
from .keyboard_ui import KeyboardOverlay
from .pinch import ContactPinch, OneEuro, pinch_point

WORDS = 'hello is his good night world hello is his good night world'.split()


def calibration_phase(elapsed):
    if elapsed < 2:
        return 'OPEN: keep thumb and index clearly apart', False, elapsed
    if elapsed < 4:
        return 'PINCH: touch thumb and index fingertips', True, elapsed-2
    return 'OPEN AGAIN: separate thumb and index', False, elapsed-4


def export_word(attempt, root):
    d=attempt.data
    segments=d['segments']
    eligible=(d['status']=='accepted' and len(segments)==1
              and segments[0]['boundary']=='keyboard_release'
              and len(segments[0]['trace_kb'])>=4 and d.get('missing_while_held',0)==0)
    d['training_eligible']=eligible
    write_json(attempt.path/'attempt.json',d)
    if eligible:
        with (root/'accepted.jsonl').open('a') as f:
            f.write(json.dumps(dict(participant=d['participant'],target=d['target'],
                     trace_kb=segments[0]['trace_kb'],keyboard=d['keyboard'],
                     source_attempt=str(attempt.path),boundary_source='keyboard',session=root.name))+'\n')


def analyze(root):
    """First repetition proposes settings; second repetition checks prompt agreement."""
    root=Path(root)
    meta=json.loads((root/'session.json').read_text())
    if meta['mode']!='calibration':
        raise ValueError('Choose a calibration session, not a word session.')
    sets=[[],[]]
    for p in root.glob('attempt_*/attempt.json'):
        d=json.loads(p.read_text())
        if d['status']!='accepted':continue
        frames=[json.loads(l) for l in (p.parent/'frames.jsonl').read_text().splitlines()]
        sets[d['repetition']].append(frames)
    if not all(sets):
        raise ValueError('Accept at least one trial from each repetition before analysis.')
    def score(trials,on,off):
        tp=tn=fp=fn=0
        for frames in trials:
            state=False
            for f in frames:
                r=f['ratio']
                state=(r < (off if state else on)) if r is not None else False
                if f['phase_elapsed'] < .7:continue
                expected=f['expected_pinch']
                tp+=bool(state and expected);tn+=bool(not state and not expected)
                fp+=bool(state and not expected);fn+=bool(not state and expected)
        bal=.5*(tp/max(tp+fn,1)+tn/max(tn+fp,1))
        return dict(balanced_accuracy=bal,tp=tp,tn=tn,fp=fp,fn=fn)
    options=[(score(sets[0],on,off)['balanced_accuracy'],on,off)
             for on in np.arange(.10,.271,.01) for off in np.arange(.20,.411,.01) if off>=on+.02]
    _,on,off=max(options,key=lambda x:(x[0],-abs(x[1]-.2)-abs(x[2]-.27)))
    report=dict(label_basis='Prompt compliance, not manually verified contact; transition windows excluded.',
                training_trials=len(sets[0]),heldout_trials=len(sets[1]),
                proposed=dict(on=round(float(on),3),off=round(float(off),3)),
                current_heldout=score(sets[1],.2,.27),proposed_heldout=score(sets[1],on,off),
                applied=False,next_step='Review failures in video and compare positions/hands before applying or training.')
    write_json(root/'calibration_analysis.json',report)
    print(json.dumps(report,indent=2))
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['words','calibration'],default='words')
    ap.add_argument('--participant',default='shaun')
    ap.add_argument('--out',default='data/diagnostic')
    ap.add_argument('--analyze',help='analyze an existing calibration session directory')
    ap.add_argument('--camera',type=int,default=0)
    args=ap.parse_args()
    if args.analyze:
        analyze(args.analyze);return
    import pygame as pg
    from .tracking import HandTracker
    from .gesture_det import GestureDetector,assign
    from .runtime import configure_decoder
    from types import SimpleNamespace
    config=configure_decoder(SimpleNamespace(mode='guided',lexicon=None,decoder=None,ckpt=None,
                                             prior_weight=None,resample_hz=None))
    decoder=None
    if args.mode=='words':
        from webswipe.decoder.live import LiveDecoder,LiveShark2,load_lexicon
        if config.decoder=='neural':
            import torch
            torch.set_num_threads(2)
            decoder=LiveDecoder(config.ckpt,load_lexicon(config.lexicon),config.prior_weight,config.resample_hz)
        else:
            decoder=LiveShark2(load_lexicon(config.lexicon),config.prior_weight,config.resample_hz)
    plans=([dict(target=w,side='Right') for w in WORDS] if args.mode=='words' else
           [dict(target=f'{side} {position}',side=side,position=position,repetition=rep)
            for rep in range(2) for side in ['Right','Left'] for position in ['left','middle','right']])
    root=Path(args.out)/(args.mode+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns())[-6:])
    root.mkdir(parents=True)
    cap=cv.VideoCapture(args.camera);cap.set(cv.CAP_PROP_FRAME_WIDTH,640);cap.set(cv.CAP_PROP_FRAME_HEIGHT,480)
    ok,frame=cap.read()
    if not ok:
        cap.release();raise SystemExit('Cannot read camera')
    h,w=frame.shape[:2];ui=KeyboardOverlay(w,h)
    keyboard=dict(pitch=ui.pitch,x0=ui.x0,y0=ui.y0,frame=[w,h])
    write_json(root/'session.json',dict(schema=1,mode=args.mode,participant=args.participant,
               plans=plans,decoder=vars(config),keyboard=keyboard,created=time.time()))
    print('Saving to',root,flush=True)
    tracker=HandTracker();detector=GestureDetector('models/gesture_320.onnx')
    contact={side:ContactPinch() for side in ['Right','Left']};smooth=OneEuro()
    pg.init();screen=pg.display.set_mode((1280,960),pg.RESIZABLE);pg.display.set_caption('WebSwipe diagnostic recorder')
    state='ready';index=0;attempt=None;held=False;trace=[];start_t=None;tail_t=None
    running=True;missing=0;status=''
    try:
        while running:
            finish_key=False
            for e in pg.event.get():
                if e.type==pg.QUIT or (e.type==pg.KEYDOWN and e.key in (pg.K_q,pg.K_ESCAPE)):
                    running=False
                elif e.type==pg.WINDOWFOCUSLOST and state in ('recording','tail'):
                    held=False;attempt.close();attempt.feedback('interrupted');state='ready'
                    status='Window lost focus; attempt saved as interrupted. Press S to retry.'
                elif e.type==pg.KEYDOWN and e.key==pg.K_s and state=='ready':
                    attempt=Attempt(root,plans[index]['target'],index,args.participant,keyboard)
                    attempt.data.update(plans[index],boundary_source='keyboard' if args.mode=='words' else 'prompt')
                    state='recording';held=False;trace=[];missing=0;start_t=None;smooth.reset();status=''
                    for c in contact.values():c.reset_state()
                elif e.type==pg.KEYDOWN and e.key==pg.K_SPACE and state=='recording' and args.mode=='words':
                    held=True
                elif e.type==pg.KEYUP and e.key==pg.K_SPACE and held and state=='recording':
                    held=False;finish_key=True
                elif e.type==pg.KEYDOWN and state=='review':
                    feedback={pg.K_a:'accepted',pg.K_r:'user_retry',pg.K_f:'tracking_failure',pg.K_n:'skipped'}.get(e.key)
                    if feedback:
                        attempt.feedback(feedback)
                        if args.mode=='words':export_word(attempt,root)
                        if feedback in ('accepted','skipped'):index+=1
                        state='complete' if index==len(plans) else 'ready'
                    elif e.key==pg.K_p:
                        from .guided import replay
                        replay(attempt.path)
            if not running:break
            ok,frame=cap.read()
            if not ok:raise RuntimeError('Camera stopped; partial attempt preserved')
            frame=cv.flip(frame,1);hands,t=tracker(frame);dets=detector(frame)
            plan=plans[min(index,len(plans)-1)];side=plan['side']
            by={s:max([x for x in hands if x.side==s],key=lambda x:x.score,default=None) for s in contact}
            hand=by[side];ratios={};touches={};scores={}
            for s,c in contact.items():
                obj=by[s]
                if obj is None:c.reset_state();touches[s]=False;ratios[s]=None;scores[s]=None
                else:touches[s],ratios[s]=c.update(obj.px);scores[s]=assign(dets,[obj])[0]
            cur=smooth(pinch_point(hand.px),t/1000) if hand is not None else None
            if hand is None:smooth.reset()
            event=None;prompt='';expected=None;phase_elapsed=0
            if state in ('recording','tail'):
                if start_t is None:start_t=t
                elapsed=(t-start_t)/1000
                if args.mode=='calibration':prompt,expected,phase_elapsed=calibration_phase(elapsed)
                keep=args.mode=='words' and held and cur is not None and state=='recording'
                if held and hand is None:missing+=1
                if keep:
                    if not trace:event='start'
                    trace.append([*ui.to_kb(cur).tolist(),t])
                if finish_key:
                    attempt.segment(trace,'keyboard_release');event='end';state='tail';tail_t=t
                attempt.frame(frame,dict(t_ms=t,hands=[dict(side=x.side,score=float(x.score),px=x.px.tolist(),world=x.world.tolist()) for x in hands],
                    cursor=cur.tolist() if cur is not None else None,pinched=touches[side],ratio=ratios[side],
                    contact=touches[side],detector_scores=scores[side],ratios=ratios,all_detector_scores=scores,
                    key_held=held,active=held,kept=bool(keep),event=event,expected_pinch=expected,
                    phase_elapsed=phase_elapsed,prompt=prompt,side=side))
                if (args.mode=='calibration' and elapsed>=6) or (state=='tail' and t-tail_t>=700):
                    attempt.data['missing_while_held']=missing
                    if args.mode=='calibration':attempt.close();attempt.data['status']='pending_review';write_json(attempt.path/'attempt.json',attempt.data)
                    else:attempt.finish(decoder,ui.kb)
                    state='review'
            img=frame.copy();ui.draw(img)
            from .draw import draw_landmarks,draw_trace
            for obj in hands:draw_landmarks(img,obj.px)
            draw_trace(img,[ui.to_img(p[:2]) for p in trace])
            if cur is not None:cv.circle(img,tuple(cur.astype(int)),7,(0,255,0) if held else (0,180,255),-1)
            if args.mode=='calibration':
                x=int(w*dict(left=.22,middle=.50,right=.78)[plan['position']]);cv.circle(img,(x,int(h*.62)),40,(0,220,255),2)
            label=(f"{side} hand | Target: {plan['position']}" if args.mode=='calibration' else plan['target'].upper())
            title=f"{min(index+1,len(plans))}/{len(plans)} {label} [{state}]"
            if state=='ready':
                lines=[title,'S: start | Q: save and quit', 'Keep the selected hand near the circle.' if args.mode=='calibration' else 'Position on first key. Hold SPACE while tracing; release at last key.']
            elif state=='recording':
                lines=[title,prompt if args.mode=='calibration' else ('DRAW NOW - SPACE HELD' if held else 'Hold SPACE with your left hand; trace with your right.'),
                       'Keep the other hand down. Slowly vary hand angle.' if args.mode=='calibration' else 'Keep thumb/index touching throughout the word.']
            elif state=='tail':lines=[title,'Trace stopped. Saving release video...']
            elif state=='review':
                preds=' | '.join('/'.join(s['candidates']) for s in attempt.data['segments'])
                lines=[title,preds[:75] if preds else 'Did you follow the prompts correctly?', 'A accept | R retry mistake | F tracking failed | N skip | P replay']
            else:lines=['COMPLETE - recordings saved. Q to exit.']
            if status:lines.append(status[:75])
            cv.rectangle(img,(0,0),(w,130),(20,20,20),-1)
            for j,line in enumerate(lines):cv.putText(img,line,(8,25+j*28),cv.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv.LINE_AA)
            surface=pg.image.frombuffer(cv.cvtColor(img,cv.COLOR_BGR2RGB).tobytes(),(w,h),'RGB')
            screen.fill((0,0,0));sw,sh=screen.get_size();scale=min(sw/w,sh/h);size=(int(w*scale),int(h*scale))
            screen.blit(pg.transform.scale(surface,size),((sw-size[0])//2,(sh-size[1])//2));pg.display.flip()
    finally:
        if attempt is not None and state in ('recording','tail'):
            attempt.close();attempt.feedback('aborted')
        cap.release();pg.quit();print('Saved:',root,flush=True)


if __name__=='__main__':main()
