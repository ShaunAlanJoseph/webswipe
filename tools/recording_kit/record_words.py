"""Guided word collection with actual Space key-up boundaries; no word predictor."""
import argparse
import json
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from recording_core import Attempt, write_json
from cursor import OneEuro, pinch_point

HERE = Path(__file__).resolve().parent
WORDS = ('hello is his this good night world word work help home house you we the '
         'there their thank please people light right little look hello is his good night world').split()
PLANS = [dict(target=w, practice=True) for w in ['hello', 'world']] + [
    dict(target=w, practice=False) for w in WORDS]


class Keyboard:
    def __init__(self, w, h):
        self.pitch = w * .45 / 10.33
        self.x0, self.y0 = w * .51 + self.pitch / 2, h * .60
        self.meta = dict(pitch=self.pitch, x0=self.x0, y0=self.y0, frame=[w, h])
        self.keys = {c: np.array([self.x0 + (i + row * .5) * self.pitch, self.y0 + row * self.pitch])
                     for row, letters in enumerate(['qwertyuiop', 'asdfghjkl', 'zxcvbnm'])
                     for i, c in enumerate(letters)}

    def to_kb(self, p):
        return [float(p[0] - self.x0), float(self.y0 - p[1])]

    def to_img(self, p):
        return [p[0] + self.x0, self.y0 - p[1]]

    def draw(self, image):
        overlay = image.copy()
        for c, p in self.keys.items():
            half = self.pitch * .45
            cv.rectangle(overlay, tuple((p-half).astype(int)), tuple((p+half).astype(int)), (30,30,30), -1)
        cv.addWeighted(overlay, .45, image, .55, 0, image)
        for c, (x,y) in self.keys.items():
            cv.putText(image, c.upper(), (int(x-6), int(y+6)), 0, .5, (255,255,255), 1, cv.LINE_AA)


def save_feedback(attempt, status, root):
    attempt.feedback(status)
    d = attempt.data
    segs = d['segments']
    eligible = (status == 'accepted' and not d['practice'] and len(segs) == 1
                and segs[0]['boundary'] == 'keyboard_release' and len(segs[0]['trace_kb']) >= 4
                and d.get('missing_while_held', 0) == 0)
    d['training_eligible'] = eligible
    write_json(attempt.path/'attempt.json', d)
    if eligible:
        with (root/'accepted.jsonl').open('a') as f:
            f.write(json.dumps(dict(participant=d['participant'], target=d['target'],
                trace_kb=segs[0]['trace_kb'], keyboard=d['keyboard'],
                source_attempt=f'words/{root.name}/{attempt.path.name}', boundary_source='keyboard',
                session=root.name))+'\n')


def main():
    import pygame as pg
    from tracking import HandTracker
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--participant', required=True)
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--seconds', type=int, default=400)
    args = ap.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'session.json', dict(schema=1, mode='words', participant=args.participant,
        plans=PLANS, budget_seconds=args.seconds, boundary_source='keyboard', decoder=None))
    cap = cv.VideoCapture(args.camera)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    tracker = None
    attempt = None
    state = 'ready'; index = 0; held = False; trace = []; missing = 0
    start_clock = None; tail_t = None; accepted = 0; skipped = 0
    ended = 'quit'; running = True; smooth = OneEuro(); ui = None
    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('Cannot read camera. Close other camera apps or try camera 1.')
        h,w = frame.shape[:2]; ui = Keyboard(w,h)
        tracker = HandTracker(model=str(HERE/'models/hand_landmarker.task'))
        pg.init(); screen = pg.display.set_mode((1280,960), pg.RESIZABLE)
        pg.display.set_caption('WebSwipe words - S start, hold SPACE to trace, A accept, R retry')
        while running:
            finish = False
            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key in (pg.K_q, pg.K_ESCAPE)):
                    running = False
                elif e.type == pg.WINDOWFOCUSLOST and state in ('recording','tail'):
                    attempt.close(); save_feedback(attempt, 'interrupted', root)
                    state = 'ready'; held = False; smooth.reset()
                elif e.type == pg.KEYDOWN and e.key == pg.K_s and state == 'ready':
                    if start_clock is None: start_clock = time.monotonic()
                    attempt = Attempt(root, PLANS[index]['target'], index, args.participant, ui.meta)
                    attempt.data.update(PLANS[index], boundary_source='keyboard')
                    state = 'recording'; trace = []; missing = 0; held = False; smooth.reset()
                elif e.type == pg.KEYDOWN and e.key == pg.K_SPACE and state == 'recording':
                    held = True
                elif e.type == pg.KEYUP and e.key == pg.K_SPACE and state == 'recording' and held:
                    held = False; finish = True
                elif e.type == pg.KEYDOWN and state == 'review':
                    status = {pg.K_a:'accepted', pg.K_r:'user_retry', pg.K_f:'tracking_failure', pg.K_n:'skipped'}.get(e.key)
                    if status:
                        save_feedback(attempt, status, root)
                        if status in ('accepted','skipped'):
                            if not PLANS[index]['practice']:
                                accepted += status == 'accepted'; skipped += status == 'skipped'
                            index += 1
                        if index == len(PLANS): ended = 'list_finished'; running = False
                        state = 'ready'
            if not running: break
            if start_clock is not None and time.monotonic()-start_clock >= args.seconds:
                ended = 'time_budget'; break
            ok, frame = cap.read()
            if not ok: raise RuntimeError('Camera stopped; partial recording will be preserved.')
            frame = cv.flip(frame,1); hands,t = tracker(frame)
            hand = max((x for x in hands if x.side == 'Right'), key=lambda x:x.score, default=None)
            if hand is None: smooth.reset()
            cur = smooth(pinch_point(hand.px), t/1000) if hand is not None else None
            event = None
            if state in ('recording','tail'):
                keep = state == 'recording' and held and cur is not None
                if held and hand is None: missing += 1
                if keep:
                    event = 'start' if not trace else None
                    trace.append([*ui.to_kb(cur), t])
                if finish:
                    attempt.segment(trace, 'keyboard_release'); event = 'end'; state = 'tail'; tail_t = t
                ratio = (float(np.linalg.norm(hand.px[4]-hand.px[8]) / max(np.linalg.norm(hand.px[0]-hand.px[9]),1e-6))
                         if hand is not None else None)
                attempt.frame(frame, dict(t_ms=t, hands=[dict(side=x.side, score=float(x.score),
                    px=x.px.tolist(), world=x.world.tolist()) for x in hands],
                    cursor=cur.tolist() if cur is not None else None, ratio=ratio,
                    pinched=None, contact=None, detector_scores=None,
                    key_held=held, active=held, kept=bool(keep), event=event))
                if state == 'tail' and t-tail_t >= 700:
                    attempt.close(); attempt.data.update(status='pending_review', missing_while_held=missing)
                    write_json(attempt.path/'attempt.json',attempt.data); state = 'review'
            image = frame.copy(); ui.draw(image)
            if len(trace)>1:
                cv.polylines(image,[np.array([ui.to_img(p[:2]) for p in trace],dtype=np.int32)],False,(0,220,255),2)
            if cur is not None: cv.circle(image,tuple(cur.astype(int)),5,(0,255,0) if held else (0,180,255),-1)
            plan = PLANS[index]
            label = f'PRACTICE {index+1}/2' if plan['practice'] else f'WORD {index-1}/{len(WORDS)}'
            remaining = args.seconds if start_clock is None else max(0,int(args.seconds-(time.monotonic()-start_clock)))
            lines = [f"{label}: {plan['target'].upper()} | {remaining//60}:{remaining%60:02d} left"]
            if state == 'ready': lines += ['S: start | Q: save and quit', 'Pinch RIGHT hand; place cursor on the first letter.']
            elif state == 'recording': lines += ['DRAW - SPACE HELD' if held else 'Hold SPACE with LEFT hand to begin the swipe.',
                                               'Keep pinched. Release SPACE on the LAST letter.']
            elif state == 'tail': lines += ['Trace stopped. Open your fingers; saving release video...']
            else: lines += ['A: accept | R: retry | F: tracking failed | N: skip',
                            'Check the path. No predicted word is shown.']
            if hand is None: lines += ['RIGHT hand not detected: bring it fully into view.']
            cv.rectangle(image,(0,0),(w,130),(20,20,20),-1)
            for j,line in enumerate(lines):cv.putText(image,line,(7,23+j*25),0,.44,(255,255,255),1,cv.LINE_AA)
            surface = pg.image.frombuffer(cv.cvtColor(image,cv.COLOR_BGR2RGB).tobytes(),(w,h),'RGB')
            sw,sh = screen.get_size(); scale = min(sw/w,sh/h); size = (int(w*scale),int(h*scale))
            screen.fill((0,0,0)); screen.blit(pg.transform.scale(surface,size),((sw-size[0])//2,(sh-size[1])//2));pg.display.flip()
    finally:
        if attempt is not None and state in ('recording','tail','review'):
            attempt.close()
            # At timeout a completed but unreviewed attempt remains available for manual inspection.
            if state != 'review':save_feedback(attempt,'aborted',root)
        cap.release()
        if tracker is not None:tracker.lm.close()
        pg.quit()
        eligible = sum(1 for _ in (root/'accepted.jsonl').open()) if (root/'accepted.jsonl').exists() else 0
        write_json(root/'summary.json',dict(end_reason=ended, accepted_words=accepted,
            eligible_words=eligible, skipped_words=skipped, planned_words=len(WORDS),
            elapsed_seconds=0 if start_clock is None else round(time.monotonic()-start_clock,1)))
        print(f'Words saved: {accepted} accepted, {eligible} structurally eligible. End: {ended}')


if __name__ == '__main__': main()
