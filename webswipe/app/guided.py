"""Guided diagnostic attempts: clean video, per-frame telemetry, and explicit feedback."""
import argparse
import json
import time
import uuid
from pathlib import Path

import cv2 as cv
import numpy as np

WORDS = ('hello is hello his this good night world word work help home house how '
         'are you we the there their then these those thank thanks please people '
         'small large light right left look love little long low high go to on '
         'in it of for from over under hello').split()


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


class Attempt:
    def __init__(self, root, target, index, participant, keyboard):
        self.path = root / ('attempt_' + uuid.uuid4().hex[:10])
        self.path.mkdir()
        self.data = dict(schema=1, target=target, word_index=index, participant=participant,
                         keyboard=keyboard, status='recording', segments=[], frame_count=0,
                         created=time.time(), video='camera.avi', telemetry='frames.jsonl',
                         mirrored=True, video_fps=30, timing='Use telemetry timestamps; AVI is nominal 30 fps.')
        self.writer = None
        self.log = (self.path / 'frames.jsonl').open('w')
        self.closed = False
        write_json(self.path / 'attempt.json', self.data)

    def frame(self, image, telemetry):
        if self.writer is None:
            h, w = image.shape[:2]
            self.writer = cv.VideoWriter(str(self.path / 'camera.avi'), cv.VideoWriter_fourcc(*'MJPG'), 30, (w, h))
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                raise RuntimeError('Cannot save video (MJPG). Recording stopped; check codec/disk space.')
        self.writer.write(image)
        telemetry = dict(telemetry, frame_index=self.data['frame_count'])
        self.log.write(json.dumps(telemetry) + '\n')
        self.log.flush()
        self.data['frame_count'] += 1

    def segment(self, trace, boundary='release'):
        self.data['segments'].append(dict(trace_kb=[list(p) for p in trace], boundary=boundary,
                                          candidates=[], decode_ms=None))

    def finish(self, decoder, kb):
        self.close()
        self.data['status'] = 'pending_review'
        write_json(self.path / 'attempt.json', self.data)
        for segment in self.data['segments']:
            if len(segment['trace_kb']) >= 4:
                try:
                    cands, ms = decoder(segment['trace_kb'], kb)
                    segment.update(candidates=cands, decode_ms=ms)
                except Exception as exc:
                    segment['decode_error'] = str(exc)
        write_json(self.path / 'attempt.json', self.data)

    def feedback(self, status):
        self.data['status'] = status
        segments = self.data['segments']
        self.data['training_eligible'] = (status == 'accepted' and len(segments) == 1
                                         and len(segments[0]['trace_kb']) >= 4
                                         and segments[0]['boundary'] == 'release')
        write_json(self.path / 'attempt.json', self.data)

    def close(self):
        if self.closed:
            return
        if self.writer is not None:
            self.writer.release()
        self.log.close()
        self.closed = True


def replay(path):
    """Replay synchronized raw video with stored landmarks, cursor and event boundaries."""
    from .draw import draw_landmarks
    from .keyboard_ui import KeyboardOverlay
    path = Path(path)
    meta = json.loads((path / 'attempt.json').read_text())
    frames = [json.loads(line) for line in (path / 'frames.jsonl').read_text().splitlines()]
    if not frames:
        return
    cap = cv.VideoCapture(str(path / 'camera.avi'))
    win = 'Attempt replay - space pause, arrows/A/D step, Q close'
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    cv.resizeWindow(win, 1280, 960)
    i, paused = 0, False
    try:
        while True:
            cap.set(cv.CAP_PROP_POS_FRAMES, i)
            ok, img = cap.read()
            if not ok:
                break
            f = frames[i]
            ui = KeyboardOverlay(img.shape[1], img.shape[0])
            # Restore recorded geometry so replay survives future layout changes.
            ui.pitch = meta['keyboard']['pitch']
            ui.x0, ui.y0 = meta['keyboard']['x0'], meta['keyboard']['y0']
            from webswipe.decoder.features import Keyboard, LETTERS
            ui.kb = Keyboard.from_pitch(ui.pitch, ui.pitch)
            ui.key_px = {c: ui.to_img(ui.kb.centers[k]) for k, c in enumerate(LETTERS)}
            ui.draw(img)
            for h in f['hands']:
                draw_landmarks(img, np.asarray(h['px']))
            for seg in meta['segments']:
                pts = [ui.to_img(p[:2]).astype(int) for p in seg['trace_kb'] if p[2] <= f['t_ms']]
                if len(pts) > 1:
                    cv.polylines(img, [np.asarray(pts)], False, (0, 220, 255), 2)
            if f['cursor'] is not None:
                cv.circle(img, tuple(np.asarray(f['cursor']).astype(int)), 7,
                          (0, 255, 0) if f['pinched'] else (0, 0, 255), -1)
            lines = [f"Target {meta['target']} | frame {i+1}/{len(frames)} | {f['t_ms']} ms",
                     f"pinch {f['pinched']} ratio {f['ratio']} event {f['event']} active {f['active']}",
                     'Space pause | A/D step | Q close']
            cv.rectangle(img, (0, 0), (img.shape[1], 90), (20, 20, 20), -1)
            for y, line in enumerate(lines):
                cv.putText(img, line, (8, 23+y*27), cv.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
            cv.imshow(win, img)
            delay = 30 if paused or i == len(frames)-1 else max(1, min(1000, frames[i+1]['t_ms']-f['t_ms']))
            key = cv.waitKeyEx(delay)
            if key in (27, ord('q')) or cv.getWindowProperty(win, cv.WND_PROP_VISIBLE) < 1:
                break
            if key == 32:
                paused = not paused
            elif key in (ord('a'), 81, 2424832, 65361):
                i, paused = max(0, i-1), True
            elif key in (ord('d'), 83, 2555904, 65363):
                i, paused = min(len(frames)-1, i+1), True
            elif not paused:
                i = min(len(frames)-1, i+1)
                if i == len(frames)-1:
                    paused = True
    finally:
        cap.release()
        cv.destroyWindow(win)


class Guided:
    def __init__(self, args, ui, decoder):
        self.words = Path(args.words).read_text().lower().split() if args.words else WORDS
        if not self.words or any(not w.isascii() or not w.isalpha() for w in self.words):
            raise ValueError('Word list must contain English words, separated by spaces or newlines.')
        self.root = Path(args.guided_out) / (time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
        self.root.mkdir(parents=True)
        self.ui, self.decoder, self.args = ui, decoder, args
        self.index, self.state, self.attempt = 0, 'ready', None
        self.buttons, self.pending = [], None
        self.keyboard = dict(pitch=ui.pitch, x0=ui.x0, y0=ui.y0, frame=[args.width, args.height])
        write_json(self.root / 'session.json', dict(schema=1, participant=args.participant, words=self.words,
                                                   settings=vars(args), keyboard=self.keyboard))
        print(f'Guided recordings -> {self.root}', flush=True)

    def mouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            for (x0,y0,x1,y1), key in self.buttons:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    self.pending = key
                    break

    def start(self):
        self.attempt = Attempt(self.root, self.words[self.index], self.index, self.args.participant, self.keyboard)
        self.state = 'recording'

    def finish(self, trace, active):
        if active:
            self.attempt.segment(trace, 'manual_finish')
        self.attempt.finish(self.decoder, self.ui.kb)
        self.state = 'review'

    def feedback(self, status):
        self.attempt.feedback(status)
        if self.attempt.data['training_eligible']:
            rec = dict(participant=self.args.participant, target=self.words[self.index],
                       trace_kb=self.attempt.data['segments'][0]['trace_kb'], keyboard=self.keyboard,
                       source_attempt=str(self.attempt.path), session=self.root.name)
            with (self.root / 'accepted.jsonl').open('a') as f:
                f.write(json.dumps(rec) + '\n')
        if status in ('accepted', 'skipped'):
            self.index += 1
        self.state = 'complete' if self.index == len(self.words) else 'ready'

    def draw(self, img):
        h, w = img.shape[:2]
        cv.rectangle(img, (0, 32), (w, 225), (20,20,20), -1)
        target = self.words[self.index] if self.index < len(self.words) else 'Complete'
        lines = [f'{self.index+1 if self.index<len(self.words) else self.index}/{len(self.words)}  {target.upper()}  [{self.state}]']
        if self.state == 'review':
            segments = self.attempt.data['segments']
            lines.append(f'{len(segments)} segment(s). Accept a good swipe even if prediction is wrong.')
            lines.append(' | '.join('/'.join(s['candidates']) or '(no prediction)' for s in segments)[:70] or 'No swipe detected')
            options = [('Accept [A]', ord('a')), ('My mistake [R]', ord('r')), ('Tracking failed [F]', ord('f')),
                       ('Replay [P]', ord('p')), ('Skip [N]', ord('n'))]
        elif self.state == 'recording':
            lines += ['Swipe once. Broken pinches do not stop this recording.', 'Release your fingers, then press SPACE or Finish.']
            options = [('Finish [SPACE]', 32)]
        elif self.state == 'ready':
            lines += ['Press Start, then trace the word. Keep your hands visible.', 'Q saves an unfinished attempt and exits.']
            options = [('Start [SPACE]', 32)]
        else:
            lines += ['All words finished. Your recordings are saved.', 'Press Q to close.']
            options = []
        for j,line in enumerate(lines):
            cv.putText(img,line,(8,58+j*28),cv.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1,cv.LINE_AA)
        self.buttons = []
        for j,(label,key) in enumerate(options):
            x,y = 8+(j%3)*205, 145+(j//3)*38
            box = (x,y,min(w-4,x+197),y+30)
            cv.rectangle(img,box[:2],box[2:],(70,70,70),-1)
            cv.putText(img,label,(x+5,y+21),cv.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv.LINE_AA)
            self.buttons.append((box,key))
        return img

    def close(self):
        if self.attempt is not None and self.state == 'recording':
            self.attempt.close()
            self.attempt.feedback('aborted')
        print(f'Saved guided session: {self.root}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--replay', help='path to an attempt directory')
    group.add_argument('--summary', help='path to a guided session directory')
    a = ap.parse_args()
    if a.replay:
        replay(a.replay)
    else:
        from collections import Counter
        root = Path(a.summary)
        attempts = [json.loads(p.read_text()) for p in sorted(root.glob('attempt_*/attempt.json'))]
        print(json.dumps(dict(attempts=len(attempts),
                              feedback=dict(Counter(r['status'] for r in attempts)),
                              multiple_segments=sum(len(r['segments']) > 1 for r in attempts),
                              no_segments=sum(not r['segments'] for r in attempts),
                              training_eligible=sum(r.get('training_eligible', False) for r in attempts)), indent=2))
