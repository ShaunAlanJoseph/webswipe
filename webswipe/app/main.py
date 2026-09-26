# WebSwipe live app: typing mode and recording mode
# loop / mode structure follows kinivi/hand-gesture-recognition-mediapipe app.py (Apache-2.0)
import argparse
import time

import cv2 as cv
import numpy as np

from .cvfpscalc import CvFpsCalc
from .draw import draw_landmarks, draw_trace, text
from .keyboard_ui import KeyboardOverlay
from .pinch import OneEuro, PinchDetector, pinch_point
from .recorder import Recorder
from .tracking import HandTracker


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["type", "record"], default="type")
    ap.add_argument("--participant", default="p0")
    ap.add_argument("--phrases", default="data/phrases.txt")
    ap.add_argument("--ckpt", help="decoder checkpoint (type mode)")
    ap.add_argument("--lexicon", default="data/shen_features.npz")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--hand", choices=["Right", "Left"], default="Right", help="dominant hand")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    return ap.parse_args()


def pick(hands, side):
    hs = [h for h in hands if h.side == side]
    return max(hs, key=lambda h: h.score) if hs else None


class Session:
    """Target words for recording mode."""
    def __init__(self, path):
        self.phrases = [l.split() for l in open(path) if l.strip()]
        self.p, self.w = 0, 0

    @property
    def phrase(self):
        return self.phrases[self.p % len(self.phrases)]

    @property
    def word(self):
        return self.phrase[self.w]

    def next(self):
        self.w += 1
        if self.w >= len(self.phrase):
            self.p, self.w = self.p + 1, 0


def main():
    a = get_args()
    cap = cv.VideoCapture(a.camera)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, a.height)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("cannot read from camera")
    H, W = frame.shape[:2]

    tracker = HandTracker()
    ui = KeyboardOverlay(W, H)
    pinch, other_pinch = PinchDetector(), PinchDetector()
    smooth = OneEuro()
    fps = CvFpsCalc(buffer_len=10)
    other = "Left" if a.hand == "Right" else "Right"

    decoder = None
    if a.mode == "type" and a.ckpt:
        from webswipe.decoder.live import LiveDecoder, load_lexicon
        decoder = LiveDecoder(a.ckpt, load_lexicon(a.lexicon))
    rec = Recorder(a.participant) if a.mode == "record" else None
    sess = Session(a.phrases) if a.mode == "record" else None

    typed, cands, last_ms = [], [], 0.0
    trace_img, trace_kb, was_on, other_was_on = [], [], False, False
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv.flip(frame, 1)
        hands, t_ms = tracker(frame)
        if rec:
            rec.frame(t_ms, hands)

        dom = pick(hands, a.hand)
        on, ratio = False, None
        if dom is not None:
            on, ratio = pinch.update(dom.px)
            cur = smooth(pinch_point(dom.px))
        else:
            pinch.reset_state(); smooth.reset()

        if on and not was_on:
            trace_img, trace_kb = [], []
            if rec:
                rec.start()
        if on:
            trace_img.append(cur)
            trace_kb.append([*ui.to_kb(cur).tolist(), t_ms])
        if was_on and not on:
            if len(trace_kb) >= 4:
                if rec:
                    rec.save(sess.word, trace_kb,
                             {"pitch": ui.pitch, "x0": ui.x0, "y0": ui.y0, "frame": [W, H]},
                             {"phrase": " ".join(sess.phrase), "word_index": sess.w, "hand": a.hand})
                    sess.next()
                elif decoder:
                    cands, last_ms = decoder(trace_kb, ui.kb)
                    if cands:
                        typed.append(cands[0])
            elif rec:
                rec.cancel()
        was_on = on

        # non-dominant hand pinch: delete last word (type) / skip word (record)
        oh = pick(hands, other)
        o_on = other_pinch.update(oh.px)[0] if oh is not None else False
        if o_on and not other_was_on:
            if rec:
                sess.next()
            elif typed:
                typed.pop()
        other_was_on = o_on

        # draw
        img = ui.draw(frame, highlight=ui.key_at(cur) if dom is not None else None)
        for h in hands:
            draw_landmarks(img, h.px)
        draw_trace(img, trace_img if on else [])
        if dom is not None:
            cv.circle(img, tuple(int(v) for v in cur), 10, (0, 255, 0) if on else (0, 0, 255), -1)
        text(img, f"FPS {fps.get():.0f}  pinch {'ON' if on else 'off'}"
                  + (f"  r={ratio:.2f} thr={pinch.thr:.2f}" if ratio is not None else ""), (10, 25))
        if rec:
            text(img, f"[{a.participant}] saved {rec.count}", (10, 55))
            text(img, " ".join(sess.phrase), (10, 90), 0.7)
            text(img, f"swipe: {sess.word}", (10, 125), 1.0, (0, 255, 255), 2)
        else:
            text(img, " ".join(typed[-8:]) + "_", (10, 60), 0.8)
            if cands:
                text(img, "  ".join(cands) + f"   ({last_ms:.0f} ms)", (10, 95), 0.6, (0, 255, 255))

        if not a.headless:
            cv.imshow("WebSwipe", img)
            key = cv.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("b") and typed:
                typed.pop()
            if key == ord("n") and sess:
                sess.next()
        n += 1
        if a.max_frames and n >= a.max_frames:
            break
    cap.release()
    cv.destroyAllWindows()
    if a.headless:
        print(f"ran {n} frames, last fps {fps.get():.1f}")


if __name__ == "__main__":
    main()
