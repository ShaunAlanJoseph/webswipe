# WebSwipe live app: typing mode and recording mode
# loop / mode structure follows kinivi/hand-gesture-recognition-mediapipe app.py (Apache-2.0)
import argparse
import glob
import json
import time

import cv2 as cv
import numpy as np

from .controls import Candidates, OtherHandControl, SwipeSegmenter
from .cvfpscalc import CvFpsCalc
from .draw import draw_landmarks, draw_trace, text
from .keyboard_ui import KeyboardOverlay
from .pinch import ContactPinch, OneEuro, PinchDetector, pinch_point
from .recorder import Recorder
from .tap import EventLog, TapTyper
from .tracking import HandTracker


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["type", "tap", "record"], default="type",
                    help="type = swipe typing, tap = per-key baseline, record = collect the webcam dataset")
    ap.add_argument("--autocorrect", action="store_true", help="tap mode: snap words to the lexicon")
    ap.add_argument("--log", action="store_true", help="type/tap: log events to data/sessions for WPM/CER")
    ap.add_argument("--participant", default="p0")
    ap.add_argument("--phrases", default="data/phrases.txt")
    ap.add_argument("--start-phrase", type=int, default=-1,
                    help="record mode: phrase to start at (default: continue after this participant's last one)")
    ap.add_argument("--ckpt", help="neural decoder checkpoint (type mode)")
    ap.add_argument("--decoder", choices=["neural", "shark2"],
                    help="default: neural if --ckpt is given, else SHARK2 (works without a trained model)")
    ap.add_argument("--lexicon", default="data/shen_features.npz")
    ap.add_argument("--prior-weight", type=float, default=0.0, help="wordfreq unigram prior weight (0 = off)")
    ap.add_argument("--resample-hz", type=float, default=0.0, help="resample traces to this rate before decoding")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--hand", choices=["Right", "Left"], default="Right", help="dominant hand")
    ap.add_argument("--pinch", choices=["contact", "contact+detector", "detector", "rule"], default="contact",
                    help="contact = fixed thumb-touches-index rule on MediaPipe landmarks; detector = fine-tuned "
                         "HaGRID YOLOv10n; contact+detector = both must agree; rule = old self-calibrating threshold")
    ap.add_argument("--contact-on", type=float, default=0.2, help="contact: pinch below this tip gap / palm length")
    ap.add_argument("--contact-off", type=float, default=0.3, help="contact: release above this")
    ap.add_argument("--gesture-model", default="models/gesture_320.onnx")
    ap.add_argument("--det-on", type=float, default=0.5, help="detector: pinch turns on at this score")
    ap.add_argument("--det-off", type=float, default=0.3, help="detector: pinch turns off below this score")
    ap.add_argument("--det-start", type=float, default=0.3,
                    help="contact+detector: detector pinch score needed (with the tips touching) to start a pinch")
    ap.add_argument("--det-votes", type=int, default=2, help="detector frames out of the last 3 needed to start")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    return ap.parse_args()


def load_lexicon_any(path):
    from webswipe.decoder.live import load_lexicon
    return load_lexicon(path)


def pick(hands, side):
    hs = [h for h in hands if h.side == side]
    return max(hs, key=lambda h: h.score) if hs else None


class Session:
    """Target words for recording mode."""
    def __init__(self, path, start=0):
        self.phrases = [l.split() for l in open(path) if l.strip()]
        self.p, self.w = start, 0

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


def resume_phrase(participant, out_dir="data/recordings"):
    """Next phrase index after the last one this participant recorded, so restarts don't repeat phrases."""
    last = -1
    for path in glob.glob(f"{out_dir}/{participant}_*.jsonl"):
        for line in open(path):
            if line.strip():
                last = max(last, json.loads(line).get("phrase_index", -1))
    return last + 1


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
    if a.pinch == "rule":
        pinch, other_pinch = PinchDetector(), PinchDetector()
    else:
        pinch, other_pinch = (ContactPinch(a.contact_on, a.contact_off), ContactPinch(a.contact_on, a.contact_off))
    det = None
    if "detector" in a.pinch:
        from .gesture_det import GestureDetector, StateFilter, assign
        det = GestureDetector(a.gesture_model)
        dom_state = StateFilter("ok", a.det_on, a.det_off, a.det_votes)
        other_state = StateFilter("ok", a.det_on, a.det_off, a.det_votes)
        other_fist = StateFilter("fist", a.det_on, a.det_off, a.det_votes)
    print(f"pinch: {a.pinch}" + (f" ({a.gesture_model})" if det else ""))
    fist_since, det_label = None, ""
    smooth = OneEuro()
    fps = CvFpsCalc(buffer_len=10)
    other = "Left" if a.hand == "Right" else "Right"

    decoder = None
    if a.mode == "type":
        from webswipe.decoder.live import LiveDecoder, LiveShark2, load_lexicon
        kind = a.decoder or ("neural" if a.ckpt else "shark2")
        if kind == "neural":
            if not a.ckpt:
                raise SystemExit("--decoder neural needs --ckpt")
            decoder = LiveDecoder(a.ckpt, load_lexicon(a.lexicon), a.prior_weight, a.resample_hz)
        else:
            # SHARK2 uses its own tuned defaults unless overridden
            decoder = LiveShark2(load_lexicon(a.lexicon), a.prior_weight or 0.25, a.resample_hz or 50.0)
        print(f"decoder: {kind}")
    tapper = TapTyper(ui.kb, load_lexicon_any(a.lexicon) if a.autocorrect else None) if a.mode == "tap" else None
    log = EventLog(a.participant, a.mode) if a.log and a.mode != "record" else None
    rec = Recorder(a.participant) if a.mode == "record" else None
    sess = None
    if a.mode == "record":
        start = a.start_phrase if a.start_phrase >= 0 else resume_phrase(a.participant)
        sess = Session(a.phrases, start)
        print(f"recording {a.participant} from phrase {start}")

    words, last_ms = Candidates(), 0.0
    seg, other_ctl = SwipeSegmenter(), OtherHandControl()
    trace_img, trace_kb = [], []
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
        oh = pick(hands, other)
        on, ratio = False, None
        if det is not None:
            dets = det(frame)
            dom_sc = assign(dets, [dom])[0] if dom is not None else None
            oh_sc = assign(dets, [oh])[0] if oh is not None else None
            det_label = " ".join(f"{k}:{v:.2f}" for k, v in (dom_sc or {}).items() if v >= 0.25)
        if dom is not None:
            touch, ratio = pinch.update(dom.px)
            if a.pinch == "detector":
                on, _ = dom_state.update(dom_sc)
            elif a.pinch == "contact+detector":
                # fingertips must touch, and the detector must see a pinch to start one (it keeps going while
                # the tips stay together, so a blurred mid-swipe frame cannot cut a word)
                voted = dom_state.started(dom_sc, a.det_start)
                on = touch and (dom_state.on or voted)
                dom_state.on = on
            else:
                on = touch
            cur = smooth(pinch_point(dom.px), t_ms / 1000)
        else:
            pinch.reset_state(); smooth.reset()
            if det is not None:
                dom_state.reset_state()

        event, keep = seg.update(dom is not None, on, t_ms)
        if event == "start":
            trace_img, trace_kb = [], []
            if tapper:
                key = tapper.tap(ui.to_kb(cur))
                if log:
                    log(t_ms, "tap", key=key, pos=ui.to_kb(cur).round(2).tolist())
            if rec:
                rec.start()
        if keep:
            trace_img.append(cur)
            trace_kb.append([*ui.to_kb(cur).tolist(), t_ms])
        if event == "end":
            if len(trace_kb) >= 4:
                if rec:
                    rec.save(sess.word, trace_kb,
                             {"pitch": ui.pitch, "x0": ui.x0, "y0": ui.y0, "frame": [W, H]},
                             {"phrase": " ".join(sess.phrase), "phrase_index": sess.p, "word_index": sess.w,
                              "hand": a.hand})
                    sess.next()
                elif decoder:
                    cands, last_ms = decoder(trace_kb, ui.kb)
                    words.commit(cands)
                    if log:
                        log(t_ms, "word", cands=cands, n_points=len(trace_kb), decode_ms=round(last_ms, 1),
                            start=trace_kb[0][2])
            elif rec:
                rec.cancel()

        # non-dominant hand: short pinch = next candidate (type) / skip word (record); hold = delete last word
        if det is not None:
            o_touch = other_pinch.update(oh.px)[0] if oh is not None else False
            if a.pinch == "detector":
                o_on = other_state.update(oh_sc)[0] if oh is not None else False
            else:
                o_voted = other_state.started(oh_sc, a.det_start) if oh is not None else False
                o_on = o_touch and (other_state.on or o_voted)
                other_state.on = o_on
            # detector only: a fist held 400 ms on the other hand deletes the last word (fires once per fist)
            fisted = other_fist.update(oh_sc)[0] if oh is not None else False
            if fisted and fist_since is None:
                fist_since = t_ms
            elif not fisted:
                fist_since = None
        else:
            o_on = other_pinch.update(oh.px)[0] if oh is not None else False
        action = other_ctl.update(o_on, t_ms)
        if det is not None and fist_since is not None and t_ms - fist_since >= 400 and not action:
            action, fist_since = "delete", float("inf")
        if action and rec:
            sess.next()
        elif action and tapper:
            if action == "next":
                out = tapper.space()
                if log and out:
                    log(t_ms, "space", raw=out[0], word=out[1])
            else:
                tapper.backspace()
                if log:
                    log(t_ms, "backspace")
        elif action == "next":
            words.next()
            if log:
                log(t_ms, "next_candidate", word=words.typed[-1] if words.typed else None)
        elif action == "delete":
            words.delete()
            if log:
                log(t_ms, "delete")

        # draw
        img = ui.draw(frame, highlight=ui.key_at(cur) if dom is not None else None)
        for h in hands:
            draw_landmarks(img, h.px)
        draw_trace(img, trace_img if seg.active else [])
        if dom is not None:
            cv.circle(img, tuple(int(v) for v in cur), 10, (0, 255, 0) if on else (0, 0, 255), -1)
        text(img, f"FPS {fps.get():.0f}  pinch {'ON' if on else 'off'} [{a.pinch}]"
                  + (f"  r={ratio:.2f} thr={pinch.thr:.2f}" if ratio is not None else "")
                  + (f"  {det_label}" if det is not None else ""), (10, 25))
        if rec:
            text(img, f"[{a.participant}] saved {rec.count}", (10, 55))
            text(img, " ".join(sess.phrase), (10, 90), 0.7)
            text(img, f"swipe: {sess.word}", (10, 125), 1.0, (0, 255, 255), 2)
        elif tapper:
            text(img, tapper.text[-60:] + "_", (10, 60), 0.8)
        else:
            text(img, " ".join(words.typed[-8:]) + "_", (10, 60), 0.8)
            if words.cands:
                shown = [f"[{c}]" if i == words.idx else c for i, c in enumerate(words.cands)]
                text(img, "  ".join(f"{i + 1}:{c}" for i, c in enumerate(shown)) + f"   ({last_ms:.0f} ms)",
                     (10, 95), 0.6, (0, 255, 255))

        if not a.headless:
            cv.imshow("WebSwipe", img)
            key = cv.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("b"):
                tapper.backspace() if tapper else words.delete()
            if ord("1") <= key <= ord("4"):
                words.choose(key - ord("1"))
            if key == ord("n") and sess:
                sess.next()
        n += 1
        if a.max_frames and n >= a.max_frames:
            break
    cap.release()
    cv.destroyAllWindows()
    if log:
        log.close()
        print(f"events -> {log.path}")
    if a.headless:
        print(f"ran {n} frames, last fps {fps.get():.1f}")


if __name__ == "__main__":
    main()
