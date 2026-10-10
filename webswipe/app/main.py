# WebSwipe live app: typing mode and recording mode
# loop / mode structure follows kinivi/hand-gesture-recognition-mediapipe app.py (Apache-2.0)
import argparse
import glob
import json
import time

import cv2 as cv
import numpy as np

from .controls import Candidates, OtherHandControl, SwipeSegmenter, SwipeStartBuffer, GuardedStartBuffer
from .cvfpscalc import CvFpsCalc
from .draw import draw_landmarks, draw_trace, text
from .keyboard_ui import KeyboardOverlay
from .pinch import ContactPinch, DetectorRelease, OneEuro, PinchDetector, pinch_point
from .recorder import Recorder
from .tap import EventLog, TapTyper
from .tracking import HandTracker


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["type", "tap", "record", "guided"], default="type",
                    help="type = swipe typing, tap = per-key baseline, record = collect the webcam dataset")
    ap.add_argument("--words", help="guided mode: optional text file of target words")
    ap.add_argument("--guided-out", default="data/guided", help="guided video/telemetry output directory")
    ap.add_argument("--autocorrect", action="store_true", help="tap mode: snap words to the lexicon")
    ap.add_argument("--log", action="store_true", help="type/tap: log events to data/sessions for WPM/CER")
    ap.add_argument("--participant", default="p0")
    ap.add_argument("--phrases", default="data/phrases.txt")
    ap.add_argument("--start-phrase", type=int, default=-1,
                    help="record mode: phrase to start at (default: continue after this participant's last one)")
    ap.add_argument("--ckpt", help="neural decoder checkpoint (type mode)")
    ap.add_argument("--decoder", choices=["neural", "shark2"],
                    help="default: use the installed neural checkpoint when available, else SHARK2")
    ap.add_argument("--lexicon", default=None, help="default: 10k vocabulary when installed, else D1 vocabulary")
    ap.add_argument("--prior-weight", type=float, default=None, help="wordfreq unigram prior weight (0 = off)")
    ap.add_argument("--resample-hz", type=float, default=None, help="resample traces to this rate before decoding")
    ap.add_argument("--endpoint-weight", type=float, default=0.0,
                    help="neural decoder: optional soft start/end key penalty (default off)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fresh-camera", action="store_true", help="keep only the newest camera frame during inference")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--hand", choices=["Right", "Left"], default="Right", help="dominant hand")
    ap.add_argument("--pinch", choices=["contact", "contact+detector", "detector", "rule"], default="contact",
                    help="contact = fixed thumb-touches-index rule on MediaPipe landmarks; detector = fine-tuned "
                         "HaGRID YOLOv10n; contact+detector = both must agree; rule = old self-calibrating threshold")
    ap.add_argument("--contact-on", type=float, default=0.2, help="contact: pinch below this tip gap / palm length")
    ap.add_argument("--contact-off", type=float, default=0.27, help="contact: release above this")
    ap.add_argument("--gesture-model", default="models/gesture_320.onnx")
    ap.add_argument("--gesture-input", choices=["full", "hand"], default="full",
                    help="hand uses landmark crops; requires a checkpoint trained on those crops")
    ap.add_argument("--det-on", type=float, default=0.5, help="detector: pinch turns on at this score")
    ap.add_argument("--det-off", type=float, default=0.3, help="detector: pinch turns off below this score")
    ap.add_argument("--det-start", type=float, default=0.3,
                    help="contact+detector: detector pinch score needed (with the tips touching) to start a pinch")
    ap.add_argument("--det-votes", type=int, default=2, help="detector frames out of the last 3 needed to start")
    ap.add_argument("--start-lookback-ms", type=float, default=700,
                    help="recover up to this many milliseconds of pinch-like movement before approval (0 disables)")
    ap.add_argument("--guarded-start", action="store_true",
                    help="experimental contact+detector start recovery; validate before deployment")
    ap.add_argument("--decoder-threads", type=int, default=2, help="CPU threads for neural word decoding")
    ap.add_argument("--release-ms", type=float, default=80,
                    help="sustained release needed to finish a swipe (milliseconds)")
    ap.add_argument("--fist-delete", action="store_true",
                    help="enable the legacy extra fist-to-delete shortcut (off by default)")
    ap.add_argument("--other-pinch", choices=["contact", "same"], default="contact",
                    help="other-hand controls: contact rule, or same detector mode as typing hand")
    ap.add_argument("--hide-cursor", action="store_true", help="hide cursor and moving key highlight; C toggles live")
    ap.add_argument("--hide-trail", action="store_true", help="hide live swipe trail; T toggles live")
    ap.add_argument("--hide-skeleton", action="store_true", help="hide hand skeleton; H toggles live")
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
    if a.guarded_start and a.pinch != "contact+detector":
        raise SystemExit("--guarded-start requires --pinch contact+detector")
    from .runtime import configure_decoder
    a = configure_decoder(a)
    if a.start_lookback_ms < 0 or a.decoder_threads < 1:
        raise SystemExit("Lookback must be nonnegative and decoder threads must be positive")
    from .controls import ReleaseDelay
    cap = cv.VideoCapture(a.camera)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, a.height)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("cannot read from camera")
    H, W = frame.shape[:2]
    if a.fresh_camera:
        from .camera import LatestFrameCamera
        cap = LatestFrameCamera(cap)

    if not a.headless:
        cv.namedWindow("WebSwipe", cv.WINDOW_NORMAL)
        cv.resizeWindow("WebSwipe", 1280, 960)

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
    detector_release = DetectorRelease()
    fist_since, det_label = None, ""
    smooth = OneEuro()
    fps = CvFpsCalc(buffer_len=10)
    other = "Left" if a.hand == "Right" else "Right"

    decoder = None
    if a.mode in ("type", "guided"):
        from webswipe.decoder.live import LiveDecoder, LiveShark2, load_lexicon
        kind = a.decoder or ("neural" if a.ckpt else "shark2")
        if kind == "neural":
            if not a.ckpt:
                raise SystemExit("--decoder neural needs --ckpt")
            import torch
            torch.set_num_threads(a.decoder_threads)
            decoder = LiveDecoder(a.ckpt, load_lexicon(a.lexicon), a.prior_weight, a.resample_hz, a.endpoint_weight)
        else:
            # SHARK2 uses its own tuned defaults unless overridden
            decoder = LiveShark2(load_lexicon(a.lexicon), a.prior_weight, a.resample_hz)
        print(f"decoder: {kind}" + (f" ({a.ckpt})" if kind == "neural" else ""))
    tapper = TapTyper(ui.kb, load_lexicon_any(a.lexicon) if a.autocorrect else None) if a.mode == "tap" else None
    log = EventLog(a.participant, a.mode) if a.log and a.mode != "record" else None
    rec = Recorder(a.participant) if a.mode == "record" else None
    sess = None
    if a.mode == "record":
        start = a.start_phrase if a.start_phrase >= 0 else resume_phrase(a.participant)
        sess = Session(a.phrases, start)
        print(f"recording {a.participant} from phrase {start}")

    guided = None
    if a.mode == "guided":
        if a.headless:
            raise SystemExit("Guided recording needs an interactive window")
        from .guided import Guided
        guided = Guided(a, ui, decoder)
        guided.keyboard['frame'] = [W, H]
        cv.setMouseCallback("WebSwipe", guided.mouse)

    words, last_ms = Candidates(), 0.0
    seg, other_ctl = SwipeSegmenter(release_ms=a.release_ms), OtherHandControl()
    other_release = ReleaseDelay(a.release_ms, start_votes=2)
    trace_img, trace_kb = [], []
    start_buffer = (GuardedStartBuffer(a.start_lookback_ms) if a.guarded_start
                    else SwipeStartBuffer(a.start_lookback_ms))
    recovered_start_ms = 0
    show_cursor, show_trail, show_skeleton = not a.hide_cursor, not a.hide_trail, not a.hide_skeleton
    n = 0
    try:
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
            on, ratio, touch, cur = False, None, False, None
            dom_sc, oh_sc = None, None
            if det is not None:
                if a.gesture_input == "hand":
                    from .hand_crop import detect_hands
                    dets = detect_hands(det, frame, hands)
                else:
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
                    proposed = touch and (dom_state.on or voted)
                    if proposed:
                        on = detector_release.update(touch, ratio, dom_sc, t_ms)
                    else:
                        detector_release.update(touch, ratio, dom_sc, t_ms)  # reset after true separation
                        on = False
                    dom_state.on = on
                else:
                    on = touch
                cur = smooth(pinch_point(dom.px), t_ms / 1000)
            else:
                pinch.reset_state(); smooth.reset(); detector_release.reset()
                if det is not None:
                    dom_state.reset_state()

            can_record = guided is None or guided.state == "recording"
            start_buffer.update(cur if can_record and a.mode != "tap" else None, t_ms,
                                touch, ratio, dom_sc, active=seg.active)
            if a.guarded_start and not seg.active and not start_buffer.ready:
                on = False
                dom_state.on = False
            event, keep = seg.update(dom is not None, on, t_ms) if guided is None or guided.state == "recording" else (None, False)
            if event == "start":
                prefix = start_buffer.take_before(t_ms)
                recovered_start_ms = t_ms - prefix[0][2] if prefix else 0
                trace_img = [np.asarray(p[:2]) for p in prefix]
                trace_kb = [[*ui.to_kb(p[:2]).tolist(), p[2]] for p in prefix]
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
                start_buffer.clear()
                if guided is not None:
                    guided.attempt.segment(trace_kb)
                elif len(trace_kb) >= 4:
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
                if a.other_pinch == "contact":
                    o_on = o_touch
                elif a.pinch == "detector":
                    o_on = other_state.update(oh_sc)[0] if oh is not None else False
                else:
                    o_voted = other_state.started(oh_sc, a.det_start) if oh is not None else False
                    o_on = o_touch and (other_state.on or o_voted)
                    other_state.on = o_on
                # Optional legacy shortcut; disabled normally because a pinching hand can be classified as a fist.
                fisted = other_fist.update(oh_sc)[0] if a.fist_delete and oh is not None else False
                if fisted and fist_since is None:
                    fist_since = t_ms
                elif not fisted:
                    fist_since = None
            else:
                o_on = other_pinch.update(oh.px)[0] if oh is not None else False
            if oh is None:
                other_pinch.reset_state()
                if det is not None:
                    other_state.reset_state()
                    other_fist.reset_state()
            o_on = other_release.update(o_on, t_ms)
            action = other_ctl.update(o_on, t_ms)
            if a.fist_delete and det is not None and fist_since is not None and t_ms - fist_since >= 400 and not action:
                action, fist_since = "delete", float("inf")
            if guided is not None:
                action = None
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

            if guided is not None and guided.state == "recording":
                guided.attempt.frame(frame, dict(
                    t_ms=t_ms, captured_unix=time.time(), hands=[dict(side=h.side, score=float(h.score),
                        px=h.px.tolist(), world=h.world.tolist()) for h in hands],
                    cursor=cur.tolist() if cur is not None else None,
                    cursor_kb=ui.to_kb(cur).tolist() if cur is not None else None,
                    pinched=bool(on), contact=bool(touch), ratio=float(ratio) if ratio is not None else None,
                    detector_scores=dom_sc, other_detector_scores=oh_sc, other_pinched=bool(o_on),
                    event=event, active=bool(seg.active), kept=bool(keep),
                    recovered_start_ms=recovered_start_ms if event == "start" else 0,
                    start_pending=bool(start_buffer.points),
                    start_ready=bool(getattr(start_buffer, "ready", False)),
                    release_override=detector_release.blocked if a.pinch == "contact+detector" else False,
                    release_since_ms=seg.release_since, lost_since_ms=seg.lost_since))

            # draw
            img = ui.draw(frame, highlight=ui.key_at(cur) if show_cursor and dom is not None else None)
            if show_skeleton:
                for h in hands:
                    draw_landmarks(img, h.px)
            if guided is not None and guided.state == "review":
                for recorded in guided.attempt.data['segments']:
                    draw_trace(img, [ui.to_img(p[:2]) for p in recorded['trace_kb']])
            elif show_trail:
                draw_trace(img, trace_img if seg.active else [])
            if show_cursor and dom is not None:
                cv.circle(img, tuple(int(v) for v in cur), 10, (0, 255, 0) if on else (0, 0, 255), -1)
            text(img, f"FPS {fps.get():.0f}  pinch {'ON' if on else 'off'} [{a.pinch}]"
                      + (f"  r={ratio:.2f} thr={pinch.thr:.2f}" if ratio is not None else "")
                      + (f"  {det_label}" if det is not None else ""), (10, 25))
            text(img, f"Other hand: {'PINCH' if o_on else 'open'} [{a.other_pinch}]", (10, H - 15), 0.5)
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

            if guided is not None:
                guided.draw(img)
            if guided is None or guided.state == "recording":
                cue = ("RELEASE CONFIRMED" if a.pinch == "contact+detector" and detector_release.blocked else
                       "PINCH REGISTERED - MOVE NOW" if on else
                       "KEEP PINCHED - brief tracking gap" if seg.active else
                       "HOLD STILL - checking pinch" if touch else
                       "POSITION ON FIRST KEY, THEN PINCH")
                color = (50, 220, 50) if on else (0, 210, 255)
                cv.rectangle(img, (0, 227), (W, 255), (20, 20, 20), -1)
                text(img, cue, (10, 247), 0.55, color)
            text(img, f"C cursor {'on' if show_cursor else 'off'} | T trail {'on' if show_trail else 'off'} | H skeleton {'on' if show_skeleton else 'off'}", (10, H - 38), 0.45)
            if not a.headless:
                cv.imshow("WebSwipe", img)
                key = cv.waitKey(1) & 0xFF
                if guided is not None:
                    key = guided.pending if guided.pending is not None else key
                    guided.pending = None
                if key in (27, ord("q")) or cv.getWindowProperty("WebSwipe", cv.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("c"):
                    show_cursor = not show_cursor
                elif key == ord("t"):
                    show_trail = not show_trail
                elif key == ord("h"):
                    show_skeleton = not show_skeleton
                if guided is not None:
                    if key == 32 and guided.state == "ready":
                        guided.start()
                        seg = SwipeSegmenter(release_ms=a.release_ms)
                        pinch.reset_state()
                        detector_release.reset()
                        smooth.reset()
                        if det is not None:
                            dom_state.reset_state()
                        trace_img, trace_kb = [], []
                        start_buffer.clear()
                    elif key == 32 and guided.state == "recording":
                        guided.finish(trace_kb, seg.active)
                    elif guided.state == "review":
                        status = {ord('a'): 'accepted', ord('r'): 'user_retry',
                                  ord('f'): 'tracking_failure', ord('n'): 'skipped'}.get(key)
                        if status:
                            guided.feedback(status)
                        elif key == ord('p'):
                            from .guided import replay
                            replay(guided.attempt.path)
                    n += 1
                    if a.max_frames and n >= a.max_frames:
                        break
                    continue
                if key == ord("b"):
                    tapper.backspace() if tapper else words.delete()
                if ord("1") <= key <= ord("4"):
                    words.choose(key - ord("1"))
                if key == ord("n") and sess:
                    sess.next()
            n += 1
            if a.max_frames and n >= a.max_frames:
                break
    finally:
        if guided is not None:
            guided.close()
        cap.release()
        cv.destroyAllWindows()
    if log:
        log.close()
        print(f"events -> {log.path}")
    if a.headless:
        print(f"ran {n} frames, last fps {fps.get():.1f}")


if __name__ == "__main__":
    main()
