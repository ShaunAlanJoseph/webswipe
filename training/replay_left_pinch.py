"""Cache hand tracking and detector scores for a recorded session.

Runs MediaPipe hand tracking (VIDEO mode, 2 hands) and the hand-crop gesture
detector over a recording's camera.avi, using the frame timestamps from
frames.jsonl, at native rate and at two 10 fps phases (0 ms and 50 ms offset).
Each pass writes <out>/<pass>.jsonl, one row per processed frame:
  {frame_index, t_ms, hands: [{px, world, side, score}],
   models: {crop: {scores: per-hand class scores, dets: raw detections}}}
"side" is the user's hand (the recording is mirrored). score_replay.py replays
the left-hand controls on these rows without loading any model.

Tested with mediapipe==1.0.1, onnxruntime==1.30.0, numpy==2.5.3,
opencv-contrib-python==5.0.0.93 (CPU only).

Example:
    python training/replay_left_pinch.py --recording data/left_quick_pinches \
        --out runs/left_pinch_replay
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PASSES = {'native': (None, 0), '10fps_phase0': (10, 0), '10fps_phase50': (10, 50)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--recording', type=Path, default=REPO / 'data/left_quick_pinches',
                   help='folder with camera.avi and frames.jsonl')
    p.add_argument('--detector', type=Path, default=REPO / 'models/gesture_crop_20261010.onnx')
    p.add_argument('--hand-landmarker', type=Path, default=REPO / 'models/hand_landmarker.task')
    p.add_argument('--out', type=Path, default=REPO / 'runs/left_pinch_replay')
    p.add_argument('--passes', nargs='+', default=list(PASSES), choices=list(PASSES))
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TF_CPP_MIN_LOG_LEVEL='3')
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()

    def progress(**kw):
        kw['elapsed_s'] = round(time.time() - start, 1)
        print(json.dumps(kw), flush=True)

    try:
        import cv2 as cv, numpy as np, mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        from webswipe.app.gesture_det import GestureDetector, assign
        from webswipe.app.hand_crop import detect_hands
        root = args.recording
        detector = GestureDetector(str(args.detector), threads=2)
        rows = [json.loads(l) for l in (root / 'frames.jsonl').read_text().splitlines()]
        counts = {}
        for name in args.passes:
            hz, offset = PASSES[name]
            opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(args.hand_landmarker)),
                                                running_mode=vision.RunningMode.VIDEO, num_hands=2,
                                                min_hand_detection_confidence=.5, min_tracking_confidence=.5)
            lm = vision.HandLandmarker.create_from_options(opts)
            cap = cv.VideoCapture(str(root / 'camera.avi'))
            count = 0
            last_bucket = -1
            previous_t = -1
            with (out / (name + '.jsonl')).open('w') as stream:
                for row in rows:
                    ok, frame = cap.read()
                    assert ok
                    t = row['t_ms']
                    if hz:
                        bucket = int((t - offset) // (1000 / hz))
                        if t < offset or bucket <= last_bucket:
                            continue
                        last_bucket = bucket
                    timestamp = max(t, previous_t + 1)
                    previous_t = timestamp
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                    result = lm.detect_for_video(image, timestamp)
                    hands = []
                    for lms, world, side in zip(result.hand_landmarks, result.hand_world_landmarks, result.handedness):
                        hands.append(SimpleNamespace(
                            px=np.array([[p.x * frame.shape[1], p.y * frame.shape[0]] for p in lms]),
                            world=np.array([[p.x, p.y, p.z] for p in world]),
                            side={'Left': 'Right', 'Right': 'Left'}.get(side[0].category_name, side[0].category_name),
                            score=side[0].score))
                    dets = detect_hands(detector, frame, hands)
                    stream.write(json.dumps({
                        'frame_index': row['frame_index'], 't_ms': t,
                        'hands': [{'px': h.px.tolist(), 'world': h.world.tolist(), 'side': h.side, 'score': h.score}
                                  for h in hands],
                        'models': {'crop': {'scores': assign(dets, hands),
                                            'dets': [(c, s, list(map(float, b))) for c, s, b in dets]}}}) + '\n')
                    count += 1
                    if count % 100 == 0:
                        stream.flush()
                        progress(stage='inference', pass_name=name, frames=count, t_ms=t)
            lm.close()
            cap.release()
            counts[name] = count
            progress(stage='pass_complete', pass_name=name, frames=count)
        (out / 'result.json').write_text(json.dumps({
            'finished': True, 'counts': counts, 'recording': root.name, 'detector': args.detector.name,
            'scope': 'Fresh tracking at each cadence; the detector supplies all left-hand scores. '
                     'No word decoding or live camera timing.'}, indent=2))
        progress(stage='complete', counts=counts)
    except Exception as e:
        (out / 'failure.json').write_text(json.dumps({'error': str(e), 'traceback': traceback.format_exc()}))
        raise


if __name__ == '__main__':
    main()
