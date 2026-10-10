"""Fine-tune the gesture detector on square hand crops (produces gesture_crop_20261010).

Starts from gesture_candidate_320.pt (train_gesture_continued.py). Training data:
  - reviewed main-camera frames from prepare_adaptation_frames.py, boxed with
    MediaPipe (IMAGE mode, rightmost hand, 12% padding); "ok" rows repeated 3x,
    "no_gesture" rows 10x;
  - every 4th frame of the earlier train sessions and of the HaGRID replay set.
Validation: the adaptation development rows, every 4th earlier development
frame and every 4th HaGRID validation image.
Every frame is cut to a square around each annotated box (1.35x the longer side,
clipped to the frame) so the detector sees fingertip detail at 320 px.

Writes crop_320.pt, crop_320.onnx, training_report.json and result.json to --out.
Needs a CUDA GPU. Tested with ultralytics==8.4.174, mediapipe==1.0.1, onnx,
onnxslim, onnxruntime.

Example:
    python training/train_gesture_crop.py --adaptation-dir data/detector_adaptation \
        --frames-dir data/detector_frames --init models/gesture_candidate_320.pt
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import shutil
import time
import traceback
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--adaptation-dir', type=Path, default=REPO / 'data/detector_adaptation',
                   help='adaptation_labels.json and images.zip (or the extracted images)')
    p.add_argument('--frames-dir', type=Path, default=REPO / 'data/detector_frames',
                   help='manifest.json, frames_*.zip and hagrid_subset.zip')
    p.add_argument('--init', type=Path, default=REPO / 'models/gesture_candidate_320.pt')
    p.add_argument('--hand-landmarker', type=Path, default=REPO / 'models/hand_landmarker.task')
    p.add_argument('--helpers', type=Path, default=REPO / 'tools/kaggle/vision_runner.py')
    p.add_argument('--out', type=Path, default=REPO / 'runs/detector_crop')
    p.add_argument('--workdir', type=Path, default=REPO / 'runs/work')
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--time-budget-s', type=float, default=2400,
                   help='stop after the current epoch once this much time has passed')
    return p.parse_args()


def main():
    args = parse_args()
    CFG = {'lr': args.lr, 'epochs': args.epochs, 'input': 'square hand crop, annotated box x1.35; clipped to frame'}
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def write(name, v):
        p = OUT / name
        t = p.with_suffix('.tmp')
        t.write_text(json.dumps(v, indent=2))
        t.replace(p)

    def progress(**v):
        print({'elapsed_s': time.time() - started, **v}, flush=True)

    try:
        import torch, numpy as np, cv2 as cv, mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        from ultralytics import YOLO
        assert torch.cuda.is_available(), 'GPU required; refusing CPU training'
        torch.set_num_threads(2)
        newroot = args.adaptation_dir
        newmeta = json.loads((newroot / 'adaptation_labels.json').read_text())
        image_root = newroot
        if (newroot / 'images.zip').exists():
            image_root = work / 'new_images'
            zipfile.ZipFile(newroot / 'images.zip').extractall(image_root)
        opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(args.hand_landmarker)),
                                            running_mode=vision.RunningMode.IMAGE, num_hands=2)
        tracker = vision.HandLandmarker.create_from_options(opts)
        new = {'train': [], 'development': []}
        skipped = []
        progress(stage='box_reviewed_frames', images=len(newmeta['rows']))
        for row in newmeta['rows']:
            assert row['t_ms'] < 180000 and (row['role'] != 'train' or row['t_ms'] < 90000)
            path = next(image_root.rglob(row['file']))
            frame = cv.imread(str(path))
            h, w = frame.shape[:2]
            result = tracker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame, cv.COLOR_BGR2RGB)))
            if not result.hand_landmarks:
                skipped.append(row)
                continue
            # The tracing hand is the right hand; take the rightmost hand if two are visible.
            lms = max(result.hand_landmarks, key=lambda pts: pts[0].x)
            px = np.array([[p.x * w, p.y * h] for p in lms])
            lo = px.min(0)
            hi = px.max(0)
            pad = (hi - lo) * .12
            lo = np.maximum(lo - pad, 0)
            hi = np.minimum(hi + pad, [w, h])
            size = hi - lo
            if min(size) < 20 or size[0] > w * .85:
                skipped.append(row)
                continue
            center = (lo + hi) / 2
            box = [0 if row['label'] == 'ok' else 3, center[0] / w, center[1] / h, size[0] / w, size[1] / h]
            new[row['role']].append({'path': str(path), 'boxes': [box],
                                     'state': row['label'] if row['label'] == 'ok' else 'none',
                                     'seg': row['condition'] + '_' + row['label'], 'source': row})
        tracker.close()
        manifest = json.loads((args.frames_dir / 'manifest.json').read_text())
        os.environ['VISION_CFG'] = json.dumps({'smoke': False})
        spec = importlib.util.spec_from_file_location('helpers', args.helpers)
        v = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v)
        v.INPUT = str(args.frames_dir.resolve())
        v.WS = str(work)
        sessions = v.load_sessions()
        rng = random.Random(42)
        hag_train, hag_val = v.load_hagrid(rng)
        oldtrain = [r for name, info in manifest['sessions'].items() if info['role'] == 'train' for r in sessions[name]][::4]
        olddev = [r for name, info in manifest['sessions'].items() if info['role'] == 'development' for r in sessions[name]][::4]
        replay = hag_train[::4]
        train = oldtrain + replay
        # Repeat new contact and separated samples; distinct counts stay in the report.
        for r in new['train']:
            train += [r] * (3 if r['state'] == 'ok' else 10)
        rng.shuffle(train)
        val = new['development'] + olddev + hag_val[::4]
        assert new['train'] and new['development']
        cropdir = work / 'crop_images'
        cropdir.mkdir(exist_ok=True)
        crop_cache = {}

        def crop_records(rows):
            answer = []
            for row in rows:
                key = (row['path'], json.dumps(row['boxes']))
                if key not in crop_cache:
                    frame = cv.imread(row['path'])
                    h, w = frame.shape[:2]
                    made = []
                    for box in row['boxes']:
                        cls, cx, cy, bw, bh = box
                        side = max(bw * w, bh * h) * 1.35
                        x0 = max(0, int(np.floor(cx * w - side / 2)))
                        y0 = max(0, int(np.floor(cy * h - side / 2)))
                        x1 = min(w, int(np.ceil(cx * w + side / 2)))
                        y1 = min(h, int(np.ceil(cy * h + side / 2)))
                        if min(x1 - x0, y1 - y0) < 20:
                            continue
                        boxes = []
                        for lab, ox, oy, ow, oh in row['boxes']:
                            left = max(x0, (ox - ow / 2) * w)
                            right = min(x1, (ox + ow / 2) * w)
                            top = max(y0, (oy - oh / 2) * h)
                            bottom = min(y1, (oy + oh / 2) * h)
                            area = max(0, right - left) * max(0, bottom - top)
                            if area < .5 * ow * w * oh * h:
                                continue
                            boxes.append([lab, ((left + right) / 2 - x0) / (x1 - x0), ((top + bottom) / 2 - y0) / (y1 - y0),
                                          (right - left) / (x1 - x0), (bottom - top) / (y1 - y0)])
                        if not boxes:
                            continue
                        name = hashlib.sha256((str(key) + str(box)).encode()).hexdigest() + '.jpg'
                        path = cropdir / name
                        cv.imwrite(str(path), frame[y0:y1, x0:x1], [cv.IMWRITE_JPEG_QUALITY, 95])
                        made.append({**row, 'path': str(path), 'boxes': boxes})
                    crop_cache[key] = made
                answer.extend(crop_cache[key])
            return answer

        progress(stage='crop_training_images')
        train = crop_records(train)
        val = crop_records(val)
        data = v.make_dataset('hand_crop_adaptation', train, val)
        initial = args.init
        model = YOLO(str(initial))
        report = {'finished': False, 'config': CFG, 'initial': initial.name,
                  'train_unique_new': len(new['train']), 'development_unique_new': len(new['development']),
                  'new_rows': {k: [r['source'] for r in rows] for k, rows in new.items()}, 'skipped': skipped,
                  'old_train': len(oldtrain), 'hagrid_replay': len(replay), 'weighted_train_size': len(train),
                  'validation_crop_count': len(val), 'unique_cropped_inputs': len(crop_cache),
                  'limitations': 'Single-person adaptation on visually sampled intervals with automatic boxes. '
                                 'Final-minute videos and teammate data excluded from training/selection.'}
        write('training_report.json', report)

        def epoch_end(trainer):
            progress(stage='training', epoch=trainer.epoch + 1, loss=str(trainer.loss_items))
            if time.time() - started > args.time_budget_s:
                trainer.stop = True
        model.add_callback('on_train_epoch_end', epoch_end)
        model.train(data=data, epochs=CFG['epochs'], imgsz=320, batch=32, device=0, workers=2, seed=42,
                    project=str(OUT / 'runs'), name='adapt', exist_ok=True,
                    optimizer='AdamW', lr0=CFG['lr'], lrf=.2, warmup_epochs=1, patience=5, amp=True, verbose=False,
                    plots=False, degrees=7, scale=.25, mosaic=0, mixup=0, save_period=5)
        chosen = OUT / 'runs/adapt/weights/best.pt'
        assert chosen.exists()
        winner = YOLO(str(chosen))
        exported = winner.export(format='onnx', imgsz=320, simplify=True, device='cpu')
        shutil.copy2(chosen, OUT / 'crop_320.pt')
        shutil.copy2(exported, OUT / 'crop_320.onnx')
        report.update(finished=True, elapsed_s=time.time() - started)
        write('training_report.json', report)
        write('result.json', {'finished': True, 'elapsed_s': time.time() - started,
                              'checkpoint': 'crop_320.pt', 'onnx': 'crop_320.onnx'})
        progress(stage='complete')
    except Exception as e:
        write('failure.json', {'error': str(e), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
