"""Adapt the crop detector to a difficult-lighting recording (produces gesture_lighting_20261010).

Starts from gesture_crop_20261010.pt (train_gesture_crop.py) and uses the same
hand-crop recipe with a lower learning rate and fewer epochs. New data:
  - the main-camera adaptation rows with cached hand boxes
    (adaptation_cached_boxes.json: VIDEO-mode MediaPipe tracks at the reviewed
    frames, so no hand tracking runs here);
  - reviewed frames from the difficult-lighting standing recording
    (lighting_adaptation_labels.json + lighting_samples.zip, boxes included).
Repeats: lighting "ok" rows 6x, other "ok" rows 3x, all "no_gesture" rows 10x.
Old sessions and HaGRID are subsampled 1 in 4 exactly as in train_gesture_crop.py.

Writes lighting_320.pt, lighting_320.onnx, training_report.json and result.json
to --out. Needs a CUDA GPU. Tested with ultralytics==8.4.174, onnx, onnxslim,
onnxruntime.

Example:
    python training/train_gesture_lighting.py --adaptation-dir data/detector_adaptation \
        --lighting-dir data/detector_lighting --frames-dir data/detector_frames \
        --init models/gesture_crop_20261010.pt
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
                   help='adaptation_cached_boxes.json and images.zip (or the extracted images)')
    p.add_argument('--lighting-dir', type=Path, default=REPO / 'data/detector_lighting',
                   help='lighting_adaptation_labels.json and lighting_samples.zip')
    p.add_argument('--frames-dir', type=Path, default=REPO / 'data/detector_frames',
                   help='manifest.json, frames_*.zip and hagrid_subset.zip')
    p.add_argument('--init', type=Path, default=REPO / 'models/gesture_crop_20261010.pt')
    p.add_argument('--helpers', type=Path, default=REPO / 'tools/kaggle/vision_runner.py')
    p.add_argument('--out', type=Path, default=REPO / 'runs/detector_lighting')
    p.add_argument('--workdir', type=Path, default=REPO / 'runs/work')
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--epochs', type=int, default=8)
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
        import torch, numpy as np, cv2 as cv
        from ultralytics import YOLO
        assert torch.cuda.is_available(), 'GPU required; refusing CPU training'
        torch.set_num_threads(2)
        newroot = args.adaptation_dir
        newmeta = json.loads((newroot / 'adaptation_cached_boxes.json').read_text())
        image_root = newroot
        if (newroot / 'images.zip').exists():
            image_root = work / 'new_images'
            zipfile.ZipFile(newroot / 'images.zip').extractall(image_root)
        new = {'train': [], 'development': []}
        skipped = newmeta['skipped']
        progress(stage='use_cached_annotation_boxes', images=len(newmeta['rows']))
        for row in newmeta['rows']:
            assert row['t_ms'] < 180000 and (row['role'] != 'train' or row['t_ms'] < 90000)
            path = next(image_root.rglob(row['file']))
            new[row['role']].append({'path': str(path), 'boxes': row['boxes'],
                                     'state': row['label'] if row['label'] == 'ok' else 'none',
                                     'seg': row['condition'] + '_' + row['label'], 'source': row})
        lightroot = args.lighting_dir
        lightmeta = json.loads((lightroot / 'lighting_adaptation_labels.json').read_text())
        lightimages = lightroot
        if (lightroot / 'lighting_samples.zip').exists():
            lightimages = work / 'lighting_images'
            zipfile.ZipFile(lightroot / 'lighting_samples.zip').extractall(lightimages)
        for row in lightmeta['rows']:
            assert row['t_ms'] < 180000 and (row['role'] != 'train' or row['t_ms'] < 90000)
            path = next(lightimages.rglob(row['file']))
            new[row['role']].append({'path': str(path), 'boxes': row['boxes'],
                                     'state': row['label'] if row['label'] == 'ok' else 'none',
                                     'seg': 'lighting_' + row['label'], 'source': row})

        manifest = json.loads((args.frames_dir / 'manifest.json').read_text())
        os.environ['VISION_CFG'] = json.dumps({'smoke': False})
        spec = importlib.util.spec_from_file_location('helpers', args.helpers)
        v = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v)
        v.INPUT = str(args.frames_dir.resolve())
        v.WS = str(work)
        progress(stage='load_replay')
        sessions = v.load_sessions()
        rng = random.Random(42)
        hag_train, hag_val = v.load_hagrid(rng)
        oldtrain = [r for name, info in manifest['sessions'].items() if info['role'] == 'train' for r in sessions[name]][::4]
        olddev = [r for name, info in manifest['sessions'].items() if info['role'] == 'development' for r in sessions[name]][::4]
        replay = hag_train[::4]
        train = oldtrain + replay
        # Repeat new contact and separated samples; distinct counts stay in the report.
        for r in new['train']:
            train += [r] * ((6 if r['seg'].startswith('lighting_') else 3) if r['state'] == 'ok' else 10)
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
        report['box_source'] = 'Cached VIDEO-mode hand boxes instead of IMAGE-mode tracking; workers=0.'
        report['lighting_samples'] = {'train': sum(r['role'] == 'train' for r in lightmeta['rows']),
                                      'development': sum(r['role'] == 'development' for r in lightmeta['rows']),
                                      'review': lightmeta['visual_review']}
        write('training_report.json', report)

        def epoch_end(trainer):
            progress(stage='training', epoch=trainer.epoch + 1, loss=str(trainer.loss_items))
            if time.time() - started > args.time_budget_s:
                trainer.stop = True
        model.add_callback('on_train_epoch_end', epoch_end)
        model.train(data=data, epochs=CFG['epochs'], imgsz=320, batch=32, device=0, workers=0, seed=42,
                    project=str(OUT / 'runs'), name='adapt', exist_ok=True,
                    optimizer='AdamW', lr0=CFG['lr'], lrf=.2, warmup_epochs=1, patience=5, amp=True, verbose=False,
                    plots=False, degrees=7, scale=.25, mosaic=0, mixup=0, save_period=5)
        chosen = OUT / 'runs/adapt/weights/best.pt'
        assert chosen.exists()
        winner = YOLO(str(chosen))
        exported = winner.export(format='onnx', imgsz=320, simplify=True, device='cpu')
        shutil.copy2(chosen, OUT / 'lighting_320.pt')
        shutil.copy2(exported, OUT / 'lighting_320.onnx')
        report.update(finished=True, elapsed_s=time.time() - started)
        write('training_report.json', report)
        write('result.json', {'finished': True, 'elapsed_s': time.time() - started,
                              'checkpoint': 'lighting_320.pt', 'onnx': 'lighting_320.onnx'})
        progress(stage='complete')
    except Exception as e:
        write('failure.json', {'error': str(e), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
