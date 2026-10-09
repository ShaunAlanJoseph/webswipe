"""Continue fine-tuning the YOLOv10n gesture detector on the October 10 frame sessions.

Starts from the earlier own+HaGRID checkpoint, trains one candidate per learning
rate, picks the candidate on the development session only, then scores the
held-out standing session and exports the winner to ONNX
(gesture_candidate_320.pt / .onnx). That checkpoint is the starting point of
train_gesture_crop.py.

Needs a CUDA GPU. Tested with ultralytics==8.4.174, onnx, onnxslim, onnxruntime.

Example:
    python training/train_gesture_continued.py --frames-dir data/detector_frames \
        --init models/F_own_hagrid_320.pt --out runs/detector_continued
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import shutil
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--frames-dir', type=Path, default=REPO / 'data/detector_frames',
                   help='manifest.json, frames_*.zip and hagrid_subset.zip')
    p.add_argument('--init', type=Path, default=REPO / 'models/F_own_hagrid_320.pt',
                   help='starting checkpoint (sha256 must match manifest.json)')
    p.add_argument('--helpers', type=Path, default=REPO / 'tools/kaggle/vision_runner.py',
                   help='data/eval helpers shared with the earlier detector runs')
    p.add_argument('--out', type=Path, default=REPO / 'runs/detector_continued')
    p.add_argument('--workdir', type=Path, default=REPO / 'runs/work',
                   help='scratch space for unpacked frames and YOLO datasets')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--max-hours', type=float, default=2.0, help='training budget across all candidates')
    p.add_argument('--learning-rates', type=float, nargs='+', default=[0.0001, 0.0003])
    p.add_argument('--smoke', action='store_true', help='tiny subsets, one epoch, no export')
    return p.parse_args()


def main():
    args = parse_args()
    CFG = {'smoke': args.smoke, 'epochs': args.epochs, 'max_hours': args.max_hours,
           'learning_rates': args.learning_rates}
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)

    def write(name, value):
        p = OUT / name
        t = p.with_suffix(p.suffix + '.tmp')
        t.write_text(json.dumps(value, indent=2))
        t.replace(p)

    try:
        import torch
        from ultralytics import YOLO
        if not torch.cuda.is_available():
            raise RuntimeError('GPU required: refusing CPU training')
        manifest = json.loads((args.frames_dir / 'manifest.json').read_text())
        init = args.init
        assert hashlib.sha256(init.read_bytes()).hexdigest() == manifest['initial_checkpoint']['sha256']
        os.environ['VISION_CFG'] = json.dumps({'smoke': CFG['smoke']})
        spec = importlib.util.spec_from_file_location('helpers', args.helpers)
        v = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v)
        v.INPUT = str(args.frames_dir.resolve())
        v.WS = str(args.workdir.resolve())
        sessions = v.load_sessions()
        for name, info in manifest['sessions'].items():
            assert len(sessions[name]) == info['frames'], (name, len(sessions[name]), info['frames'])
        rng = random.Random(0)
        hag_train, hag_val = v.load_hagrid(rng)
        assert hag_train and hag_val, 'HaGRID replay data missing'
        train = [r for name, x in manifest['sessions'].items() if x['role'] == 'train' for r in sessions[name]]
        dev = [r for name, x in manifest['sessions'].items() if x['role'] == 'development' for r in sessions[name]]
        test = [r for name, x in manifest['sessions'].items() if x['role'] == 'test' for r in sessions[name]]
        # File identities must be disjoint across roles.
        assert not set(r['path'] for r in train) & set(r['path'] for r in dev + test)
        assert not set(r['path'] for r in dev) & set(r['path'] for r in test)
        if CFG['smoke']:
            train = train[::40]
            dev = dev[::20]
            hag_val = hag_val[::4]
        train += hag_train
        rng.shuffle(train)
        report = {'config': CFG, 'manifest': manifest, 'gpu': torch.cuda.get_device_name(0),
                  'train_images': len(train), 'development_images': len(dev), 'test_images': len(test),
                  'hagrid_validation_images': len(hag_val), 'candidates': {}, 'finished': False}
        write('results.json', report)
        cmap = {s: s for s in v.STATES}
        deadline = time.monotonic() + CFG['max_hours'] * 3600

        def evaluate(model, records):
            fs = v.frame_scores(model, records, 320, cmap)
            out = {f'thr{thr}': v.metrics(records, fs, thr) for thr in [.25, .5, .6]}
            for seg in sorted(set(r['seg'] for r in records)):
                indices = [i for i, r in enumerate(records) if r['seg'] == seg]
                out[seg] = {f'thr{thr}': v.metrics([records[i] for i in indices], [fs[i] for i in indices], thr)
                            for thr in [.25, .5, .6]}
            return out

        baseline = YOLO(str(init))
        report['baseline_development'] = evaluate(baseline, dev)
        report['baseline_hagrid'] = v.evaluate(baseline, {'hagrid': hag_val}, 320, cmap)
        write('results.json', report)
        data = v.make_dataset('overnight', train, dev + hag_val)
        for lr in CFG['learning_rates']:
            if time.monotonic() > deadline:
                break
            name = f'lr{lr:g}'.replace('.', 'p')
            model = YOLO(str(init))
            run_dir = OUT / 'runs' / name

            def epoch_end(trainer):
                print({'candidate': name, 'epoch': trainer.epoch + 1, 'loss': str(trainer.loss_items)}, flush=True)
                if time.monotonic() > deadline:
                    trainer.stop = True
            model.add_callback('on_train_epoch_end', epoch_end)
            t = time.time()
            model.train(data=data, epochs=1 if CFG['smoke'] else CFG['epochs'], imgsz=320, batch=32, device=0,
                        workers=2, seed=0, project=str(OUT / 'runs'), name=name, exist_ok=True, plots=False,
                        optimizer='AdamW', lr0=lr, lrf=.1, warmup_epochs=1, patience=8, amp=True, verbose=False,
                        degrees=10, scale=.3, save=True, save_period=5)
            last = run_dir / 'weights/last.pt'
            assert last.is_file()
            candidate = YOLO(str(last))
            shutil.copyfile(last, OUT / (name + '.pt'))
            dv = evaluate(candidate, dev)
            score = .5 * (dv['thr0.25']['ok']['recall'] + 1 - dv['thr0.25']['ok']['false_rate'])
            report['candidates'][name] = {'learning_rate': lr, 'seconds': round(time.time() - t, 1), 'development': dv,
                                          'hagrid': v.evaluate(candidate, {'hagrid': hag_val}, 320, cmap),
                                          'selection_score': score, 'checkpoint': name + '.pt'}
            write('results.json', report)
        if not report['candidates']:
            raise RuntimeError('No training candidate completed')
        selected = max(report['candidates'], key=lambda n: report['candidates'][n]['selection_score'])
        report['selected'] = selected
        if not CFG['smoke']:
            # The held-out camera session is scored only after the candidate is chosen.
            report['baseline_camera_test'] = evaluate(baseline, test)
            winner = YOLO(str(OUT / (selected + '.pt')))
            report['candidate_camera_test'] = evaluate(winner, test)
            b = report['baseline_camera_test']['thr0.25']['ok']
            c = report['candidate_camera_test']['thr0.25']['ok']
            report['eligible_for_live_trial'] = (c['recall'] >= b['recall'] and c['false_rate'] <= b['false_rate'] + .01)
            report['live_trial_required'] = True
            exported = winner.export(format='onnx', imgsz=320, simplify=True, device='cpu')
            shutil.copyfile(exported, OUT / 'gesture_candidate_320.onnx')
            shutil.copyfile(OUT / (selected + '.pt'), OUT / 'gesture_candidate_320.pt')
        report['finished'] = True
        report['finished_at'] = time.time()
        write('results.json', report)
        print(json.dumps(report), flush=True)
    except Exception as e:
        write('failure.json', {'type': type(e).__name__, 'error': str(e), 'time': time.time()})
        raise


if __name__ == '__main__':
    main()
