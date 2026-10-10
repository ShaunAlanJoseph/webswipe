"""Extract labelled main-camera frames for detector adaptation.

For each manually reviewed word interval in the sitting and standing benchmark
recordings, saves frames every 200 ms inside the stable-contact interval
(label "ok") and 100-500 ms after the release bracket ("no_gesture").
Words ending before 90 s are training rows, the rest development rows.
The 180-240 s block of each recording is never read.

Writes <out>/images/*.jpg, <out>/images.zip and <out>/adaptation_labels.json,
the inputs of train_gesture_crop.py.

Example:
    python training/prepare_adaptation_frames.py --recording data/benchmark_20261010 \
        --boundaries data/benchmark_20261010/physical_boundaries_dev.json \
        --out data/detector_adaptation
"""
import argparse
import bisect
import json
import zipfile
from collections import Counter
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--recording', type=Path, default=REPO / 'data/benchmark_20261010',
                   help='folder with sitting/ and standing/ (camera.avi, frames.jsonl)')
    p.add_argument('--boundaries', type=Path, default=REPO / 'data/benchmark_20261010/physical_boundaries_dev.json',
                   help='reviewed contact intervals and release brackets per word')
    p.add_argument('--out', type=Path, default=REPO / 'data/detector_adaptation')
    p.add_argument('--source-session', default='20261010_155846_9c687f')
    return p.parse_args()


def main():
    args = parse_args()
    D = args.out
    D.mkdir(parents=True, exist_ok=True)
    images = D / 'images'
    images.mkdir(exist_ok=True)
    labels = json.loads(args.boundaries.read_text())
    rows = []
    for condition in ('sitting', 'standing'):
        src = args.recording / condition
        frames = [json.loads(l) for l in (src / 'frames.jsonl').read_text().splitlines()]
        ts = [f['t_ms'] for f in frames]
        cap = cv2.VideoCapture(str(src / 'camera.avi'))
        for lab in labels:
            if lab['condition'] != condition:
                continue
            a, b = lab['stable_contact_interval_ms']
            lo, hi = lab['release_bracket_ms']
            role = 'train' if b < 90000 else 'development'
            for label, start, end in [('ok', a, b), ('no_gesture', hi + 100, hi + 500)]:
                for requested in range(start, end + 1, 200):
                    idx = min(bisect.bisect_left(ts, requested), len(ts) - 1)
                    t = ts[idx]
                    name = f'{condition}_{idx:05d}_{label}.jpg'
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, frame = cap.read()
                    assert ok
                    cv2.imwrite(str(images / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    rows.append({'file': name, 'condition': condition, 'frame_index': idx, 't_ms': t, 'label': label,
                                 'role': role, 'trial_id': lab['trial_id'],
                                 'basis': 'Contact interval inferred from manually reviewed raw-video samples; '
                                          'release negatives start after clearly separated sample.'})
        cap.release()
    with zipfile.ZipFile(D / 'images.zip', 'w', zipfile.ZIP_STORED) as z:
        for p in images.glob('*.jpg'):
            z.write(p, p.name)
    (D / 'adaptation_labels.json').write_text(json.dumps({
        'rows': rows,
        'train_policy': 'Only <90s visually sampled words. 90-180s development; 180-240s never used for training '
                        'or checkpoint selection. Teammate excluded.',
        'source_session': args.source_session}, indent=2))
    print(Counter((r['role'], r['label']) for r in rows))


if __name__ == '__main__':
    main()
