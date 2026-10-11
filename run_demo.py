"""Launch the live demo with the validated settings in configs/demo.json.

Extra arguments are passed to webswipe.app.main, e.g. `--camera 1`.
`--check` loads all four models without opening the camera.
"""
from pathlib import Path
import os, sys, json
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True


def main():
    args = sys.argv[1:]
    config = json.loads((ROOT / 'configs' / 'demo.json').read_text())['arguments']
    if args == ['--check']:
        import cv2, torch, mediapipe
        from webswipe.app.gesture_det import GestureDetector
        from webswipe.decoder.live import LiveDecoder, load_lexicon
        from webswipe.app.tracking import HandTracker
        torch.set_num_threads(2)
        for flag in ('--gesture-model', '--gesture-reference-model'):
            GestureDetector(config[config.index(flag) + 1])
        LiveDecoder(config[config.index('--ckpt') + 1],
                    load_lexicon(config[config.index('--lexicon') + 1]), 0, 50, 1)
        tracker = HandTracker()
        tracker.lm.close()
        print('All four models load. No camera opened and no inference run.')
        return
    sys.argv = ['webswipe.app.main', *config, *args]
    from webswipe.app.main import main as app_main
    app_main()


if __name__ == '__main__':
    main()
