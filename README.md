# WebSwipe
Mid-air word-gesture typing from a single webcam. You pinch your thumb and index finger on the first letter of a word, trace through its letters on an on-screen keyboard, and release; WebSwipe turns the trace into word suggestions. The other hand cycles suggestions or deletes words. No headset, depth camera or calibration is needed.

Course project, Computer Vision, IIIT Kottayam. Team: Venkat Jaswanth (2023BCD0021), Shaun Alan Joseph (2023BCD0006), Akhil K (2023BCD0018), Nivien M. (2023BCS0054). The full write-up is [`docs/WebSwipe_Project_Documentation.pdf`](docs/WebSwipe_Project_Documentation.pdf).
## How it works
Each webcam frame goes through four stages:
1. **Hand tracking.** MediaPipe Hand Landmarker finds 21 landmarks per hand (`webswipe/app/tracking.py`); the pinch point is smoothed with a One Euro filter (`pinch.py`).
2. **Pinch detection.** A YOLOv10n gesture detector, pretrained on HaGRID and fine-tuned on our own webcam frames, scores each hand crop (`gesture_det.py`, `hand_crop.py`). A second detector acts as a reference that can veto a pinch reading (`gesture_reference.py`), and release and start buffers keep a word from being cut or started late (`controls.py`, `left_pinch.py`).
3. **Trace to features.** While the right hand pinches, the cursor path over the keyboard is recorded, resampled to 50 Hz and converted to the features of Shen et al. (2023): normalised position, key proximity and shape (`webswipe/decoder/features.py`).
4. **Decoding.** A BiLSTM with self-attention and a CTC output layer, ported from Shen et al.'s TensorFlow code to PyTorch (`webswipe/decoder/model.py`), scores the trace; a lexicon beam search returns the top four words (`decode.py`, `live.py`). SHARK2, the classic template matcher, is included as a baseline (`webswipe/baselines/shark2.py`).

The decoder was trained on the Shen et al. HoloLens mid-air dataset with webcam-like degradation added during training, then personalised on our own webcam swipes. The detector was fine-tuned on hand crops and then on low-light frames.
## Run the demo
The four model files and the 10k-word lexicon are in the repository, so the demo needs no training or data download. It runs on the CPU.

Set up once (needs internet and 64-bit Python 3.13):
```
sh setup_mac_linux.sh        # Windows: setup_windows.bat
```
This creates `.venv`, installs `requirements.txt`, and checks that all four models load without opening the camera. Then run:
```
sh run_mac_linux.sh          # Windows: run_windows.bat
.venv/bin/python run_demo.py --camera 1    # extra flags are passed to the app
.venv/bin/python run_demo.py --check       # load the models only, no camera
```
`run_demo.py` starts `webswipe.app.main` with the validated settings in `configs/demo.json`.

To type a word, face the camera in even front lighting with your right hand fully visible. Place the right-hand pinch point on the first letter, pinch, trace through the letters while keeping the fingertips touching, then open them clearly. The first suggestion is added automatically.

| Input | Action |
|---|---|
| Short left-hand pinch | Cycle to the next suggestion |
| Left-hand pinch held about 0.6 s | Delete the last word |
| `1`–`4` | Pick a suggestion |
| `B` | Delete the last word |
| `C` / `T` / `H` | Toggle cursor / trace / hand skeleton (all start hidden) |
| `Q` or `Esc` | Quit |

Typing events are logged locally to `data/sessions/`; no video is saved in demo mode.
## Repository layout
| Path | Contents |
|---|---|
| `webswipe/app/` | Live app: tracking, pinch detection, controls, keyboard overlay, recording and diagnostic modes |
| `webswipe/decoder/` | PyTorch port of the Shen et al. decoder: features, model, training, augmentation, live decoding |
| `webswipe/baselines/` | SHARK2 baseline |
| `webswipe/eval/` | Leave-one-subject-out evaluation on the Shen et al. data |
| `training/` | Scripts that produced the final detector and decoder models |
| `tools/` | Frame capture and auto-labelling, recording checks, Kaggle runners, recording kit |
| `notebooks/` | Colab/Kaggle notebook for decoder training and evaluation |
| `experiments/` | Early simulation experiments E1–E10 (see `experiments/RESULTS.md`) |
| `results/` | Evaluation outputs and training logs |
| `models/` | The four models used by the demo |
| `configs/demo.json` | Validated demo settings |
| `docs/` | Report, reproducibility supplement, intro submission, original plan, LaTeX source |
| `tests/` | Unit tests |
## Results
These are the headline numbers from the report; `docs/` and `results/` have the details.

| Measure | Before | After |
|---|---|---|
| Complete capture of 29 webcam swipes (pinch pipeline changes) | 8/29 | 23/29 |
| Correct first suggestion on those swipes (decoder personalisation) | 14/29 | 21/29 |
| Difficult-lighting replay, capture / top-1 / top-4 | – | 15/15, 11/15, 14/15 |
| Top-1, 9 held-out headset subjects, simulated webcam degradation | 74.6% | 82.9% |
| Tuned SHARK2 baseline on the same folds | – | 77.1% |

The webcam replays reuse recordings made during development by one user, so they show that the changes work on real webcam input but are not an independent user study. Section 4.2 of the report ("Evaluation scope") discusses this.
## Development
Install the extra dependencies for training, evaluation and tests:
```
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q tests
```
**Data.** Raw data is not in the repository; `data/README.md` lists each dataset and how to obtain or record it. The Shen et al. data comes from github.com/CambridgeIIS/mid_air_gesture_keyboard_decoder (`data/data_all.pkl`, linked as `data/shen_midair.pkl`).

**Decoder.** Build features, then train:
```
python -m webswipe.decoder.shen_data                                   # -> data/shen_features.npz
python -m webswipe.decoder.train --epochs 200                          # random 95/5 split, as in the original
python -m webswipe.decoder.train --split subject --test-subject 0:3 --aug-prob 0.5   # leave one subject out, with augmentation
```
GPU training runs on Colab or Kaggle: `bash tools/make_colab_bundle.sh`, then run `notebooks/train_decoder_colab.ipynb`. With the Kaggle CLI configured and `KAGGLE_USERNAME` set, `python tools/kaggle/push.py bundle` uploads the bundle and `python tools/kaggle/push.py train:0` runs `tools/kaggle/runner.py` as a GPU kernel.

**Evaluation** on the Shen et al. data, leave-one-subject-out (`results/README.md` explains each table):
```
python -m webswipe.eval.shark2_d1
python -m webswipe.eval.robustness_d1 --resample-hz 50 --ckpt clean=checkpoints/d1_loso0_clean_last.pt
python -m webswipe.eval.lm_prior_d1 --resample-hz 50 --ckpt clean=checkpoints/d1_loso0_clean_last.pt
python -m webswipe.eval.summarize_loso
```
**Gesture detector.** `tools/capture_frames.py` records labelled frames, `tools/label_frames.py` auto-labels hand boxes with MediaPipe, and `tools/kaggle/push_vision.py` fine-tunes the HaGRID YOLOv10n detector on Kaggle (`tools/kaggle/vision_runner.py`). `training/` holds the hand-crop, low-light and decoder-personalisation steps that produced the files in `models/`.

**Recording.** `tools/recording_kit/` is the stand-alone kit teammates used to record webcam data. Inside the app, `--mode record` records phrase sessions, `--mode guided` records prompted words with video for review (`python -m webswipe.app.guided --replay data/guided/SESSION/attempt_ID`), and `python -m webswipe.app.diagnostic` records word boundaries and pinch calibration trials. `--mode tap` is the per-key baseline.
## Licences and credits
No licence is granted for this project's own code; all rights are reserved by the authors. Third-party parts keep their own licences:
- The app skeleton derives from kinivi/hand-gesture-recognition-mediapipe (Apache-2.0); see `webswipe/app/LICENSE-kinivi` and `webswipe/app/NOTICE`.
- `models/hand_landmarker.task` is Google's MediaPipe Hand Landmarker (Apache-2.0); see `licenses/`.
- The gesture detectors are fine-tuned from the released HaGRID YOLOv10n gesture model (HaGRIDv2, Nuzhdin et al., arXiv:2412.01508; dataset CC BY-SA 4.0).
- The decoder architecture and the mid-air dataset are from Shen, Dudley and Kristensson, *Fast and Robust Mid-Air Gesture Typing for AR Headsets using 3D Trajectory Decoding*, IEEE TVCG 2023. Their repository has no licence, so it is cited, not redistributed.
