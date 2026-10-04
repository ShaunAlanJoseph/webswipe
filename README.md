# WebSwipe

Mid-air word-gesture typing from a single webcam. Builds on Shen et al. (IEEE TVCG 2023); see `PLAN.md`.

## Setup
```
python3.13 -m venv .venv
.venv/bin/pip install mediapipe opencv-python numpy scipy pandas pytest
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
mkdir -p models && curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```
Shen et al. data: clone github.com/CambridgeIIS/mid_air_gesture_keyboard_decoder and link
`data/data_all.pkl` to `data/shen_midair.pkl`.

## Decoder (PyTorch port of Shen et al.)
```
python -m webswipe.decoder.shen_data                      # features -> data/shen_features.npz
python -m webswipe.decoder.train --epochs 200             # random 95/5 split, as in the original
python -m webswipe.decoder.train --split subject --test-subject 0:3   # leave one subject out
python -m webswipe.decoder.train --data data/webcam_features.npz --init checkpoints/shen_d1.pt --lr 3e-5
```
Webcam-like augmentation during training (domain randomisation over frame rate, jitter, pinch timing):
`--aug-prob 0.5`. Webcam recordings -> features: `python -m webswipe.decoder.webcam_data --resample-hz 50`.

GPU training and all evaluations (do not train on the laptop): `bash tools/make_colab_bundle.sh`, then run
`notebooks/train_decoder_colab.ipynb` on Colab (bundle in `MyDrive/webswipe/`) or Kaggle (bundle as a dataset);
the first cell detects which. Setup steps are in the notebook's first cell.
Headless alternative (Kaggle CLI, token in `~/.kaggle`): `python tools/kaggle/push.py bundle` uploads the bundle
as a private dataset, `python tools/kaggle/push.py train:0` runs `tools/kaggle/runner.py` as a private GPU
kernel (`smoke`, `train:<subjects>`, `repro`, `eval:<subjects> --cpu --after train:<subjects>`).

## Evaluation (D1, leave-one-subject-out; see `results/README.md`)
```
pip install wordfreq                                       # unigram prior, independent of the study data
python -m webswipe.eval.shark2_d1                          # SHARK2 baseline, tuned on the other subjects
python -m webswipe.eval.robustness_d1 --resample-hz 50 --ckpt clean=checkpoints/d1_loso0_clean_last.pt
python -m webswipe.eval.lm_prior_d1 --resample-hz 50 --ckpt clean=checkpoints/d1_loso0_clean_last.pt
python -m webswipe.eval.plot_robustness
```

## Live app
```
python tools/make_phrases.py                                          # phrase list for recording
python -m webswipe.app.main --mode record --participant shaun         # record webcam dataset (resumes
                                                                      #   after this participant's last phrase)
python tools/check_recordings.py                                      # per-participant report + flagged gestures
python -m webswipe.app.main                                           # live typing, SHARK2 (no model needed)
python -m webswipe.app.main --ckpt checkpoints/d1_loso0_aug_last.pt --prior-weight 0.25 --resample-hz 50   # neural
python -m webswipe.app.main --mode tap --autocorrect                  # per-key baseline for the user study
python -m webswipe.app.main --participant shaun --log                 # any typing mode: events -> data/sessions/
```
Tap mode: pinch on a key = that letter; other hand short pinch = space (word snapped to the lexicon with
`--autocorrect`), hold = backspace.
Dominant hand: pinch thumb and index, trace the word, release (a swipe survives tracking dropouts up to
200 ms). Other hand: short pinch = swap the last word for the next candidate, hold (0.6 s) = delete it;
in recording mode any other-hand pinch skips the word. Keys: `1`-`4` pick a candidate, `b` delete,
`n` skip, `q` quit. SHARK2 defaults: smoothing, 50 Hz, prior 0.25 (see `results/README.md`).

## Tests
```
python -m pytest -q tests
```
