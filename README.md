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
GPU training: `tools/make_colab_bundle.sh`, upload `colab_bundle.zip` to Drive, run `notebooks/train_decoder_colab.ipynb`.

## Live app
```
python tools/make_phrases.py                                          # phrase list for recording
python -m webswipe.app.main --mode record --participant shaun         # record webcam dataset
python -m webswipe.app.main --ckpt checkpoints/shen_d1.pt             # live typing
```
Dominant hand: pinch thumb and index, trace the word, release. Other hand: pinch to delete the last word
(typing) or skip the word (recording). Keys: `q` quit, `b` delete, `n` skip.

## Tests
```
python -m pytest -q tests
```
