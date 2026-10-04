# WebSwipe implementation plan

Goal: mid-air word-gesture typing from one webcam. Final report must show novelty and improvement over
existing systems, name the datasets, and explain the fine-tuning.

## Base paper
Shen, Dudley, Kristensson. *Fast and Robust Mid-Air Gesture Typing for AR Headsets using 3D Trajectory Decoding.*
IEEE TVCG 2023. Code: github.com/CambridgeIIS/mid_air_gesture_keyboard_decoder (cloned to `references/repos/shen2023`).

- Model: 2-layer BiLSTM (400 units) + self-attention + CTC over letters. TensorFlow 2.4, no released weights.
- Input: 50 resampled points x 39 features (normalised x,y + 26 key-proximity flags + 11 shape features).
  Their public code uses x,y only (z is dropped), so the pipeline maps directly onto 2D webcam traces.
- Data: `data/data_all.pkl`, HoloLens 2 fingertip traces, 9 subjects, 26,071 words, 96 keyboard-size conditions.

What we change: port to PyTorch, replace the headset tracker with a webcam front end (MediaPipe), pretrain on
headset and touchscreen data, fine-tune on webcam data.

## System
```
webcam -> MediaPipe HandLandmarker -> One Euro filter -> pinch detector (scale-normalised + online GMM)
       -> keyboard-plane trajectory -> Shen features -> BiLSTM-CTC (fine-tuned) -> lexicon beam search + LM
       -> candidates; non-dominant hand: space / delete / pick candidate
```
App skeleton: kinivi/hand-gesture-recognition-mediapipe, ported from `mp.solutions` to the MediaPipe Tasks API.

## Datasets
| ID | Dataset | Use |
|---|---|---|
| D1 | Shen et al. mid-air (HoloLens), 26k words, 9 users | pretraining, reproduction |
| D2 | swipe.futo.org, ~1M touchscreen swipes, MIT | extra pretraining (optional) |
| D3 | Ours: webcam traces, 4 users x ~200-300 words (MacKenzie phrase set) | fine-tuning + test, split by user |
| D4 | HaGRIDv2 landmark annotations | pinch classifier (optional) |

## Fine-tuning and comparisons
Training stages: D1 (+D2) pretrain -> fine-tune on D3 (3 users) -> test on held-out user (leave-one-user-out).

| Comparison | Shows |
|---|---|
| SHARK2 template decoder | classic baseline |
| FUTO Swipe (released weights, zero-shot) — optional, see note | strongest off-the-shelf model |
| Shen model trained on D1, no fine-tune | base paper as-is on webcam data |
| Shen model trained on D3 only | value of pretraining |
| Ours: D1(+D2) -> D3 fine-tuned | main result |
| Per-key webcam keyboard (tap) | the camera-only per-key approach from the intro |

Metrics: word accuracy (top-1 / top-4), CER, WPM (live), pinch F1, end-to-end latency.

## Schedule (4 weeks, from Sep 26)
| Week | Dates | Work |
|---|---|---|
| 1 | Sep 26 - Oct 2 | PyTorch port of Shen pipeline, reproduce on D1 (Colab). App skeleton: landmarks, keyboard overlay, pinch, trace logger |
| 2 | Oct 3 - Oct 9 | Record D3 (each teammate ~1 hr). SHARK2 + FUTO baselines. D2 pretraining |
| 3 | Oct 10 - Oct 16 | Fine-tuning + ablations. Decoder + LM in live app. Bimanual controls |
| 4 | Oct 17 - Oct 23 | User study (WPM/CER vs per-key), write Sections 4-5, update Section 3 |

## Hardware
Laptop: CPU only, webcam 640x480 @ 30 fps, MediaPipe ~12 fps with no hand in view (to re-measure while tracking).
Training: Colab / Kaggle free GPUs.

## Notes
- The Shen repo has no licence: academic use with citation; consider emailing the authors.
- Gesture2Text and RingGesture have no public code, so they are cited, not compared.
- FUTO (checked Oct 2, swipe.futo.tech): dataset huggingface.co/datasets/futo-org/swipe.futo.org is MIT
  (1M QWERTY swipes) -> fine for D2. Model weights huggingface.co/futo-org/futo-swipe are under a custom
  "FUTO Model License" (read the terms before benchmarking/publishing); inference is a C++ GPL library
  (gitlab.futo.org/keyboard/swipe-library) with no Python bindings and an undocumented input format.
  Zero-shot FUTO is therefore optional: only if time allows after the main results.

## Status (Oct 2, branch webswipe-eval)
- Done: SHARK2 baseline + webcam-degradation robustness + word-frequency prior on D1 (`results/README.md`);
  training augmentation; Colab/Kaggle notebook + bundle; live app: SHARK2 default, dropout tolerance,
  candidate selection, tap mode, event log; recording resume + checker; paper Sec. 3-4 revision (`paper/`).
- Oct 5: neural decoders trained and evaluated on Kaggle for all 9 held-out D1 subjects (`results/README.md` sec. 4):
  aug beats clean by +8.3 top-1 on the webcam preset on 9/9 subjects; both beat SHARK2.
- Next (needs people): record D3, fine-tune (`tools/kaggle/push.py`), user study.
