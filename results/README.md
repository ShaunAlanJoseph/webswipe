# WebSwipe — decoder evaluation on D1 (Oct 2)

All numbers: Shen et al. HoloLens mid-air data (D1), **leave-one-subject-out**: train/tune on 8 subjects,
test on subject 0 (1,845 word gestures). Closed lexicon = all 1,996 D1 words. Top-1 / top-4 word accuracy.
No model or setting was chosen on the test subject unless a row says so.

## 1. Baseline: SHARK2 (`webswipe/baselines/shark2.py`, `webswipe/eval/shark2_d1.py`)
Template matcher, shape + location channels; channel weights tuned on 1,500 gestures from the other 8 subjects.

| Variant | Top-1 | Top-4 | Shape only | Location only |
|---|---|---|---|---|
| bbox normalisation (original SHARK2) | 70.2 | 98.9 | 41.0 | 70.0 |
| RMS normalisation (sim. E6) | 70.4 | 98.8 | 40.7 | 70.0 |
| RMS + Savitzky–Golay smoothing (same filter as the neural features) | **72.5** | **99.0** | 43.7 | 72.4 |

- The **location channel does almost all the work**; shape adds ≤ 0.3 points (as in simulation E5).
  In the webcam app this is fine: the cursor is drawn on the on-screen keyboard, so the user aims at keys,
  and the location channel is defined in overlay pixels (no camera calibration involved).
- Top-4 is ~99% but top-1 ~72%: the right word is almost always a candidate, but common short words lose
  to rarer neighbours (to→top, you→toy, of→off, for→four, in→on). Only 6% of errors are words with an
  identical key path (to/too). → a word-frequency prior is the obvious fix (section 3).

## 2. Robustness to webcam-like input (`webswipe/eval/robustness_d1.py`)
Test traces degraded with `webswipe/decoder/augment.py`: frame rate reduced by re-sampling on the real D1
timestamps (D1 is ~50 Hz), i.i.d. jitter in key widths (1 key ≈ 50 px in the 640×480 overlay), and pinch
timing error (up to 100 ms cut from each end). Same random seed per condition, so every decoder sees
identical traces. `webcam` preset = 12 fps (measured MediaPipe rate on the laptop) + 0.15 key jitter + 100 ms.

SHARK2 rows (top-1 %). `@50Hz` = trace re-sampled onto a uniform 50 Hz clock before decoding
(`--resample-hz 50`, also available in the live app):

| Condition | SHARK2 | + smoothing | SHARK2 @50Hz | **+ smoothing @50Hz** |
|---|---|---|---|---|
| clean (50 Hz) | 70.4 | 72.5 | 71.6 | **73.2** |
| 30 fps | 75.3 | **77.5** | 76.1 | 76.8 |
| 12 fps | 78.2 | 73.6 | 78.9 | **79.4** |
| jitter 0.1 | 47.8 | 68.4 | 55.7 | **71.4** |
| jitter 0.2 | 38.6 | 59.0 | 44.7 | **64.8** |
| jitter 0.3 | 35.0 | 51.2 | 39.1 | **57.4** |
| webcam: 12 fps + 0.15 + 100 ms | 72.4 | 70.7 | 72.7 | **73.3** |
| harsh: 10 fps + 0.3 + 100 ms | 62.1 | 60.4 | 62.1 | **63.8** |

- Raw SHARK2 is very fragile to jitter at 50 Hz: arc-length resampling follows every zig-zag. Smoothing first
  recovers ~+20 points.
- A frame-count smoothing window over-smooths at low frame rates (7 frames = 140 ms at 50 Hz but 580 ms at
  12 fps), so smoothing alone *hurts* at 12 fps. Putting the trace on a uniform 50 Hz clock first fixes this
  and is the best or near-best SHARK2 variant everywhere (D1 itself mixes 10 and 20 ms frame gaps).
  → "SHARK2 + smoothing @50Hz" is the strong baseline the neural decoder has to beat.

**Neural rows: pending the GPU run** (`notebooks/train_decoder_colab.ipynb`, sections 1-2). Two models,
same budget and held-out subject: `clean` (as in Shen et al.) vs `aug` (`--aug-prob 0.5`). The question is
whether `aug@50Hz` beats `SHARK2+smooth@50Hz` on the webcam/harsh rows without losing on clean.
A first CPU attempt (20 epochs) was abandoned: ~5 min/epoch on the shared laptop.

## 3. Word-frequency prior (`webswipe/eval/lm_prior_d1.py`, `webswipe/decoder/lm.py`)
Unigram prior from `wordfreq` (general English, independent of D1). score = log-likelihood + λ · log p(word).
SHARK2's λ tuned on the other subjects: λ = 0.5.

| SHARK2 + smoothing | λ=0 | 0.25 | **0.5 (tuned)** | 1 | 2 |
|---|---|---|---|---|---|
| clean | 72.5 | 81.5 | **84.2** | 83.9 | 78.0 |
| webcam preset | 70.7 | 76.3 | **77.7** | 75.8 | 69.5 |

With 50 Hz resampling (`results/lm_prior_d1_shark2_rs50.json`), SHARK2 + smoothing + prior (λ = 0.5):
**84.6** clean, **81.0** webcam preset. This is the strongest baseline so far.

**Accuracy vs reachability.** The D1-optimal λ = 0.5 is tuned on D1's phrases, which are mostly common
words. On the live overlay it makes 7.8% of the 1,996-word lexicon impossible to get as top-1 *even with a
perfect swipe* (bank→back, beer→be; 0.3% not even in the top 4). At λ = 0.25: 3.6% not top-1, every word
within the top 3, for −2.4 / −1.7 points on D1 (clean / webcam). With λ = 0: 16 words (identical paths, e.g.
met/meet). The live app therefore uses λ = 0.25 plus candidate selection with the other hand.

**Neural + prior: pending the GPU run.** The neural decoder's λ cannot be tuned on its own training subjects
(overfit logits); report the full λ curve, and fix λ on a second held-out subject before quoting one number.

## Caveats
- D1 is headset data. The webcam conditions are simulated degradations, not real webcam traces; D3 is needed.
- One held-out subject; the other 8 should be rotated through for the final report (needs GPU).
- Neural models are trained on Colab/Kaggle (100 epochs in the notebook; the original trains for 200).
