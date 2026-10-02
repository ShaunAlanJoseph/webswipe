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

NEURAL_ROBUSTNESS_PLACEHOLDER

## 3. Word-frequency prior (`webswipe/eval/lm_prior_d1.py`, `webswipe/decoder/lm.py`)
Unigram prior from `wordfreq` (general English, independent of D1). score = log-likelihood + λ · log p(word).
SHARK2's λ tuned on the other subjects: λ = 0.5.

| SHARK2 + smoothing | λ=0 | 0.25 | **0.5 (tuned)** | 1 | 2 |
|---|---|---|---|---|---|
| clean | 72.5 | 81.5 | **84.2** | 83.9 | 78.0 |
| webcam preset | 70.7 | 76.3 | **77.7** | 75.8 | 69.5 |

NEURAL_PRIOR_PLACEHOLDER

## Caveats
- D1 is headset data. The webcam conditions are simulated degradations, not real webcam traces; D3 is needed.
- One held-out subject; the other 8 should be rotated through for the final report (needs GPU).
- Neural models here are short CPU runs (20 epochs); the original trains for 200.
