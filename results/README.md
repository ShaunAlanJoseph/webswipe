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

**Neural rows (Kaggle, Oct 4-5):** see section 4 for all 9 held-out subjects. Subject 0, top-1 @50Hz:

| Condition | SHARK2+smooth | clean | aug |
|---|---|---|---|
| clean (50 Hz) | 73.2 | **96.0** | 95.8 |
| jitter 0.3 | 57.4 | 94.5 | **95.8** |
| webcam: 12 fps + 0.15 + 100 ms | 73.3 | 89.1 | **92.8** |
| harsh: 10 fps + 0.3 + 100 ms | 63.8 | 84.1 | **88.9** |

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

**Neural + prior (section 4):** the prior does **not** help the neural decoders: λ = 0 is best or tied for
both models in both conditions (all 9 subjects). The CTC model already prefers frequent words from its
training data; the prior only matters for SHARK2.

## 4. Leave-one-subject-out over all 9 D1 subjects (Kaggle, Oct 4-5)
Each subject held out in turn; `clean` and `aug` trained on the other 8 (100 epochs, last epoch reported),
SHARK2 sigmas and λ tuned on the other 8. Full tables: `loso_summary.md` (9 subjects),
`loso_summary_excl9.md` (without subject 9); per subject: `*_s<id>.json`, `robustness_d1_s<id>.png`.
Top-1 %, mean ± sd over held-out subjects, every decoder @50Hz:

| Condition | SHARK2+smooth | clean | aug | aug > clean |
|---|---|---|---|---|
| clean (50 Hz), 9 subjects | 67.8 ± 20.3 | 86.3 ± 12.9 | 86.6 ± 15.9 | 4/9 |
| webcam preset, 9 subjects | 69.0 ± 9.1 | 74.6 ± 12.5 | **82.9 ± 10.0** | **9/9** |
| harsh preset, 9 subjects | 61.3 ± 7.8 | 69.3 ± 13.4 | **78.5 ± 11.3** | **9/9** |
| clean (50 Hz), without subject 9 | 74.5 ± 1.4 | 90.5 ± 3.2 | 91.8 ± 2.6 | 4/8 |
| webcam preset, without subject 9 | 70.6 ± 8.4 | 75.6 ± 13.0 | **83.4 ± 10.6** | 8/8 |

- **Neural beats SHARK2 everywhere**, by ~16 points on clean traces and ~14 on the webcam preset (aug).
- **Augmentation is the main result:** +8.3 (webcam) and +9.2 (harsh) points over `clean`, better on
  **every** held-out subject, while costing nothing on clean traces (+0.3 on average, 4/9 subjects).
- With the prior, SHARK2 reaches 77.1 ± 7.4 on the webcam preset (tuned λ), still 5.8 points below `aug`
  without a prior.
- **Subject 9 is an outlier in the data, not the models.** Its traces were recorded ~2 months before the
  others, average 26 ms/frame (others 16.7), and 94.6% of them contain a single-frame jump of more than one
  key width (others 16-28%); path length is 1.63× the ideal key path (others 1.3-1.4×). SHARK2 gets 13.8%
  on it, and 50 Hz resampling drops the neural models from 87.0/86.0 (raw) to 52.7/44.8. Report the
  9-subject numbers and give the 8-subject ones as a sensitivity check.

**Reproduction of Shen et al.** (random 95/5 split, 200 epochs, as in their code; `shen_d1_random.json`):
92.2% top-1 / 97.3% top-4 at the last epoch (best 93.2%). Compare against the number reported in the paper
before quoting it.

## Caveats
- D1 is headset data. The webcam conditions are simulated degradations, not real webcam traces; D3 is needed.
- Neural LOSO models are trained for 100 epochs (the original trains for 200); one seed per model.
