# WebSwipe — Decoder simulation experiments (progress, Sep 25)

Code: `experiments/run_experiments.py` (numpy, runs in ~15 s). Figures: `experiments/results/`.
**All experiments are synthetic**: ideal gestures on a QWERTY layout (units = key widths), corrupted with
i.i.d. isotropic Gaussian noise standing in for monocular landmark error. Lexicon: 3,000 words. No webcam data yet.

| Exp | Question | Figure |
|---|---|---|
| E1 | Word accuracy vs landmark noise: word-gesture vs per-key decoding | `e1_accuracy_vs_noise.png` |
| E2 | Is the shape channel invariant to user–camera distance (30–90 cm)? | `e2_distance_invariance.png` |
| E3 | Is the noise offset common to all candidates, and does ranking variance fall as O(1/N)? | `e3_offset_vs_N.png` |
| E4 | Decoder latency vs lexicon size | `e4_latency.png` |

## Findings
- **E1:** raw per-key typing collapses under noise (87% → 35% → 9% at σ = 0.2 / 0.3 / 0.4 key widths). The word-gesture
  shape decoder degrades gradually (91% / 89% / 82%). **But** per-key decoding *with lexicon correction* stays ≥ 97% up to σ = 0.6,
  better than the shape-only channel. The shape channel is also ambiguous without noise (99.5% at σ = 0), because
  translation-invariant shapes collide. → Our claim that "per-key admits no comparable averaging" only holds for
  per-key typing *without* a lexicon; the decoder needs the location channel (SHARK2-style) as well as shape.
- **E2:** with no noise, accuracy is 100% at every distance (scale and translation invariance confirmed). With fixed
  image-space noise, accuracy drops from 99% (30 cm) to about 93% (90 cm): normalisation removes scale but not the loss of
  signal-to-noise ratio as the hand gets smaller in the frame.
- **E3:** the variance of the ranking margin falls with N, but with a log–log slope of **≈ −0.64, not −1**. The mean
  offset drifts and differs between candidates as N grows. Likely cause: bounding-box scale normalisation
  (max − min) grows with noise and with the number of samples, which adds a systematic bias. → Proposed fix: normalise by RMS
  radius, then re-test the O(1/N) claim.
- **E4:** the brute-force decoder takes 0.3 ms/word (500 words) to 6.3 ms/word (10k words) on CPU, well within a real-time budget.

## Next steps
1. Add a location channel and a language-model prior, then re-run E1 against per-key + lexicon.
2. Swap bbox normalisation for RMS normalisation, then re-run E3 and revise the proof in Sec. 3.5 to match.
3. Measure real MediaPipe landmark noise (σ in key widths) from webcam recordings to find the operating point on E1.

---

# Batch 2 (E5–E10): `run_experiments_2.py` (~3 min) and `e10_median.py`
Still **synthetic**. Tuned values (decoder weight λ, pinch thresholds) were chosen on separate validation data, not on the test set.

| Exp | Question | Figure |
|---|---|---|
| E5 | Does adding a location channel (SHARK2-style) fix the E1 weakness? | `e5_shark2_vs_perkey.png` |
| E6 | Does RMS normalisation restore the O(1/N) claim? | `e6_normalisation_O1N.png` |
| E7 | Robustness to hand drift, speed-dependent blur noise, and pinch timing error | `e7_noise_ladder.png` |
| E8 | Ablation: lexicon size (1k–43k) × resample count N | `e8_lexicon_N_ablation.png` |
| E9 | Raw vs EMA vs Kalman vs One Euro: jitter–lag trade-off, and accuracy after filtering | `e9_filters.png` |
| E10 | Pinch detection: fixed px threshold vs scale-normalised vs online GMM, over distance and lighting | `e10_pinch_detection.png`, `e10b_pinch_detection_median.png` |

## Findings
- **E5:** shape + location (λ = 0.3 chosen on validation) raises top-1 accuracy over shape-only at every noise level
  (0.85 → 0.90 at σ = 0.4; 0.57 → 0.81 at σ = 1.2). Per-key + lexicon still wins up to σ ≈ 0.8. **Above σ ≈ 1.0 the
  gesture decoder wins** (0.81 vs 0.76 at σ = 1.2), which supports the "advantage in the low-fidelity regime" claim, but only
  at high noise. The E1 + E5 crossover is the honest version of contribution 2.
- **E6:** RMS normalisation gives a log–log slope of **−0.98** (bbox: −0.66), so **O(1/N) holds with RMS normalisation.**
  The offset is still not identical across candidates (spread ≈ 0.02), so the "common offset" part of the proof needs
  weakening to "approximately common". Shape top-1 accuracy is also slightly better with RMS at σ ≥ 0.4.
  → Switch the decoder to RMS normalisation and restate the proof.
- **E7:** starting from 0.94 (i.i.d. noise only), accuracy drops with hand drift (0.90), pinch timing error (0.90) and
  speed-dependent noise (0.85, the most harmful). All combined: 0.82 with shape + location vs 0.70 with shape only.
  Motion blur is the main thing to design against.
- **E8:** accuracy falls with lexicon size (0.98 at 1k → 0.84 at 43k, σ = 0.3). **N barely matters above 16–32**, so N = 32 is enough.
  Lexicon size, not resampling, is the bottleneck; a language-model prior is the next lever.
- **E9:** at a lag of 50 ms or less, One Euro gives the lowest jitter at rest (0.073 vs 0.114 Kalman, 0.116 EMA, 0.205 raw), and
  sits below the other filters' trade-off curves. Filtering before decoding raises top-1 accuracy from 0.79 (raw) to 0.93–0.95. Fills TBD result (c).
- **E10:** the first run (no temporal filtering) gave low F1 for all detectors, because single-frame outliers split pinches
  into fragments. That was a simulator/pipeline flaw, not a detector property. With a 5-frame median filter applied equally
  to all detectors (`e10b`): in normal light, all three reach F1 ≥ 0.96 up to 70 cm, and at 90 cm **online GMM 0.96 vs
  normalised 0.88 vs fixed px 0.67**. In low light, GMM is best at 30–50 cm (0.99 / 0.97), but **every detector degrades at 70–90 cm**
  (GMM 0.74 / 0.51). Supports contribution 3 at normal range; low light at distance remains an open problem.
  The signal model (noise levels, outlier rate) is assumed, not measured. Webcam recordings are needed.

## Updated next steps
1. Measure real MediaPipe noise σ (key widths) and the pinch-distance distribution from webcam sessions, then place the operating point on E1/E5/E10.
2. Adopt RMS normalisation + shape and location channels + One Euro + median-filtered GMM pinch detection as the default pipeline.
3. Add a word-frequency or language-model prior (E8 shows lexicon size is the bottleneck).

---

# E1 across keyboard layouts (`e1_layouts.py`, ~30 s)
Same 300 test words, noise levels and seed for each layout. Figure: `results/e1_layouts.png`.

Word-gesture top-1 accuracy:

| Layout | σ=0.2 | σ=0.4 | σ=0.6 | σ=1.0 |
|---|---|---|---|---|
| QWERTY | 0.95 | 0.84 | 0.69 | 0.60 |
| Alphabetical | 0.94 | 0.86 | 0.71 | 0.59 |
| Colemak | 0.92 | 0.74 | 0.63 | 0.48 |
| Workman | 0.92 | 0.72 | 0.65 | 0.53 |
| Dvorak | 0.88 | 0.76 | 0.57 | 0.50 |

- Gesture decoding is about 0.1 worse on Dvorak, Colemak and Workman than on QWERTY or alphabetical at σ = 0.4–1.0.
  These layouts put common letters on the home row, so word paths are shorter and look more alike. This matches
  earlier findings that typing-optimised layouts are not good for gesture typing.
- Per-key accuracy barely depends on layout (raw: 0.07–0.11 at σ = 0.4; with lexicon: 0.99–1.00), because every layout uses the same key grid.
- With 300 words the standard error is about ±0.025, so gaps under ~0.05 (e.g. QWERTY vs alphabetical) are not meaningful.
