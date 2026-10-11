# Training the final models
These scripts produced the three trained models the demo uses. Each one is the script that ran on a Kaggle GPU (or CPU) worker, with paths turned into command-line arguments; training logic, hyperparameters, seeds and selection rules are unchanged. The model files in `models/` are byte-identical (sha256) to the outputs of these runs. Every script documents its inputs, outputs and pinned package versions in its docstring and `--help`.
## Model lineage
```
F_own_hagrid_320.pt        detector fine-tuned on our s1-s3 frames + HaGRID (tools/kaggle/vision_runner.py, 9 Oct)
  -> train_gesture_continued.py  -> gesture_candidate_320.pt            (10 Oct 03:49)
     -> train_gesture_crop.py    -> gesture_crop_20261010.pt / .onnx     (10 Oct 17:29)
        -> train_gesture_lighting.py -> gesture_lighting_20261010.onnx   (10 Oct 22:08)
d1_loso0_aug_last.pt       decoder trained on D1 with subject 0 held out (webswipe.decoder.train, 4 Oct)
  -> personalize_decoder.py      -> decoder_personalized_20261010.pt     (10 Oct 18:12)
```
| Script | What it does |
|---|---|
| `train_gesture_continued.py` | Continues detector fine-tuning on the 10 October frame sessions, one candidate per learning rate; the `lr0p0003` candidate was selected |
| `prepare_adaptation_frames.py` | Cuts labelled frames out of the sitting and standing benchmark videos for detector adaptation |
| `train_gesture_crop.py` | Fine-tunes the detector on square hand crops, so it sees fingertip detail at 320 px |
| `train_gesture_lighting.py` | Adapts the crop detector to a difficult-lighting recording |
| `personalize_decoder.py` | Fine-tunes the D1 decoder on one user's webcam swipes |
| `replay_left_pinch.py` | Caches hand tracking and detector scores for a recorded session |
| `score_replay.py` | Replays the left-hand controls on cached rows and scores each prompted quick or held pinch |

The detector scripts need a CUDA GPU and `ultralytics`; the decoder script runs on a GPU or, slowly, on the CPU. Outputs go to `runs/` by default.
## Inputs
Defaults are relative to the repository root. The data folders are local only (see `data/README.md`):
- `data/detector_frames/`: labelled frame sessions `frames_s1`-`s3.zip`, the three 10 October single-user sessions with their `manifest.json`, and `hagrid_subset.zip`.
- `data/benchmark_20261010/`: the sitting and standing benchmark recordings, with `physical_boundaries_dev.json`.
- `data/detector_adaptation/`: adaptation labels, images and `adaptation_cached_boxes.json`.
- `data/detector_lighting/`: the difficult-lighting labels and samples.
- `data/decoder_adaptation/`: the personalisation payload of one user's swipes.
- `data/left_quick_pinches/`: ten recorded quick left-hand pinches.
- Intermediate weights in `models/` (not tracked): `F_own_hagrid_320.pt`, `gesture_candidate_320.pt`, `gesture_crop_20261010.pt`, `d1_loso0_aug_last.pt`.

Two inputs were produced by one-off commands that were not kept: the cached hand boxes in `adaptation_cached_boxes.json`, and the box labels in the lighting set. The decoder payload was built from replays of the benchmark recordings by a script that is also not included. Their outputs are in the data folders above, so every training step can still be rerun.
