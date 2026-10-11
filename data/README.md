# Data
Only three small files in this folder are tracked: this README, `lexicon_10k.txt` (the 10,000-word vocabulary the demo decodes into) and `phrases.txt` (the phrase list used for recording sessions). Everything else is raw or derived data. It stays out of git because it is large, because the webcam frames and videos show the participants' faces, and because the Shen et al. data has no licence for redistribution.

| Path | What it is | How to get it |
|---|---|---|
| `shen_midair.pkl` | Shen et al. HoloLens 2 mid-air traces: 9 subjects, 26,071 words (dataset D1) | `data/data_all.pkl` from github.com/CambridgeIIS/mid_air_gesture_keyboard_decoder |
| `shen_features.npz` | D1 converted to decoder features | `python -m webswipe.decoder.shen_data` |
| `hagrid_subset.zip` | 3,576 HaGRIDv2 images mixed into detector fine-tuning (CC BY-SA 4.0) | Built from the public HaGRID sample by the Kaggle kernel `tools/kaggle/prep_hagrid.py` |
| `frames/` | Labelled webcam frames for detector fine-tuning; sessions `s1`–`s3` are three locations, the others are later single-user sessions | `tools/capture_frames.py`, then `tools/label_frames.py` |
| `guided/` | Guided word recordings with video, landmarks and per-frame pinch decisions | `python -m webswipe.app.main --mode guided` |
| `diagnostic/` | Word-boundary and pinch-calibration recordings | `python -m webswipe.app.diagnostic` |
| `sessions/` | Typing event logs from the live app (`--log`) | Written by the app |
| `recordings/` | Teammate recordings made with `tools/recording_kit/`, plus the low-light (`lighting/`) and pinch-control (`pinch/`) recordings | Recorded with the kit or the app |
| `detector_frames/`, `benchmark_20261010/`, `detector_adaptation/`, `detector_lighting/`, `decoder_adaptation/`, `left_quick_pinches/` | Inputs of the scripts in `training/` that produced the final models | See `training/README.md` |
