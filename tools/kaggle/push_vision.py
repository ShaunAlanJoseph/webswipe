# Upload labelled webcam frames as a private Kaggle dataset and push vision fine-tuning runs.
#   python tools/kaggle/push_vision.py frames s1 s2          # create or version shauniiitk/webswipe-frames
#   python tools/kaggle/push_vision.py run smoke --cpu --smoke --folds A:s1:s2
#   python tools/kaggle/push_vision.py run xloc --folds A:s1:s2 B:s2:s1 --imgsz 320
# Fold syntax: name:train_sessions:test_sessions, sessions joined with '+'; empty test = final model (HaGRID val only).
# Status / outputs: kaggle kernels status <id>; kaggle kernels output <id> -p <dir>
import argparse
import os
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "kaggle_build"
KAGGLE = str(ROOT / ".venv/bin/kaggle")
# another account (e.g. a teammate's quota): KAGGLE_CONFIG_DIR=<dir with their access token> KAGGLE_USER=<name>
USER = os.environ.get("KAGGLE_USER", "shauniiitk")
DATASET = f"{USER}/webswipe-frames"
HAGRID = "innominate817/hagrid-sample-30k-384p"
BASE = ROOT / "models/hagrid/YOLOv10n_gestures.pt"


def frames(sessions):
    d = BUILD / "frames"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for s in sessions:
        src = ROOT / "data/frames" / s
        recs = [json.loads(line) for line in (src / "labels.jsonl").read_text().splitlines()]
        with zipfile.ZipFile(d / f"frames_{s}.zip", "w", zipfile.ZIP_STORED) as z:
            z.write(src / "labels.jsonl", f"{s}/labels.jsonl")
            for r in recs:
                z.write(src / r["file"], f"{s}/{r['file']}")
        print(f"{s}: {len(recs)} frames, {(d / f'frames_{s}.zip').stat().st_size / 1e6:.0f} MB")
    shutil.copy(BASE, d / BASE.name)
    if (ROOT / "data/hagrid_subset.zip").exists():  # from prep_hagrid.py; lets runs skip the public dataset
        shutil.copy(ROOT / "data/hagrid_subset.zip", d / "hagrid_subset.zip")
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": "webswipe-frames", "id": DATASET, "licenses": [{"name": "other"}]}, indent=1))
    exists = subprocess.run([KAGGLE, "datasets", "status", DATASET], capture_output=True, text=True)
    if exists.returncode == 0 and "ready" in exists.stdout.lower():
        cmd = [KAGGLE, "datasets", "version", "-p", str(d), "-m", "frames: " + " ".join(sessions)]
    else:
        cmd = [KAGGLE, "datasets", "create", "-p", str(d)]
    subprocess.run(cmd, check=True)


def run(a):
    folds = []
    for f in a.folds:
        name, tr, te = f.split(":")
        folds.append({"name": name, "train": tr.split("+"), "test": [t for t in te.split("+") if t]})
    cfg = {"folds": folds, "mixes": a.mixes, "imgsz": a.imgsz, "epochs": a.epochs, "seed": a.seed, "smoke": a.smoke,
           "max_hours": a.max_hours}
    slug = f"webswipe-vision-{a.name}"
    d = BUILD / slug
    d.mkdir(parents=True, exist_ok=True)
    src = (ROOT / "tools/kaggle/vision_runner.py").read_text().replace('"__CFG__"', json.dumps(json.dumps(cfg)))
    (d / "vision_runner.py").write_text(src)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{slug}", "title": slug, "code_file": "vision_runner.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": not a.cpu, "enable_internet": True,
        "dataset_sources": [DATASET] + ([] if a.no_hagrid_mount else [HAGRID]), "kernel_sources": [], "competition_sources": []}, indent=1))
    cmd = [KAGGLE, "kernels", "push", "-p", str(d)]
    if not a.cpu:
        cmd += ["--accelerator", "NvidiaTeslaT4"]
    subprocess.run(cmd, check=True)
    print(f"{USER}/{slug}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frames")
    f.add_argument("sessions", nargs="+")
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--folds", nargs="+", required=True)
    r.add_argument("--mixes", nargs="+", default=["own+hagrid", "own"])
    r.add_argument("--imgsz", nargs="+", type=int, default=[320])
    r.add_argument("--epochs", type=int, default=40)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--cpu", action="store_true")
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--max-hours", type=float, default=2.0, help="skip trainings after this many hours; watchdog kills the kernel 15 min later")
    r.add_argument("--no-hagrid-mount", action="store_true", help="use hagrid_subset.zip from our dataset instead")
    a = ap.parse_args()
    frames(a.sessions) if a.cmd == "frames" else run(a)
