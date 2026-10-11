# Upload the bundle as a private Kaggle dataset and push runs as private script kernels.
#   python tools/kaggle/push.py bundle            # create or version $KAGGLE_USERNAME/webswipe-bundle
#   python tools/kaggle/push.py smoke             # GPU kernel webswipe-smoke
#   python tools/kaggle/push.py train:0           # GPU kernel webswipe-train-0
#   python tools/kaggle/push.py eval:0 --cpu --after train:0
# Status / outputs: kaggle kernels status <id>; kaggle kernels output <id> -p <dir>
import argparse
import os
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "kaggle_build"
KAGGLE = str(ROOT / ".venv/bin/kaggle")
USER = os.environ["KAGGLE_USERNAME"]  # same variable the kaggle CLI reads
DATASET = f"{USER}/webswipe-bundle"


def slug(run):
    return "webswipe-" + run.replace(":", "-").replace(",", "-")


def bundle():
    subprocess.run(["bash", str(ROOT / "tools/make_colab_bundle.sh")], check=True, cwd=ROOT)
    d = BUILD / "bundle"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "colab_bundle.zip", d / "colab_bundle.zip")
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": "webswipe-bundle", "id": DATASET, "licenses": [{"name": "other"}]}, indent=1))
    exists = subprocess.run([KAGGLE, "datasets", "status", DATASET], capture_output=True, text=True)
    if exists.returncode == 0 and "ready" in exists.stdout.lower():
        cmd = [KAGGLE, "datasets", "version", "-p", str(d), "-m", "update bundle"]
    else:
        cmd = [KAGGLE, "datasets", "create", "-p", str(d)]
    subprocess.run(cmd, check=True)


def kernel(run, cpu, after):
    d = BUILD / slug(run)
    d.mkdir(parents=True, exist_ok=True)
    src = (ROOT / "tools/kaggle/runner.py").read_text().replace('"__RUN__"', json.dumps(run))
    (d / "runner.py").write_text(src)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{slug(run)}", "title": slug(run), "code_file": "runner.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": not cpu, "enable_internet": True,
        "dataset_sources": [DATASET], "kernel_sources": [f"{USER}/{slug(r)}" for r in after],
        "competition_sources": []}, indent=1))
    cmd = [KAGGLE, "kernels", "push", "-p", str(d)]
    if not cpu:
        cmd += ["--accelerator", "NvidiaTeslaT4"]
    subprocess.run(cmd, check=True)
    print(f"{USER}/{slug(run)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--after", nargs="*", default=[], help="runs whose outputs this kernel reads")
    a = ap.parse_args()
    bundle() if a.run == "bundle" else kernel(a.run, a.cpu, a.after)
