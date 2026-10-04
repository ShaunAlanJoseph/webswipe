# Kaggle kernel script: same steps as notebooks/train_decoder_colab.ipynb, one run per kernel.
# tools/kaggle/push.py fills in RUN and uploads this as a private script kernel.
#   smoke          1 tiny epoch + tiny evaluations
#   train:0,1      clean + aug models, leave-one-subject-out, for each listed D1 subject (GPU)
#   repro          Shen et al. reproduction: random 95/5 split, 200 epochs (GPU)
#   eval:0,1       SHARK2 tuning, robustness, word-frequency prior, plot, per subject (CPU; checkpoints
#                  come from the train kernels attached as inputs)
import glob
import json
import os
import subprocess
import sys
import time

RUN = os.environ.get("WEBSWIPE_RUN", "__RUN__")
INPUT = os.environ.get("WEBSWIPE_INPUT", "/kaggle/input")
OUT = os.environ.get("WEBSWIPE_OUT", "/kaggle/working/out")
WS = os.environ.get("WEBSWIPE_WS", "/tmp/ws")
EPOCHS = int(os.environ.get("WEBSWIPE_EPOCHS", 100))


def sh(cmd, log=None):
    print(f"$ {cmd}", flush=True)
    t0 = time.time()
    with open(log, "a") if log else open(os.devnull, "w") as f:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            f.write(line); f.flush()
    if p.wait():
        raise SystemExit(f"failed ({p.returncode}): {cmd}")
    print(f"  [{time.time() - t0:.0f} s]", flush=True)


def sh_parallel(jobs):
    """(cmd, log) pairs, one per GPU (Kaggle 'GPU' = 2x T4); falls back to sequential on fewer GPUs."""
    import torch
    if torch.cuda.device_count() < len(jobs):
        for cmd, log in jobs:
            sh(cmd, log)
        return
    t0, procs = time.time(), []
    for gpu, (cmd, log) in enumerate(jobs):
        print(f"$ [gpu {gpu}] {cmd}", flush=True)
        procs.append(subprocess.Popen(cmd, shell=True, stdout=open(log, "a"), stderr=subprocess.STDOUT,
                                      env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}))
    codes = [p.wait() for p in procs]
    for cmd, log in jobs:
        print(f"--- {log}\n" + open(log).read(), flush=True)
    print(f"  [{time.time() - t0:.0f} s]", flush=True)
    if any(codes):
        raise SystemExit(f"failed {codes}: {[c for c, _ in jobs]}")


def setup():
    sh(f"rm -rf {WS} && mkdir -p {WS}/data {OUT}/checkpoints {OUT}/results {OUT}/logs")
    zips = glob.glob(f"{INPUT}/**/colab_bundle.zip", recursive=True)
    if zips:
        sh(f"unzip -q {zips[0]} -d {WS}")
    else:  # Kaggle unpacked the zip into a folder
        pkg = glob.glob(f"{INPUT}/**/webswipe/__init__.py", recursive=True)
        assert pkg, "bundle not found: attach the webswipe-bundle dataset"
        sh(f"cp -r {os.path.dirname(os.path.dirname(pkg[0]))}/* {WS}/")
    sh(f"mv {WS}/*.npz {WS}/shen_midair.pkl {WS}/data/")
    os.chdir(WS)
    sh("pip -q install wordfreq 2>&1 | tail -2")
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv || echo 'no GPU'")
    sh(f"python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())'")


def ckpt(name):
    hits = glob.glob(f"{OUT}/checkpoints/{name}") + glob.glob(f"{INPUT}/**/checkpoints/{name}", recursive=True)
    assert hits, f"checkpoint {name} not found: attach the train kernel as an input"
    return hits[0]


def subjects(spec):
    return [int(s) for s in spec.split(",")]


def smoke():
    log = f"{OUT}/logs/smoke.log"
    sh("python -u -m webswipe.decoder.train --split subject --test-subject 0:0 --epochs 1 --max-train 256 "
       "--aug-prob 0.5 --out /tmp/smoke.pt", log)
    sh("python -m webswipe.eval.robustness_d1 --resample-hz 50 --max-test 20 --ckpt smoke=/tmp/smoke_last.pt "
       f"--out {OUT}/results/smoke_rob.json", log)
    sh("python -m webswipe.eval.lm_prior_d1 --resample-hz 50 --max-test 20 --n-tune 50 "
       f"--ckpt smoke=/tmp/smoke_last.pt --out {OUT}/results/smoke_lm.json", log)


def train(spec):
    for s in subjects(spec):
        every = 10 if s == 0 else 50
        sh_parallel([(f"python -u -m webswipe.decoder.train --split subject --test-subject 0:{s} --epochs {EPOCHS} "
                      f"--eval-every {every} --aug-prob {p} --out {OUT}/checkpoints/d1_loso{s}_{name}.pt",
                      f"{OUT}/logs/d1_loso{s}_{name}.log") for name, p in [("clean", 0.0), ("aug", 0.5)]])


def repro():
    sh(f"python -u -m webswipe.decoder.train --epochs 200 --eval-every 10 --out {OUT}/checkpoints/shen_d1_random.pt",
       f"{OUT}/logs/shen_d1_random.log")


def evaluate(spec):
    for s in subjects(spec):
        log, res = f"{OUT}/logs/eval_s{s}.log", f"{OUT}/results"
        # SHARK2 sigmas tuned on the 8 other subjects for this held-out subject
        sh(f"python -m webswipe.eval.shark2_d1 --test-subject {s} --out {res}/shark2_d1_s{s}.json", log)
        sig = json.load(open(f"{res}/shark2_d1_s{s}.json"))["rms"]
        ck = f"clean={ckpt(f'd1_loso{s}_clean_last.pt')} aug={ckpt(f'd1_loso{s}_aug_last.pt')}"
        sh(f"python -m webswipe.eval.robustness_d1 --test-subject {s} --resample-hz 50 "
           f"--shark2-sigma {sig['sigma_shape']} {sig['sigma_loc']} --ckpt {ck} --out {res}/robustness_d1_s{s}.json",
           log)
        sh(f"python -m webswipe.eval.lm_prior_d1 --test-subject {s} --resample-hz 50 --ckpt {ck} "
           f"--out {res}/lm_prior_d1_s{s}.json", log)
        sh(f"python -m webswipe.eval.plot_robustness --json {res}/robustness_d1_s{s}.json "
           f"--out {res}/robustness_d1_s{s}.png", log)


if __name__ == "__main__":
    kind, _, arg = RUN.partition(":")
    print("run:", RUN, flush=True)
    setup()
    {"smoke": lambda _: smoke(), "train": train, "repro": lambda _: repro(), "eval": evaluate}[kind](arg)
    print("done:", RUN, flush=True)
