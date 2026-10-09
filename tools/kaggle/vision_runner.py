# Kaggle kernel script: fine-tune the released HaGRID YOLOv10n gesture detector on our webcam frames.
# tools/kaggle/push_vision.py fills in CFG and uploads this as a private script kernel.
# Classes keep their HaGRID names so the pretrained class weights carry over:
#   ok = pinch, palm = open hand, fist, no_gesture = any other hand.
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile

CFG = json.loads(os.environ.get("VISION_CFG", "__CFG__"))
# CFG: {"folds": [{"name": "A", "train": ["s1"], "test": ["s2"]}], "mixes": ["own+hagrid", "own"],
#       "imgsz": [320], "epochs": 40, "seed": 0, "smoke": false}
INPUT, OUT, WS = "/kaggle/input", "/kaggle/working/out", "/tmp/ws"
CLASSES = ["ok", "palm", "fist", "no_gesture"]
STATES = ["ok", "palm", "fist"]                      # what the keyboard reacts to; anything else is "none"
SEG_STATE = {"pinch": "ok", "open": "palm", "palm": "palm", "fist": "fist", "none": "none"}
# released 34-class model -> our states (generous: every pinch-like / open-hand class counts)
BASE_MAP = {"ok": "ok", "grip": "ok", "palm": "palm", "stop": "palm", "four": "palm", "fist": "fist"}
# HaGRID sample images used for mixing: gesture class -> our class; classes not listed are left out
HAGRID_MAP = {"ok": "ok", "palm": "palm", "stop": "palm", "fist": "fist", "call": "no_gesture",
              "dislike": "no_gesture", "like": "no_gesture", "mute": "no_gesture", "one": "no_gesture",
              "peace": "no_gesture", "peace_inverted": "no_gesture", "rock": "no_gesture", "three": "no_gesture",
              "three2": "no_gesture", "two_up": "no_gesture", "two_up_inverted": "no_gesture"}
SMOKE = CFG.get("smoke", False)


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def find(pattern):
    return sorted(glob.glob(f"{INPUT}/**/{pattern}", recursive=True) + glob.glob(f"{WS}/hagrid/**/{pattern}", recursive=True))


# ---------------------------------------------------------------- data
def load_sessions():
    """{session: [record]} with absolute image paths; frames come zipped or already unpacked."""
    for z in find("frames_*.zip"):
        zipfile.ZipFile(z).extractall(f"{WS}/frames")
    out = {}
    for lab in find("labels.jsonl") + sorted(glob.glob(f"{WS}/frames/**/labels.jsonl", recursive=True)):
        root = os.path.dirname(lab)
        recs = [json.loads(line) for line in open(lab)]
        for r in recs:
            r["path"] = f"{root}/{r['file']}"
            r["state"] = SEG_STATE[r["seg"]]
        out[os.path.basename(root)] = [r for r in recs if os.path.exists(r["path"])]
    return out


def load_hagrid(rng):
    """(train, val) records from the HaGRID 30k sample, mapped to our four classes."""
    for z in find("hagrid_subset.zip"):  # our re-upload of the subset (the public dataset fails to mount on GPU)
        zipfile.ZipFile(z).extractall(f"{WS}/hagrid")
    ann = {os.path.basename(a)[:-5]: a for a in find("ann_train_val/*.json")}
    train, val = [], []
    for cls, ours in HAGRID_MAP.items():
        key = ours in STATES
        n_tr, n_va = (400, 100) if key else (100, 30)
        if SMOKE:
            n_tr, n_va = 12, 6
        files = find(f"train_val_{cls}/*.jpg")
        rng.shuffle(files)
        recs_ann = json.load(open(ann[cls]))
        for i, f in enumerate(files[:n_tr + n_va]):
            a = recs_ann.get(os.path.basename(f)[:-4])
            if a is None:
                continue
            boxes = []
            for (x, y, w, h), lab in zip(a["bboxes"], a["labels"]):
                c = HAGRID_MAP.get(lab, "no_gesture")
                boxes.append([CLASSES.index(c), x + w / 2, y + h / 2, w, h])
            rec = {"path": f, "boxes": boxes, "state": ours if key else "none", "seg": f"hagrid_{cls}"}
            (val if i < n_va else train).append(rec)
    return train, val


def write_yolo(recs, root, split):
    os.makedirs(f"{root}/images/{split}", exist_ok=True)
    os.makedirs(f"{root}/labels/{split}", exist_ok=True)
    for i, r in enumerate(recs):
        ext = os.path.splitext(r["path"])[1]
        shutil.copy(r["path"], f"{root}/images/{split}/{i:06d}{ext}")
        with open(f"{root}/labels/{split}/{i:06d}.txt", "w") as f:
            f.write("".join(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n" for b in r["boxes"]))


def make_dataset(name, train, val):
    root = f"{WS}/ds_{name}"
    shutil.rmtree(root, ignore_errors=True)
    write_yolo(train, root, "train")
    write_yolo(val, root, "val")
    with open(f"{root}/data.yaml", "w") as f:
        f.write(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n")
        f.write("".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
    return f"{root}/data.yaml"


# ---------------------------------------------------------------- evaluation
def frame_scores(model, recs, imgsz, cmap):
    """Per frame: {state: best confidence}, plus the best box per state. cmap: model class name -> state."""
    out = []
    names = model.names
    for i in range(0, len(recs), 64):
        res = model.predict([r["path"] for r in recs[i:i + 64]], imgsz=imgsz, conf=0.05, verbose=False)
        for r in res:
            sc, bx = {s: 0.0 for s in STATES}, {}
            for c, p, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xywhn.tolist()):
                s = cmap.get(names[int(c)])
                if s and p > sc[s]:
                    sc[s], bx[s] = p, b
            out.append((sc, bx))
    return out


def iou(a, b):
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def average_precision(scores, truth):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = fp = 0
    ap = prev_recall = 0.0
    pos = sum(truth)
    if not pos:
        return None
    for i in order:
        tp, fp = tp + truth[i], fp + (not truth[i])
        if truth[i]:
            ap += tp / (tp + fp) * (tp / pos - prev_recall)
            prev_recall = tp / pos
    return ap


def metrics(recs, fs, thr):
    """Frame-level results: what state the keyboard would see in each frame."""
    labels = STATES + ["none"]
    conf = {g: {p: 0 for p in labels} for g in labels}
    ious = {s: [] for s in STATES}
    for r, (sc, bx) in zip(recs, fs):
        best = max(STATES, key=lambda s: sc[s])
        pred = best if sc[best] >= thr else "none"
        conf[r["state"]][pred] += 1
        if pred == r["state"] and pred in bx:
            gt = [b for b in r["boxes"] if CLASSES[int(b[0])] == pred]
            if gt:
                ious[pred].append(iou(bx[pred], gt[0][1:]))
    out = {"n": len(recs), "confusion": conf, "accuracy": sum(conf[s][s] for s in labels) / max(len(recs), 1)}
    for s in STATES:
        tp = conf[s][s]
        pred_s = sum(conf[g][s] for g in labels)
        true_s = sum(conf[s].values())
        p, rc = tp / max(pred_s, 1), tp / max(true_s, 1)
        out[s] = {"precision": p, "recall": rc, "f1": 2 * p * rc / max(p + rc, 1e-9), "frames": true_s,
                  "false_rate": (pred_s - tp) / max(len(recs) - true_s, 1),
                  "ap": average_precision([f[0][s] for f in fs], [r["state"] == s for r in recs]),
                  "mean_iou": sum(ious[s]) / len(ious[s]) if ious[s] else None}
    return out


def evaluate(model, sets, imgsz, cmap):
    out = {}
    for name, recs in sets.items():
        if not recs:
            continue
        t = time.time()
        fs = frame_scores(model, recs, imgsz, cmap)
        out[name] = {f"thr{thr}": metrics(recs, fs, thr) for thr in (0.25, 0.5)}
        hard = [i for i, r in enumerate(recs) if r.get("filled")]  # frames where MediaPipe lost the hand
        if hard:
            out[name]["mediapipe_lost_thr0.25"] = metrics([recs[i] for i in hard], [fs[i] for i in hard], 0.25)
        out[name]["sec"] = round(time.time() - t, 1)
    return out


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WS, exist_ok=True)
    sh(f"{sys.executable} -m pip install -q ultralytics onnx onnxslim onnxruntime")
    import torch
    from ultralytics import YOLO
    rng = random.Random(CFG.get("seed", 0))
    device = 0 if torch.cuda.is_available() else "cpu"
    base = (find("YOLOv10n_gestures.pt") or [None])[0]
    if base is None:
        base = f"{WS}/YOLOv10n_gestures.pt"
        sh(f"wget -q -O {base} https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/"
           "YOLOv10n_gestures.pt")
    sessions = load_sessions()
    hag_train, hag_val = load_hagrid(rng)
    rng.shuffle(hag_train)  # load_hagrid returns class by class; a fraction must sample every class
    if SMOKE:
        sessions = {k: v[::40] for k, v in sessions.items()}
    res = {"cfg": CFG, "device": str(device), "gpu": torch.cuda.get_device_name(0) if device == 0 else None,
           "sessions": {k: len(v) for k, v in sessions.items()}, "hagrid": [len(hag_train), len(hag_val)],
           "runs": {}}
    print(json.dumps(res), flush=True)
    own_map = {c: c for c in STATES}

    def save():
        json.dump(res, open(f"{OUT}/vision_results.json", "w"), indent=1)

    t_start = time.time()
    # watchdog: a hung or crawling run must not drain the GPU quota (one ran ~5 h on Oct 9)
    import threading
    limit = CFG.get("max_hours", 2.0) * 3600 + 900

    def kill():
        res["killed"] = f"watchdog after {limit / 3600:.2f} h"
        save()
        print("WATCHDOG: time limit reached, exiting", flush=True)
        os._exit(3)
    threading.Timer(limit, kill).start()
    for fold in CFG["folds"]:
        train_own = [r for s in fold["train"] for r in sessions[s]]
        test_sets = {"hagrid_val": hag_val}
        if fold["test"]:
            test_sets[f"own_{'+'.join(fold['test'])}"] = [r for s in fold["test"] for r in sessions[s]]
        for imgsz in CFG["imgsz"]:
            key = f"{fold['name']}/released/{imgsz}"
            res["runs"][key] = {"eval": evaluate(YOLO(base), test_sets, imgsz, BASE_MAP)}
            print(key, json.dumps(res["runs"][key])[:600], flush=True)
            save()
            for mix in CFG["mixes"]:
                key = f"{fold['name']}/{mix}/{imgsz}"
                if (time.time() - t_start) / 3600 > CFG.get("max_hours", 2.0):
                    res["runs"][key] = {"skipped": "max_hours reached"}
                    print(key, "skipped: max_hours reached", flush=True)
                    continue
                # "own+hagrid" = all HaGRID images, "own+hagrid25" = 25% of them
                frac = int(mix.split("hagrid")[1] or 100) / 100 if "hagrid" in mix else 0
                train = train_own + hag_train[:int(len(hag_train) * frac)]
                rng.shuffle(train)
                # validation during training uses HaGRID only, so the test session never guides model choice
                data = make_dataset(key.replace("/", "_").replace("+", "_"), train, hag_val)
                t = time.time()
                model = YOLO(base)
                model.train(data=data, epochs=1 if SMOKE else CFG["epochs"], imgsz=imgsz, batch=32, device=device,
                            workers=4, seed=CFG.get("seed", 0), project=f"{WS}/runs",
                            name=key.replace("/", "_").replace("+", "_"), exist_ok=True, plots=False,
                            degrees=10, scale=0.5, patience=1000, amp=device == 0, verbose=False)
                last = f"{WS}/runs/{key.replace('/', '_').replace('+', '_')}/weights/last.pt"
                tuned = YOLO(last)
                run = {"train_images": len(train), "train_sec": round(time.time() - t, 1),
                       "names": tuned.names, "eval": evaluate(tuned, test_sets, imgsz, own_map)}
                dst = f"{OUT}/{key.replace('/', '_').replace('+', '_')}"
                shutil.copy(last, dst + ".pt")
                onnx = YOLO(last).export(format="onnx", imgsz=imgsz, simplify=True)
                shutil.move(onnx, dst + ".onnx")
                res["runs"][key] = run
                print(key, json.dumps(run)[:900], flush=True)
                save()
    save()
    print("DONE", flush=True)
    os._exit(0)  # do not wait for the watchdog timer thread


if __name__ == "__main__":
    main()
