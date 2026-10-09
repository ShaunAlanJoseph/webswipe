# Kaggle CPU kernel: copy the HaGRID images the vision runner uses into one zip.
# GPU workers fail to mount the public HaGRID sample (ERRORED_MOUNTING_DATASET), so the subset is
# re-uploaded inside our own private dataset (push_vision.py frames picks up data/hagrid_subset.zip).
import glob
import json
import os
import random
import zipfile

INPUT, OUT = "/kaggle/input", "/kaggle/working"
# same classes and counts as vision_runner.load_hagrid: train + val per class
PER_CLASS = {c: 500 for c in ("ok", "palm", "stop", "fist")}
PER_CLASS.update({c: 130 for c in ("call", "dislike", "like", "mute", "one", "peace", "peace_inverted", "rock",
                                   "three", "three2", "two_up", "two_up_inverted")})

rng = random.Random(0)
ann = {os.path.basename(a)[:-5]: a for a in glob.glob(f"{INPUT}/**/ann_train_val/*.json", recursive=True)}
n = 0
with zipfile.ZipFile(f"{OUT}/hagrid_subset.zip", "w", zipfile.ZIP_STORED) as z:
    for cls, k in PER_CLASS.items():
        files = sorted(glob.glob(f"{INPUT}/**/train_val_{cls}/*.jpg", recursive=True))
        rng.shuffle(files)
        recs = json.load(open(ann[cls]))
        keep = {}
        for f in files:
            iid = os.path.basename(f)[:-4]
            if iid in recs:
                z.write(f, f"hagrid_subset/train_val_{cls}/{iid}.jpg")
                keep[iid] = recs[iid]
            if len(keep) == k:
                break
        z.writestr(f"hagrid_subset/ann_train_val/{cls}.json", json.dumps(keep))
        n += len(keep)
        print(cls, len(keep), flush=True)
print("images", n, "zip MB", round(os.path.getsize(f"{OUT}/hagrid_subset.zip") / 1e6), flush=True)
