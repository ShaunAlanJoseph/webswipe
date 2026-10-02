# SHARK2 baseline on the Shen et al. mid-air data (D1), same lexicon and test subject as the neural decoder
import argparse
import json
from pathlib import Path

import numpy as np

from ..baselines.shark2 import Shark2, shen_keyboard
from ..decoder.shen_data import load_words

SIGMA_SHAPE = [0.04, 0.08, 0.16, 0.32, 0.64, 1.28, np.inf]
SIGMA_LOC = [0.25, 0.5, 1.0, np.inf]


def dist_table(dec, recs):
    out = []
    for r in recs:
        key, kb = shen_keyboard(r["candidate"])
        out.append(dec.distances(r["coords"], kb, key))
    return out


def accuracy(dec, recs, dists, ss, sl, ks=(1, 4)):
    hits = np.zeros(len(ks))
    for r, (ds, dl) in zip(recs, dists):
        order = np.argsort(-Shark2.logp(ds, dl, ss, sl))[:max(ks)]
        top = [dec.words[i] for i in order]
        hits += [r["word"] in top[:k] for k in ks]
    return (hits / len(recs)).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/shen_midair.pkl")
    ap.add_argument("--test-subject", type=int, default=0)
    ap.add_argument("--n-tune", type=int, default=1500)
    ap.add_argument("--out", default="results/shark2_d1.json")
    a = ap.parse_args()

    recs = load_words(a.pkl)
    words = sorted({r["word"] for r in recs})  # same closed lexicon as train.py
    rng = np.random.default_rng(0)
    train = [r for r in recs if r["subject"] != a.test_subject]
    test = [r for r in recs if r["subject"] == a.test_subject]
    tune = [train[i] for i in rng.choice(len(train), a.n_tune, replace=False)]
    print(f"lexicon {len(words)}  tune {len(tune)} (other subjects)  test {len(test)} (subject {a.test_subject})")

    res = {"lexicon": len(words), "test_subject": a.test_subject, "n_test": len(test)}
    for norm, smooth in [("bbox", False), ("rms", False), ("rms", True)]:
        dec = Shark2(words, norm=norm, smooth=smooth)
        dt = dist_table(dec, tune)
        grid = {(ss, sl): accuracy(dec, tune, dt, ss, sl)[0] for ss in SIGMA_SHAPE for sl in SIGMA_LOC
                if not (ss == np.inf and sl == np.inf)}
        ss, sl = max(grid, key=grid.get)
        dtest = dist_table(dec, test)
        row = {"sigma_shape": ss, "sigma_loc": sl, "tune_top1": grid[(ss, sl)]}
        row["top1"], row["top4"] = accuracy(dec, test, dtest, ss, sl)
        # single channels on the test subject, each with its tuned sigma
        row["shape_only_top1"] = accuracy(dec, test, dtest, ss if ss != np.inf else 0.08, np.inf)[0]
        row["location_only_top1"] = accuracy(dec, test, dtest, np.inf, sl if sl != np.inf else 1.0)[0]
        name = norm + ("+smooth" if smooth else "")
        res[name] = row
        print(name, json.dumps(row, default=str), flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
