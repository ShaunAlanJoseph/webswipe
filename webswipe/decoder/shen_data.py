# load the Shen et al. mid-air dataset and cache features
import argparse

import numpy as np
import pandas as pd

from .features import LETTERS, Keyboard, featurize


def load_words(pkl):
    """One record per word gesture, same filtering as data_process.py."""
    df = pd.read_pickle(pkl)
    recs = []
    for _, row in df.iterrows():
        phrase, sugg = row["phrase"], row["word suggestions"]
        if len(phrase) != len(sugg):
            continue
        for word, coords, s in zip(phrase, row["lift on coords"], sugg):
            # original kept a gesture only if the target word was among the suggestions
            if word not in s or len(coords) < 4:
                continue
            word = word.lower()
            if not word or any(ch not in LETTERS for ch in word):
                continue  # original mapped apostrophes etc. to 'a'; we drop them
            recs.append({"subject": int(row["subject id"]), "candidate": int(row["candidate id"]),
                         "word": word, "coords": np.asarray(coords, float)[:, :2]})
    return recs


def build(pkl, out):
    recs = load_words(pkl)
    X, keep = [], []
    for i, r in enumerate(recs):
        f = featurize(r["coords"], Keyboard.shen(r["candidate"]))
        if f is not None:
            X.append(f)
            keep.append(i)
    recs = [recs[i] for i in keep]
    np.savez_compressed(out, X=np.stack(X),
                        words=np.array([r["word"] for r in recs]),
                        subject=np.array([r["subject"] for r in recs]),
                        candidate=np.array([r["candidate"] for r in recs]))
    print(f"{len(recs)} gestures, {len(set(r['word'] for r in recs))} unique words, "
          f"{len(set(r['subject'] for r in recs))} subjects -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/shen_midair.pkl")
    ap.add_argument("--out", default="data/shen_features.npz")
    a = ap.parse_args()
    build(a.pkl, a.out)
