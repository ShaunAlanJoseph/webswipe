# phrase list for recording, taken from the Shen dataset so the vocabulary matches D1
import argparse
import random

import pandas as pd

LETTERS = set("abcdefghijklmnopqrstuvwxyz")

ap = argparse.ArgumentParser()
ap.add_argument("--pkl", default="data/shen_midair.pkl")
ap.add_argument("--out", default="data/phrases.txt")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

seen, phrases = set(), []
for p in pd.read_pickle(a.pkl)["phrase"]:
    words = [w.lower() for w in p]
    s = " ".join(words)
    if s not in seen and all(w and set(w) <= LETTERS for w in words):
        seen.add(s)
        phrases.append(s)
random.Random(a.seed).shuffle(phrases)
open(a.out, "w").write("\n".join(phrases) + "\n")
print(f"{len(phrases)} phrases -> {a.out}")
