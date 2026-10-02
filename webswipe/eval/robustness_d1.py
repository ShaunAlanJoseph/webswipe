# D1 held-out subject under webcam-like degradation: SHARK2 vs neural decoders on identical traces
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..baselines.shark2 import Shark2, shen_keyboard
from ..decoder.augment import Degrade
from ..decoder.decode import Lexicon
from ..decoder.features import featurize, resample_time
from ..decoder.model import ShenDecoder
from ..decoder.shen_data import load_words

CONDITIONS = {
    "clean (50 Hz)": Degrade(),
    "30 fps": Degrade(fps=30),
    "12 fps": Degrade(fps=12),
    "jitter 0.1": Degrade(jitter=0.1),
    "jitter 0.2": Degrade(jitter=0.2),
    "jitter 0.3": Degrade(jitter=0.3),
    "webcam: 12 fps + 0.15 + 100 ms": Degrade(fps=12, jitter=0.15, trunc_ms=100),
    "harsh: 10 fps + 0.3 + 100 ms": Degrade(fps=10, jitter=0.3, trunc_ms=100),
}


def degraded_set(test, cond, seed, hz=0.0):
    """Degraded traces; hz > 0 resamples each one onto a uniform hz clock afterwards."""
    rng = np.random.default_rng(seed)
    out = []
    for r in test:
        _, kb = shen_keyboard(r["candidate"])
        c, t = cond.apply(r["coords"], r["t_ms"], kb.key_w, rng)
        out.append(resample_time(c, t, hz) if hz > 0 else c)
    return out


def eval_shark2(dec, test, traces, ss, sl):
    hits = np.zeros(2)
    for r, tr in zip(test, traces):
        key, kb = shen_keyboard(r["candidate"])
        top = dec.topk(tr, kb, key, ss, sl, k=4)
        hits += [r["word"] == top[0], r["word"] in top]
    return (hits / len(test)).tolist()


@torch.no_grad()
def eval_neural(model, lex, test, traces, bs=128):
    feats, ok = [], []
    for r, tr in zip(test, traces):
        _, kb = shen_keyboard(r["candidate"])
        f = featurize(tr, kb)
        ok.append(f is not None)
        feats.append(f if f is not None else np.zeros((50, 39), np.float32))
    X = torch.from_numpy(np.stack(feats))
    hits = np.zeros(2)
    for i in range(0, len(X), bs):
        logits = model(X[i:i + bs])
        for j, lg in enumerate(logits):
            if not ok[i + j]:
                continue  # too few frames to decode counts as an error
            top = lex.topk(lg, 4)
            w = test[i + j]["word"]
            hits += [w == top[0], w in top]
    return (hits / len(test)).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/shen_midair.pkl")
    ap.add_argument("--test-subject", type=int, default=0)
    ap.add_argument("--ckpt", nargs="*", default=[], help="name=path pairs")
    ap.add_argument("--shark2-sigma", nargs=2, type=float, default=[0.64, 0.25],
                    help="shape, location sigma tuned on other subjects (results/shark2_d1.json, rms)")
    ap.add_argument("--resample-hz", type=float, default=0.0,
                    help="also evaluate every decoder on traces resampled to this rate (variant suffix '@Hz')")
    ap.add_argument("--out", default="results/robustness_d1.json")
    a = ap.parse_args()

    recs = load_words(a.pkl)
    words = sorted({r["word"] for r in recs})
    test = [r for r in recs if r["subject"] == a.test_subject]
    lex = Lexicon(words)
    sharks = {"SHARK2": Shark2(words, norm="rms"), "SHARK2+smooth": Shark2(words, norm="rms", smooth=True)}
    models = {}
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        m = ShenDecoder()
        m.load_state_dict(torch.load(path, map_location="cpu")["model"])
        models[name] = m.eval()

    res = {"test_subject": a.test_subject, "n_test": len(test), "lexicon": len(words), "conditions": {}}
    for ci, (cname, cond) in enumerate(CONDITIONS.items()):
        variants = {"": degraded_set(test, cond, seed=100 + ci)}
        if a.resample_hz:  # same seed: identical degraded traces, only resampled
            variants[f"@{a.resample_hz:g}Hz"] = degraded_set(test, cond, seed=100 + ci, hz=a.resample_hz)
        row = {}
        for suf, traces in variants.items():
            row.update({k + suf: eval_shark2(d, test, traces, *a.shark2_sigma) for k, d in sharks.items()})
            for name, m in models.items():
                row[name + suf] = eval_neural(m, lex, test, traces)
        res["conditions"][cname] = row
        print(cname, "  ".join(f"{k}: {v[0]:.3f}/{v[1]:.3f}" for k, v in row.items()), flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
