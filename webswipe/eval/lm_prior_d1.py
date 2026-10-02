# effect of a unigram word-frequency prior (wordfreq, independent of D1) on SHARK2 and neural decoders
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..baselines.shark2 import Shark2, shen_keyboard
from ..decoder.decode import Lexicon
from ..decoder.features import featurize, resample_time
from ..decoder.lm import unigram_logprior
from ..decoder.model import ShenDecoder
from ..decoder.shen_data import load_words
from .robustness_d1 import CONDITIONS, degraded_set

LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0]


def accuracy(scores, targets, lp, lam):
    """scores: (n, V) log-likelihoods; returns top-1, top-4."""
    s = scores + lam * lp
    top = np.argsort(-s, axis=1)[:, :4]
    return [float(np.mean(top[:, 0] == targets)), float(np.mean((top == targets[:, None]).any(1)))]


def shark2_scores(dec, recs, traces, ss, sl):
    out = []
    for r, tr in zip(recs, traces):
        key, kb = shen_keyboard(r["candidate"])
        out.append(Shark2.logp(*dec.distances(tr, kb, key), ss, sl))
    return np.stack(out)


@torch.no_grad()
def neural_scores(model, lex, recs, traces):
    out = []
    for r, tr in zip(recs, traces):
        _, kb = shen_keyboard(r["candidate"])
        f = featurize(tr, kb)
        if f is None:
            out.append(np.full(len(lex.words), -1e9))  # undecodable: counts as an error
            continue
        out.append(lex.scores(model(torch.from_numpy(f)[None])[0]).numpy())
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/shen_midair.pkl")
    ap.add_argument("--test-subject", type=int, default=0)
    ap.add_argument("--ckpt", nargs="*", default=[], help="name=path pairs")
    ap.add_argument("--conditions", nargs="+", default=["clean (50 Hz)", "webcam: 12 fps + 0.15 + 100 ms"])
    ap.add_argument("--n-tune", type=int, default=1500)
    ap.add_argument("--resample-hz", type=float, default=0.0, help="uniform-clock resampling for every decoder")
    ap.add_argument("--out", default="results/lm_prior_d1.json")
    a = ap.parse_args()

    recs = load_words(a.pkl)
    words = sorted({r["word"] for r in recs})
    widx = {w: i for i, w in enumerate(words)}
    lp = unigram_logprior(words)
    test = [r for r in recs if r["subject"] == a.test_subject]
    tgt = np.array([widx[r["word"]] for r in test])
    shark = Shark2(words, norm="rms", smooth=True)
    ss, sl = 0.64, 0.25

    # SHARK2 has no training data, so its prior weight can be tuned on the other subjects
    rng = np.random.default_rng(0)
    others = [r for r in recs if r["subject"] != a.test_subject]
    tune = [others[i] for i in rng.choice(len(others), a.n_tune, replace=False)]
    hz = a.resample_hz
    st = shark2_scores(shark, tune, [resample_time(r["coords"], r["t_ms"], hz) if hz else r["coords"]
                                     for r in tune], ss, sl)
    tt = np.array([widx[r["word"]] for r in tune])
    tune_acc = {lam: accuracy(st, tt, lp, lam)[0] for lam in LAMBDAS}
    lam_shark = max(tune_acc, key=tune_acc.get)
    print("SHARK2 prior weight tuned on other subjects:", lam_shark, tune_acc, flush=True)

    lex = Lexicon(words)
    models = {}
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        m = ShenDecoder()
        m.load_state_dict(torch.load(path, map_location="cpu")["model"])
        models[name] = m.eval()

    res = {"resample_hz": hz, "lambdas": LAMBDAS, "shark2_tuned_lambda": lam_shark, "shark2_tune_top1": tune_acc,
           "conditions": {}}
    for cname in a.conditions:
        ci = list(CONDITIONS).index(cname)
        traces = degraded_set(test, CONDITIONS[cname], seed=100 + ci, hz=hz)  # same traces as robustness_d1
        scores = {"SHARK2+smooth": shark2_scores(shark, test, traces, ss, sl)}
        for name, m in models.items():
            scores[name] = neural_scores(m, lex, test, traces)
        res["conditions"][cname] = {k: {str(lam): accuracy(s, tgt, lp, lam) for lam in LAMBDAS}
                                    for k, s in scores.items()}
        for k, v in res["conditions"][cname].items():
            print(cname, k, "  ".join(f"λ={lam}: {t1:.3f}/{t4:.3f}" for lam, (t1, t4) in v.items()), flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
