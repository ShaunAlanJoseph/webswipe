# E1 on different keyboard layouts
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_experiments import OUT, load_lexicon, normalise, resample

# (letters, column offset of first letter) per row; non-letter keys are skipped
LAYOUTS = {
    "QWERTY": [("qwertyuiop", 0), ("asdfghjkl", 0), ("zxcvbnm", 0)],
    "Dvorak": [("pyfgcrl", 3), ("aoeuidhtns", 0), ("qjkxbmwvz", 1)],
    "Colemak": [("qwfpgjluy", 0), ("arstdhneio", 0), ("zxcvbkm", 0)],
    "Workman": [("qdrwbjfup", 0), ("ashtgyneoi", 0), ("zxmcvkl", 0)],
    "Alphabetical": [("abcdefghij", 0), ("klmnopqrs", 0), ("tuvwxyz", 0)],
}
ROW_OFFSET = [0.0, 0.25, 0.75]
SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


def make_keys(rows):
    return {c: np.array([start + i + ROW_OFFSET[r] + 0.5, r + 0.5])
            for r, (letters, start) in enumerate(rows) for i, c in enumerate(letters)}


def run_layout(name, lex, tests, trials):
    keys = make_keys(LAYOUTS[name])
    assert len(keys) == 26, name
    chars = np.array(list(keys))
    key_xy = np.stack([keys[c] for c in chars])
    poly = lambda w: np.stack([keys[c] for c in w])

    T = normalise(np.stack([resample(poly(w), 32) for w in lex]))
    by_len = {}
    for w in lex:
        by_len.setdefault(len(w), []).append(w)
    P = {L: np.stack([poly(w) for w in ws]) for L, ws in by_len.items()}

    rng = np.random.default_rng(0)
    res = {k: [] for k in ["gesture_top1", "gesture_top3", "perkey_raw", "perkey_lex_top1"]}
    for s in SIGMAS:
        g1 = g3 = pr = pl = 0
        for w in tests:
            g = resample(poly(w), 60)
            g = g + rng.normal(0, s, g.shape)
            top = np.argsort(((T - normalise(resample(g, 32))) ** 2).sum(-1).mean(-1))[:3]
            g1 += lex[top[0]] == w
            g3 += w in [lex[i] for i in top]

            taps = poly(w) + rng.normal(0, s, (len(w), 2))
            d = ((taps[:, None] - key_xy[None]) ** 2).sum(-1)
            pr += "".join(chars[d.argmin(1)]) == w

            taps = poly(w) + rng.normal(0, s, (len(w), 2))
            pl += by_len[len(w)][int(np.argmin(((P[len(w)] - taps) ** 2).sum((-1, -2))))] == w
        for k, v in zip(res, [g1, g3, pr, pl]):
            res[k].append(v / trials)
    print(f"  {name:12s} sigma=0.4: gesture@1={res['gesture_top1'][4]:.3f}  "
          f"perkey_raw={res['perkey_raw'][4]:.3f}  perkey+lex={res['perkey_lex_top1'][4]:.3f}")
    return res


if __name__ == "__main__":
    lex = load_lexicon(3000)
    trials = 300
    tests = [lex[i] for i in np.random.default_rng(42).choice(len(lex), trials)]
    R = {name: run_layout(name, lex, tests, trials) for name in LAYOUTS}

    titles = {"gesture_top1": "Word-gesture (top-1)", "gesture_top3": "Word-gesture (top-3)",
              "perkey_raw": "Per-key raw", "perkey_lex_top1": "Per-key + lexicon (top-1)"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True, sharey=True)
    for ax, (k, t) in zip(axes.flat, titles.items()):
        for name in LAYOUTS:
            ax.plot(SIGMAS, R[name][k], "o-", ms=4, label=name)
        ax.set_title(t); ax.grid(alpha=.3)
    for ax in axes[1]:
        ax.set_xlabel("Landmark noise σ (key widths)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Word accuracy")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"E1 across keyboard layouts (lexicon={len(lex)}, {trials} words)")
    plt.tight_layout(); plt.savefig(OUT / "e1_layouts.png", dpi=150); plt.close()
    (OUT / "results_e1_layouts.json").write_text(json.dumps({"sigmas": SIGMAS, **R}, indent=2))
