# decoder simulation experiments E1-E4 (synthetic data)
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(0)

# keyboard
ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
ROW_OFFSET = [0.0, 0.25, 0.75]
KEYS = {c: np.array([i + ROW_OFFSET[r] + 0.5, r + 0.5])
        for r, row in enumerate(ROWS) for i, c in enumerate(row)}
KEY_CHARS = np.array(list(KEYS))
KEY_XY = np.stack([KEYS[c] for c in KEY_CHARS])


def load_lexicon(size, min_len=3, max_len=10):
    words = set()
    for w in Path("/usr/share/dict/cracklib-small").read_text().split():
        w = w.lower()
        if w.isalpha() and min_len <= len(w) <= max_len and len(set(w)) > 1:
            words.add(w)
    words = sorted(words)
    idx = RNG.choice(len(words), size=min(size, len(words)), replace=False)
    return [words[i] for i in sorted(idx)]


# gestures
def polyline(word):
    return np.stack([KEYS[c] for c in word])


def resample(pts, n):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    if cum[-1] == 0:
        return np.repeat(pts[:1], n, axis=0)
    t = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(t, cum, pts[:, d]) for d in range(2)], axis=1)


def normalise(g):
    # center and scale to unit bbox
    g = g - g.mean(axis=-2, keepdims=True)
    span = (g.max(axis=-2) - g.min(axis=-2)).max(axis=-1)
    return g / np.maximum(span, 1e-9)[..., None, None]


class ShapeDecoder:
    def __init__(self, lexicon, n=32):
        self.lex, self.n = lexicon, n
        self.T = normalise(np.stack([resample(polyline(w), n) for w in lexicon]))

    def scores(self, gesture):
        q = normalise(resample(gesture, self.n))
        return ((self.T - q) ** 2).sum(-1).mean(-1)

    def topk(self, gesture, k=3):
        return [self.lex[i] for i in np.argsort(self.scores(gesture))[:k]]


def noisy_gesture(word, sigma, frames=60):
    g = resample(polyline(word), frames)
    return g + RNG.normal(0, sigma, g.shape)


# per-key
def perkey_raw(word, sigma):
    taps = polyline(word) + RNG.normal(0, sigma, (len(word), 2))
    d = ((taps[:, None] - KEY_XY[None]) ** 2).sum(-1)
    return "".join(KEY_CHARS[d.argmin(1)])


class PerKeyLexicon:
    # taps + nearest word of same length
    def __init__(self, lexicon):
        self.by_len = {}
        for w in lexicon:
            self.by_len.setdefault(len(w), []).append(w)
        self.P = {L: np.stack([polyline(w) for w in ws]) for L, ws in self.by_len.items()}

    def decode(self, word, sigma, k=3):
        taps = polyline(word) + RNG.normal(0, sigma, (len(word), 2))
        s = ((self.P[len(word)] - taps) ** 2).sum((-1, -2))
        return [self.by_len[len(word)][i] for i in np.argsort(s)[:k]]


# E1
def e1(lexicon, trials=200):
    dec, pkl = ShapeDecoder(lexicon), PerKeyLexicon(lexicon)
    sigmas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    tests = [lexicon[i] for i in RNG.choice(len(lexicon), trials)]
    res = {k: [] for k in ["gesture_top1", "gesture_top3", "perkey_raw",
                           "perkey_lex_top1", "perkey_lex_top3"]}
    for s in sigmas:
        g1 = g3 = pr = p1 = p3 = 0
        for w in tests:
            top = dec.topk(noisy_gesture(w, s))
            g1 += top[0] == w; g3 += w in top
            pr += perkey_raw(w, s) == w
            ptop = pkl.decode(w, s)
            p1 += ptop[0] == w; p3 += w in ptop
        for k, v in zip(res, [g1, g3, pr, p1, p3]):
            res[k].append(v / trials)
        print(f"  E1 sigma={s:.1f}  gesture@1={g1/trials:.3f}  perkey_raw={pr/trials:.3f}  "
              f"perkey+lex@1={p1/trials:.3f}")

    plt.figure(figsize=(6.4, 4.2))
    plt.plot(sigmas, res["gesture_top1"], "o-", label="Word-gesture (top-1)")
    plt.plot(sigmas, res["gesture_top3"], "o--", label="Word-gesture (top-3)")
    plt.plot(sigmas, res["perkey_lex_top1"], "s-", label="Per-key + lexicon (top-1)")
    plt.plot(sigmas, res["perkey_raw"], "^-", label="Per-key raw")
    plt.xlabel("Landmark noise σ (key widths)"); plt.ylabel("Word accuracy")
    plt.title(f"E1: Decoding accuracy vs noise (lexicon={len(lexicon)})")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "e1_accuracy_vs_noise.png", dpi=150); plt.close()
    return {"sigmas": sigmas, **res}


# E2
def e2(lexicon, trials=150, pixel_sigma=0.15):
    # scale = 50 / distance
    dec = ShapeDecoder(lexicon)
    dists = [30, 40, 50, 60, 70, 90]
    tests = [lexicon[i] for i in RNG.choice(len(lexicon), trials)]
    clean, noisy = [], []
    for d in dists:
        s = 50.0 / d
        c = n = 0
        for w in tests:
            g = resample(polyline(w), 60) * s + RNG.uniform(-5, 5, 2)
            c += dec.topk(g, 1)[0] == w
            n += dec.topk(g + RNG.normal(0, pixel_sigma, g.shape), 1)[0] == w
        clean.append(c / trials); noisy.append(n / trials)
        print(f"  E2 d={d}cm  clean={c/trials:.3f}  fixed-pixel-noise={n/trials:.3f}")

    plt.figure(figsize=(6.4, 4.2))
    plt.plot(dists, clean, "o-", label="No noise (scale + translation only)")
    plt.plot(dists, noisy, "s-", label=f"Fixed image-space noise σ={pixel_sigma}")
    plt.xlabel("Simulated user–camera distance (cm)"); plt.ylabel("Top-1 accuracy")
    plt.ylim(0, 1.05); plt.title("E2: Distance invariance of the shape channel")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "e2_distance_invariance.png", dpi=150); plt.close()
    return {"distances_cm": dists, "clean": clean, "fixed_pixel_noise": noisy,
            "pixel_sigma": pixel_sigma}


# E3
def e3(lexicon, sigma=0.3, reps=300, n_words=20, competitors=50):
    Ns = [8, 16, 32, 64, 128, 256]
    words = [lexicon[i] for i in RNG.choice(len(lexicon), n_words, replace=False)]
    spread, margin_var, mean_off = [], [], []
    for N in Ns:
        dec = ShapeDecoder(lexicon, n=N)
        sp, mv, mo = [], [], []
        for w in words:
            wi = dec.lex.index(w)
            cand = np.concatenate([[wi], RNG.choice(len(lexicon), competitors, replace=False)])
            clean_path = resample(polyline(w), N)
            T = dec.T[cand]
            base = ((T - normalise(clean_path)) ** 2).sum(-1).mean(-1)
            noisy = clean_path + RNG.normal(0, sigma, (reps,) + clean_path.shape)
            offs = ((T[None] - normalise(noisy)[:, None]) ** 2).sum(-1).mean(-1) - base
            mean_c = offs.mean(0)
            sp.append(mean_c[1:].std()); mo.append(mean_c.mean())
            margin = offs[:, 1:] - offs[:, :1]
            mv.append(margin.var(0).mean())
        spread.append(float(np.mean(sp))); margin_var.append(float(np.mean(mv)))
        mean_off.append(float(np.mean(mo)))
        print(f"  E3 N={N:3d}  mean offset={mean_off[-1]:.4f}  "
              f"cross-candidate spread={spread[-1]:.4f}  margin var={margin_var[-1]:.2e}")
    slope = float(np.polyfit(np.log(Ns), np.log(margin_var), 1)[0])

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].loglog(Ns, margin_var, "o-", label=f"measured (slope {slope:.2f})")
    ref = margin_var[0] * Ns[0] / np.array(Ns)
    ax[0].loglog(Ns, ref, "k--", label="O(1/N) reference")
    ax[0].set_xlabel("Resampled points N"); ax[0].set_ylabel("Var of margin change")
    ax[0].set_title("E3a: ranking noise vs N"); ax[0].legend(); ax[0].grid(alpha=.3, which="both")
    ax[1].plot(Ns, mean_off, "o-", label="mean offset (all candidates)")
    ax[1].plot(Ns, spread, "s-", label="spread of offset across candidates")
    ax[1].set_xscale("log"); ax[1].set_xlabel("Resampled points N")
    ax[1].set_title(f"E3b: noise offset is (nearly) common, σ={sigma}")
    ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "e3_offset_vs_N.png", dpi=150); plt.close()
    return {"N": Ns, "sigma": sigma, "mean_offset": mean_off,
            "cross_candidate_spread": spread, "margin_variance": margin_var,
            "loglog_slope": slope}


# E4
def e4(sizes=(500, 1000, 2000, 5000, 10000), queries=50):
    out = []
    for sz in sizes:
        lex = load_lexicon(sz)
        dec = ShapeDecoder(lex)
        gs = [noisy_gesture(lex[i], 0.2) for i in RNG.choice(len(lex), queries)]
        t = time.perf_counter()
        for g in gs:
            dec.topk(g)
        ms = (time.perf_counter() - t) / queries * 1e3
        out.append(ms)
        print(f"  E4 lexicon={len(lex):5d}  {ms:.2f} ms/word")
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(sizes, out, "o-"); plt.xlabel("Lexicon size"); plt.ylabel("ms per word (CPU, numpy)")
    plt.title("E4: Decoder latency"); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(OUT / "e4_latency.png", dpi=150); plt.close()
    return {"lexicon_sizes": list(sizes), "ms_per_word": out}


if __name__ == "__main__":
    lex = load_lexicon(3000)
    print(f"Lexicon: {len(lex)} words")
    results = {"lexicon_size": len(lex)}
    print("E1 ..."); results["E1"] = e1(lex)
    print("E2 ..."); results["E2"] = e2(lex)
    print("E3 ..."); results["E3"] = e3(lex)
    print("E4 ..."); results["E4"] = e4()
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Saved figures + results.json to {OUT}")
