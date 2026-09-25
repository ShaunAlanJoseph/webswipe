# experiments E5-E10 (synthetic data)
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import OUT, load_lexicon, normalise, polyline, resample

rng = np.random.default_rng(1)
FPS = 30


# normalisers
def norm_bbox(g):
    return normalise(g)


def norm_rms(g):
    g = g - g.mean(axis=-2, keepdims=True)
    r = np.sqrt((g ** 2).sum(-1).mean(-1))
    return g / np.maximum(r, 1e-9)[..., None, None]


# decoder
class Shark2:
    def __init__(self, lexicon, n=32, norm=norm_bbox):
        self.lex, self.n, self.norm = lexicon, n, norm
        self.L = np.stack([resample(polyline(w), n) for w in lexicon])
        self.S = norm(self.L)

    def dists(self, g):
        q = resample(g, self.n)
        s = ((self.S - self.norm(q)) ** 2).sum(-1).mean(-1)
        l = ((self.L - q) ** 2).sum(-1).mean(-1)
        return s, l

    @staticmethod
    def combine(s, l, lam):
        return l if lam == np.inf else s + lam * l

    def topk(self, g, lam, k=3):
        s, l = self.dists(g)
        return [self.lex[i] for i in np.argsort(self.combine(s, l, lam))[:k]]


def iid_gesture(word, sigma, frames=60):
    g = resample(polyline(word), frames)
    return g + rng.normal(0, sigma, g.shape)


def timed_gesture(word, hold=0):
    # min-jerk path between keys
    P = polyline(word)
    pts = [P[0]] * (hold + 1)
    for a, b in zip(P[:-1], P[1:]):
        L = np.linalg.norm(b - a)
        nf = max(2, round((0.12 + 0.08 * L) * FPS))
        tau = np.linspace(0, 1, nf + 1)[1:]
        s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
        pts += list(a + (b - a) * s[:, None])
    pts += [P[-1]] * hold
    g = np.array(pts)
    speed = np.concatenate([[0], np.linalg.norm(np.diff(g, axis=0), axis=1) * FPS])
    return g, speed


def perkey_lex_top1(by_len, P, word, sigma):
    taps = polyline(word) + rng.normal(0, sigma, (len(word), 2))
    s = ((P[len(word)] - taps) ** 2).sum((-1, -2))
    return by_len[len(word)][int(np.argmin(s))]


def pick_lambda(dec, words, sigma, grid):
    acc = np.zeros(len(grid))
    for w in words:
        s, l = dec.dists(iid_gesture(w, sigma))
        for j, lam in enumerate(grid):
            acc[j] += dec.lex[int(np.argmin(Shark2.combine(s, l, lam)))] == w
    return grid[int(np.argmax(acc))], (acc / len(words)).tolist()


# E5
LAM_GRID = [0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, np.inf]


def e5(lex, trials=200):
    dec = Shark2(lex)
    val = [lex[i] for i in rng.choice(len(lex), 150)]
    lam, val_acc = pick_lambda(dec, val, 0.4, LAM_GRID)
    print(f"  E5 lambda chosen on validation (sigma=0.4): {lam}  val acc per lambda: "
          + ", ".join(f"{a:.2f}" for a in val_acc))
    by_len = {}
    for w in lex:
        by_len.setdefault(len(w), []).append(w)
    P = {L: np.stack([polyline(w) for w in ws]) for L, ws in by_len.items()}

    sigmas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    test = [lex[i] for i in rng.choice(len(lex), trials)]
    res = {k: [] for k in ["shape", "location", "shark2", "shark2_top3", "perkey_lex"]}
    for sg in sigmas:
        c = dict.fromkeys(res, 0)
        for w in test:
            s, l = dec.dists(iid_gesture(w, sg))
            c["shape"] += lex[int(np.argmin(s))] == w
            c["location"] += lex[int(np.argmin(l))] == w
            order = np.argsort(Shark2.combine(s, l, lam))[:3]
            c["shark2"] += lex[order[0]] == w
            c["shark2_top3"] += w in [lex[i] for i in order]
            c["perkey_lex"] += perkey_lex_top1(by_len, P, w, sg) == w
        for k in res:
            res[k].append(c[k] / trials)
        print(f"  E5 sigma={sg:.1f}  " + "  ".join(f"{k}={c[k]/trials:.3f}" for k in res))

    plt.figure(figsize=(6.4, 4.2))
    for k, lab, m in [("shark2", f"Shape+location (λ={lam})", "o-"),
                      ("shark2_top3", "Shape+location top-3", "o--"),
                      ("shape", "Shape only", "^-"), ("location", "Location only", "v-"),
                      ("perkey_lex", "Per-key + lexicon", "s-")]:
        plt.plot(sigmas, res[k], m, label=lab)
    plt.xlabel("Landmark noise σ (key widths)"); plt.ylabel("Top-1 word accuracy")
    plt.title("E5: SHARK2-style decoder vs per-key + lexicon")
    plt.grid(alpha=.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(OUT / "e5_shark2_vs_perkey.png", dpi=150); plt.close()
    return {"lambda": lam, "val_acc_per_lambda": dict(zip(map(str, LAM_GRID), val_acc)),
            "sigmas": sigmas, **res}, lam


# E6
def e6(lex, sigma=0.3, reps=300, n_words=20, competitors=50):
    Ns = [8, 16, 32, 64, 128, 256]
    words = [lex[i] for i in rng.choice(len(lex), n_words, replace=False)]
    out = {}
    for name, norm in [("bbox", norm_bbox), ("rms", norm_rms)]:
        mv, sp, mo = [], [], []
        for N in Ns:
            T_all = norm(np.stack([resample(polyline(w), N) for w in lex]))
            m_, s_, o_ = [], [], []
            for w in words:
                cand = np.concatenate([[lex.index(w)], rng.choice(len(lex), competitors, replace=False)])
                T = T_all[cand]
                clean = resample(polyline(w), N)
                base = ((T - norm(clean)) ** 2).sum(-1).mean(-1)
                noisy = clean + rng.normal(0, sigma, (reps,) + clean.shape)
                offs = ((T[None] - norm(noisy)[:, None]) ** 2).sum(-1).mean(-1) - base
                mean_c = offs.mean(0)
                s_.append(mean_c[1:].std()); o_.append(mean_c.mean())
                m_.append((offs[:, 1:] - offs[:, :1]).var(0).mean())
            mv.append(float(np.mean(m_))); sp.append(float(np.mean(s_))); mo.append(float(np.mean(o_)))
        slope = float(np.polyfit(np.log(Ns), np.log(mv), 1)[0])
        dec = Shark2(lex, n=32, norm=norm)
        acc = []
        for sg in [0.2, 0.4, 0.6]:
            tw = [lex[i] for i in rng.choice(len(lex), 150)]
            acc.append(float(np.mean([lex[int(np.argmin(dec.dists(iid_gesture(w, sg))[0]))] == w for w in tw])))
        out[name] = {"margin_variance": mv, "cross_candidate_spread": sp, "mean_offset": mo,
                     "loglog_slope": slope, "shape_top1_at_sigma_0.2_0.4_0.6": acc}
        print(f"  E6 {name:4s}: slope={slope:.2f}  spread@N=8..256 {sp[0]:.4f}->{sp[-1]:.4f}  "
              f"shape top-1 {acc}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for name, m in [("bbox", "o-"), ("rms", "s-")]:
        ax[0].loglog(Ns, out[name]["margin_variance"], m,
                     label=f"{name} (slope {out[name]['loglog_slope']:.2f})")
        ax[1].plot(Ns, out[name]["cross_candidate_spread"], m, label=name)
    v0 = out["rms"]["margin_variance"][0]
    ax[0].loglog(Ns, v0 * Ns[0] / np.array(Ns), "k--", label="O(1/N) reference")
    ax[0].set_xlabel("N"); ax[0].set_ylabel("Var of margin change"); ax[0].legend()
    ax[0].set_title("E6a: ranking variance vs N"); ax[0].grid(alpha=.3, which="both")
    ax[1].set_xscale("log"); ax[1].set_xlabel("N"); ax[1].set_ylabel("Offset spread across candidates")
    ax[1].set_title("E6b: is the noise offset common to all candidates?"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "e6_normalisation_O1N.png", dpi=150); plt.close()
    out["N"], out["sigma"] = Ns, sigma
    return out


# E7
def corrupt(word, cond, sigma0=0.2):
    g, speed = timed_gesture(word)
    n = len(g)
    if cond in ("speed", "all"):
        sd = sigma0 * (1 + speed / max(np.median(speed[speed > 0]), 1e-9))
    else:
        sd = np.full(n, sigma0)
    g = g + rng.normal(0, 1, g.shape) * sd[:, None]
    if cond in ("drift", "all"):
        g = g + np.cumsum(rng.normal(0, 0.04, g.shape), axis=0)
    if cond in ("timing", "all"):
        a, b = rng.integers(0, 4, 2)
        g = g[a:n - b] if n - a - b >= 2 else g
    return g


def e7(lex, lam, trials=250):
    dec = Shark2(lex)
    conds = ["iid", "drift", "speed", "timing", "all"]
    labels = ["i.i.d. only\n(σ=0.2)", "+ hand drift\n(random walk)", "+ speed-dependent\nblur noise",
              "+ pinch timing\nerror (≤100 ms)", "all combined"]
    test = [lex[i] for i in rng.choice(len(lex), trials)]
    res = {"shape": [], "shark2": []}
    for c in conds:
        s1 = s2 = 0
        for w in test:
            s, l = dec.dists(corrupt(w, c))
            s1 += lex[int(np.argmin(s))] == w
            s2 += lex[int(np.argmin(Shark2.combine(s, l, lam)))] == w
        res["shape"].append(s1 / trials); res["shark2"].append(s2 / trials)
        print(f"  E7 {c:7s} shape={s1/trials:.3f}  shark2={s2/trials:.3f}")
    x = np.arange(len(conds))
    plt.figure(figsize=(8, 4.2))
    plt.bar(x - 0.2, res["shape"], 0.4, label="Shape only")
    plt.bar(x + 0.2, res["shark2"], 0.4, label="Shape + location")
    plt.xticks(x, labels, fontsize=8); plt.ylabel("Top-1 accuracy"); plt.ylim(0, 1)
    plt.title("E7: Robustness to realistic corruption"); plt.legend(); plt.grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(OUT / "e7_noise_ladder.png", dpi=150); plt.close()
    return {"conditions": conds, **res}


# E8
def e8(lam, sigma=0.3, trials=120):
    sizes = [1000, 5000, 10000, 20000, 100000]
    Ns = [8, 16, 32, 64, 128]
    acc = np.zeros((len(sizes), len(Ns)))
    real_sizes = []
    for i, sz in enumerate(sizes):
        lex = load_lexicon(sz)
        real_sizes.append(len(lex))
        test = [lex[k] for k in rng.choice(len(lex), trials)]
        gs = [iid_gesture(w, sigma) for w in test]
        for j, N in enumerate(Ns):
            dec = Shark2(lex, n=N)
            acc[i, j] = np.mean([lex[int(np.argmin(Shark2.combine(*dec.dists(g), lam)))] == w
                                 for g, w in zip(gs, test)])
        print(f"  E8 lexicon={len(lex):6d}  " + "  ".join(f"N={N}:{a:.2f}" for N, a in zip(Ns, acc[i])))
    plt.figure(figsize=(6.4, 4.4))
    plt.imshow(acc, cmap="viridis", vmin=acc.min(), vmax=1, aspect="auto")
    for i in range(len(sizes)):
        for j in range(len(Ns)):
            plt.text(j, i, f"{acc[i, j]:.2f}", ha="center", va="center",
                     color="white" if acc[i, j] < (acc.min() + 1) / 2 else "black", fontsize=9)
    plt.xticks(range(len(Ns)), Ns); plt.yticks(range(len(sizes)), real_sizes)
    plt.xlabel("Resampled points N"); plt.ylabel("Lexicon size")
    plt.colorbar(label="Top-1 accuracy")
    plt.title(f"E8: Lexicon size × N (σ={sigma}, shape+location)")
    plt.tight_layout(); plt.savefig(OUT / "e8_lexicon_N_ablation.png", dpi=150); plt.close()
    return {"lexicon_sizes": real_sizes, "N": Ns, "sigma": sigma, "accuracy": acc.tolist()}


# E9
def f_raw(x, _):
    return x


def f_ema(x, a):
    y = np.empty_like(x); y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = a * x[t] + (1 - a) * y[t - 1]
    return y


def f_oneeuro(x, p, dcutoff=1.0):
    # one euro filter (Casiez 2012)
    mincutoff, beta = p
    te = 1.0 / FPS
    alpha = lambda fc: 1.0 / (1.0 + 1.0 / (2 * np.pi * fc * te))
    y = np.empty_like(x); y[0] = x[0]; dx = np.zeros(2)
    for t in range(1, len(x)):
        dx = alpha(dcutoff) * (x[t] - y[t - 1]) * FPS + (1 - alpha(dcutoff)) * dx
        a = alpha(mincutoff + beta * np.linalg.norm(dx))
        y[t] = a * x[t] + (1 - a) * y[t - 1]
    return y


def f_kalman(x, q, r=0.15 ** 2):
    dt = 1.0 / FPS
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
    Q, H, R = q * G @ G.T, np.eye(2, 4), r * np.eye(2)
    s, P = np.array([*x[0], 0, 0]), np.eye(4)
    y = np.empty_like(x); y[0] = x[0]
    for t in range(1, len(x)):
        s, P = F @ s, F @ P @ F.T + Q
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        s = s + K @ (x[t] - H @ s); P = (np.eye(4) - K @ H) @ P
        y[t] = s[:2]
    return y


def jitter_lag(filt, p, words, sigma=0.15, hold=15):
    J, Lg = [], []
    shifts = np.arange(0, 8.01, 0.25)
    for w in words:
        g, speed = timed_gesture(w, hold=hold)
        y = filt(g + rng.normal(0, sigma, g.shape), p)
        J.append(np.sqrt(((y[5:hold] - g[5:hold]) ** 2).sum(-1).mean()))
        mv = np.where(speed > 0.5)[0]
        t = np.arange(len(g))
        err = [np.mean(((y[mv] - np.stack([np.interp(mv - d, t, g[:, k]) for k in range(2)], 1)) ** 2).sum(-1))
               for d in shifts]
        Lg.append(shifts[int(np.argmin(err))] / FPS * 1000)
    return float(np.mean(J)), float(np.mean(Lg))


def e9(lex, lam):
    words = [lex[i] for i in rng.choice(len(lex), 40)]
    configs = {
        "Raw": [(f_raw, None)],
        "EMA": [(f_ema, a) for a in [0.9, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1]],
        "Kalman (CV)": [(f_kalman, q) for q in [1000, 300, 100, 30, 10, 3, 1]],
        "One Euro": [(f_oneeuro, (mc, b)) for mc in [0.3, 1.0, 3.0] for b in [0.0, 0.05, 0.3, 1.0, 3.0]],
    }
    pts = {}
    for name, cs in configs.items():
        pts[name] = [(p, *jitter_lag(f, p, words)) for f, p in cs]
        best = min(pts[name], key=lambda z: z[1] if z[2] <= 50 else 1e9)
        print(f"  E9 {name:12s} best (lag<=50ms): param={best[0]} jitter={best[1]:.3f} lag={best[2]:.0f} ms")

    # decode after filtering
    dec = Shark2(lex)
    test = [lex[i] for i in rng.choice(len(lex), 200)]
    dec_acc = {}
    for name, cs in configs.items():
        best_p = min(pts[name], key=lambda z: z[1] if z[2] <= 50 else 1e9)[0]
        f = cs[0][0]
        c = 0
        for w in test:
            g, _ = timed_gesture(w)
            y = f(g + rng.normal(0, 0.4, g.shape), best_p)
            c += lex[int(np.argmin(Shark2.combine(*dec.dists(y), lam)))] == w
        dec_acc[name] = c / len(test)
        print(f"  E9 decode after {name:12s}: top-1={dec_acc[name]:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, m in zip(pts, ["*", "o-", "s-", "^"]):
        J = [z[1] for z in pts[name]]; L = [z[2] for z in pts[name]]
        ax[0].plot(L, J, m, label=name, ms=9 if name == "Raw" else 6)
    ax[0].axvline(50, color="gray", ls=":", lw=1)
    ax[0].set_xlabel("Lag (ms)"); ax[0].set_ylabel("Jitter at rest (RMS, key widths)")
    ax[0].set_title("E9a: Jitter–lag trade-off (σ=0.15, 30 fps)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].bar(list(dec_acc), list(dec_acc.values()), color="C0")
    ax[1].set_ylim(0, 1); ax[1].set_ylabel("Top-1 accuracy")
    ax[1].set_title("E9b: Decoding after filtering (σ=0.4, lag ≤ 50 ms)")
    for i, v in enumerate(dec_acc.values()):
        ax[1].text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout(); plt.savefig(OUT / "e9_filters.png", dpi=150); plt.close()
    return {"tradeoff": {k: [{"param": str(p), "jitter": j, "lag_ms": l} for p, j, l in v]
                         for k, v in pts.items()},
            "decode_top1_sigma0.4": dec_acc}


# E10
def pinch_session(dist_cm, low_light, frames=900):
    # simulated pinch distance stream
    S = 100.0 * (50.0 / dist_cm) * rng.uniform(0.85, 1.15)
    r_open, r_pinch = rng.uniform(0.35, 0.6), rng.uniform(0.07, 0.13)
    sig_px = 5.0 if low_light else 2.5
    label = np.zeros(frames, bool)
    t = 0
    while t < frames:
        t += int(rng.integers(20, 60))
        d = int(rng.integers(15, 45))
        label[t:t + d] = True
        t += d
    ratio = np.where(label, r_pinch, r_open) + rng.normal(0, 0.03, frames) * np.where(label, 0.3, 1)
    ratio = np.convolve(ratio, np.ones(3) / 3, mode="same")
    noise = rng.normal(0, sig_px * np.sqrt(2), frames) * np.where(label, 2.0, 1.0)
    out = rng.random(frames) < 0.03
    dist_px = np.abs(ratio * S + noise + out * 0.3 * S)
    scale_meas = S + rng.normal(0, sig_px * 1.4, frames)
    return dist_px, scale_meas, label


def hysteresis(x, thr, h):
    s = np.zeros(len(x), bool); on = False
    for t, v in enumerate(x):
        th = thr[t] if np.ndim(thr) else thr
        on = v < th * (1 + h) if on else v < th * (1 - h)
        s[t] = on
    return s


def gmm_threshold_stream(r, window=300, every=15, warm=60, default=0.25):
    thr = np.full(len(r), default)
    mu, var, w = np.array([0.1, 0.45]), np.array([0.01, 0.01]), np.array([0.4, 0.6])
    cur = default
    for t in range(len(r)):
        if t >= warm and t % every == 0:
            x = r[max(0, t - window):t]
            for _ in range(5):
                p = w * np.exp(-(x[:, None] - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
                g = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
                nk = g.sum(0) + 1e-9
                mu = (g * x[:, None]).sum(0) / nk
                var = np.maximum((g * (x[:, None] - mu) ** 2).sum(0) / nk, 1e-4)
                w = nk / nk.sum()
            if mu[0] > mu[1]:
                mu, var, w = mu[::-1], var[::-1], w[::-1]
            grid = np.linspace(mu[0], mu[1], 200)
            d = w * np.exp(-(grid[:, None] - mu) ** 2 / (2 * var)) / np.sqrt(var)
            cur = grid[int(np.argmin(np.abs(d[:, 0] - d[:, 1])))]
        thr[t] = cur
    return thr


def events(state, min_len=3):
    ev, t = [], 0
    while t < len(state):
        if state[t]:
            s = t
            while t < len(state) and state[t]:
                t += 1
            if t - s >= min_len:
                ev.append((s, t))
        t += 1
    return ev


def event_prf(pred, label):
    P, T = events(pred), events(label, 1)
    iou = lambda a, b: max(0, min(a[1], b[1]) - max(a[0], b[0])) / (max(a[1], b[1]) - min(a[0], b[0]))
    used, tp = set(), 0
    for p in P:
        for i, g in enumerate(T):
            if i not in used and iou(p, g) >= 0.5:
                used.add(i); tp += 1; break
    pr = tp / max(len(P), 1); rc = tp / max(len(T), 1)
    return pr, rc, 2 * pr * rc / max(pr + rc, 1e-9)


def e10(h=0.1, sessions=8):
    # tune thresholds at 50cm
    cal = [pinch_session(50, False) for _ in range(6)]
    px_grid, r_grid = np.linspace(5, 50, 46), np.linspace(0.1, 0.4, 31)
    T_px = px_grid[int(np.argmax([np.mean([event_prf(hysteresis(d, T, h), l)[2] for d, s, l in cal])
                                  for T in px_grid]))]
    T_r = r_grid[int(np.argmax([np.mean([event_prf(hysteresis(d / s, T, h), l)[2] for d, s, l in cal])
                                for T in r_grid]))]
    print(f"  E10 calibrated at 50 cm: fixed px threshold={T_px:.1f}px  normalised ratio={T_r:.3f}")
    dists = [30, 50, 70, 90]
    res = {}
    for light in [False, True]:
        for name in ["Fixed px threshold", "Scale-normalised", "Normalised + online GMM"]:
            res[(name, light)] = []
        for dcm in dists:
            f = {k: [] for k in ["Fixed px threshold", "Scale-normalised", "Normalised + online GMM"]}
            for _ in range(sessions):
                d, s, l = pinch_session(dcm, light)
                s_run = f_ema(s[:, None].repeat(2, 1), 0.05)[:, 0]
                r = d / s_run
                f["Fixed px threshold"].append(event_prf(hysteresis(d, T_px, h), l))
                f["Scale-normalised"].append(event_prf(hysteresis(r, T_r, h), l))
                f["Normalised + online GMM"].append(event_prf(hysteresis(r, gmm_threshold_stream(r), h), l))
            for k, v in f.items():
                res[(k, light)].append(np.mean(v, axis=0).tolist())
            print(f"  E10 {'low ' if light else 'norm'} light d={dcm}cm  " +
                  "  ".join(f"{k}: F1={np.mean(v, axis=0)[2]:.2f}" for k, v in f.items()))

    plt.figure(figsize=(6.8, 4.4))
    for (name, light), v in res.items():
        c = {"Fixed px threshold": "C3", "Scale-normalised": "C1", "Normalised + online GMM": "C0"}[name]
        plt.plot(dists, [z[2] for z in v], "o--" if light else "o-", color=c,
                 label=f"{name} ({'low' if light else 'normal'} light)")
    plt.xlabel("User–camera distance (cm)"); plt.ylabel("Pinch event F1 (IoU ≥ 0.5)"); plt.ylim(0, 1.02)
    plt.title("E10: Pinch detection across distance and lighting\n(thresholds calibrated at 50 cm, normal light)")
    plt.grid(alpha=.3); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(OUT / "e10_pinch_detection.png", dpi=150); plt.close()
    return {"calibrated_px": float(T_px), "calibrated_ratio": float(T_r), "distances_cm": dists,
            "prf": {f"{k} | {'low' if lt else 'normal'} light": v for (k, lt), v in res.items()}}


if __name__ == "__main__":
    lex = load_lexicon(3000)
    print(f"Lexicon: {len(lex)} words")
    R = {}
    print("E5 ..."); R["E5"], lam = e5(lex)
    print("E6 ..."); R["E6"] = e6(lex)
    print("E7 ..."); R["E7"] = e7(lex, lam)
    print("E8 ..."); R["E8"] = e8(lam)
    print("E9 ..."); R["E9"] = e9(lex, lam)
    print("E10 ..."); R["E10"] = e10()
    (OUT / "results2.json").write_text(json.dumps(R, indent=2, default=str))
    print(f"Saved figures + results2.json to {OUT}")
