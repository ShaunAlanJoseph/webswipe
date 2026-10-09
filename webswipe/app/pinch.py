# self-calibrating pinch detector: scale-normalised distance + online GMM threshold
from collections import deque

import numpy as np

THUMB_TIP, INDEX_TIP, WRIST, MIDDLE_MCP = 4, 8, 0, 9


class OneEuro:
    # Casiez et al. 2012
    def __init__(self, freq=30.0, mincutoff=0.3, beta=0.3, dcutoff=1.0):
        self.freq, self.mincutoff, self.beta, self.dcutoff = freq, mincutoff, beta, dcutoff
        self.x = self.dx = self.t = None

    def _alpha(self, fc):
        return 1.0 / (1.0 + self.freq / (2 * np.pi * fc))

    def __call__(self, x, t=None):
        """t: timestamp in seconds; when given, the rate follows the real frame interval
        (MediaPipe runs at ~12 fps on the laptop, not the nominal 30)."""
        x = np.asarray(x, float)
        if t is not None and self.t is not None and t > self.t:
            self.freq = 1.0 / (t - self.t)
        self.t = t
        if self.x is None:
            self.x, self.dx = x, np.zeros_like(x)
            return x
        dx = (x - self.x) * self.freq
        self.dx = self._alpha(self.dcutoff) * dx + (1 - self._alpha(self.dcutoff)) * self.dx
        a = self._alpha(self.mincutoff + self.beta * np.linalg.norm(self.dx))
        self.x = a * x + (1 - a) * self.x
        return self.x

    def reset(self):
        self.x = self.dx = self.t = None


class PinchDetector:
    def __init__(self, default_thr=0.25, hyst=0.1, window=300, refit_every=15, warmup=60, scale_alpha=0.05,
                 max_pinch_mode=0.3, max_thr=0.4):
        self.thr = default_thr
        self.hyst = hyst
        self.hist = deque(maxlen=window)
        self.med = deque(maxlen=5)
        self.refit_every, self.warmup = refit_every, warmup
        self.scale_alpha = scale_alpha
        self.max_pinch_mode, self.max_thr = max_pinch_mode, max_thr
        self.scale = None
        self.n = 0
        self.on = False
        self.gmm = (np.array([0.1, 0.45]), np.array([0.01, 0.01]), np.array([0.4, 0.6]))

    def ratio(self, px):
        s = np.linalg.norm(px[WRIST] - px[MIDDLE_MCP])
        self.scale = s if self.scale is None else (1 - self.scale_alpha) * self.scale + self.scale_alpha * s
        return np.linalg.norm(px[THUMB_TIP] - px[INDEX_TIP]) / max(self.scale, 1e-6)

    def update(self, px):
        self.med.append(self.ratio(px))
        r = float(np.median(self.med))
        self.hist.append(r)
        self.n += 1
        if self.n >= self.warmup and self.n % self.refit_every == 0:
            self._refit()
        lo, hi = self.thr * (1 - self.hyst), self.thr * (1 + self.hyst)
        self.on = r < hi if self.on else r < lo
        return self.on, r

    def _refit(self, iters=5):
        x = np.array(self.hist)
        mu, var, w = self.gmm
        for _ in range(iters):
            p = w * np.exp(-(x[:, None] - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
            g = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
            nk = g.sum(0) + 1e-9
            mu = (g * x[:, None]).sum(0) / nk
            var = np.maximum((g * (x[:, None] - mu) ** 2).sum(0) / nk, 1e-4)
            w = nk / nk.sum()
        if mu[0] > mu[1]:
            mu, var, w = mu[::-1], var[::-1], w[::-1]
        # only trust the fit if the two modes are well separated
        # ...and only if the lower mode looks like a real pinch: with an open hand held in view both modes are
        # "open" and the threshold used to drift up until a relaxed hand counted as pinched (seen Oct 8)
        if mu[1] - mu[0] > 2 * (np.sqrt(var[0]) + np.sqrt(var[1])) / 2 and mu[0] < self.max_pinch_mode:
            grid = np.linspace(mu[0], mu[1], 200)
            d = w * np.exp(-(grid[:, None] - mu) ** 2 / (2 * var)) / np.sqrt(var)
            self.thr = min(float(grid[np.argmin(np.abs(d[:, 0] - d[:, 1]))]), self.max_thr)
        self.gmm = (mu, var, w)

    def reset_state(self):
        self.on = False
        self.med.clear()


class ContactPinch:
    """Fixed contact rule: pinched while the thumb tip touches the index tip.

    r = tip gap / palm length (same as PinchDetector). On below `on`, off above `off`; nothing adapts, so it
    cannot drift. Thresholds from the recordings (s1-s3): pinch median r 0.13-0.16, open hand 0.8-1.0.
    """

    def __init__(self, on=0.2, off=0.3, scale_alpha=0.05):
        self.thr, self.off_thr = on, off
        self.scale_alpha = scale_alpha
        self.scale = None
        self.med = deque(maxlen=3)
        self.on = False

    def update(self, px):
        s = np.linalg.norm(px[WRIST] - px[MIDDLE_MCP])
        self.scale = s if self.scale is None else (1 - self.scale_alpha) * self.scale + self.scale_alpha * s
        self.med.append(np.linalg.norm(px[THUMB_TIP] - px[INDEX_TIP]) / max(self.scale, 1e-6))
        r = float(np.median(self.med))
        self.on = r < self.off_thr if self.on else r < self.thr
        return self.on, r

    def reset_state(self):
        self.on = False
        self.med.clear()


def pinch_point(px):
    # midpoint of thumb and index tips: stays still while pinching
    return (px[THUMB_TIP] + px[INDEX_TIP]) / 2
