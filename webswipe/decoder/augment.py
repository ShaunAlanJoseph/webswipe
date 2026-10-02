# make headset traces look like webcam traces: lower frame rate, landmark jitter, pinch timing error
from dataclasses import dataclass

import numpy as np


@dataclass
class Degrade:
    fps: float = 0.0        # 0 = keep native rate (~50 Hz in D1)
    jitter: float = 0.0     # i.i.d. Gaussian noise per frame, in key widths
    trunc_ms: float = 0.0   # drop up to this much from each end (pinch onset/release error)

    def __call__(self, coords, t_ms, key_w, rng):
        return self.apply(coords, t_ms, key_w, rng)[0]

    def apply(self, coords, t_ms, key_w, rng):
        """Returns degraded (coords, t_ms)."""
        c, t = np.asarray(coords, float), np.asarray(t_ms, float) - t_ms[0]
        if self.trunc_ms > 0:
            a, b = rng.uniform(0, self.trunc_ms, 2)
            keep = (t >= a) & (t <= t[-1] - b)
            if keep.sum() >= 2:
                c, t = c[keep], t[keep] - t[keep][0]
        if self.fps > 0:
            # camera samples the hand at frame times with a random phase
            ft = np.arange(rng.uniform(0, 1000 / self.fps), t[-1] + 1e-9, 1000 / self.fps)
            if len(ft) < 2:
                ft = np.array([0.0, t[-1]])
            c, t = np.stack([np.interp(ft, t, c[:, k]) for k in range(2)], 1), ft
        if self.jitter > 0:
            c = c + rng.normal(0, self.jitter * key_w, c.shape)
        return c, t


def random_webcam(rng):
    """Training-time domain randomisation over plausible webcam conditions."""
    return Degrade(fps=float(rng.choice([0, 30, 20, 15, 12, 10])),
                   jitter=float(rng.uniform(0, 0.3)),
                   trunc_ms=float(rng.choice([0, 50, 100])))
