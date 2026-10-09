import numpy as np
THUMB_TIP, INDEX_TIP = 4, 8

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

def pinch_point(px):
    # midpoint of thumb and index tips: stays still while pinching
    return (px[THUMB_TIP] + px[INDEX_TIP]) / 2
