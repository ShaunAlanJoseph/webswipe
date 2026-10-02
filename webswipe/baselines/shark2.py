# SHARK2 template decoder (Kristensson & Zhai 2004): shape + location channels, no language model
import numpy as np
from scipy.signal import savgol_filter

from ..decoder.features import LETTERS, Keyboard

N_POINTS = 50


def resample_arc(c, n=N_POINTS):
    """n points equally spaced along the path (c: (T, 2))."""
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    if cum[-1] == 0:
        return np.repeat(c[:1], n, axis=0)
    t = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(t, cum, c[:, 0]), np.interp(t, cum, c[:, 1])], 1)


def norm_shape(g, how="bbox"):
    """Remove translation and scale. g: (..., n, 2)."""
    g = g - g.mean(-2, keepdims=True)
    if how == "bbox":  # original SHARK2: largest side of the bounding box
        s = (g.max(-2) - g.min(-2)).max(-1)
    else:  # RMS radius, less sensitive to outlier points (simulation E6)
        s = np.sqrt((g ** 2).sum(-1).mean(-1))
    return g / np.maximum(s, 1e-9)[..., None, None]


class Shark2:
    def __init__(self, words, norm="bbox", n=N_POINTS, smooth=False):
        # smooth: same Savitzky-Golay filter as the neural decoder's features, applied before resampling
        self.words, self.norm, self.n, self.smooth = list(words), norm, n, smooth
        self._cache = {}

    def templates(self, kb_key, kb):
        if kb_key not in self._cache:
            idx = {c: i for i, c in enumerate(LETTERS)}
            # a word's ideal trace is the polyline through its key centres
            L = np.stack([resample_arc(kb.centers[[idx[c] for c in w]], self.n) for w in self.words])
            self._cache[kb_key] = (L.astype(np.float32), norm_shape(L, self.norm).astype(np.float32))
        return self._cache[kb_key]

    def distances(self, trace, kb, kb_key):
        """Mean point-to-point distance per word: shape (normalised units), location (key widths)."""
        L, S = self.templates(kb_key, kb)
        c = np.asarray(trace, float)[:, :2]
        if self.smooth and len(c) >= 7:
            c = np.stack([savgol_filter(c[:, k], 7, 3, mode="nearest") for k in range(2)], 1)
        q = resample_arc(c, self.n).astype(np.float32)
        d_shape = np.linalg.norm(S - norm_shape(q, self.norm), axis=-1).mean(-1)
        d_loc = np.linalg.norm(L - q, axis=-1).mean(-1) / kb.key_w
        return d_shape, d_loc

    @staticmethod
    def logp(d_shape, d_loc, sigma_shape, sigma_loc):
        # each channel is a Gaussian likelihood on its distance; channels are combined as independent
        return -0.5 * (d_shape / sigma_shape) ** 2 - 0.5 * (d_loc / sigma_loc) ** 2

    def topk(self, trace, kb, kb_key, sigma_shape, sigma_loc, k=4):
        s = self.logp(*self.distances(trace, kb, kb_key), sigma_shape, sigma_loc)
        return [self.words[i] for i in np.argsort(-s)[:k]]


def shen_keyboard(candidate):
    return ("shen", int(candidate)), Keyboard.shen(candidate)
