# trajectory features, ported from Shen et al. 2023 (utils.py / data_process.py)
from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

LETTERS = "abcdefghijklmnopqrstuvwxyz"
N_POINTS = 50
N_FEATURES = 39
N_CLASSES = 29
BOUNDARY, BLANK = 27, 28

# key centres in kb-layout.txt units (y up)
_ROWS = [("qwertyuiop", 0, 0), ("asdfghjkl", 3, -6), ("zxcvbnm", 6, -12)]
LAYOUT_UNITS = {c: (x0 + 6 * i, y) for row, x0, y in _ROWS for i, c in enumerate(row)}

# key size conditions used in the Shen study, indexed by candidate id - 1
_hw = np.linspace(0.375, 1, 20) * 2
_w = np.linspace(0.167, 1, 20) * 3
SHEN_CANDIDATES = [(a, b) for a in _hw for b in _w]


@dataclass
class Keyboard:
    centers: np.ndarray  # (26, 2), alphabetical order, y up
    key_w: float
    width: float
    height: float

    @property
    def tolerance(self):
        return np.sqrt(2) * self.key_w / 2

    @classmethod
    def from_pitch(cls, pitch_x, pitch_y, origin=(0.0, 0.0)):
        # pitch = distance between neighbouring key centres
        u = np.array([LAYOUT_UNITS[c] for c in LETTERS], float)
        centers = np.asarray(origin) + u / 6 * [pitch_x, pitch_y]
        return cls(centers, 50 / 60 * pitch_x, 620 / 60 * pitch_x, 170 / 60 * pitch_y)

    @classmethod
    def shen(cls, candidate_id):
        hw, w = SHEN_CANDIDATES[int(candidate_id) - 1]
        return cls.from_pitch(60 * w, 60 * hw * w)


def resample_index(c, n=N_POINTS):
    t = np.linspace(0, len(c) - 1, n)
    i = np.arange(len(c))
    return np.stack([np.interp(t, i, c[:, 0]), np.interp(t, i, c[:, 1])], 1)


def resample_time(coords, t_ms, hz=50.0):
    """Interpolate a trace onto a uniform clock, so frame-based filters see the training-time rate."""
    c, t = np.asarray(coords, float), np.asarray(t_ms, float)
    if len(c) < 2 or t[-1] <= t[0]:
        return c
    ft = np.arange(t[0], t[-1] + 1e-9, 1000.0 / hz)
    return np.stack([np.interp(ft, t, c[:, k]) for k in range(c.shape[1])], 1)


def shape_features(s):
    prev = np.vstack([s[:1], s[:-1]])
    nxt = np.vstack([s[1:], s[-1:]])
    grad = s - prev

    with np.errstate(divide="ignore", invalid="ignore"):
        v1, v2 = -grad, nxt - s
        cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
        # clip: original gave 0 or -1 at random on straight segments due to rounding
        ang = np.arccos(np.clip(cos, -1, 1))
        curv = np.stack([np.cos(ang), np.sin(ang)], 1)

        p = np.vstack([s[:1], s[:1], s, s[-1:], s[-1:]])
        win = np.stack([p[j:j + len(s)] for j in range(5)], 1)  # (T, 5, 2)
        dx = win[:, :, 0].max(1) - win[:, :, 0].min(1)
        dy = win[:, :, 1].max(1) - win[:, :, 1].min(1)
        aspect = (dy - dx) / (dy + dx)

        sv = win[:, -1] - win[:, 0]
        r = sv[:, 1] / sv[:, 0]
        sa = np.arctan(np.abs(r)) * np.sign(r)
        slope = np.stack([np.cos(sa), np.sin(sa)], 1)

        seg = np.linalg.norm(np.diff(win, axis=1), axis=2).sum(1)
        curl = seg / np.maximum(dx, dy)

        d0 = win[:, :1] - win
        cross = sv[:, None, 0] * d0[:, :, 1] - sv[:, None, 1] * d0[:, :, 0]
        lin = ((cross / np.abs(sv).sum(1, keepdims=True)) ** 2).mean(1)

    out = np.column_stack([grad, grad, curv, aspect, slope, curl, lin])
    # original used nan_to_num only; also clip so inf never reaches the model
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -100, 100)


def featurize(coords, kb, n=N_POINTS):
    """coords: (T, 2) trace in keyboard frame (y up). Returns (n, 39) or None."""
    c = np.asarray(coords, float)[:, :2]
    if len(c) < 4:
        return None
    if len(c) >= 7:
        c = np.stack([savgol_filter(c[:, k], 7, 3, mode="nearest") for k in range(2)], 1)
    c = resample_index(c, n)
    near = (np.linalg.norm(c[:, None] - kb.centers[None], axis=2) < kb.tolerance).astype(float)
    cn = c / [kb.width, kb.height]
    return np.hstack([cn, near, shape_features(cn)]).astype(np.float32)


def encode_word(word, max_letters=12):
    # 1..26 = letters, 27 = word boundary, 28 = CTC blank
    return [BOUNDARY] + [LETTERS.index(ch) + 1 for ch in word[:max_letters]] + [BOUNDARY]


def endpoint_cost(trace_kb, kb, words):
    """Soft start/end mismatch in key pitches; independent of the intended word.

    Cap severe mismatches so missing/incorrect boundaries cannot dominate all
    sequence evidence. Use the raw recorded endpoints before feature smoothing.
    """
    trace = np.asarray(trace_kb, float)
    first = np.array([LETTERS.index(w[0]) for w in words])
    last = np.array([LETTERS.index(w[-1]) for w in words])
    pitch = kb.width / (620 / 60)
    cost = (np.square(kb.centers[first] - trace[0, :2]).sum(1)
            + np.square(kb.centers[last] - trace[-1, :2]).sum(1)) / (2 * pitch ** 2)
    return np.minimum(cost, 12)
