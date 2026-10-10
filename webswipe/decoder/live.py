# load a trained decoder for the live app
import time

import numpy as np
import torch

from .decode import Lexicon
from .features import endpoint_cost, featurize, resample_time
from .model import ShenDecoder


class LiveDecoder:
    def __init__(self, ckpt, lexicon_words, prior_weight=0.0, resample_hz=0.0, endpoint_weight=0.0):
        """prior_weight > 0 adds a wordfreq unigram prior; resample_hz > 0 puts the trace on a uniform
        clock before featurising (the decoder's smoothing window is in frames, tuned for ~50 Hz)."""
        if not np.isfinite(endpoint_weight) or endpoint_weight < 0:
            raise ValueError("endpoint_weight must be finite and nonnegative")
        self.endpoint_weight = endpoint_weight
        self.model = ShenDecoder()
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
        self.model.eval()
        words = sorted(set(lexicon_words))
        logprior = None
        if prior_weight > 0:
            from .lm import unigram_logprior
            logprior = unigram_logprior(words)
        self.lex = Lexicon(words, logprior)
        self.prior_weight, self.resample_hz = prior_weight, resample_hz

    @torch.no_grad()
    def __call__(self, trace_kb, kb, k=4):
        t0 = time.perf_counter()
        tr = np.asarray(trace_kb, float)
        c = resample_time(tr[:, :2], tr[:, 2], self.resample_hz) if self.resample_hz else tr[:, :2]
        f = featurize(c, kb)
        if f is None:
            return [], 0.0
        logits = self.model(torch.from_numpy(f)[None])[0]
        if self.endpoint_weight:
            scores = self.lex.scores(logits, self.prior_weight)
            penalty = torch.as_tensor(endpoint_cost(tr, kb, self.lex.words), dtype=scores.dtype, device=scores.device)
            indices = torch.topk(scores - self.endpoint_weight * penalty, min(k, len(self.lex.words))).indices.tolist()
            candidates = [self.lex.words[i] for i in indices]
        else:
            candidates = self.lex.topk(logits, k, self.prior_weight)
        return candidates, (time.perf_counter() - t0) * 1000


class LiveShark2:
    """SHARK2 template decoder with the live interface; needs no trained model.
    D1 setting (results/README.md): RMS shape + location, smoothing, 50 Hz. Prior 0.25, not the D1-optimal 0.5:
    at 0.5, 7.8% of the lexicon cannot be typed as top-1 even with a perfect swipe (0.3% not even top-4)."""

    def __init__(self, lexicon_words, prior_weight=0.25, resample_hz=50.0, sigma_shape=0.64, sigma_loc=0.25):
        from ..baselines.shark2 import Shark2
        self.words = sorted(set(lexicon_words))
        self.dec = Shark2(self.words, norm="rms", smooth=True)
        self.logprior = np.zeros(len(self.words))
        if prior_weight > 0:
            from .lm import unigram_logprior
            self.logprior = unigram_logprior(self.words)
        self.prior_weight, self.resample_hz = prior_weight, resample_hz
        self.sigma = (sigma_shape, sigma_loc)

    def __call__(self, trace_kb, kb, k=4):
        t0 = time.perf_counter()
        tr = np.asarray(trace_kb, float)
        if len(tr) < 2:
            return [], 0.0
        c = resample_time(tr[:, :2], tr[:, 2], self.resample_hz) if self.resample_hz else tr[:, :2]
        s = self.dec.logp(*self.dec.distances(c, kb, id(kb)), *self.sigma) + self.prior_weight * self.logprior
        top = np.argsort(-s)[:k]
        return [self.words[i] for i in top], (time.perf_counter() - t0) * 1000


def load_lexicon(path):
    if path.endswith(".npz"):
        return list(np.load(path)["words"])
    return [w.strip().lower() for w in open(path) if w.strip()]
