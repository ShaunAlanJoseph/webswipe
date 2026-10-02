# load a trained decoder for the live app
import time

import numpy as np
import torch

from .decode import Lexicon
from .features import featurize, resample_time
from .model import ShenDecoder


class LiveDecoder:
    def __init__(self, ckpt, lexicon_words, prior_weight=0.0, resample_hz=0.0):
        """prior_weight > 0 adds a wordfreq unigram prior; resample_hz > 0 puts the trace on a uniform
        clock before featurising (the decoder's smoothing window is in frames, tuned for ~50 Hz)."""
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
        return self.lex.topk(logits, k, self.prior_weight), (time.perf_counter() - t0) * 1000


def load_lexicon(path):
    if path.endswith(".npz"):
        return list(np.load(path)["words"])
    return [w.strip().lower() for w in open(path) if w.strip()]
