# load a trained decoder for the live app
import time

import numpy as np
import torch

from .decode import Lexicon
from .features import featurize
from .model import ShenDecoder


class LiveDecoder:
    def __init__(self, ckpt, lexicon_words):
        self.model = ShenDecoder()
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
        self.model.eval()
        self.lex = Lexicon(sorted(set(lexicon_words)))

    @torch.no_grad()
    def __call__(self, trace_kb, kb, k=4):
        t0 = time.perf_counter()
        f = featurize(np.asarray(trace_kb)[:, :2], kb)
        if f is None:
            return [], 0.0
        logits = self.model(torch.from_numpy(f)[None])[0]
        return self.lex.topk(logits, k), (time.perf_counter() - t0) * 1000


def load_lexicon(path):
    if path.endswith(".npz"):
        return list(np.load(path)["words"])
    return [w.strip().lower() for w in open(path) if w.strip()]
