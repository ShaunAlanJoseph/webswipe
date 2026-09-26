# greedy and lexicon-constrained CTC decoding
import numpy as np
import torch
import torch.nn.functional as F

from .features import BLANK, LETTERS, encode_word


def greedy(logits):
    """logits: (B, T, C) -> list of strings"""
    out = []
    for seq in logits.argmax(-1).tolist():
        s, prev = [], None
        for k in seq:
            if k != prev and 1 <= k <= 26:
                s.append(LETTERS[k - 1])
            prev = k
        out.append("".join(s))
    return out


class Lexicon:
    def __init__(self, words, logprior=None):
        self.words = list(words)
        enc = [encode_word(w) for w in self.words]
        self.lens = torch.tensor([len(e) for e in enc])
        self.targets = torch.zeros(len(enc), int(self.lens.max()), dtype=torch.long)
        for i, e in enumerate(enc):
            self.targets[i, :len(e)] = torch.tensor(e)
        self.logprior = torch.zeros(len(enc)) if logprior is None else torch.as_tensor(logprior, dtype=torch.float)

    @torch.no_grad()
    def scores(self, logits, prior_weight=0.0):
        """logits: (T, C) for one gesture -> log-likelihood of every word (V,)"""
        V, T = len(self.words), logits.shape[0]
        logp = logits.log_softmax(-1)[:, None].expand(T, V, -1)
        nll = F.ctc_loss(logp, self.targets, torch.full((V,), T, dtype=torch.long), self.lens,
                         blank=BLANK, reduction="none", zero_infinity=True)
        return -nll + prior_weight * self.logprior

    def topk(self, logits, k=4, prior_weight=0.0):
        s = self.scores(logits, prior_weight)
        idx = torch.topk(s, min(k, len(self.words))).indices.tolist()
        return [self.words[i] for i in idx]


def lexicon_accuracy(logits, words, lex, ks=(1, 4)):
    hits = np.zeros(len(ks))
    for lg, w in zip(logits, words):
        top = lex.topk(lg, max(ks))
        hits += [w in top[:k] for k in ks]
    return (hits / len(words)).tolist()
