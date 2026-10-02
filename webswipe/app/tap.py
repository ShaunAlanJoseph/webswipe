# per-key (tap) typing: the camera-only baseline from the intro, for the user study
import json
import time
from pathlib import Path

import numpy as np

from webswipe.decoder.features import LETTERS


class TapTyper:
    """Pinch on a key = that letter; space commits the word, optionally snapped to the lexicon
    (best same-length word by summed squared tap-to-key distance, as in simulation E1/E5)."""

    def __init__(self, kb, lexicon=None):
        self.kb = kb
        self.idx = {c: i for i, c in enumerate(LETTERS)}
        self.by_len = {}
        for w in sorted(set(lexicon or [])):
            if w and all(c in self.idx for c in w):
                self.by_len.setdefault(len(w), []).append(w)
        self.paths = {L: np.stack([kb.centers[[self.idx[c] for c in w]] for w in ws])
                      for L, ws in self.by_len.items()}
        self.typed, self.letters, self.taps = [], [], []

    def tap(self, pos_kb):
        """pos_kb: (x, y) in keyboard frame. Returns the letter typed (nearest key centre)."""
        p = np.asarray(pos_kb, float)
        c = LETTERS[int(np.argmin(np.linalg.norm(self.kb.centers - p, axis=1)))]
        self.letters.append(c)
        self.taps.append(p)
        return c

    def space(self):
        """Commit the current letters as a word. Returns (raw, committed) or None if empty."""
        if not self.letters:
            return None
        raw = "".join(self.letters)
        word = raw
        if len(raw) in self.paths:
            d = ((self.paths[len(raw)] - np.stack(self.taps)) ** 2).sum((1, 2))
            word = self.by_len[len(raw)][int(np.argmin(d))]
        self.typed.append(word)
        self.letters, self.taps = [], []
        return raw, word

    def backspace(self):
        if self.letters:
            self.letters.pop()
            self.taps.pop()
        elif self.typed:
            self.typed.pop()

    @property
    def text(self):
        return " ".join(self.typed + (["".join(self.letters)] if self.letters else []))


class EventLog:
    """One JSON line per event (typed word, tap, delete, ...) for WPM / error-rate analysis."""

    def __init__(self, participant, mode, out_dir="data/sessions"):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.path = Path(out_dir) / f"{participant}_{mode}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.f = open(self.path, "a")

    def __call__(self, t_ms, kind, **data):
        self.f.write(json.dumps({"t": t_ms, "event": kind, **data}) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()
