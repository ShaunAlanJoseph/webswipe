# per-frame state machines for the two hands; pure logic, so they can be tested without a camera


class SwipeSegmenter:
    """Dominant hand: turns per-frame pinch states into word gestures.

    A gesture survives short tracking dropouts (hand not detected for up to max_gap_ms): without this,
    one missed MediaPipe frame mid-swipe commits a truncated word.
    """

    def __init__(self, max_gap_ms=200):
        self.max_gap_ms = max_gap_ms
        self.active = False
        self.lost_since = None

    def update(self, hand_seen, pinched, t_ms):
        """Returns (event, record): event is "start", "end" or None; record = append this frame's point."""
        if not hand_seen:
            if not self.active:
                return None, False
            if self.lost_since is None:
                self.lost_since = t_ms
            if t_ms - self.lost_since > self.max_gap_ms:
                self.active, self.lost_since = False, None
                return "end", False
            return None, False  # hold the gesture open, no point this frame
        self.lost_since = None
        if pinched and not self.active:
            self.active = True
            return "start", True
        if not pinched and self.active:
            self.active = False
            return "end", False
        return None, self.active


class OtherHandControl:
    """Non-dominant hand: short pinch = next candidate for the last word, long pinch = delete it."""

    def __init__(self, hold_ms=600):
        self.hold_ms = hold_ms
        self.down_at = None
        self.fired = False

    def update(self, pinched, t_ms):
        """Returns "next", "delete" or None."""
        if pinched:
            if self.down_at is None:
                self.down_at, self.fired = t_ms, False
            elif not self.fired and t_ms - self.down_at >= self.hold_ms:
                self.fired = True
                return "delete"  # fire while still held, so the user sees it happen
            return None
        if self.down_at is None:
            return None
        short = not self.fired
        self.down_at = None
        return "next" if short else None


class Candidates:
    """Typed words plus the candidate list of the last word, so it can be swapped after the fact."""

    def __init__(self):
        self.typed, self.cands, self.idx = [], [], 0

    def commit(self, cands):
        if cands:
            self.typed.append(cands[0])
            self.cands, self.idx = list(cands), 0

    def choose(self, i):
        if self.cands and self.typed and 0 <= i < len(self.cands):
            self.idx = i
            self.typed[-1] = self.cands[i]

    def next(self):
        if self.cands:
            self.choose((self.idx + 1) % len(self.cands))

    def delete(self):
        if self.typed:
            self.typed.pop()
        self.cands, self.idx = [], 0
