# per-frame state machines for the two hands; pure logic, so they can be tested without a camera


class SwipeSegmenter:
    """Dominant hand: turns per-frame pinch states into word gestures.

    A gesture survives short tracking dropouts (hand not detected for up to max_gap_ms): without this,
    one missed MediaPipe frame mid-swipe commits a truncated word.
    """

    def __init__(self, max_gap_ms=200, release_ms=0):
        self.max_gap_ms = max_gap_ms
        self.release_ms = release_ms
        self.release_since = None
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
                self.release_since = None
                return "end", False
            return None, False  # hold the gesture open, no point this frame
        self.lost_since = None
        if pinched:
            self.release_since = None
        if pinched and not self.active:
            self.active = True
            return "start", True
        if not pinched and self.active:
            if self.release_since is None:
                self.release_since = t_ms
            if t_ms - self.release_since < self.release_ms:
                return None, False  # exclude uncertain/released points from the word
            self.release_since = None
            self.active = False
            return "end", False
        return None, self.active


class ReleaseDelay:
    """Bridge brief false readings; sustained release turns off after delay_ms."""

    def __init__(self, delay_ms=120):
        self.delay_ms = delay_ms
        self.on = False
        self.off_since = None

    def update(self, detected, t_ms):
        if detected:
            self.on, self.off_since = True, None
        elif self.on:
            if self.off_since is None:
                self.off_since = t_ms
            if t_ms - self.off_since >= self.delay_ms:
                self.on, self.off_since = False, None
        return self.on


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


class SwipeStartBuffer:
    """Keep a bounded, contiguous pinch-like prefix while start approval is pending.

    Strict contact qualifies without detector approval. Near-contact qualifies
    only with a confident pinch prediction. Ordinary hovering cannot contribute.
    """

    def __init__(self, lookback_ms=700, approach_ratio=0.4, approach_score=0.6):
        from collections import deque
        self.points = deque()
        self.lookback_ms = lookback_ms
        self.approach_ratio = approach_ratio
        self.approach_score = approach_score

    def update(self, cursor, t_ms, contact, ratio, scores=None, active=False):
        confident_approach = (ratio is not None and ratio <= self.approach_ratio
                             and (scores or {}).get('ok', 0.0) >= self.approach_score)
        if active or cursor is None or not (contact or confident_approach):
            self.clear()
            return
        self.points.append((float(cursor[0]), float(cursor[1]), t_ms))
        while self.points and t_ms - self.points[0][2] > self.lookback_ms:
            self.points.popleft()

    def take_before(self, t_ms):
        points = [p for p in self.points if p[2] < t_ms]
        self.clear()
        return points

    def clear(self):
        self.points.clear()


class GuardedStartBuffer:
    """Experimental start recovery: anchor a short lead-in on a tight contact.

    Contact alone can be a near-pinch false positive. Before tighter closure,
    retain only a short lead-in. Once closure is observed, preserve that entire
    prefix while waiting for detector approval, rather than sliding away its
    first letter. Expired prefixes are rejected until separation, not truncated.
    This is development policy, not proof of physical finger contact.
    """

    def __init__(self, lookback_ms=700, anchor_ratio=0.16, max_wait_ms=2500):
        from collections import deque
        self.points = deque()
        self.lookback_ms = lookback_ms
        self.anchor_ratio = anchor_ratio
        self.max_wait_ms = max_wait_ms
        self.ready = False
        self.expired = False

    def update(self, cursor, t_ms, contact, ratio, scores=None, active=False):
        if active or cursor is None or not contact or ratio is None:
            self.clear()
            return
        if self.expired:
            return
        self.points.append((float(cursor[0]), float(cursor[1]), t_ms))
        if not self.ready:
            while self.points and t_ms - self.points[0][2] > self.lookback_ms:
                self.points.popleft()
            self.ready = ratio <= self.anchor_ratio
        if self.ready and t_ms - self.points[0][2] > self.max_wait_ms:
            self.points.clear()
            self.ready, self.expired = False, True

    def take_before(self, t_ms):
        points = [p for p in self.points if p[2] < t_ms] if self.ready else []
        self.clear()
        return points

    def clear(self):
        self.points.clear()
        self.ready = False
        self.expired = False
