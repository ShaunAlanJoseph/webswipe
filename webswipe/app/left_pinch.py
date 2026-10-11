from webswipe.app.controls import ContactAppearanceGate, ReleaseDelay

class FusedLeftPinch:
    """Confirmed pinch for the non-dominant hand's cycle/delete controls.

    Preserve the established contact gate for weaker appearance evidence. A new
    high-confidence pinch can start on close raw fingertips before the contact
    median catches up (up to 250 ms of evidence alignment). Static near-pinches
    cannot use that shortcut without a recent opening or a very small tip gap.

    Two moderate-confidence observations can also confirm a rapid closure from
    a recently verified open pose, before smoothed contact crosses its threshold.
    That fallback requires measured closing movement and leaves legacy holds alone.

    Fast episodes also accept 200 ms of confident visual release to avoid a
    tracking glitch becoming a 600 ms delete. Separate geometry/appearance
    timers avoid mixing unrelated release evidence. Suppress an old contact
    latch after visual release and debounce fast re-entry for 200 ms.
    """
    def __init__(self, release_ms=80, approval_score=.3):
        self.on = False
        self.release_ms = release_ms
        self.approval_score = approval_score
        self.gate = ContactAppearanceGate(approval_score)
        self.legacy = ReleaseDelay(release_ms, start_votes=2)
        self.mode = None
        self.geometry_since = self.appearance_since = None
        self.last_open = self.last_nonpinch = None
        self.suppress_legacy = False
        self.last_end = None
        self.open_anchor = None
        self.close_votes = 0
        self.close_observed_at = None

    def update(self, contact, raw_ratio, scores, t_ms):
        scores = scores or {}
        ok = scores.get('ok', 0.0)
        if ok < self.approval_score:
            self.last_nonpinch = t_ms
            if raw_ratio is not None and raw_ratio > .35:
                self.last_open = t_ms
        # Two agreeing observations can rescue a brief closure whose tracked
        # tips never cross the smoothed contact threshold. Require actual closing
        # movement from a recently appearance-confirmed open pose, not a static
        # near-pinch or an isolated classifier spike.
        if (ok < self.approval_score and not contact and raw_ratio is not None
            and max(scores.get('palm', 0), scores.get('no_gesture', 0)) >= .8):
            self.open_anchor = (t_ms, raw_ratio)
        closing = (raw_ratio is not None and raw_ratio <= .27
                   and ok >= max(.7, self.approval_score)
                   and self.open_anchor is not None
                   and 0 <= t_ms - self.open_anchor[0] <= 250
                   and self.open_anchor[1] - raw_ratio >= .07)
        if closing:
            consecutive = (self.close_observed_at is not None
                           and 0 < t_ms - self.close_observed_at <= 150)
            self.close_votes = self.close_votes + 1 if consecutive else 1
            self.close_observed_at = t_ms
        else:
            self.close_votes = 0
            self.close_observed_at = None
        corroborated = (self.close_votes >= 2
                        and (self.last_end is None or t_ms - self.last_end >= 200))
        strong = ((self.last_end is None or t_ms-self.last_end >= 200)
                  and raw_ratio is not None and raw_ratio <= .27 and ok >= max(.8, self.approval_score)
                  and (raw_ratio <= .18 or (self.last_open is not None and t_ms-self.last_open <= 350))
                  and self.last_nonpinch is not None and t_ms-self.last_nonpinch <= 250)
        legacy = self.legacy.update(self.gate.update(contact,scores,active=self.legacy.on),t_ms)
        strong = strong or (corroborated and not legacy)
        if not legacy:
            self.suppress_legacy = False
        if self.mode == 'legacy':
            self.on = legacy
            if not self.on:
                self.mode = None
                self.last_end = t_ms
            return self.on
        if self.mode is None:
            if strong:
                self.mode = 'fast'
                self.last_open = None
                self.geometry_since = self.appearance_since = None
                self.on = True
            elif legacy and not self.suppress_legacy:
                self.mode = 'legacy'
                self.on = True
            return self.on
        # Fast onset explicitly trusted the model; let confident release evidence
        # end it even if the tracker keeps the fingertip positions together.
        separated = raw_ratio is None or (not contact and raw_ratio > .27)
        opened = max(scores.get('palm',0),scores.get('no_gesture',0)) >= .8 and ok < self.approval_score
        self.geometry_since = (t_ms if self.geometry_since is None else self.geometry_since) if separated else None
        self.appearance_since = (t_ms if self.appearance_since is None else self.appearance_since) if opened else None
        if ((self.geometry_since is not None and t_ms-self.geometry_since >= self.release_ms)
            or (self.appearance_since is not None and t_ms-self.appearance_since >= 200)):
            self.on = False
            self.mode = None
            self.suppress_legacy = legacy
            self.last_end = t_ms
        return self.on
