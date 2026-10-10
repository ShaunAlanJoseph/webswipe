"""Optional release veto for a lighting-adapted pinch detector."""

def checked_scores(primary, reference, threshold=0.5):
    """A reference open-hand prediction vetoes pinch; it never creates one.

    Keep all other primary scores for telemetry and start-buffer behavior.
    Neither dictionary is mutated. Missing reference evidence does not veto.
    """
    scores = dict(primary or {})
    reference = reference or {}
    if max(reference.get("no_gesture", 0.0), reference.get("palm", 0.0)) >= threshold:
        scores["ok"] = 0.0
    return scores
