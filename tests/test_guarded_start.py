from webswipe.app.controls import GuardedStartBuffer


def test_delayed_approval_keeps_first_letter_after_anchor():
    b = GuardedStartBuffer()
    for t in range(0, 1301, 50):
        b.update((250-t/20, 0), t, True, .12, {'fist': .9})
    assert b.ready
    points = b.take_before(1300)
    assert points[0] == (250., 0., 0)
    assert points[-1][2] == 1250
    assert not b.ready and not b.points


def test_near_hover_never_approves_start_even_with_false_detector_vote():
    b = GuardedStartBuffer()
    for t in range(0, 3000, 50):
        b.update((t, 0), t, True, .195, {'ok': .95})
        assert not b.ready
    assert b.take_before(3000) == []


def test_anchor_keeps_short_leadin_but_drops_old_hover():
    b = GuardedStartBuffer()
    for t in range(0, 1501, 50):
        b.update((t, 0), t, True, .19 if t < 1000 else .12)
    pts = b.take_before(1500)
    assert pts[0][2] == 300  # 700 ms before tighter closure, not before approval
    assert pts[-1][2] == 1450


def test_expiry_rejects_whole_prefix_until_release_instead_of_losing_letter():
    b = GuardedStartBuffer(max_wait_ms=1000)
    for t in range(0, 1600, 50):
        b.update((t, 0), t, True, .1)
    assert b.expired and not b.ready and not b.points
    b.update((0, 0), 1600, False, .4)
    b.update((1, 0), 1650, True, .1)
    assert b.ready and list(b.points) == [(1., 0., 1650)]


def test_tracking_loss_and_active_gesture_clear_pending_prefix():
    b = GuardedStartBuffer()
    b.update((1, 2), 0, True, .1)
    b.update(None, 50, False, None)
    assert not b.ready and not b.points
    b.update((3, 4), 100, True, .1)
    b.update((5, 6), 150, True, .1, active=True)
    assert not b.ready and not b.points
