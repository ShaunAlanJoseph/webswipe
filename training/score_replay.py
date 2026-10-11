"""Score left-hand pinch controls offline on cached replay rows.

Replays the app's left-hand control logic over rows written by
replay_left_pinch.py (no model is loaded) and counts, per prompted trial,
whether exactly one expected action fired ("next" for a quick pinch that
cycles suggestions, "delete" for a held pinch).

Two control paths are available:
  fused     webswipe.app.left_pinch.FusedLeftPinch (the demo's left-hand path)
  baseline  ContactPinch + ContactAppearanceGate(0.3) + ReleaseDelay(80 ms, 2 start votes)

Trial windows come from the recording's trials.json ([start_ms, end_ms + extra]).
Quiet intervals are spans where the user was told not to pinch; any action
there is reported in quiet_events.

Example (10 quick left pinches, native rate and both 10 fps phases):
    for p in native 10fps_phase0 10fps_phase50; do
      python training/score_replay.py --rows runs/left_pinch_replay/$p.jsonl \
          --trials data/left_quick_pinches/trials.json --quiet 0:8000 48000:51066 \
          --out runs/left_pinch_replay/$p.score.json
    done
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from webswipe.app.pinch import ContactPinch
from webswipe.app.controls import ContactAppearanceGate, ReleaseDelay, OtherHandControl
from webswipe.app.left_pinch import FusedLeftPinch


def read(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines()]


def run(rows, new):
    """Return (events, telemetry). new=True uses FusedLeftPinch, else the baseline gate chain."""
    pinch = ContactPinch()
    gate = ContactAppearanceGate(.3)
    release = ReleaseDelay(80, start_votes=2)
    ctl = OtherHandControl()
    candidate = FusedLeftPinch()
    events = []
    telemetry = []
    for r in rows:
        ids = [i for i, h in enumerate(r['hands']) if h['side'] == 'Left']
        i = max(ids, key=lambda i: r['hands'][i]['score']) if ids else None
        contact = False
        raw = None
        scores = {}
        if i is not None:
            px = np.array(r['hands'][i]['px'])
            contact, ratio = pinch.update(px)
            raw = float(np.linalg.norm(px[4] - px[8]) / max(pinch.scale, 1e-6))
            scores = r['models']['crop']['scores'][i]
        if new:
            active = candidate.update(contact, raw, scores, r['t_ms'])
        else:
            active = release.update(gate.update(contact, scores, active=release.on), r['t_ms'])
        if i is None:
            pinch.reset_state()
        action = ctl.update(active, r['t_ms'])
        if action:
            events.append({'t_ms': r['t_ms'], 'action': action})
        telemetry.append({'t_ms': r['t_ms'], 'active': bool(active), 'raw_ratio': raw, 'contact': bool(contact),
                          'scores': scores, 'action': action})
    return events, telemetry


def summarize(events, trials, quiet, extra):
    rows = []
    for t in trials:
        ev = [e for e in events if t['start_ms'] <= e['t_ms'] <= t['end_ms'] + extra]
        want = t.get('expected_action', 'next_candidate')
        want = 'next' if want == 'next_candidate' else want
        rows.append({'id': t['id'], 'expected': want, 'duration_ms': t['end_ms'] - t['start_ms'], 'events': ev,
                     'single_expected': len(ev) == 1 and ev[0]['action'] == want})
    return {'n': len(rows), 'any_action': sum(bool(t['events']) for t in rows),
            'single_expected': sum(t['single_expected'] for t in rows),
            'multiple': sum(len(t['events']) > 1 for t in rows),
            'deletes': sum(e['action'] == 'delete' for e in events),
            'quiet_events': [e for e in events if any(a <= e['t_ms'] < b for a, b in quiet)],
            'trials': rows, 'events': events}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--rows', type=Path, required=True, help='<pass>.jsonl from replay_left_pinch.py')
    p.add_argument('--trials', type=Path, required=True, help="the recording's trials.json")
    p.add_argument('--recorded-controls-only', action='store_true',
                   help='keep only kind=control trials with status=recorded and an end_ms '
                        '(benchmark recordings that also contain word trials)')
    p.add_argument('--quiet', nargs='*', default=[], metavar='START:END',
                   help='no-pinch intervals in ms, e.g. 0:8000')
    p.add_argument('--extra-ms', type=int, default=0, help='grace period after each trial end')
    p.add_argument('--mode', choices=['fused', 'baseline', 'both'], default='both')
    p.add_argument('--out', type=Path, help='write the full summary JSON here')
    p.add_argument('--telemetry', type=Path, help='write per-frame telemetry JSON here (fused, or baseline if only one)')
    return p.parse_args()


def main():
    args = parse_args()
    rows = read(args.rows)
    trials = json.loads(args.trials.read_text())
    if args.recorded_controls_only:
        trials = [t for t in trials if t['kind'] == 'control' and t.get('status') == 'recorded' and t.get('end_ms')]
    quiet = [tuple(int(x) for x in q.split(':')) for q in args.quiet]
    modes = {'fused': [True], 'baseline': [False], 'both': [False, True]}[args.mode]
    results = {}
    for new in modes:
        key = 'fused' if new else 'baseline'
        ev, te = run(rows, new)
        results[key] = summarize(ev, trials, quiet, args.extra_ms)
        if args.telemetry and (new or len(modes) == 1):
            args.telemetry.write_text(json.dumps(te))
    print(json.dumps({k: {a: v[a] for a in ['n', 'any_action', 'single_expected', 'multiple', 'deletes', 'quiet_events']}
                      for k, v in results.items()}, indent=2))
    if args.out:
        args.out.write_text(json.dumps({'rows': args.rows.name, 'frames': len(rows), 'results': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
