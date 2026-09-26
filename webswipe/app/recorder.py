# save gestures for the webcam dataset (D3), one JSON line per word
import json
import time
from collections import deque
from pathlib import Path


class Recorder:
    def __init__(self, participant, out_dir="data/recordings", pre_frames=30):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.path = Path(out_dir) / f"{participant}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.participant = participant
        self.pre = deque(maxlen=pre_frames)
        self.frames = None
        self.count = 0

    def frame(self, t_ms, hands):
        f = {"t": t_ms, "hands": [{"side": h.side, "score": round(h.score, 3),
                                   "px": h.px.round(2).tolist(), "world": h.world.round(5).tolist()}
                                  for h in hands]}
        if self.frames is None:
            self.pre.append(f)
        else:
            self.frames.append(f)

    def start(self):
        self.frames = list(self.pre)
        self.start_idx = len(self.frames)

    def cancel(self):
        self.frames = None

    def save(self, target, trace, keyboard, extra=None):
        rec = {"participant": self.participant, "target": target, "trace_kb": trace,
               "keyboard": keyboard, "pinch_start_frame": self.start_idx, "frames": self.frames,
               "saved": time.time(), **(extra or {})}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.frames = None
        self.count += 1
