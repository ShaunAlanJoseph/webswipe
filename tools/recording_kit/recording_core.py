import json
import time
import uuid
import cv2 as cv

def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

class Attempt:
    def __init__(self, root, target, index, participant, keyboard):
        self.path = root / ('attempt_' + uuid.uuid4().hex[:10])
        self.path.mkdir()
        self.data = dict(schema=1, target=target, word_index=index, participant=participant,
                         keyboard=keyboard, status='recording', segments=[], frame_count=0,
                         created=time.time(), video='camera.avi', telemetry='frames.jsonl',
                         mirrored=True, video_fps=30, timing='Use telemetry timestamps; AVI is nominal 30 fps.')
        self.writer = None
        self.log = (self.path / 'frames.jsonl').open('w')
        self.closed = False
        write_json(self.path / 'attempt.json', self.data)

    def frame(self, image, telemetry):
        if self.writer is None:
            h, w = image.shape[:2]
            self.writer = cv.VideoWriter(str(self.path / 'camera.avi'), cv.VideoWriter_fourcc(*'MJPG'), 30, (w, h))
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                raise RuntimeError('Cannot save video (MJPG). Recording stopped; check codec/disk space.')
        self.writer.write(image)
        telemetry = dict(telemetry, frame_index=self.data['frame_count'])
        self.log.write(json.dumps(telemetry) + '\n')
        self.log.flush()
        self.data['frame_count'] += 1

    def segment(self, trace, boundary='release'):
        self.data['segments'].append(dict(trace_kb=[list(p) for p in trace], boundary=boundary,
                                          candidates=[], decode_ms=None))

    def finish(self, decoder, kb):
        self.close()
        self.data['status'] = 'pending_review'
        write_json(self.path / 'attempt.json', self.data)
        for segment in self.data['segments']:
            if len(segment['trace_kb']) >= 4:
                try:
                    cands, ms = decoder(segment['trace_kb'], kb)
                    segment.update(candidates=cands, decode_ms=ms)
                except Exception as exc:
                    segment['decode_error'] = str(exc)
        write_json(self.path / 'attempt.json', self.data)

    def feedback(self, status):
        self.data['status'] = status
        segments = self.data['segments']
        self.data['training_eligible'] = (status == 'accepted' and len(segments) == 1
                                         and len(segments[0]['trace_kb']) >= 4
                                         and segments[0]['boundary'] == 'release')
        write_json(self.path / 'attempt.json', self.data)

    def close(self):
        if self.closed:
            return
        if self.writer is not None:
            self.writer.release()
        self.log.close()
        self.closed = True
