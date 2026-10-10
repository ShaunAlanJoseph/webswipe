"""Optional camera reader that keeps only the newest frame during slow inference."""
import threading
import time


class LatestFrameCamera:
    def __init__(self, capture, timeout=2.0):
        self.capture = capture
        self.timeout = timeout
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.frame = None
        self.sequence = self.delivered = 0
        self.captured_at = None
        self.failed = False
        self.worker = threading.Thread(target=self._pump, name='webswipe-camera', daemon=True)
        self.worker.start()

    def _pump(self):
        try:
            while not self.stop.is_set():
                ok, frame = self.capture.read()
                with self.condition:
                    if not ok:
                        self.failed = True
                        self.condition.notify_all()
                        return
                    self.frame = frame
                    self.captured_at = time.monotonic()
                    self.sequence += 1
                    self.condition.notify_all()
        finally:
            with self.condition:
                self.failed = True
                self.condition.notify_all()

    def read(self):
        with self.condition:
            fresh = self.condition.wait_for(
                lambda: self.sequence > self.delivered or self.failed or self.stop.is_set(),
                timeout=self.timeout)
            if not fresh or self.sequence <= self.delivered:
                return False, None
            self.delivered = self.sequence
            # Each capture read owns a new array; the worker only replaces its reference.
            return True, self.frame

    def release(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        self.worker.join(timeout=self.timeout)
        # Avoid releasing a backend concurrently with a blocked read (unsafe in OpenCV).
        if not self.worker.is_alive():
            self.capture.release()
