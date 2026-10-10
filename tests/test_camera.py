import threading
from webswipe.app import camera as m

def test_slow_consumer_gets_latest_frame_and_eof():
    import queue
    class Capture:
        def __init__(self): self.q=queue.Queue();self.released=False
        def read(self): return self.q.get(timeout=1)
        def release(self): self.released=True
    c=Capture();reader=m.LatestFrameCamera(c)
    c.q.put((True,1));assert reader.read()==(True,1)
    for n in range(2,9):c.q.put((True,n))
    with reader.condition:
        assert reader.condition.wait_for(lambda:reader.sequence==8,timeout=1)
    assert reader.read()==(True,8)
    c.q.put((False,None));assert reader.read()==(False,None)
    reader.release();assert c.released

def test_camera_failure_wakes_reader():
    class Capture:
        def read(self):return False,None
        def release(self):pass
    c=m.LatestFrameCamera(Capture());assert c.read()==(False,None);c.release()
