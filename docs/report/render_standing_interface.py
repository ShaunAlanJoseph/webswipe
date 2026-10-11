"""Render the standing view using the archived frame and frozen UI helpers."""
from pathlib import Path
import sys
import cv2 as cv
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'evidence/runtime_source'))
from webswipe.app.keyboard_ui import KeyboardOverlay
from webswipe.app.draw import text
frame = cv.imread(str(ROOT / 'evidence/standing_interface_frame.png'))
assert frame is not None
height, width = frame.shape[:2]
image = KeyboardOverlay(width, height).draw(frame)
text(image, 'C cursor off | T trail off | H skeleton off', (10, height - 38), 0.45)
assert cv.imwrite(str(ROOT / 'images/app_replay_standing.png'), image)
