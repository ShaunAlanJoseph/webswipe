# MediaPipe Tasks hand tracker
import time
from dataclasses import dataclass

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL = "models/hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")


@dataclass
class Hand:
    px: np.ndarray       # (21, 2) pixel coords
    world: np.ndarray    # (21, 3) metric coords from MediaPipe
    side: str            # "Left" / "Right" as seen in the mirrored image
    score: float


class HandTracker:
    def __init__(self, model=MODEL, num_hands=2, det_conf=0.5, track_conf=0.5):
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model),
            running_mode=vision.RunningMode.VIDEO, num_hands=num_hands,
            min_hand_detection_confidence=det_conf, min_tracking_confidence=track_conf)
        self.lm = vision.HandLandmarker.create_from_options(opts)
        self.t0 = time.monotonic()
        self.last_ms = -1

    def __call__(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB))
        ms = max(int((time.monotonic() - self.t0) * 1000), self.last_ms + 1)
        self.last_ms = ms
        r = self.lm.detect_for_video(img, ms)
        hands = []
        for lms, wl, hd in zip(r.hand_landmarks, r.hand_world_landmarks, r.handedness):
            px = np.array([[p.x * w, p.y * h] for p in lms])
            world = np.array([[p.x, p.y, p.z] for p in wl])
            # the app mirrors the frame before tracking, and the Tasks API then reports the opposite hand
            # (the user's right hand came back as "Left"); swap so "Right" is the user's right hand
            side = {"Left": "Right", "Right": "Left"}.get(hd[0].category_name, hd[0].category_name)
            hands.append(Hand(px, world, side, hd[0].score))
        return hands, ms
