"""Webcam head-yaw tracking with MediaPipe Face Landmarker."""
import math
import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")


def _capture_api():
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW            # MSMF is very slow to open
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


def list_cameras():
    """[(index, name)] of cameras. Linux knows names; elsewhere the indexes are probed."""
    base = "/sys/class/video4linux"
    if os.path.isdir(base):
        out = []
        for dev in sorted(os.listdir(base)):
            try:
                with open(f"{base}/{dev}/index") as f:
                    if f.read().strip() != "0":      # skip metadata nodes
                        continue
                with open(f"{base}/{dev}/name") as f:
                    out.append((int(dev.removeprefix("video")), f.read().strip()))
            except (OSError, ValueError):
                continue
        return sorted(out)
    out = []
    for i in range(4):
        cap = cv2.VideoCapture(i, _capture_api())
        if cap.isOpened():
            out.append((i, f"Camera {i}"))
        cap.release()
    return out


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


class HeadTracker:
    def __init__(self, camera=0, width=640, height=480, fps=15):
        ensure_model()
        self.cap = cv2.VideoCapture(camera, _capture_api())
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {camera}")
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.min_interval = 1.0 / fps
        self._last = 0.0
        self._t0 = time.monotonic()

    def read(self):
        """Return (yaw_degrees or None, frame). Blocks to cap the frame rate."""
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.monotonic() - self._t0) * 1000)
        res = self.landmarker.detect_for_video(img, ts)
        if not res.facial_transformation_matrixes:
            return None, frame
        return _yaw(np.array(res.facial_transformation_matrixes[0])), frame

    def close(self):
        self.cap.release()
        self.landmarker.close()


def _yaw(m):
    """Yaw in degrees from MediaPipe's 4x4 face transform (sign fixed by calibration)."""
    r = m[:3, :3] / np.linalg.norm(m[:3, :3], axis=0)
    return math.degrees(math.atan2(r[0, 2], r[2, 2]))
