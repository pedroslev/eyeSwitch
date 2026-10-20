"""Webcam head-yaw tracking with MediaPipe Face Landmarker."""
import math
import os
import sys
import time
import urllib.request
from collections import namedtuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from virtualcam import LABEL as VIRTUAL_LABEL

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
# Landmark ids (MediaPipe face mesh with iris): outer corner, inner corner, iris center.
# (The lid landmarks are too close together at webcam resolution to be a usable signal.)
RIGHT_EYE = (33, 133, 468)
LEFT_EYE = (263, 362, 473)

NOSE_TIP = 1

# What the tracker measures per frame. yaw/pitch in degrees; tx/ty/tz = head position in the camera
# frame (shown for debugging only: it says where you sit, not where you look); h/v = iris offset
# inside the eye in eye widths (right/down positive); nx/ny = where the nose tip sits relative to
# the eyes, in eye distances along/across the eye line: a head-pointing cue that does not depend on
# how far or off-centre you sit. Signs are arbitrary (camera-dependent); calibration absorbs them.
Face = namedtuple("Face", "yaw pitch tx ty tz h v nx ny")
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
                    name = f.read().strip()
                if name != VIRTUAL_LABEL:            # never offer our own virtual camera as input
                    out.append((int(dev.removeprefix("video")), name))
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


TRACK_FPS = 15          # face tracking rate; the camera may run faster to feed a virtual camera


# Tried in this order when the resolution is "auto". Not 4K: the iris landmarks stop improving and the
# extra pixels only cost CPU. At 640x480 an eye is ~20 px wide, at 1080p ~54 px.
SIZES = ((1920, 1080), (1280, 720), (640, 480))


class HeadTracker:
    def __init__(self, camera=0, sink=None, blendshapes=False, height="auto"):
        """Opens the camera at the best resolution it really delivers (`height` "auto"), or at 1080/720
        when asked and possible. `sink(frame)` receives every camera frame (for the virtual camera, at
        30 fps so calls look good) while tracking still runs at TRACK_FPS."""
        ensure_model()
        self.sink = sink
        self.cap = cv2.VideoCapture(camera, _capture_api())
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {camera}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))    # raw formats cannot carry HD over USB
        self.cap.set(cv2.CAP_PROP_FPS, 30 if sink else TRACK_FPS)
        self.width, self.height = self._pick_size(height)
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
            output_face_blendshapes=blendshapes,
        )
        self.shapes = None           # {blendshape name: 0..1} of the last frame, when enabled
        self.landmarks = self.matrix = self.frame_wh = None
        self.blink = None            # mean eyelid closure score 0..1 (needs blendshapes)
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.min_interval = 0.8 / TRACK_FPS          # 0.8: do not skip frames to timing jitter
        self._last = 0.0
        self._t0 = time.monotonic()

    def _pick_size(self, height):
        """The first size the camera really delivers: the best one, or the asked height if it can."""
        wanted = SIZES if height == "auto" else [z for z in SIZES if z[1] == int(height)] or SIZES
        got = wanted[-1]
        for w, h in wanted:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            ok, frame = self.cap.read()                  # the camera may need a frame to switch modes
            if ok:
                got = (frame.shape[1], frame.shape[0])
                if got == (w, h):
                    break
        return got

    def read(self):
        """Return (Face or None, frame) at about TRACK_FPS. Extra frames go to the sink."""
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return None, None
            if self.sink is not None:
                self.sink(frame)
            now = time.monotonic()
            if now - self._last >= self.min_interval:
                break
        self._last = now
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect_for_video(img, int((now - self._t0) * 1000))
        self.shapes = ({c.category_name: c.score for c in res.face_blendshapes[0]}
                       if res.face_blendshapes else None)
        self.landmarks = res.face_landmarks[0] if res.face_landmarks else None   # for the debug view
        self.blink = (None if not self.shapes else
                      (self.shapes["eyeBlinkLeft"] + self.shapes["eyeBlinkRight"]) / 2)
        if not res.facial_transformation_matrixes:
            return None, frame
        self.matrix, self.frame_wh = np.array(res.facial_transformation_matrixes[0]), (frame.shape[1], frame.shape[0])
        face = _face(self.matrix, res.face_landmarks[0], frame.shape[1], frame.shape[0])
        return face, frame

    def raw(self):
        """The last frame's raw geometry: (478x3 landmarks, 4x4 face pose, frame size). Numbers only,
        no image. Saved with calibrations so other models can be tried on the same data offline."""
        if self.landmarks is None or self.matrix is None:
            return None
        pts = np.array([[p.x, p.y, p.z] for p in self.landmarks], np.float32)
        return pts, self.matrix.astype(np.float32), self.frame_wh, (dict(self.shapes) if self.shapes else None)

    def close(self):
        self.cap.release()
        self.landmarker.close()


def _face(m, lm, w, h):
    """Head pose from MediaPipe's 4x4 face transform plus iris features from the landmarks.

    Signs are arbitrary (they depend on the camera); calibration absorbs them.
    """
    r = m[:3, :3] / np.linalg.norm(m[:3, :3], axis=0)
    yaw = math.degrees(math.atan2(r[0, 2], r[2, 2]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, r[1, 2]))))
    eyes = [_eye(lm, ids, w, h) for ids in (RIGHT_EYE, LEFT_EYE)]
    hh, vv = np.mean(eyes, axis=0)
    nx, ny = _nose(lm, w, h)
    return Face(yaw, pitch, float(m[0, 3]), float(m[1, 3]), float(m[2, 3]), float(hh), float(vv), nx, ny)


def _nose(lm, w, h):
    """Nose tip relative to the midpoint of the outer eye corners, in units of the eye distance."""
    right, left, nose = (np.array([lm[i].x * w, lm[i].y * h]) for i in (RIGHT_EYE[0], LEFT_EYE[0], NOSE_TIP))
    axis = left - right
    dist = np.linalg.norm(axis)
    axis /= dist
    perp = np.array([-axis[1], axis[0]])
    off = nose - (left + right) / 2
    return float(off @ axis / dist), float(off @ perp / dist)


def _eye(lm, ids, w, h):
    """(h, v) of one eye: iris offset along and across the eye-corner line, in eye widths."""
    outer, inner, iris = (np.array([lm[i].x * w, lm[i].y * h]) for i in ids)
    left, right = (outer, inner) if outer[0] < inner[0] else (inner, outer)
    axis = right - left
    width = np.linalg.norm(axis)
    axis /= width
    perp = np.array([-axis[1], axis[0]])         # points down the image for a level head
    off = iris - (left + right) / 2
    return off @ axis / width, off @ perp / width
