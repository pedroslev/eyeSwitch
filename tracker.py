"""Webcam head-yaw tracking with MediaPipe Face Landmarker."""
import math
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.request
from collections import namedtuple

import cv2
import numpy as np

from handworker import HEADER
from virtualcam import LABEL as VIRTUAL_LABEL

# OpenCV spreads even a colour conversion over every core: the same 0.2 ms of wall time cost 13x more CPU
# with 12 threads than with one (2.6 vs 0.2 ms), and the whole pipeline 43.5 vs 23 ms per 1080p frame.
cv2.setNumThreads(1)

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


HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
                  "gesture_recognizer/float16/1/gesture_recognizer.task")
HAND_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture_recognizer.task")
# A hand recognition costs ~31 ms of CPU even with no hand in view (the palm detector shrinks its input, so a
# small picture is as expensive as a large one). So it runs rarely, 3 times a second (~10% of a core), and at
# 10 per second only while a hand is around, which is when a palm-then-fist needs timing.
HAND_INTERVAL_IDLE, HAND_INTERVAL_NEAR, HAND_NEAR_S = 0.3, 0.1, 3.0
HAND_WIDTH = 640
SIGNAL_POSES = ("Open_Palm", "Closed_Fist")    # only these mean someone is signalling; a hand resting on the
                                                 # keyboard is "None" and must not speed anything up


def ensure_model(path=MODEL_PATH, url=MODEL_URL):
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)


# Tried in this order when the resolution is "auto". Not 4K: the iris landmarks stop improving and the
# extra pixels only cost CPU. At 640x480 an eye is ~20 px wide, at 1080p ~54 px.
SIZES = ((1920, 1080), (1280, 720), (640, 480))


class HandProcess:
    """Client of handworker.py: one request in flight, answers read without blocking."""
    NOTHING_NEW = object()
    MAX_RESTARTS = 3

    def __init__(self, model_path):
        self.model_path, self.restarts = model_path, 0
        self.proc = self.reader = None
        self.busy, self.alive, self._answer = False, True, queue.Queue()
        self._start()

    def _start(self):
        self.proc = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "handworker.py"),
                                      self.model_path],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.busy, self._answer = False, queue.Queue()
        threading.Thread(target=self._read_answers, args=(self.proc, self._answer), daemon=True).start()

    @staticmethod
    def _read_answers(proc, out):
        for line in proc.stdout:
            out.put(line.decode().strip())
        out.put(None)                                       # the process ended

    def submit(self, rgb, ts_ms):
        try:
            self.proc.stdin.write(HEADER.pack(rgb.shape[1], rgb.shape[0], ts_ms) + rgb.tobytes())
            self.proc.stdin.flush()
            self.busy = True
        except (BrokenPipeError, OSError):
            self._died()

    def poll(self):
        """The newest verdict ((name, score) or None) if an answer came, else NOTHING_NEW."""
        verdict = self.NOTHING_NEW
        while True:
            try:
                line = self._answer.get_nowait()
            except queue.Empty:
                return verdict
            if line is None:
                self._died()
                return self.NOTHING_NEW
            if line == "ready":
                continue
            self.busy = False
            name, _, score = line.partition(" ")
            verdict = None if name == "-" else (name, float(score))

    def _died(self):
        """A crash of the recognizer costs only the hand signal: try again a few times, then give up."""
        self.busy = False
        if self.restarts < self.MAX_RESTARTS:
            self.restarts += 1
            print("hand recognizer stopped; restarting it", file=sys.stderr)
            self._start()
        else:
            self.alive = False

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()


class HeadTracker:
    def __init__(self, camera=0, sink=None, blendshapes=False, height="auto", fps=30, hands=False):
        """Opens the camera at the best resolution it really delivers (`height` "auto"), or at 1080/720
        when asked and possible. Tracking runs at `fps` (15 or 30). `sink(frame)` receives every camera
        frame (for the virtual camera, at 30 fps so calls look good). With `hands`, the MediaPipe gesture
        recognizer also watches for a hand (`hand` is its latest verdict)."""
        ensure_model()
        import mediapipe as mp                       # imported here, in the engine thread, so the window
        from mediapipe.tasks import python as mp_python      # appears at once instead of after ~1 s of imports
        from mediapipe.tasks.python import vision
        self._mp = mp
        self.sink = sink
        self.cap = cv2.VideoCapture(camera, _capture_api())
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {camera}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))    # raw formats cannot carry HD over USB
        self.cap.set(cv2.CAP_PROP_FPS, 30 if (sink or fps >= 30) else 15)
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
        self.face_enabled, self.hand, self.recognizer, self._last_hand, self._hand_seen = True, None, None, 0.0, -1e9
        if hands:
            ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL)
            self.recognizer = HandProcess(HAND_MODEL_PATH)
        self.max_fps = self.fps = fps
        self.min_interval = 0.8 / fps                # 0.8: do not skip frames to timing jitter
        self.camera_fps, self._grabbed = 30.0, None     # what the camera really delivers, measured
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

    def set_face_enabled(self, on):
        """Skip the face model altogether (the engine is suspended and only watches for the hand)."""
        self.face_enabled = on

    def _read_hand(self, frame, now):
        """Hand the frame to the recognizer process (when it is free) and pick up its latest answer."""
        verdict = self.recognizer.poll()
        if verdict is not HandProcess.NOTHING_NEW:
            self.hand = verdict
            if self.hand and self.hand[0] in SIGNAL_POSES and self.hand[1] >= 0.5:
                self._hand_seen = now
        if not self.recognizer.busy:
            self._last_hand = now
            width = min(HAND_WIDTH, frame.shape[1])
            small = cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])), interpolation=cv2.INTER_AREA)
            self.recognizer.submit(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), int((now - self._t0) * 1000))
        if not self.recognizer.alive:
            self.hand = None

    @property
    def hand_recent(self):
        """A palm or a fist was seen in the last few seconds: worth looking more often (the closing of a hand
        passes through poses that read as "None", which this window of time covers)."""
        return time.monotonic() - self._hand_seen < HAND_NEAR_S

    def set_rate(self, fps):
        """Track at `fps` (never above the rate this tracker was opened with). The camera keeps delivering
        frames; the ones in between are grabbed but not even decoded, which is where the CPU goes."""
        self.fps = min(max(fps, 1), self.max_fps)
        self.min_interval = 0.8 / self.fps

    def read(self):
        """Return (Face or None, frame) at the tracking rate. Extra frames go to the sink."""
        while True:
            if not self.cap.grab():
                return None, None
            now = time.monotonic()
            if self._grabbed is not None and now > self._grabbed:
                self.camera_fps += 0.05 * (1.0 / (now - self._grabbed) - self.camera_fps)
            self._grabbed = now
            due = now - self._last >= self.min_interval
            if self.sink is None and not due:
                continue                                    # not needed: skip the JPEG decode altogether
            ok, frame = self.cap.retrieve()
            if not ok:
                return None, None
            if self.sink is not None:
                self.sink(frame)
            if due:
                break
        self._last = now
        if self.recognizer is not None and (self.recognizer.busy or now - self._last_hand >= (
                HAND_INTERVAL_NEAR if now - self._hand_seen < HAND_NEAR_S else HAND_INTERVAL_IDLE)):
            self._read_hand(frame, now)
        if not self.face_enabled:
            self.shapes = self.landmarks = self.blink = None
            return None, frame
        mp = self._mp
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
        if self.recognizer is not None:
            self.recognizer.close()
        if self.recognizer is not None:
            self.recognizer.close()


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
