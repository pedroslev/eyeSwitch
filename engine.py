"""Tracking engine: webcam loop in a background thread, plus config and calibration storage."""
import copy
import json
import math
import os
import sys
import threading
import time

import numpy as np

import backends
import preview
import virtualcam
from backends.base import parse_keys
from camwatch import CameraWatcher
from gaze import FixationFilter, GazeModel, OneEuro
from gestures import Detector, GestureModel
from samples import PAD_AFTER, BlinkDetector, clean_capture, clean_indices
from tracker import Face, HeadTracker

CONFIG_DIR = backends.config_dir()
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")
CALIB_PATH = os.path.join(CONFIG_DIR, "calibration.json")
DEFAULTS = {
    "camera": 0,
    "delay_ms": 300,            # facing another monitor this long switches to it
    "enabled": True,
    "mode": "monitor",          # "monitor": head direction only; "window": gaze, needs full calibration
    "window_delay_ms": 500,     # looking at another window of the same monitor this long focuses it
    "typing_wait_s": 2.0,       # window mode: no same-monitor window changes this soon after typing
    "show_gaze": False,         # draw the estimated gaze point (for testing)
    "virtual_camera": False,    # Linux: republish the camera as "eyeSwitch Camera" for video calls
    "calibration_preview": True,  # show the camera view in a corner while calibrating
    "camera_height": "auto",    # "auto" = the camera's best (up to 1080p), or 1080 / 720; a gaze calibration
                                # only fits the resolution it was made at
    "stability": 3,             # window mode, 1..5: how steady the gaze point is (higher = steadier, slower)
    "edge_px": 0,               # window mode: stay on the active window within this many px of it (0 = auto)
    "gestures": {               # facial gestures that press keys; bindings come from the Gestures dialog
        "enabled": False,
        "pause_s": 6.0,         # after a key gesture, no automatic focus changes for this long (menus, overview)
        "bindings": [],         # {"label", "action": keys|click|toggle, "keys", "hold_ms", "model"}
    },
}

SMOOTH = 0.4            # EMA factor for the yaw signal
HYSTERESIS = 0.25       # must be this much closer to the target than to the current monitor
MOUSE_QUIET = 1.5       # seconds without touching the mouse before we act
FAR_FACTOR = 1.5        # ignore poses farther than this many spacings from every monitor
EDGE_MARGIN = 0.04      # gaze must be this far (monitor widths) inside a monitor to move there
WINDOW_POLL = 0.1       # seconds between "which window is under the gaze" lookups
RETRY = 4.0             # seconds of calm before taking the camera back from another app
RANGE_LIMIT = 0.5       # pose this far outside the calibrated range (fraction of it): do not act on it

# Candidate feature sets for the gaze model; the one that cross-validates best wins. Head position
# (tx, ty, tz) is deliberately absent: it says where you sit, not where you look, and a model fitted
# with it moved the estimate ~1000 px per cm of seat shift while being less accurate.
_F = Face._fields
GAZE_COLUMN_SETS = [[_F.index(n) for n in names] for names in (
    ("yaw", "pitch", "h", "v"), ("yaw", "pitch", "h", "v", "nx", "ny"))]
STABILITY = {1: (2, 0.8), 2: (3, 0.9), 3: (4, 1.0), 4: (5, 1.1), 5: (6, 1.25)}   # level -> (confirm frames, radius x)


def load_config():
    cfg = copy.deepcopy(DEFAULTS)
    try:
        with open(CONFIG_PATH) as f:
            cfg.update(json.load(f))
    except (FileNotFoundError, ValueError):
        pass
    return cfg


def save_config(cfg):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def pick_target(yaw, centers, current):
    """Index of the monitor the head is facing, or None if out of range."""
    dist = [abs(yaw - c) for c in centers]
    best = int(np.argmin(dist))
    s = sorted(centers)
    spacing = min(b - a for a, b in zip(s, s[1:]))
    if dist[best] > FAR_FACTOR * spacing:
        return None
    if best != current and dist[best] > dist[current] * (1 - HYSTERESIS):
        return current
    return best


def nearest_monitor(x, y, monitors):
    def dist(m):
        return math.hypot(max(m.x - x, 0, x - (m.x + m.w)), max(m.y - y, 0, y - (m.y + m.h)))
    return min(range(len(monitors)), key=lambda i: dist(monitors[i]))


def gaze_monitor(x, y, monitors, current):
    """Monitor under the gaze point; leaving `current` needs to be clearly inside the other one."""
    best = nearest_monitor(x, y, monitors)
    if best != current:
        m = monitors[best]
        margin = EDGE_MARGIN * m.w
        if not (m.x + margin <= x < m.x + m.w - margin):
            return current
    return best


def clamp_to_monitors(x, y, monitors):
    """Nearest point on a monitor: an estimate past the screen edge means "at the edge"."""
    m = monitors[nearest_monitor(x, y, monitors)]
    return min(max(x, m.x), m.x + m.w - 1), min(max(y, m.y), m.y + m.h - 1)


def edge_margin_px(cfg, model):
    """Distance past a window's edge the gaze must travel before focus leaves that window."""
    if cfg["edge_px"] > 0:
        return cfg["edge_px"]
    return min(max(0.6 * (model.error_px or 0), 40), 250)


def stay_on_active(wid, active, bounds, point, margin):
    """Window to focus: keep `active` while the gaze is within `margin` of its edges."""
    if active and bounds and wid != active:
        x, y, w, h = bounds
        if x - margin <= point[0] < x + w + margin and y - margin <= point[1] < y + h + margin:
            return active
    return wid


def switch(xs, monitors, idx, last_window, last_pointer, wid=None):
    """Move to monitor idx: focus `wid` (or its last/top window) and bring the pointer along."""
    if wid is None:
        wid = last_window.get(idx)
        if wid is None or not xs.exists(wid) or xs.window_monitor(wid, monitors) != idx:
            wid = xs.top_window_on(monitors, idx)
    m = monitors[idx]
    x, y = last_pointer.get(idx, (m.x + m.w // 2, m.y + m.h // 2))
    xs.warp(x, y)
    if wid:
        xs.activate(wid)


class Camera:
    """The webcam, its optional virtual copy, and a watcher that notices other apps wanting it.

    A webcam can be streamed by one process only. When another app opens the device we let go
    (it will usually fail once, then work) and take it back after RETRY seconds without news.
    With the virtual camera on, calls use "eyeSwitch Camera" instead and nothing is shared at all.
    """

    def __init__(self, engine):
        self.engine = engine
        self.tracker = self.vcam = self.watcher = self.key = None
        self.yielded, self.retry_at = False, 0.0

    def _wanted(self):
        cfg = self.engine.cfg
        return (cfg["camera"], bool(cfg["virtual_camera"]) and virtualcam.AVAILABLE,
                self.engine.want_shapes or bool(self.engine.active_gestures()) or cfg["mode"] == "window",
                cfg["camera_height"])

    def _watch(self, cam):
        path = f"/dev/video{cam}"
        if self.watcher and self.watcher.path == path:
            return
        self._stop_watcher()
        if sys.platform.startswith("linux"):
            try:
                self.watcher = CameraWatcher(path)
            except OSError:
                self.watcher = None

    def _stop_watcher(self):
        if self.watcher:
            self.watcher.stop()
            self.watcher = None

    def _close(self):
        if self.tracker:
            self.tracker.close()
        if self.vcam:
            self.vcam.close()
        self.tracker = self.vcam = self.key = None

    def release(self):
        """Let go of everything (paused, or quitting)."""
        self._close()
        self._stop_watcher()
        self.yielded = False
        self.engine.vcam_state = "off"

    def fail(self, message):
        """The camera stopped working: show why and reopen on the next ensure()."""
        self._close()
        self.engine.status = message
        time.sleep(1.0)

    def ensure(self):
        """Open the camera unless we are stepping aside. True when frames can be read."""
        e, now, key = self.engine, time.monotonic(), self._wanted()
        if self.tracker and self.key != key:              # settings changed
            self._close()
        self._watch(key[0])
        if self.watcher and self.watcher.foreign_open.is_set():
            self.watcher.foreign_open.clear()
            if self.tracker:                              # someone else wants it: step aside
                self._close()
                self.yielded = True
            if self.yielded:
                self.retry_at = now + RETRY
        if self.tracker:
            return True
        if self.yielded and now < self.retry_at:
            e.status = "Camera in use by another app: waiting for it"
            time.sleep(0.1)
            return False
        return self._open(key, now)

    def _open(self, key, now):
        e, (cam, want_vcam, shapes, height) = self.engine, key
        if self.watcher:
            self.watcher.ignore_for(2.0)                  # the opens below are ours
        sink, e.vcam_state = None, "off"
        if want_vcam:
            device = virtualcam.find_device()
            if device is None:
                e.vcam_state = "not set up"
            else:
                try:
                    self.vcam = virtualcam.VirtualCamera(device)
                    sink, e.vcam_state = self.vcam.send, "active"
                except Exception as ex:
                    e.vcam_state = f"error: {ex}"
        try:
            self.tracker, self.key, self.yielded = HeadTracker(cam, sink, shapes, height), key, False
            e.active_height = self.tracker.height
            return True
        except Exception as ex:
            if self.vcam:
                self.vcam.close()
                self.vcam = None
            if self.yielded:                              # still busy: keep waiting
                self.retry_at = now + RETRY
                e.status = "Camera in use by another app: waiting for it"
                time.sleep(0.1)
            else:
                e.status = f"Camera error: {ex}"
                time.sleep(1.0)
            return False


class Engine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.status = "Starting..."
        self.yaw = None                 # smoothed head yaw in degrees, None when no face
        self.gaze = None                # steady gaze point in desktop pixels (window mode): what decisions use
        self.gaze_live = None           # the same, continuously smoothed: what the test dot shows
        self.vcam_state = "off"         # off / active / not set up / error: ...
        self.want_shapes = False        # the Gestures dialog is training: blendshapes are needed
        self.gestures_version = 0       # bumped when the bindings change
        self.gesture_scores = {}        # label -> latest score (0 relaxed, 1 gesture), for live feedback
        self.gesture_log = ""           # what the last gesture did, or why it failed
        self.shapes_seen = 0.0          # monotonic time of the last frame that carried blendshapes
        self.preview = None             # latest debug image (BGR array), rendered while preview_wanted > 0
        self.preview_wanted = 0         # how many views want it (the window, a calibration)
        self.blinking = False           # eyelids closing right now (or just reopened): estimate is held
        self.out_of_range = False       # pose is far outside what was calibrated (camera or seat moved)
        self.stale_model = False        # a saved gaze model from an older tracker: recalibrate
        self.last_capture = {}          # stats of the last capture: frames kept, blinks and motion dropped
        self.last_capture_raw = []      # raw geometry of the kept frames, aligned with the faces returned
        self.calibrating = False
        self.on_change = None           # called from the worker thread when `enabled` flips
        self.monitors = backends.Session().monitors()
        self.centers, self.calib_monitors, self.gaze_model = None, None, None
        self.calib_height = None        # camera height the gaze model was calibrated at (None: unknown)
        self.active_height = None       # camera height in use right now (None: camera not opened yet)
        self._capture = self._shape_capture = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._load_calibration()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # --- calibration -------------------------------------------------------
    def _load_calibration(self):
        try:
            with open(CALIB_PATH) as f:
                data = json.load(f)
            self.centers, self.calib_monitors = data["yaw"], data["monitors"]
            self.calib_height = data.get("camera_height")
            if data.get("gaze"):
                model = GazeModel.from_dict(data["gaze"])
                if len(model.mean) == len(Face._fields):
                    self.gaze_model = model
                else:
                    self.stale_model = True      # fitted on other features: its numbers mean nothing now
        except (FileNotFoundError, ValueError, KeyError):
            pass

    @property
    def calibrated(self):
        return self.centers is not None and self.calib_monitors == [list(m) for m in self.monitors]

    @property
    def resolution_mismatch(self):
        """The gaze model was calibrated at another camera resolution than the one in use."""
        return (self.gaze_model is not None and self.calib_height is not None
                and self.active_height is not None and self.calib_height != self.active_height)

    @property
    def window_mode(self):
        """Gaze-based window focus is on and has a model for the current layout and resolution."""
        return (self.cfg["mode"] == "window" and self.gaze_model is not None and self.calibrated
                and not self.resolution_mismatch)

    def set_calibration(self, yaws, monitors, model=None, camera_height=None):
        """Store per-monitor head yaws; `model` is the gaze model of a full calibration, made with the
        camera at `camera_height` (the height in use by default)."""
        self.centers, self.calib_monitors, self.gaze_model = list(yaws), [list(m) for m in monitors], model
        self.calib_height = camera_height or self.active_height
        self.stale_model = False
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CALIB_PATH, "w") as f:
            json.dump({"monitors": self.calib_monitors, "yaw": self.centers, "camera_height": self.calib_height,
                       "gaze": model.to_dict() if model else None}, f, indent=2)

    def begin_capture(self):
        with self._lock:
            self._capture = []

    def end_capture(self):
        """Faces worth keeping since begin_capture(): blinks (with padding) and frames where the head
        is still travelling are dropped. `last_capture` says how many went."""
        with self._lock:
            records, self._capture = self._capture or [], None
        kept, self.last_capture = clean_indices(records)
        self.last_capture_raw = [records[i][3] for i in kept]
        return [records[i][1] for i in kept]

    def capture_clean_count(self):
        """How many clean frames the capture in progress has so far (blinks and motion excluded)."""
        with self._lock:
            records = list(self._capture or [])
        return len(clean_capture(records)[0])

    def begin_shape_capture(self):
        with self._lock:
            self._shape_capture = []

    def end_shape_capture(self):
        """Blendshape dicts seen since begin_shape_capture()."""
        with self._lock:
            samples, self._shape_capture = self._shape_capture or [], None
        return samples

    def _render_preview(self, tracker, face, frame):
        lines = [self.status, f"{frame.shape[1]}x{frame.shape[0]}"]
        if face is not None:
            lines += [f"yaw {face.yaw:+6.1f}   pitch {face.pitch:+6.1f}",
                      f"iris h {face.h:+.3f}   v {face.v:+.3f}   nose {face.nx:+.2f} {face.ny:+.2f}",
                      f"head x{face.tx:+.0f} y{face.ty:+.0f} z{face.tz:+.0f} cm"]
        if self.gaze:
            lines.append(f"gaze {self.gaze[0]:.0f}, {self.gaze[1]:.0f}")
        if self.blinking:
            lines.append("BLINK: estimate held")
        if self.out_of_range:
            lines.append("OUT OF CALIBRATED RANGE")
        return preview.render(frame, tracker.landmarks, lines)

    # --- gestures ----------------------------------------------------------
    def active_gestures(self):
        """Bindings that are switched on and trained."""
        g = self.cfg["gestures"]
        return [b for b in g["bindings"] if g["enabled"] and b.get("model")]

    def gestures_changed(self):
        save_config(self.cfg)
        self.gestures_version += 1

    def _fire(self, xs, binding):
        """Run a gesture's action; returns seconds during which automatic focus stays quiet."""
        label, action = binding["label"], binding.get("action", "keys")
        try:
            if action == "keys":
                keys = parse_keys(binding.get("keys", ""))
                xs.press_keys(keys)
                self.gesture_log = f"{label}: pressed {'+'.join(keys)}"
                return float(self.cfg["gestures"].get("pause_s", 6.0))
            if action == "click":
                if self.gaze:
                    xs.warp(int(self.gaze[0]), int(self.gaze[1]))
                xs.click()
                self.gesture_log = f"{label}: clicked" + (" where you look" if self.gaze else "")
                return 1.0
            if action == "toggle":
                self.set_enabled(not self.cfg["enabled"])
                self.gesture_log = f"{label}: toggled eyeSwitch"
        except Exception as e:
            self.gesture_log = f"{label} failed: {e}"
        return 0.0

    # --- control -----------------------------------------------------------
    def set_enabled(self, value):
        self.cfg["enabled"] = value
        save_config(self.cfg)
        if self.on_change:
            self.on_change()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    # --- worker ------------------------------------------------------------
    def _run(self):
        try:
            xs = backends.Session()
            hotkey_ok = xs.grab_hotkey()
        except Exception as e:
            self.status = f"Cannot start the {backends.platform_name()} backend: {e}"
            return
        camera = Camera(self)
        last_window, last_pointer = {}, {}
        px, py = xs.pointer()
        current = xs.monitor_at(self.monitors, px, py) or 0
        mouse_time, smooth, cand, cand_since = 0.0, None, None, 0.0
        gaze_filter, filter_key, live_filter = FixationFilter(), None, OneEuro()
        win_checked, win_under, active_bounds = 0.0, None, None
        blink_detector, blink_until = BlinkDetector(), 0.0
        suppress_until, gesture_state = 0.0, (-1, [])      # (version, [(binding, model, detector)])

        while not self._stop.is_set():
            if hotkey_ok and xs.hotkey_pressed():
                self.set_enabled(not self.cfg["enabled"])
            if not (self.cfg["enabled"] or self.calibrating or self.preview_wanted > 0):
                camera.release()                  # let other apps have the camera while paused
                self.yaw = self.gaze = self.gaze_live = None
                smooth, cand = None, None
                gaze_filter.update(None)
                live_filter.update(None, time.monotonic())
                self.status = "Paused"
                time.sleep(0.1)
                continue

            if not camera.ensure():
                continue
            try:
                face, frame = camera.tracker.read()
            except Exception as e:                # show it instead of silently killing the thread
                camera.fail(f"Tracking error: {e}")
                continue
            if frame is None:
                camera.fail("Camera error: no frames (is another app using it?)")
                continue
            now = time.monotonic()
            monitors = self.monitors = xs.monitors()
            window_mode = self.window_mode
            if self.preview_wanted > 0:
                self.preview = self._render_preview(camera.tracker, face, frame)

            # Track the user's own activity so focus can be restored later.
            nx, ny = xs.pointer()
            if abs(nx - px) + abs(ny - py) > 2:
                mouse_time = now
                if not window_mode:
                    current = xs.monitor_at(monitors, nx, ny) or current
            px, py = nx, ny
            m = xs.monitor_at(monitors, px, py)
            if m is not None:
                last_pointer[m] = (px, py)
            aw = xs.active_window()
            if aw:
                wm = xs.window_monitor(aw, monitors)
                if wm is not None:
                    last_window[wm] = aw
                    if window_mode:
                        current = wm

            if gesture_state[0] != self.gestures_version:          # bindings changed: rebuild detectors
                gesture_state = (self.gestures_version, [
                    (b, GestureModel.from_dict(b["model"]), Detector(b.get("hold_ms", 250) / 1000))
                    for b in self.active_gestures()])
            detectors = gesture_state[1]

            if face is None:
                self.yaw = self.gaze = self.gaze_live = None
                gaze_filter.update(None)
                live_filter.update(None, now)
                for _, _, det in detectors:
                    det.update(None, now)
                cand = None
                self.status = "No face detected"
                continue
            if camera.tracker.shapes:
                self.shapes_seen = now
            flagged = blink_detector.update(camera.tracker.blink)
            if flagged:
                blink_until = now + PAD_AFTER          # the lids reopening still distort the estimate
            blinking = self.blinking = now < blink_until
            with self._lock:
                if self._capture is not None:
                    self._capture.append((now, face, flagged, camera.tracker.raw()))
                if self._shape_capture is not None and camera.tracker.shapes:
                    self._shape_capture.append(dict(camera.tracker.shapes))
            smooth = face.yaw if smooth is None else smooth + SMOOTH * (face.yaw - smooth)
            self.yaw = smooth

            # Where on the desktop are you looking (window mode, or to click where you look).
            calibrated = self.calibrated
            click_wanted = any(b.get("action") == "click" for b, _, _ in detectors)
            self.out_of_range = False
            if calibrated and self.gaze_model is not None and (window_mode or click_wanted):
                model, vec = self.gaze_model, np.asarray(face, float)
                level = min(max(int(self.cfg.get("stability", 3)), 1), 5)
                if filter_key != (id(model), level):                 # new model or setting: new filter
                    confirm, factor = STABILITY[level]
                    radius = min(max(2.2 * (model.noise_px or 90.0) * factor, 100.0), 320.0)
                    gaze_filter, filter_key = FixationFilter(radius, confirm), (id(model), level)
                if model.range_excess(vec) > RANGE_LIMIT:
                    self.out_of_range, self.gaze, self.gaze_live = True, None, None
                    gaze_filter.update(None)
                    live_filter.update(None, now)
                elif blinking and self.gaze is not None:
                    pass                                             # hold the estimate through a blink
                else:
                    raw = model.predict(vec)
                    g, live = gaze_filter.update(raw), live_filter.update(raw, now)
                    self.gaze = clamp_to_monitors(g[0], g[1], monitors)
                    self.gaze_live = clamp_to_monitors(live[0], live[1], monitors)
            else:
                self.gaze = self.gaze_live = None
                gaze_filter.update(None)
                live_filter.update(None, now)

            # Gestures
            shapes = camera.tracker.shapes
            for b, model, det in detectors:
                score = model.score(shapes) if shapes else None
                if score is not None:
                    self.gesture_scores[b["label"]] = score
                if det.update(score, now) and not self.calibrating:
                    suppress_until = max(suppress_until, now + self._fire(xs, b))
                    px, py = xs.pointer()                         # our own warp is not mouse activity

            if not calibrated:
                self.status = "Not calibrated: click Calibrate"
                continue
            current = min(current, len(monitors) - 1)
            busy = self.calibrating or now - mouse_time < MOUSE_QUIET or now < suppress_until

            if window_mode:
                # --- gaze: which monitor and which window are you looking at -------------------
                if self.gaze is None:
                    self.status = ("Outside the calibrated range: did the camera or your seat move? "
                                   "Recalibrate" if self.out_of_range else "No gaze estimate")
                    cand = None
                    continue
                if blinking:
                    continue                                         # decisions wait until the eyes reopen
                gx, gy = self.gaze
                mon = gaze_monitor(gx, gy, monitors, current)
                if now - win_checked >= WINDOW_POLL:
                    win_checked, win_under = now, xs.window_at(gx, gy)
                    active_bounds = xs.window_bounds(aw) if aw else None
                wid = win_under if xs.monitor_at(monitors, gx, gy) == mon else None
                wid = stay_on_active(wid, aw, active_bounds, (gx, gy), edge_margin_px(self.cfg, self.gaze_model))
                self.status = f"Looking at monitor {mon + 1}"
                if busy:
                    cand = None
                    continue
                if mon != current:
                    key, need = ("mon", mon, wid), self.cfg["delay_ms"] / 1000
                elif wid and wid != aw and xs.seconds_since_key() >= self.cfg["typing_wait_s"]:
                    key, need = ("win", wid), self.cfg["window_delay_ms"] / 1000
                else:
                    cand = None
                    continue
                if cand != key:
                    cand, cand_since = key, now
                elif now - cand_since >= need:
                    if mon != current:
                        switch(xs, monitors, mon, last_window, last_pointer, wid)
                    else:
                        xs.activate(wid)
                    current, cand = mon, None
                    px, py = xs.pointer()             # our own warp is not mouse activity
                continue

            # --- head direction only: which monitor are you facing ---------------------------
            target = pick_target(smooth, self.centers, current)
            self.status = "Looking away" if target is None else f"Facing monitor {target + 1}"
            if busy or target is None or target == current:
                cand = None
                continue
            if cand != target:
                cand, cand_since = target, now
            elif now - cand_since >= self.cfg["delay_ms"] / 1000:
                switch(xs, monitors, target, last_window, last_pointer)
                current, cand = target, None
                px, py = xs.pointer()             # our own warp is not mouse activity

        camera.release()
