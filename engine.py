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
from gestures import Detector, GestureModel, HandPoses, HandToggle
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
    "tracking_fps": 30,         # 30 or 15: tracking rate (the camera must deliver it); 30 reacts sooner and
                                # averages twice the samples, for about 60% more CPU than 15
    "adaptive_rate": True,      # slow tracking down to IDLE_FPS when nothing is happening (saves ~2/3 of the CPU)
    "stability": 3,             # window mode, 1..5: how steady the gaze point is (higher = steadier, slower)
    "edge_px": 0,               # window mode: stay on the active window within this many px of it (0 = auto)
    "gestures": {               # facial gestures that press keys; bindings come from the Gestures dialog
        "enabled": False,
        "pause_s": 6.0,         # an "overview session" after a key gesture lasts this long (and no automatic focus changes)
        "keyboard_super": True, # the Super key on the real keyboard starts that session too
        "gaze_cursor": True,    # during that session the mouse pointer follows your gaze (the overview hides any overlay)
        "dwell_ms": 1200,       # ...and looking at one spot this long clicks it (0 = off)
        "hand_toggle": False,   # an open palm closing into a fist suspends / resumes eyeSwitch (watches for a hand)
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
IDLE_FPS = 10           # tracking rate while nothing is happening
SUSPENDED_FPS = 3       # while suspended and no hand is around: just often enough to notice one
CALM_S = 1.5            # seconds without any sign of movement before slowing down
ACTIVITY_YAW_DEG = 1.5  # the head moved this much off its smoothed yaw: something is happening
SESSION_GRACE = 0.6     # seconds after Super before a dwell can click (the overview animates in)
DWELL_RADIUS = 90       # px: the gaze counts as "one spot" while it stays this close to where the dwell began
SESSION_MOUSE_PX = 12   # you moved the mouse yourself this much: the session is over, it is yours

# Candidate feature sets for the gaze model; the one that cross-validates best wins. Head position
# (tx, ty, tz) is deliberately absent: it says where you sit, not where you look, and a model fitted
# with it moved the estimate ~1000 px per cm of seat shift while being less accurate.
_F = Face._fields
GAZE_COLUMN_SETS = [[_F.index(n) for n in ("yaw", "pitch", "h", "v")]]
# A nose cue (nx, ny) used to compete with this set; on two real calibrations it never helped (120 vs 125 px,
# 166 vs 169 px) and letting cross-validation choose between sets only added noise. It is still measured and
# saved with each calibration, for offline experiments.

# Steadiness level -> (seconds a new place must hold before the decision point moves, radius factor).
# In seconds, not frames: the frame rate is a setting. Level 3 is 0.2 s (it was 4 frames = 0.27 s at 15 fps).
STABILITY = {1: (0.10, 0.8), 2: (0.15, 0.9), 3: (0.20, 1.0), 4: (0.27, 1.1), 5: (0.35, 1.25)}


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


def window_mode_cfg(engine):
    """Window mode is selected (whether or not its model is ready): the test dot needs full speed."""
    return engine.cfg["mode"] == "window"


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
                cfg["camera_height"], cfg["tracking_fps"], bool(cfg["gestures"].get("hand_toggle", False)) or self.engine.hand_wanted > 0)

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
            if self.key[:5] == key[:5] and hasattr(self.tracker, "set_hands"):
                self.tracker.set_hands(key[5])            # only the hand: no need to reopen the camera or the face model
                self.key = key
            else:
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
        e, (cam, want_vcam, shapes, height, fps, hands) = self.engine, key
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
            self.tracker, self.key, self.yielded = HeadTracker(cam, sink, shapes, height, fps, hands), key, False
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
        self.hand_wanted = 0            # the hand is being trained (or looked at): watch it, fast
        self._hand_capture, self._hand_stamp, self._poses_version = None, 0, -1
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
        self.fps = 15.0                 # tracking rate measured while running
        self.tracking_rate = None       # what the engine asked of the tracker last: full speed or IDLE_FPS
        self.session_start = self.session_until = 0.0    # "overview session": pointer follows the gaze until then
        self.in_session = False
        self.suspended = False          # flag off but still watching: only a toggle gesture / the hand can undo it
        self.hand_text = ""             # what the hand recognizer sees right now, for the Gestures screen
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
            # Calibrations saved before the height was recorded were made in window mode, which ran at 720p.
            self.calib_height = data.get("camera_height", 720)
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

    def begin_hand_capture(self):
        with self._lock:
            self._hand_capture = []

    def end_hand_capture(self):
        """Hand pose dicts (see gestures.hand_features) seen since begin_hand_capture()."""
        with self._lock:
            samples, self._hand_capture = self._hand_capture or [], None
        return samples

    def _render_preview(self, tracker, face, frame):
        lines = [self.status, f"{frame.shape[1]}x{frame.shape[0]}  tracking at {self.tracking_rate or 0:.0f} fps"]
        if face is not None:
            lines += [f"yaw {face.yaw:+6.1f}   pitch {face.pitch:+6.1f}",
                      f"iris h {face.h:+.3f}   v {face.v:+.3f}   nose {face.nx:+.2f} {face.ny:+.2f}",
                      f"head x{face.tx:+.0f} y{face.ty:+.0f} z{face.tz:+.0f} cm"]
        if self.gaze:
            lines.append(f"gaze {self.gaze[0]:.0f}, {self.gaze[1]:.0f}")
        if self.suspended:
            lines.append("SUSPENDED")
        if self.in_session:
            lines.append("OVERVIEW: the pointer follows your gaze")
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

    def _start_session(self, seconds):
        """After a key gesture (Super: the overview), let the pointer follow the gaze for a while: the system
        draws the pointer above everything and highlights what is under it, so it works where no overlay can."""
        if (self.cfg["gestures"].get("gaze_cursor", True) and self.gaze_model is not None and self.calibrated
                and not self.resolution_mismatch):
            self.session_start = time.monotonic()
            self.session_until = self.session_start + seconds
            self.in_session = True
            return True
        return False

    def _end_session(self):
        self.session_until, self.in_session = 0.0, False

    def _fire(self, xs, binding):
        """Run a gesture's action; returns seconds during which automatic focus stays quiet."""
        label, action = binding["label"], binding.get("action", "keys")
        try:
            if action == "keys":
                keys = parse_keys(binding.get("keys", ""))
                xs.press_keys(keys)
                self.gesture_log = f"{label}: pressed {'+'.join(keys)}"
                seconds = float(self.cfg["gestures"].get("pause_s", 6.0))
                if self._start_session(seconds):
                    self.gesture_log += ": the pointer follows your gaze, look at what you want"
                return seconds
            if action == "click":
                if self.gaze:
                    xs.warp(int(self.gaze[0]), int(self.gaze[1]))
                xs.click()
                self.gesture_log = f"{label}: clicked" + (" where you look" if self.gaze else "")
                self._end_session()                                   # picked something: the session is over
                return 1.0
            if action == "toggle":
                self._set_suspended(not self.suspended)
        except Exception as e:
            self.gesture_log = f"{label} failed: {e}"
        return 0.0

    # --- control -----------------------------------------------------------
    def _set_suspended(self, value):
        """Suspend: act on nothing, but keep watching so the same gesture can resume. Not the same as the
        pause of the hotkey and the tray, which lets go of the camera (nothing could see a gesture then)."""
        if value == self.suspended:
            return
        self.suspended = value
        self._end_session()
        if value:
            self.gaze = self.gaze_live = None
        self.gesture_log = "Suspended: eyeSwitch acts on nothing until the same gesture" if value else "Resumed"
        if self.on_change:
            self.on_change()

    def _resume_hint(self):
        ways = []
        if self.cfg["gestures"].get("hand_toggle"):
            ways.append("open your palm and close it")
        ways += [f"make the \u201c{b['label']}\u201d gesture" for b in self.active_gestures() if b.get("action") == "toggle"]
        return "Suspended: " + (" or ".join(ways) if ways else "use the hotkey") + " to resume"

    def set_enabled(self, value):
        self.suspended = False                              # the hotkey and the tray are explicit: they win
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
        blink_detector, blink_until, last_frame_t = BlinkDetector(), 0.0, None
        active_until = time.monotonic() + CALM_S                   # adaptive rate: full speed until then
        suppress_until, gesture_state = 0.0, (-1, [])      # (version, [(binding, model, detector)])
        dwell_anchor, dwell_since = None, 0.0
        hand_toggle = None

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
            top = time.monotonic()
            if cand is not None:
                active_until = max(active_until, top + CALM_S)          # a decision is pending: stay sharp
            force = (self.calibrating or self.preview_wanted > 0 or self.want_shapes or self.hand_wanted > 0 or top < self.session_until
                     or (window_mode_cfg(self) and self.cfg["show_gaze"]))
            forced_view = self.calibrating or self.preview_wanted > 0 or self.want_shapes or self.hand_wanted > 0
            full = (((not self.cfg.get("adaptive_rate", True)) or force or top < active_until)
                    and not (self.suspended and not forced_view))         # suspended: only watching, at the idle rate
            self.tracking_rate = camera.tracker.max_fps if full else IDLE_FPS
            if self.suspended and not forced_view and not getattr(camera.tracker, "hand_recent", False):
                self.tracking_rate = SUSPENDED_FPS                       # nobody is signalling: barely look
            camera.tracker.set_rate(self.tracking_rate)
            toggles = any(b.get("action") == "toggle" for b in self.active_gestures())
            camera.tracker.set_face_enabled(not self.suspended or toggles or forced_view)   # no face model if only a hand matters
            try:
                face, frame = camera.tracker.read()
            except Exception as e:                # show it instead of silently killing the thread
                camera.fail(f"Tracking error: {e}")
                continue
            if frame is None:
                camera.fail("Camera error: no frames (is another app using it?)")
                continue
            now = time.monotonic()
            if last_frame_t is not None and now - last_frame_t > 0:
                self.fps += 0.1 * (1.0 / (now - last_frame_t) - self.fps)        # smoothed measured rate
            last_frame_t = now
            monitors = self.monitors = xs.monitors()
            window_mode = self.window_mode
            if self.preview_wanted > 0:
                self.preview = self._render_preview(camera.tracker, face, frame)

            # Track the user's own activity so focus can be restored later.
            nx, ny = xs.pointer()
            moved = abs(nx - px) + abs(ny - py)
            if moved > SESSION_MOUSE_PX and now - self.session_start > 0.3:
                self._end_session()                                   # you took the mouse: it is yours again
            session = self.in_session = now < self.session_until
            if moved > 2:
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

            # The hand toggle: an open palm, then a fist.
            tracker = camera.tracker
            if getattr(tracker, "recognizer", None) is not None:
                if self._poses_version != self.gestures_version:           # the user's own poses, when trained
                    self._poses_version = self.gestures_version
                    poses = self.cfg["gestures"].get("hand_poses")
                    tracker.hand_poses = HandPoses.from_dict(poses) if poses else None
                tracker.hand_fast = self.hand_wanted > 0
                if tracker.hand_stamp != self._hand_stamp:
                    self._hand_stamp = tracker.hand_stamp
                    with self._lock:
                        if self._hand_capture is not None and tracker.hand_features:
                            self._hand_capture.append(dict(tracker.hand_features))
            if self.cfg["gestures"].get("hand_toggle", False) or (
                    self.hand_wanted > 0 and getattr(tracker, "recognizer", None) is not None):
                hand_toggle = hand_toggle or HandToggle()
                hand = getattr(tracker, "hand", None)
                self.hand_text = f"{hand[0]} {hand[1]:.2f}" if hand else "no hand"
                if getattr(tracker, "hand_recent", False):
                    active_until = now + CALM_S
                if (self.cfg["gestures"].get("hand_toggle", False) and hand_toggle.update(hand, now)
                        and not self.calibrating):
                    self._set_suspended(not self.suspended)
            else:
                hand_toggle, self.hand_text = None, ""

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
                self.status = self._resume_hint() if self.suspended else "No face detected"
                continue
            if camera.tracker.shapes:
                self.shapes_seen = now
            flagged = blink_detector.update(camera.tracker.blink, now)
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
            if abs(face.yaw - smooth) > ACTIVITY_YAW_DEG:       # (a blink says nothing about where you look)
                active_until = now + CALM_S

            # Where on the desktop are you looking (window mode, or to click where you look).
            calibrated = self.calibrated
            click_wanted = any(b.get("action") == "click" for b, _, _ in detectors)
            self.out_of_range = False
            if (calibrated and self.gaze_model is not None and not self.resolution_mismatch and not self.suspended
                    and (window_mode or click_wanted or session)):
                model, vec = self.gaze_model, np.asarray(face, float)
                level = min(max(int(self.cfg.get("stability", 3)), 1), 5)
                design = min(camera.tracker.max_fps, camera.tracker.camera_fps)      # what full speed delivers
                rate = max(5, round(design / 5) * 5)                  # frames per second, in steps of 5
                if filter_key != (id(model), level, rate):            # new model, setting or rate: new filter
                    seconds, factor = STABILITY[level]
                    radius = min(max(2.2 * (model.noise_px or 90.0) * factor, 100.0), 320.0)
                    gaze_filter = FixationFilter(radius, max(2, round(seconds * rate)), max(4, round(0.5 * rate)))
                    filter_key = (id(model), level, rate)
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
            # A hand in front of the face (palm covering it, or beside it) makes the face model invent an open
            # mouth: no face gesture counts while a hand is in view, and a gesture that starts to rise makes
            # the hand watcher look at once, so the hand is noticed before the hold time is over.
            hand_blocks = (getattr(tracker, "recognizer", None) is not None
                           and getattr(tracker, "hand_in_view", lambda: False)())
            starting = False
            for b, model, det in detectors:
                relevant = not self.suspended or b.get("action") == "toggle"     # suspended: only the way back
                score = model.score(shapes) if (shapes and relevant and not hand_blocks) else None
                if score is not None:
                    self.gesture_scores[b["label"]] = score
                    if score > 0.2:
                        active_until = now + CALM_S                     # a gesture is starting: do not miss its hold
                        starting = True
                if det.update(score, now) and not self.calibrating:
                    quiet = self._fire(xs, b)
                    suppress_until = now + quiet if b.get("action") == "click" else max(suppress_until, now + quiet)
                    px, py = xs.pointer()                         # our own warp is not mouse activity
                    session = now < self.session_until

            if getattr(tracker, "recognizer", None) is not None:
                tracker.hand_urgent = starting

            # Super on the keyboard opens the overview by itself: the pointer follows the gaze from there too.
            if (xs.super_tapped() and self.cfg["gestures"].get("keyboard_super", True) and not self.suspended
                    and not self.calibrating):
                seconds = float(self.cfg["gestures"].get("pause_s", 6.0))
                if self._start_session(seconds):
                    suppress_until = max(suppress_until, now + seconds)
                    px, py = xs.pointer()
                    session = now < self.session_until
                    self.gesture_log = "Super: the pointer follows your gaze, look at what you want"

            # Overview session: the pointer follows the gaze; one spot held long enough is a click.
            dwell_ms = int(self.cfg["gestures"].get("dwell_ms", 1200))
            if session and self.gaze_live is not None and not self.calibrating:
                xs.warp(int(self.gaze_live[0]), int(self.gaze_live[1]))
                px, py = xs.pointer()
                if dwell_ms > 0 and self.gaze is not None and now - self.session_start > SESSION_GRACE:
                    if dwell_anchor is None or math.hypot(self.gaze[0] - dwell_anchor[0],
                                                          self.gaze[1] - dwell_anchor[1]) > DWELL_RADIUS:
                        dwell_anchor, dwell_since = self.gaze, now
                    elif now - dwell_since >= dwell_ms / 1000:
                        xs.click()
                        self.gesture_log = f"Looked at one spot for {dwell_ms / 1000:.1f} s: clicked"
                        self._end_session()
                        suppress_until, dwell_anchor = now + 1.0, None
                        px, py = xs.pointer()
                else:
                    dwell_anchor = None
            else:
                dwell_anchor = None

            if self.suspended:
                self.status, cand = self._resume_hint(), None
                continue
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
                # Nothing can be decided while you look inside the active window, however much your eyes wander
                # there (reading, typing): full speed is only needed once the gaze leaves it.
                lx, ly = self.gaze_live or self.gaze
                margin = edge_margin_px(self.cfg, self.gaze_model)
                in_active = (active_bounds is not None and
                             active_bounds[0] - margin <= lx < active_bounds[0] + active_bounds[2] + margin and
                             active_bounds[1] - margin <= ly < active_bounds[1] + active_bounds[3] + margin)
                if not in_active or mon != current or (wid and wid != aw):
                    active_until = now + CALM_S
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
