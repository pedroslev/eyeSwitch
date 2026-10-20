"""Tracking engine: webcam loop in a background thread, plus config and calibration storage."""
import json
import os
import threading
import time

import numpy as np

import backends
from tracker import HeadTracker

CONFIG_DIR = backends.config_dir()
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")
CALIB_PATH = os.path.join(CONFIG_DIR, "calibration.json")
DEFAULTS = {"camera": 0, "delay_ms": 300, "enabled": True}

SMOOTH = 0.4            # EMA factor for the yaw signal
HYSTERESIS = 0.25       # must be this much closer to the target than to the current monitor
MOUSE_QUIET = 1.5       # seconds without touching the mouse before we act
FAR_FACTOR = 1.5        # ignore poses farther than this many spacings from every monitor


def load_config():
    try:
        with open(CONFIG_PATH) as f:
            return {**DEFAULTS, **json.load(f)}
    except (FileNotFoundError, ValueError):
        return dict(DEFAULTS)


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


def switch(xs, monitors, idx, last_window, last_pointer):
    wid = last_window.get(idx)
    if wid is None or not xs.exists(wid) or xs.window_monitor(wid, monitors) != idx:
        wid = xs.top_window_on(monitors, idx)
    m = monitors[idx]
    x, y = last_pointer.get(idx, (m.x + m.w // 2, m.y + m.h // 2))
    xs.warp(x, y)
    if wid:
        xs.activate(wid)


class Engine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.status = "Starting..."
        self.yaw = None                 # smoothed head yaw in degrees, None when no face
        self.calibrating = False
        self.on_change = None           # called from the worker thread when `enabled` flips
        self.monitors = backends.Session().monitors()
        self.centers, self.calib_monitors = None, None
        self._capture = None
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
        except (FileNotFoundError, ValueError, KeyError):
            pass

    @property
    def calibrated(self):
        return self.centers is not None and self.calib_monitors == [list(m) for m in self.monitors]

    def set_calibration(self, yaws, monitors):
        self.centers, self.calib_monitors = list(yaws), [list(m) for m in monitors]
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CALIB_PATH, "w") as f:
            json.dump({"monitors": self.calib_monitors, "yaw": self.centers}, f, indent=2)

    def begin_capture(self):
        with self._lock:
            self._capture = []

    def end_capture(self):
        with self._lock:
            samples, self._capture = self._capture or [], None
        return samples

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
        tracker, opened = None, None
        last_window, last_pointer = {}, {}
        px, py = xs.pointer()
        current = xs.monitor_at(self.monitors, px, py) or 0
        mouse_time, smooth, cand, cand_since = 0.0, None, None, 0.0

        while not self._stop.is_set():
            if hotkey_ok and xs.hotkey_pressed():
                self.set_enabled(not self.cfg["enabled"])
            if not (self.cfg["enabled"] or self.calibrating):
                if tracker:                       # release the camera while paused
                    tracker.close()
                    tracker, opened = None, None
                self.yaw, smooth, cand = None, None, None
                self.status = "Paused"
                time.sleep(0.1)
                continue

            cam = self.cfg["camera"]
            if tracker is None or opened != cam:
                if tracker:
                    tracker.close()
                    tracker = None
                try:
                    tracker, opened = HeadTracker(cam), cam
                except Exception as e:
                    self.status = f"Camera error: {e}"
                    time.sleep(1.0)
                    continue

            raw, _ = tracker.read()
            now = time.monotonic()
            monitors = self.monitors = xs.monitors()

            # Track the user's own activity so focus can be restored later.
            nx, ny = xs.pointer()
            if abs(nx - px) + abs(ny - py) > 2:
                mouse_time = now
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

            if raw is None:
                self.yaw, cand = None, None
                self.status = "No face detected"
                continue
            with self._lock:
                if self._capture is not None:
                    self._capture.append(raw)
            smooth = raw if smooth is None else smooth + SMOOTH * (raw - smooth)
            self.yaw = smooth

            if not self.calibrated:
                self.status = "Not calibrated: click Calibrate"
                continue
            current = min(current, len(self.centers) - 1)
            target = pick_target(smooth, self.centers, current)
            self.status = "Looking away" if target is None else f"Facing monitor {target + 1}"
            if (self.calibrating or target is None or target == current
                    or now - mouse_time < MOUSE_QUIET):
                cand = None
                continue
            if cand != target:
                cand, cand_since = target, now
            elif now - cand_since >= self.cfg["delay_ms"] / 1000:
                switch(xs, monitors, target, last_window, last_pointer)
                current, cand = target, None
                px, py = xs.pointer()             # our own warp is not mouse activity

        if tracker:
            tracker.close()
