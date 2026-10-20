"""Engine decision logic with a fake camera and a fake OS backend. Run: python -m unittest discover tests"""
import os
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import backends  # noqa: E402
import engine as engine_mod  # noqa: E402
from backends.base import BaseSession, Monitor  # noqa: E402
from gaze import GazeModel  # noqa: E402
from tracker import Face  # noqa: E402

MONITORS = [Monitor(0, 0, 1920, 1080), Monitor(1920, 0, 1920, 1080)]
# window id -> (x, y, w, h); A and B share monitor 1, C is on monitor 2
WINDOWS = {1: (0, 0, 960, 1080), 2: (960, 0, 960, 1080), 3: (1920, 0, 1920, 1080)}
A, B, C = 1, 2, 3


class FakeSession(BaseSession):
    def __init__(self):
        self.active, self.activated, self.warps, self.key_age = A, [], [], 99.0

    def monitors(self):
        return MONITORS

    def pointer(self):
        return 100, 100

    def warp(self, x, y):
        self.warps.append((x, y))

    def active_window(self):
        return self.active

    def window_monitor(self, wid, monitors):
        x, y, w, h = WINDOWS[wid]
        return self.center_monitor(monitors, x, y, x + w, y + h)

    def top_window_on(self, monitors, idx):
        return next((wid for wid in WINDOWS if self.window_monitor(wid, monitors) == idx), None)

    def window_at(self, x, y):
        return next((wid for wid, (wx, wy, w, h) in WINDOWS.items()
                     if wx <= x < wx + w and wy <= y < wy + h), None)

    def seconds_since_key(self):
        return self.key_age

    def exists(self, wid):
        return wid in WINDOWS

    def activate(self, wid):
        self.activated.append(wid)
        self.active = wid

    def grab_hotkey(self):
        return True

    def hotkey_pressed(self):
        return False


class FakeTracker:
    face = None

    def __init__(self, camera=0):
        pass

    def read(self):
        time.sleep(0.02)
        return FakeTracker.face, None

    def close(self):
        pass


def looking_at(x, y, yaw=0.0):
    """A face whose gaze features (h, v) map straight onto desktop pixel (x, y)."""
    return Face(yaw, 0, 0, 0, 0, x, y, 0)


def identity_model():
    w = np.zeros((9, 2))
    w[1 + 5, 0] = 1                  # x = h
    w[1 + 6, 1] = 1                  # y = v
    return GazeModel(1, np.zeros(8), np.ones(8), w, 100.0)


class EngineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saved = (engine_mod.CONFIG_DIR, engine_mod.CONFIG_PATH, engine_mod.CALIB_PATH,
                      engine_mod.HeadTracker, backends.Session)
        engine_mod.CONFIG_DIR = self.tmp.name
        engine_mod.CONFIG_PATH = os.path.join(self.tmp.name, "config.json")
        engine_mod.CALIB_PATH = os.path.join(self.tmp.name, "calibration.json")
        engine_mod.HeadTracker = FakeTracker
        self.os = FakeSession()
        backends.Session = lambda: self.os
        FakeTracker.face = None
        self.engines = []

    def tearDown(self):
        for e in self.engines:
            e.stop()
        (engine_mod.CONFIG_DIR, engine_mod.CONFIG_PATH, engine_mod.CALIB_PATH,
         engine_mod.HeadTracker, backends.Session) = self.saved
        self.tmp.cleanup()

    def start(self, mode, gaze_model=True, **cfg):
        config = {**engine_mod.DEFAULTS, "mode": mode, "delay_ms": 100, "window_delay_ms": 150,
                  "typing_wait_s": 1.0, **cfg}
        e = engine_mod.Engine(config)
        e.set_calibration([-20.0, 20.0], MONITORS, identity_model() if gaze_model else None)
        self.engines.append(e)
        return e

    def wait_for(self, cond, timeout=2.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if cond():
                return True
            time.sleep(0.02)
        return False

    # --- window mode -------------------------------------------------------
    def test_window_mode_focuses_window_under_gaze_without_moving_pointer(self):
        e = self.start("window")
        FakeTracker.face = looking_at(1400, 500)             # window B, same monitor as active A
        self.assertTrue(self.wait_for(lambda: self.os.activated == [B]))
        self.assertEqual(self.os.warps, [])
        self.assertIsNotNone(e.gaze)

    def test_window_mode_waits_while_typing(self):
        self.start("window")
        self.os.key_age = 0.2                                # typed 0.2 s ago
        FakeTracker.face = looking_at(1400, 500)
        time.sleep(0.8)
        self.assertEqual(self.os.activated, [])
        self.os.key_age = 5.0
        self.assertTrue(self.wait_for(lambda: self.os.activated == [B]))

    def test_window_mode_switches_monitor_and_focuses_that_window(self):
        self.start("window")
        FakeTracker.face = looking_at(2900, 500)             # window C on monitor 2
        self.assertTrue(self.wait_for(lambda: self.os.activated == [C]))
        self.assertEqual(len(self.os.warps), 1)

    def test_window_mode_ignores_glance_that_is_too_short(self):
        self.start("window", window_delay_ms=600)
        FakeTracker.face = looking_at(1400, 500)
        time.sleep(0.3)
        FakeTracker.face = looking_at(500, 500)              # back to A before the dwell ends
        time.sleep(0.8)
        self.assertEqual(self.os.activated, [])

    def test_window_mode_without_gaze_calibration_falls_back_to_monitor_mode(self):
        e = self.start("window", gaze_model=False)
        self.assertFalse(e.window_mode)
        FakeTracker.face = looking_at(1400, 500, yaw=20.0)   # head turned to monitor 2
        self.assertTrue(self.wait_for(lambda: C in self.os.activated))

    # --- monitor-only mode (the original behaviour) ------------------------
    def test_monitor_mode_follows_head_yaw(self):
        self.start("monitor")
        FakeTracker.face = looking_at(500, 500, yaw=20.0)    # monitor 2 by head angle
        self.assertTrue(self.wait_for(lambda: self.os.activated == [C]))
        self.assertEqual(len(self.os.warps), 1)

    def test_monitor_mode_ignores_gaze_and_windows(self):
        e = self.start("monitor")
        FakeTracker.face = looking_at(1400, 500, yaw=-20.0)  # eyes on window B, head on monitor 1
        time.sleep(0.8)
        self.assertEqual(self.os.activated, [])
        self.assertIsNone(e.gaze)

    def test_pause_releases_everything(self):
        e = self.start("window")
        e.set_enabled(False)
        FakeTracker.face = looking_at(1400, 500)
        time.sleep(0.5)
        self.assertEqual(self.os.activated, [])
        self.assertEqual(e.status, "Paused")


if __name__ == "__main__":
    unittest.main()
