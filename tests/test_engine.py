"""Engine decision logic with a fake camera and a fake OS backend. Run: python -m unittest discover tests"""
import os
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import backends  # noqa: E402
import engine as engine_mod  # noqa: E402
import virtualcam  # noqa: E402
from backends.base import BaseSession, Monitor  # noqa: E402
from gaze import GazeModel  # noqa: E402
from gestures import GestureModel  # noqa: E402
from tracker import Face  # noqa: E402

MONITORS = [Monitor(0, 0, 1920, 1080), Monitor(1920, 0, 1920, 1080)]
# window id -> (x, y, w, h); A and B share monitor 1, C is on monitor 2
WINDOWS = {1: (0, 0, 960, 1080), 2: (960, 0, 960, 1080), 3: (1920, 0, 1920, 1080)}
A, B, C = 1, 2, 3


class FakeSession(BaseSession):
    def __init__(self):
        self.active, self.activated, self.warps, self.key_age = A, [], [], 99.0
        self.pressed, self.clicks = [], 0

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

    def window_bounds(self, wid):
        return WINDOWS.get(wid)

    def press_keys(self, keys):
        if "nonsense" in keys:
            raise ValueError("unknown key: nonsense")
        self.pressed.append(list(keys))

    def click(self):
        self.clicks += 1

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
    face, opens, closes, args = None, 0, 0, None     # args: (camera, has_sink, hires) of the last open
    fail_open, no_frames, shapes_on, shapes = False, False, None, None
    frame = np.zeros((48, 64, 3), np.uint8)

    def __init__(self, camera=0, sink=None, hires=False, blendshapes=False):
        if FakeTracker.fail_open:
            raise RuntimeError("device busy")
        FakeTracker.opens += 1
        FakeTracker.args = (camera, sink is not None, hires)
        FakeTracker.shapes_on = blendshapes
        self.sink = sink
        self.landmarks = None

    @property
    def shapes(self):
        return FakeTracker.shapes if FakeTracker.shapes_on else None

    def read(self):
        time.sleep(0.02)
        if FakeTracker.no_frames:
            return None, None
        if self.sink:
            self.sink("frame")
        return FakeTracker.face, FakeTracker.frame

    def close(self):
        FakeTracker.closes += 1


class FakeWatcher:
    instance = None

    def __init__(self, path):
        self.path, self.foreign_open = path, threading.Event()
        FakeWatcher.instance = self

    def ignore_for(self, seconds):
        pass

    def stop(self):
        pass


class FakeVCam:
    instance = None

    def __init__(self, device):
        self.device, self.frames, self.closed = device, 0, False
        FakeVCam.instance = self

    def send(self, frame):
        self.frames += 1

    def close(self):
        self.closed = True


def looking_at(x, y, yaw=0.0):
    """A face whose gaze features (h, v) map straight onto desktop pixel (x, y)."""
    return Face(yaw, 0, 0, 0, 0, x, y)


def identity_model():
    w = np.zeros((8, 2))
    w[1 + 5, 0] = 1                  # x = h
    w[1 + 6, 1] = 1                  # y = v
    return GazeModel(1, np.zeros(7), np.ones(7), w, 100.0)


BROWS = GestureModel(["browInnerUp"], [1.0 / 0.7], 0.0, 4.0)    # score 1 when browInnerUp = 0.7


def binding(label="Eyebrows", action="keys", keys="super", hold_ms=100):
    return {"label": label, "action": action, "keys": keys, "hold_ms": hold_ms, "model": BROWS.to_dict()}


class EngineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saved = (engine_mod.CONFIG_DIR, engine_mod.CONFIG_PATH, engine_mod.CALIB_PATH,
                      engine_mod.HeadTracker, backends.Session, engine_mod.CameraWatcher, engine_mod.RETRY,
                      virtualcam.find_device, virtualcam.VirtualCamera, virtualcam.AVAILABLE)
        engine_mod.CONFIG_DIR = self.tmp.name
        engine_mod.CONFIG_PATH = os.path.join(self.tmp.name, "config.json")
        engine_mod.CALIB_PATH = os.path.join(self.tmp.name, "calibration.json")
        engine_mod.HeadTracker, engine_mod.CameraWatcher, engine_mod.RETRY = FakeTracker, FakeWatcher, 0.4
        virtualcam.find_device, virtualcam.VirtualCamera, virtualcam.AVAILABLE = (
            lambda: "/dev/video99", FakeVCam, True)
        self.os = FakeSession()
        backends.Session = lambda: self.os
        FakeTracker.face, FakeTracker.opens, FakeTracker.closes, FakeTracker.args = None, 0, 0, None
        FakeTracker.fail_open = FakeTracker.no_frames = False
        FakeTracker.shapes_on = FakeTracker.shapes = None
        FakeWatcher.instance = FakeVCam.instance = None
        self.engines = []

    def tearDown(self):
        for e in self.engines:
            e.stop()
        (engine_mod.CONFIG_DIR, engine_mod.CONFIG_PATH, engine_mod.CALIB_PATH,
         engine_mod.HeadTracker, backends.Session, engine_mod.CameraWatcher, engine_mod.RETRY,
         virtualcam.find_device, virtualcam.VirtualCamera, virtualcam.AVAILABLE) = self.saved
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


    # --- window edges ------------------------------------------------------
    def test_stays_on_the_active_window_near_its_edge(self):
        self.start("window", edge_px=100)                    # active window is A (x 0..960)
        FakeTracker.face = looking_at(1010, 500)             # 50 px into B: inside A's tolerance
        time.sleep(0.9)
        self.assertEqual(self.os.activated, [])
        FakeTracker.face = looking_at(1200, 500)             # clearly inside B
        self.assertTrue(self.wait_for(lambda: self.os.activated == [B]))

    def test_gaze_past_the_screen_edge_is_clamped_to_it(self):
        e = self.start("window")
        FakeTracker.face = looking_at(-500, 5000)            # the model overshoots wildly
        self.assertTrue(self.wait_for(lambda: e.gaze is not None and e.gaze == (0, 1079)))

    # --- gestures ----------------------------------------------------------
    def gestures(self, mode, *bindings, **cfg):
        e = self.start(mode, **cfg)
        e.cfg["gestures"] = {"enabled": True, "pause_s": 1.0, "bindings": list(bindings)}
        e.gestures_changed()
        return e

    def test_gesture_presses_keys(self):
        self.gestures("monitor", binding())
        FakeTracker.face = looking_at(500, 500)
        FakeTracker.shapes = {"browInnerUp": 0.0}
        self.assertTrue(self.wait_for(lambda: FakeTracker.shapes_on))      # blendshapes got switched on
        time.sleep(0.3)
        self.assertEqual(self.os.pressed, [])
        FakeTracker.shapes = {"browInnerUp": 0.7}                          # eyebrows up
        self.assertTrue(self.wait_for(lambda: self.os.pressed == [["super"]]))
        time.sleep(0.4)
        self.assertEqual(len(self.os.pressed), 1)                          # once per gesture

    def test_gesture_waits_for_the_hold_time(self):
        self.gestures("monitor", binding(hold_ms=400))
        FakeTracker.face = looking_at(500, 500)
        FakeTracker.shapes = {"browInnerUp": 0.7}
        time.sleep(0.2)
        FakeTracker.shapes = {"browInnerUp": 0.0}                          # a twitch
        time.sleep(0.5)
        self.assertEqual(self.os.pressed, [])

    def test_click_gesture_clicks_where_you_look(self):
        e = self.gestures("window", binding("Bite", "click", ""))
        FakeTracker.face = looking_at(2900, 400)
        FakeTracker.shapes = {"browInnerUp": 0.7}
        self.assertTrue(self.wait_for(lambda: self.os.clicks == 1))
        self.assertIn((2900, 400), self.os.warps)
        self.assertIn("clicked", e.gesture_log)

    def test_click_gesture_in_monitor_mode_still_uses_gaze_if_calibrated(self):
        self.gestures("monitor", binding("Bite", "click", ""))
        FakeTracker.face = looking_at(2900, 400, yaw=-20.0)                # head on monitor 1
        FakeTracker.shapes = {"browInnerUp": 0.7}
        self.assertTrue(self.wait_for(lambda: self.os.clicks == 1))
        self.assertIn((2900, 400), self.os.warps)

    def test_key_gesture_silences_automatic_focus_for_a_while(self):
        self.gestures("window", binding())                                  # pause_s = 1.0
        FakeTracker.face = looking_at(1400, 500)                            # looking at B, A is active
        FakeTracker.shapes = {"browInnerUp": 0.7}
        self.assertTrue(self.wait_for(lambda: self.os.pressed == [["super"]]))
        FakeTracker.shapes = {"browInnerUp": 0.0}
        time.sleep(0.7)
        self.assertEqual(self.os.activated, [])                             # the overview owns the focus
        self.assertTrue(self.wait_for(lambda: self.os.activated == [B], timeout=2.5))

    def test_bad_key_is_reported_not_fatal(self):
        e = self.gestures("monitor", binding(keys="nonsense"))
        FakeTracker.face = looking_at(500, 500)
        FakeTracker.shapes = {"browInnerUp": 0.7}
        self.assertTrue(self.wait_for(lambda: "failed" in e.gesture_log))
        self.assertTrue(self.engines[0]._thread.is_alive())

    def test_gestures_off_means_no_blendshapes_and_no_keys(self):
        e = self.start("monitor")
        e.cfg["gestures"] = {"enabled": False, "pause_s": 1.0, "bindings": [binding()]}
        e.gestures_changed()
        FakeTracker.face = looking_at(500, 500)
        FakeTracker.shapes = {"browInnerUp": 0.7}
        time.sleep(0.5)
        self.assertFalse(FakeTracker.shapes_on)
        self.assertEqual(self.os.pressed, [])

    def test_training_turns_blendshapes_on_and_collects_them(self):
        e = self.start("monitor")
        FakeTracker.face = looking_at(500, 500)
        FakeTracker.shapes = {"browInnerUp": 0.3}
        e.want_shapes = e.calibrating = True
        self.assertTrue(self.wait_for(lambda: FakeTracker.shapes_on))
        e.begin_shape_capture()
        time.sleep(0.4)
        samples = e.end_shape_capture()
        self.assertGreater(len(samples), 5)
        self.assertEqual(samples[0], {"browInnerUp": 0.3})

    # --- debug view --------------------------------------------------------
    def test_preview_is_rendered_only_while_someone_watches(self):
        e = self.start("monitor")
        FakeTracker.face = looking_at(500, 500)
        time.sleep(0.4)
        self.assertIsNone(e.preview)
        e.preview_wanted = 1
        self.assertTrue(self.wait_for(lambda: e.preview is not None))
        self.assertEqual(e.preview.shape[1], 480)
        self.assertEqual(e.preview.shape[2], 3)

    def test_preview_wakes_the_camera_even_when_paused(self):
        e = self.start("monitor", enabled=False)
        time.sleep(0.3)
        self.assertEqual(FakeTracker.opens, 0)
        e.preview_wanted = 1
        self.assertTrue(self.wait_for(lambda: FakeTracker.opens == 1 and e.preview is not None))
        e.preview_wanted = 0
        self.assertTrue(self.wait_for(lambda: FakeTracker.closes == 1))

    # --- sharing the camera ------------------------------------------------
    def test_yields_the_camera_when_another_app_opens_it_then_takes_it_back(self):
        e = self.start("monitor")
        FakeTracker.face = looking_at(500, 500)
        self.assertTrue(self.wait_for(lambda: FakeTracker.opens == 1))
        FakeWatcher.instance.foreign_open.set()                 # a video call app opens the device
        self.assertTrue(self.wait_for(lambda: FakeTracker.closes == 1))
        self.assertIn("another app", e.status)
        FakeTracker.fail_open = True                            # it streams: our retries fail
        time.sleep(1.0)
        self.assertEqual(FakeTracker.opens, 1)
        self.assertIn("another app", e.status)
        FakeTracker.fail_open = False                           # call over
        self.assertTrue(self.wait_for(lambda: FakeTracker.opens == 2))

    def test_new_foreign_opens_extend_the_wait(self):
        self.start("monitor")
        FakeTracker.face = looking_at(500, 500)
        self.assertTrue(self.wait_for(lambda: FakeTracker.opens == 1))
        FakeWatcher.instance.foreign_open.set()
        self.assertTrue(self.wait_for(lambda: FakeTracker.closes == 1))
        time.sleep(0.25)
        FakeWatcher.instance.foreign_open.set()                 # still being probed by the other app
        time.sleep(0.3)                                         # past the first deadline (0.4 s)
        self.assertEqual(FakeTracker.opens, 1)
        self.assertTrue(self.wait_for(lambda: FakeTracker.opens == 2))

    def test_no_frames_is_reported_as_a_camera_error(self):
        e = self.start("monitor")
        FakeTracker.no_frames = True
        self.assertTrue(self.wait_for(lambda: "no frames" in e.status))

    def test_window_mode_asks_for_high_resolution(self):
        self.start("window")
        self.assertTrue(self.wait_for(lambda: FakeTracker.args is not None))
        self.assertEqual(FakeTracker.args, (0, False, True))

    def test_monitor_mode_keeps_low_resolution(self):
        self.start("monitor")
        self.assertTrue(self.wait_for(lambda: FakeTracker.args is not None))
        self.assertEqual(FakeTracker.args, (0, False, False))

    def test_virtual_camera_receives_every_frame(self):
        e = self.start("monitor", virtual_camera=True)
        FakeTracker.face = looking_at(500, 500)
        self.assertTrue(self.wait_for(lambda: FakeVCam.instance and FakeVCam.instance.frames > 5))
        self.assertEqual(e.vcam_state, "active")
        self.assertEqual(FakeVCam.instance.device, "/dev/video99")
        self.assertEqual(FakeTracker.args, (0, True, False))

    def test_virtual_camera_not_set_up_still_tracks(self):
        virtualcam.find_device = lambda: None
        e = self.start("monitor", virtual_camera=True)
        FakeTracker.face = looking_at(500, 500, yaw=20.0)
        self.assertTrue(self.wait_for(lambda: self.os.activated == [C]))
        self.assertEqual(e.vcam_state, "not set up")

    def test_toggling_the_virtual_camera_reopens_the_camera(self):
        e = self.start("monitor")
        self.assertTrue(self.wait_for(lambda: FakeTracker.opens == 1))
        e.cfg["virtual_camera"] = True
        self.assertTrue(self.wait_for(lambda: FakeTracker.opens == 2))
        self.assertEqual(FakeTracker.args, (0, True, False))

    def test_pause_closes_camera_and_virtual_camera(self):
        e = self.start("monitor", virtual_camera=True)
        self.assertTrue(self.wait_for(lambda: FakeVCam.instance is not None))
        e.set_enabled(False)
        self.assertTrue(self.wait_for(lambda: FakeVCam.instance.closed and FakeTracker.closes >= 1))
        self.assertEqual(e.vcam_state, "off")


if __name__ == "__main__":
    unittest.main()
