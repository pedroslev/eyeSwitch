"""Engine decision logic with a fake camera and a fake OS backend. Run: python -m unittest discover tests"""
import json
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
    face, opens, closes, args = None, 0, 0, None     # args: (camera, has_sink) of the last open
    fail_open, no_frames, shapes_on, shapes = False, False, None, None
    frame = np.zeros((48, 64, 3), np.uint8)

    def __init__(self, camera=0, sink=None, blendshapes=False, height="auto", fps=30):
        if FakeTracker.fail_open:
            raise RuntimeError("device busy")
        FakeTracker.opens += 1
        FakeTracker.args = (camera, sink is not None)
        FakeTracker.height_asked, FakeTracker.fps_asked = height, fps
        self.max_fps, self.camera_fps = fps, 50.0
        FakeTracker.rate = fps
        self.height = 1080 if height == "auto" else height     # what a 1080p camera would deliver
        FakeTracker.shapes_on = blendshapes
        self.sink = sink
        self.landmarks = None

    @property
    def shapes(self):
        return FakeTracker.shapes if FakeTracker.shapes_on else None

    @property
    def blink(self):
        s = FakeTracker.shapes if isinstance(FakeTracker.shapes, dict) else None
        return None if not s else (s.get("eyeBlinkLeft", 0.0) + s.get("eyeBlinkRight", 0.0)) / 2

    def read(self):
        time.sleep(0.02)
        if FakeTracker.no_frames:
            return None, None
        if self.sink:
            self.sink("frame")
        return FakeTracker.face, FakeTracker.frame

    def set_rate(self, fps):
        FakeTracker.rate = fps

    def raw(self):
        return np.zeros((478, 3), np.float32), np.eye(4, dtype=np.float32), (64, 48), {"eyeLookUpLeft": 0.1}

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
    return Face(yaw, 0, 0, 0, 0, x, y, 0, 0)


def identity_model():
    w = np.zeros((10, 2))
    w[1 + 5, 0] = 1                  # x = h
    w[1 + 6, 1] = 1                  # y = v
    return GazeModel(1, np.zeros(9), np.ones(9), w, 100.0)


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
        FakeTracker.rate = None
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

    # --- blinks, glitches, a pose that left the calibrated range -----------
    def test_a_blink_holds_the_estimate_instead_of_following_it(self):
        e = self.start("window")
        FakeTracker.face = looking_at(500, 500)                           # window A, the active one
        FakeTracker.shapes = {"eyeBlinkLeft": 0.2, "eyeBlinkRight": 0.2}
        time.sleep(0.5)                                                   # the open-eye baseline
        FakeTracker.shapes = {"eyeBlinkLeft": 0.7, "eyeBlinkRight": 0.7}  # lids close...
        FakeTracker.face = looking_at(1400, 500)                          # ...and the estimate jumps to B
        self.assertTrue(self.wait_for(lambda: e.blinking))
        time.sleep(0.2)
        FakeTracker.shapes = {"eyeBlinkLeft": 0.2, "eyeBlinkRight": 0.2}  # eyes open, looking at A again
        FakeTracker.face = looking_at(500, 500)
        time.sleep(1.0)
        self.assertEqual(self.os.activated, [])

    def test_the_same_jump_without_a_blink_does_change_focus(self):
        self.start("window")
        FakeTracker.face = looking_at(500, 500)
        FakeTracker.shapes = {"eyeBlinkLeft": 0.2, "eyeBlinkRight": 0.2}
        time.sleep(0.5)
        FakeTracker.face = looking_at(1400, 500)                          # eyes open: a real look at B
        self.assertTrue(self.wait_for(lambda: self.os.activated == [B]))

    def test_the_shown_estimate_follows_at_once_while_the_decision_point_waits_to_be_sure(self):
        e = self.start("window")
        FakeTracker.face = looking_at(1000, 500)
        self.assertTrue(self.wait_for(lambda: e.gaze_live is not None and e.gaze is not None))
        time.sleep(0.3)
        FakeTracker.face = looking_at(1800, 500)                     # a clear move, far outside the radius
        t0, live_at, held_at = time.monotonic(), None, None
        while time.monotonic() - t0 < 2 and held_at is None:
            if live_at is None and e.gaze_live and e.gaze_live[0] > 1400:
                live_at = time.monotonic() - t0
            if e.gaze and e.gaze[0] > 1400:
                held_at = time.monotonic() - t0
            time.sleep(0.003)
        self.assertIsNotNone(live_at)
        self.assertIsNotNone(held_at)
        self.assertLess(live_at, held_at)                            # flows first; the held point confirms later

    def test_a_one_frame_glitch_does_not_change_focus(self):
        self.start("window")
        FakeTracker.face = looking_at(500, 500)
        time.sleep(0.3)
        FakeTracker.face = looking_at(1400, 500)
        time.sleep(0.03)                                                  # about one frame
        FakeTracker.face = looking_at(500, 500)
        time.sleep(0.8)
        self.assertEqual(self.os.activated, [])

    def test_a_pose_outside_the_calibrated_range_does_nothing_and_says_so(self):
        e = self.start("window")
        e.gaze_model.lo, e.gaze_model.hi = np.zeros(9), np.zeros(9)
        e.gaze_model.hi[5], e.gaze_model.hi[6] = 3840, 1080
        FakeTracker.face = looking_at(9000, 500)                          # far beyond what was calibrated
        self.assertTrue(self.wait_for(lambda: e.out_of_range))
        time.sleep(0.5)
        self.assertEqual(self.os.activated, [])
        self.assertIsNone(e.gaze)
        self.assertIn("Outside the calibrated range", e.status)
        FakeTracker.face = looking_at(1400, 500)                          # back inside: it works again
        self.assertTrue(self.wait_for(lambda: self.os.activated == [B]))

    def test_a_gaze_model_from_an_older_tracker_is_ignored(self):
        stale = GazeModel(1, np.zeros(7), np.ones(7), np.zeros((8, 2)), 100.0)
        with open(engine_mod.CALIB_PATH, "w") as f:
            json.dump({"monitors": [list(m) for m in MONITORS], "yaw": [-20.0, 20.0], "gaze": stale.to_dict()}, f)
        e = engine_mod.Engine({**engine_mod.DEFAULTS, "mode": "window"})
        self.engines.append(e)
        self.assertTrue(e.stale_model)
        self.assertIsNone(e.gaze_model)
        self.assertFalse(e.window_mode)

    def test_window_mode_always_asks_for_blendshapes_for_blink_detection(self):
        self.start("window")
        self.assertTrue(self.wait_for(lambda: FakeTracker.shapes_on))

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

    def test_a_capture_keeps_the_raw_geometry_aligned_with_the_faces(self):
        e = self.start("monitor")
        FakeTracker.face = looking_at(500, 500)
        time.sleep(0.3)
        e.begin_capture()
        time.sleep(0.4)
        faces = e.end_capture()
        self.assertGreater(len(faces), 5)
        self.assertEqual(len(e.last_capture_raw), len(faces))
        landmarks, matrix, size, shapes = e.last_capture_raw[0]
        self.assertEqual((landmarks.shape, matrix.shape, size, shapes), ((478, 3), (4, 4), (64, 48),
                                                                          {"eyeLookUpLeft": 0.1}))

    def test_the_tracking_rate_setting_reaches_the_tracker_and_defaults_to_30(self):
        self.start("monitor")
        self.assertTrue(self.wait_for(lambda: getattr(FakeTracker, "fps_asked", None) == 30))
        e = self.start("monitor", tracking_fps=15)
        self.assertTrue(self.wait_for(lambda: FakeTracker.fps_asked == 15))

    def test_the_measured_rate_is_tracked_and_shown(self):
        e = self.start("monitor")
        FakeTracker.face = looking_at(500, 500)
        self.assertTrue(self.wait_for(lambda: e.fps > 30))              # the fake camera runs near 50 per second

    def test_a_chosen_resolution_reaches_the_tracker(self):
        e = self.start("window", camera_height=720)
        self.assertTrue(self.wait_for(lambda: e.active_height == 720))
        self.assertEqual(FakeTracker.height_asked, 720)

    def test_a_gaze_model_calibrated_at_another_resolution_is_not_used(self):
        e = self.start("window", camera_height=720)
        self.assertTrue(self.wait_for(lambda: e.active_height == 720))
        e.set_calibration([-20.0, 20.0], MONITORS, identity_model(), 720)
        self.assertTrue(e.window_mode)
        e.cfg["camera_height"] = 1080                                   # changed without recalibrating
        self.assertTrue(self.wait_for(lambda: e.active_height == 1080))
        self.assertTrue(e.resolution_mismatch)
        self.assertFalse(e.window_mode)
        e.set_calibration([-20.0, 20.0], MONITORS, identity_model())    # a new calibration, at what is in use
        self.assertEqual(e.calib_height, 1080)
        self.assertTrue(e.window_mode)

    def test_a_calibration_without_a_known_height_is_never_blocked(self):
        e = self.start("window")
        self.assertTrue(self.wait_for(lambda: e.active_height == 1080))
        e.calib_height = None
        self.assertFalse(e.resolution_mismatch)
        self.assertTrue(e.window_mode)

    def test_a_calibration_saved_before_the_height_was_recorded_counts_as_720p(self):
        model = identity_model()
        with open(engine_mod.CALIB_PATH, "w") as f:
            json.dump({"monitors": [list(m) for m in MONITORS], "yaw": [-20.0, 20.0], "gaze": model.to_dict()}, f)
        e = engine_mod.Engine({**engine_mod.DEFAULTS, "mode": "window"})          # the fake camera gives 1080p
        self.engines.append(e)
        self.assertEqual(e.calib_height, 720)
        self.assertTrue(self.wait_for(lambda: e.resolution_mismatch))
        self.assertFalse(e.window_mode)

    def test_the_calibration_remembers_its_resolution_across_restarts(self):
        e = self.start("window")
        e.set_calibration([-20.0, 20.0], MONITORS, identity_model(), 1080)
        again = engine_mod.Engine({**engine_mod.DEFAULTS, "mode": "window"})
        self.engines.append(again)
        self.assertEqual(again.calib_height, 1080)
        again.active_height = 720                                       # the camera now gives something else
        self.assertTrue(again.resolution_mismatch)

    # --- adaptive tracking rate --------------------------------------------
    def calm_quickly(self):
        self.addCleanup(setattr, engine_mod, "CALM_S", engine_mod.CALM_S)
        engine_mod.CALM_S = 0.4

    def test_slows_down_when_nothing_happens_and_speeds_up_when_the_head_moves(self):
        self.calm_quickly()
        self.start("monitor")
        FakeTracker.face = looking_at(500, 500, yaw=0.0)
        self.assertTrue(self.wait_for(lambda: FakeTracker.rate == engine_mod.IDLE_FPS, 3))
        FakeTracker.face = looking_at(500, 500, yaw=15.0)                    # the head moves
        self.assertTrue(self.wait_for(lambda: FakeTracker.rate == 30, 1))
        FakeTracker.face = looking_at(500, 500, yaw=15.0)
        self.assertTrue(self.wait_for(lambda: FakeTracker.rate == engine_mod.IDLE_FPS, 3))   # calm again

    def test_the_gaze_leaving_its_fixation_wakes_the_tracker_in_window_mode(self):
        self.calm_quickly()
        self.start("window")
        FakeTracker.face = looking_at(500, 500)
        self.assertTrue(self.wait_for(lambda: FakeTracker.rate == engine_mod.IDLE_FPS, 3))
        FakeTracker.face = looking_at(1400, 500)                              # looking elsewhere
        self.assertTrue(self.wait_for(lambda: FakeTracker.rate == 30, 1))

    def test_adaptive_rate_can_be_switched_off(self):
        self.calm_quickly()
        self.start("monitor", adaptive_rate=False)
        FakeTracker.face = looking_at(500, 500)
        time.sleep(1.2)                                                       # three times CALM_S
        self.assertEqual(FakeTracker.rate, 30)

    def test_calibration_and_the_camera_view_keep_full_speed(self):
        self.calm_quickly()
        e = self.start("monitor")
        FakeTracker.face = looking_at(500, 500)
        self.assertTrue(self.wait_for(lambda: FakeTracker.rate == engine_mod.IDLE_FPS, 3))
        e.calibrating = True
        self.assertTrue(self.wait_for(lambda: FakeTracker.rate == 30, 1))
        e.calibrating = False
        self.assertTrue(self.wait_for(lambda: FakeTracker.rate == engine_mod.IDLE_FPS, 3))
        e.preview_wanted = 1
        self.assertTrue(self.wait_for(lambda: FakeTracker.rate == 30, 1))
        time.sleep(0.8)
        self.assertEqual(FakeTracker.rate, 30)

    def test_a_gesture_starting_wakes_the_tracker_before_it_triggers(self):
        self.calm_quickly()
        self.gestures("monitor", binding(hold_ms=2000))                      # never reaches its hold
        FakeTracker.face = looking_at(500, 500)
        FakeTracker.shapes = {"browInnerUp": 0.0}
        self.assertTrue(self.wait_for(lambda: FakeTracker.rate == engine_mod.IDLE_FPS, 3))
        FakeTracker.shapes = {"browInnerUp": 0.3}                             # starting, above 0.2 of the score
        self.assertTrue(self.wait_for(lambda: FakeTracker.rate == 30, 1))

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

    def test_every_mode_asks_for_the_cameras_best_resolution(self):
        for mode in ("monitor", "window"):
            e = self.start(mode)
            self.assertTrue(self.wait_for(lambda: e.active_height == 1080))
            self.assertEqual(FakeTracker.height_asked, "auto")
            e.stop()

    def test_virtual_camera_receives_every_frame(self):
        e = self.start("monitor", virtual_camera=True)
        FakeTracker.face = looking_at(500, 500)
        self.assertTrue(self.wait_for(lambda: FakeVCam.instance and FakeVCam.instance.frames > 5))
        self.assertEqual(e.vcam_state, "active")
        self.assertEqual(FakeVCam.instance.device, "/dev/video99")
        self.assertEqual(FakeTracker.args, (0, True))

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
        self.assertEqual(FakeTracker.args, (0, True))

    def test_pause_closes_camera_and_virtual_camera(self):
        e = self.start("monitor", virtual_camera=True)
        self.assertTrue(self.wait_for(lambda: FakeVCam.instance is not None))
        e.set_enabled(False)
        self.assertTrue(self.wait_for(lambda: FakeVCam.instance.closed and FakeTracker.closes >= 1))
        self.assertEqual(e.vcam_state, "off")


if __name__ == "__main__":
    unittest.main()
