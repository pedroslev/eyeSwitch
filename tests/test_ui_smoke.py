"""The windows and buttons the user actually opens. Run: python -m unittest discover tests

A dialog that was deleted by accident once made its button fail silently (the error only reaches a
terminal), and no logic test noticed. These open every screen for real, offscreen.
"""
import os
import subprocess
import sys
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from PySide6.QtWidgets import QApplication  # noqa: E402

import backends  # noqa: E402
import engine as engine_mod  # noqa: E402
import eyeswitch as es  # noqa: E402
from test_engine import FakeSession  # noqa: E402
from test_gestures import frames  # noqa: E402

SOURCES = ["eyeswitch.py", "engine.py", "tracker.py", "handworker.py", "gaze.py", "gestures.py", "samples.py", "preview.py",
           "autostart.py", "camwatch.py", "virtualcam.py", "install.py"]


class StaticCheckTest(unittest.TestCase):
    def test_no_name_is_used_without_being_defined(self):
        try:
            import pyflakes  # noqa: F401
        except ImportError:
            self.skipTest("pyflakes not installed (pip install pyflakes)")
        root = os.path.join(HERE, "..")
        files = [os.path.join(root, f) for f in SOURCES + ["backends/__init__.py", "backends/base.py", "backends/x11.py"]]
        out = subprocess.run([sys.executable, "-m", "pyflakes", *files], capture_output=True, text=True).stdout
        bad = [line for line in out.splitlines() if "undefined name" in line]
        self.assertEqual(bad, [])


class FakeGestureEngine:
    """Just what the gestures dialog and its training use, with a camera that feeds synthetic faces."""

    def __init__(self):
        self.gesture_scores, self.gesture_log, self.changes = {}, "", 0
        self.want_shapes = self.calibrating = False
        self.hand_wanted, self.hand_text = 0, ""
        self.shapes_seen, self.cap, self.captures = 0.0, None, 0
        self.relaxed, self.gesture = iter(frames(2000, 1)), iter(frames(2000, 2, browInnerUp=0.7, browOuterUpLeft=0.8))

    def gestures_changed(self):
        self.changes += 1

    def begin_shape_capture(self):
        self.cap, self.captures = [], self.captures + 1

    def end_shape_capture(self):
        out, self.cap = self.cap, None
        return out

    def tick(self):                                  # one camera frame
        if self.want_shapes:
            self.shapes_seen = time.monotonic()
            if self.cap is not None:
                self.cap.append(next(self.relaxed if self.captures == 1 else self.gesture))


class GesturesScreenTest(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.saved = (es.TrainDialog.REACT_MS, es.TrainDialog.RELAXED_MS, es.TrainDialog.GESTURE_MS)
        es.TrainDialog.REACT_MS, es.TrainDialog.RELAXED_MS, es.TrainDialog.GESTURE_MS = 50, 600, 600
        self.engine = FakeGestureEngine()
        self.cfg = {"gestures": {"enabled": True, "pause_s": 6, "bindings": [
            {"label": "Eyebrows", "action": "keys", "keys": "super", "hold_ms": 250, "model": None},
            {"label": "Bite lip", "action": "click", "keys": "", "hold_ms": 300, "model": None}]}}

    def tearDown(self):
        es.TrainDialog.REACT_MS, es.TrainDialog.RELAXED_MS, es.TrainDialog.GESTURE_MS = self.saved

    def pump(self, seconds, until=None):
        t0 = time.time()
        while time.time() - t0 < seconds and not (until and until()):
            self.engine.tick()
            self.app.processEvents()
            time.sleep(0.02)

    def test_the_screen_opens_with_the_saved_gestures(self):
        dlg = es.GesturesDialog(None, self.engine, self.cfg)
        dlg.show()
        self.app.processEvents()
        self.assertEqual([r.name.text() for r in dlg.rows], ["Eyebrows", "Bite lip"])
        self.assertEqual(dlg.rows[1].action.currentData(), "click")
        self.assertFalse(dlg.rows[1].keys.isEnabled())                       # a click needs no keys
        dlg.close()

    def test_gestures_can_be_added_edited_and_removed(self):
        dlg = es.GesturesDialog(None, self.engine, self.cfg)
        dlg.add()
        self.assertEqual(len(self.cfg["gestures"]["bindings"]), 3)
        self.assertEqual(self.cfg["gestures"]["bindings"][2]["label"], "Gesture 1")
        row = dlg.rows[0]
        row.keys.setText("alt+tab")
        row.keys.editingFinished.emit()
        self.assertEqual(self.cfg["gestures"]["bindings"][0]["keys"], "alt+tab")
        row.hold.setValue(400)
        self.assertEqual(self.cfg["gestures"]["bindings"][0]["hold_ms"], 400)
        dlg.remove(dlg.rows[2])
        self.assertEqual(len(self.cfg["gestures"]["bindings"]), 2)
        self.assertGreater(self.engine.changes, 3)                            # every edit reaches the engine
        dlg.enabled.setChecked(False)
        self.assertFalse(self.cfg["gestures"]["enabled"])

    def test_the_hand_toggle_option_writes_to_the_config_and_shows_the_hand(self):
        dlg = es.GesturesDialog(None, self.engine, self.cfg)
        self.assertFalse(dlg.hand.isChecked())
        dlg.hand.setChecked(True)
        self.assertTrue(self.cfg["gestures"]["hand_toggle"])
        self.engine.hand_text = "Open_Palm 0.93"
        dlg._live()
        self.assertIn("Open_Palm 0.93", dlg.hand_live.text())
        dlg.hand.setChecked(False)
        dlg._live()
        self.assertEqual(dlg.hand_live.text(), "")

    def test_training_the_hand_opens_and_saves_the_poses(self):
        from tests.test_gestures import HandPosesTest
        dlg = es.GesturesDialog(None, self.engine, self.cfg)
        self.assertIn("generic", dlg.hand_status.text())
        self.assertFalse(dlg.hand_reset.isEnabled())
        opened, fist, rest = HandPosesTest().poses(1)
        self.engine.hand_wanted = 0
        train = es.HandTrainDialog(None, self.engine, lambda m, msg: None)      # it can be built and cancelled
        self.assertEqual(self.engine.hand_wanted, 1)
        train.reject()
        self.assertEqual(self.engine.hand_wanted, 0)
        poses = es.gestures.HandPoses.train(opened, fist, rest)
        self.cfg["gestures"]["hand_poses"] = poses.to_dict()
        dlg._hand_status()
        self.assertIn("Trained", dlg.hand_status.text())
        dlg._reset_hand()
        self.assertNotIn("hand_poses", self.cfg["gestures"])
        self.assertIn("generic", dlg.hand_status.text())

    def test_the_overview_controls_write_to_the_config(self):
        dlg = es.GesturesDialog(None, self.engine, self.cfg)                    # the config has none of these keys yet
        self.assertTrue(dlg.cursor.isChecked())
        self.assertEqual(dlg.dwell.value(), 1200)
        dlg.cursor.setChecked(False)
        dlg.dwell.setValue(0)
        self.assertEqual((self.cfg["gestures"]["gaze_cursor"], self.cfg["gestures"]["dwell_ms"]), (False, 0))
        dlg.dwell.setValue(1500)
        self.assertEqual(self.cfg["gestures"]["dwell_ms"], 1500)
        self.assertGreaterEqual(self.engine.changes, 3)

    def test_training_a_gesture_end_to_end(self):
        dlg = es.GesturesDialog(None, self.engine, self.cfg)
        done = []
        row = dlg.rows[0]
        train = es.TrainDialog(dlg, self.engine, lambda model, msg: done.append((model, msg)))
        train.show()
        self.pump(10, until=lambda: done)
        self.assertTrue(done, "the training never finished")
        model, _ = done[0]
        self.assertTrue(model.reliable)
        self.assertEqual(set(model.names), {"browInnerUp", "browOuterUpLeft"})
        dlg._trained(row, model, "")                                           # what the dialog does with it
        self.assertEqual(row.binding["model"]["names"], model.names)
        self.assertIn("reliable", row.status.text())
        self.assertFalse(self.engine.want_shapes or self.engine.calibrating)   # the camera is released again

    def test_cancelling_the_training_releases_the_camera(self):
        train = es.TrainDialog(None, self.engine, lambda *_: None)
        self.assertTrue(self.engine.want_shapes)
        train.reject()
        self.assertFalse(self.engine.want_shapes or self.engine.calibrating)

    def test_the_live_bars_follow_the_scores(self):
        self.cfg["gestures"]["bindings"][0]["model"] = {"names": ["browInnerUp"], "weights": [1.4], "bias": 0.0, "quality": 4.0}
        dlg = es.GesturesDialog(None, self.engine, self.cfg)
        self.engine.gesture_scores["Eyebrows"], self.engine.gesture_log = 0.8, "Eyebrows: pressed super"
        dlg._live()
        self.assertEqual(dlg.rows[0].bar.value(), 80)
        self.assertEqual(dlg.log.text(), "Eyebrows: pressed super")
        self.assertEqual(dlg.rows[1].bar.value(), 0)                           # untrained: no bar


class MainWindowButtonsTest(unittest.TestCase):
    """The real window over a real (paused) engine and a stand-in desktop: every button leads somewhere."""

    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.tmp = tempfile.TemporaryDirectory()
        self.saved = (engine_mod.CONFIG_DIR, engine_mod.CONFIG_PATH, engine_mod.CALIB_PATH, backends.Session,
                      es.save_config)
        engine_mod.CONFIG_DIR = self.tmp.name
        engine_mod.CONFIG_PATH = os.path.join(self.tmp.name, "config.json")
        engine_mod.CALIB_PATH = os.path.join(self.tmp.name, "calibration.json")
        backends.Session = lambda: FakeSession()
        es.save_config = lambda cfg: None
        self.cfg = {**engine_mod.DEFAULTS, "enabled": False}                    # paused: no camera is opened
        self.engine = engine_mod.Engine(self.cfg)
        self.win = es.MainWindow(self.app, self.engine, self.cfg, False)

    def tearDown(self):
        self.win.preview_win.hide()
        self.engine.stop()
        (engine_mod.CONFIG_DIR, engine_mod.CONFIG_PATH, engine_mod.CALIB_PATH, backends.Session,
         es.save_config) = self.saved
        self.tmp.cleanup()

    def test_the_gestures_button_opens_the_gestures_screen(self):
        opened = []
        self.addCleanup(setattr, es.GesturesDialog, "exec", es.GesturesDialog.exec)
        es.GesturesDialog.exec = lambda dlg: opened.append(dlg) or 0           # do not block on the modal loop
        self.win.gestures_btn.click()
        self.assertEqual(len(opened), 1)
        self.assertIsInstance(opened[0], es.GesturesDialog)

    def test_the_camera_view_button_shows_the_camera_view_and_counts_it(self):
        self.win.preview_btn.click()
        self.assertTrue(self.win.preview_win.isVisible())
        self.assertEqual(self.engine.preview_wanted, 1)
        self.win.preview_win.hide()
        self.assertEqual(self.engine.preview_wanted, 0)

    def test_the_calibration_buttons_start_their_calibrations(self):
        started = []
        for cls, name in ((es.QuickCalibration, "quick"), (es.FullCalibration, "full")):
            self.addCleanup(setattr, cls, "__init__", cls.__init__)            # never leak the stand-ins
            cls.__init__ = lambda calib, engine, on_done, name=name: started.append(name)
        self.win.quick_btn.click()
        self.assertFalse(self.win.full_btn.isEnabled())                        # both are locked while one runs
        self.win.full_btn.setEnabled(True)                                     # (the calibration would do this)
        self.win.full_btn.click()
        self.assertEqual(started, ["quick", "full"])


if __name__ == "__main__":
    unittest.main()
