"""The full calibration flow end to end, offscreen, with a simulated camera. Run: python -m unittest discover tests"""
import os
import sys
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import eyeswitch as es  # noqa: E402
from backends.base import Monitor  # noqa: E402
from tracker import Face  # noqa: E402

MONITORS = [Monitor(0, 0, 1920, 1080), Monitor(1920, 0, 1920, 1080)]


class FakeEngine:
    def __init__(self):
        self.monitors, self.cfg = MONITORS, {"calibration_preview": False}
        self.calibrating, self.preview_wanted, self.preview = False, 0, None
        self.cap, self.saved = None, None

    def begin_capture(self):
        self.cap = []

    def end_capture(self):
        out, self.cap = self.cap or [], None
        return out

    def set_calibration(self, yaws, monitors, model=None):
        self.saved = (yaws, model)


def face_for(x, y, rng):
    """What a head looking at desktop pixel (x, y) would measure: head pose carries the signal, the
    vertical iris offset `v` is pure noise, like in a real webcam."""
    return Face(yaw=-(x - 1920) / 3840 * 80 + rng.normal(0, 0.4), pitch=(y - 540) / 1080 * 20 + rng.normal(0, 0.3),
                tx=18 + (x - 1920) / 3840 * 4 + rng.normal(0, 0.05), ty=-4 + rng.normal(0, 0.05),
                tz=-50 + rng.normal(0, 0.1), h=(x - 1920) / 3840 * 0.2 + rng.normal(0, 0.01),
                v=rng.normal(0, 0.012))


class CalibrationFlowTest(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.tmp = tempfile.TemporaryDirectory()
        self.saved = (es.CONFIG_DIR, es.FullCalibration.SETTLE_MS, es.FullCalibration.SAMPLE_MS,
                      es.FullCalibration.INTRO_MS, es.FullCalibration.CHECK_MS)
        es.CONFIG_DIR = self.tmp.name
        es.FullCalibration.SETTLE_MS, es.FullCalibration.SAMPLE_MS = 20, 120
        es.FullCalibration.INTRO_MS = es.FullCalibration.CHECK_MS = 20

    def tearDown(self):
        (es.CONFIG_DIR, es.FullCalibration.SETTLE_MS, es.FullCalibration.SAMPLE_MS,
         es.FullCalibration.INTRO_MS, es.FullCalibration.CHECK_MS) = self.saved
        self.tmp.cleanup()

    def run_flow(self, cancel_after=None):
        eng, results, rng = FakeEngine(), [], np.random.default_rng(3)
        cal = es.FullCalibration(eng, lambda ok, msg: results.append((ok, msg)))
        t0 = time.time()
        while not results and time.time() - t0 < 60:
            self.app.processEvents()
            if eng.cap is not None and cal.win is not None:       # the camera delivers a face of the shown dot
                m = MONITORS[cal.win_mon]
                eng.cap.append(face_for(m.x + cal.win.fx * m.w, m.y + cal.win.fy * m.h, rng))
            if cancel_after and cal.i >= cancel_after:
                cal.cancel()
            time.sleep(0.004)
        return eng, results, cal

    def test_full_flow_fits_verifies_and_saves(self):
        eng, results, cal = self.run_flow()
        self.assertTrue(results and results[0][0], results)
        message = results[0][1]
        self.assertIn("Check on 6 new dots", message)
        self.assertIn("good", message)
        yaws, model = eng.saved
        self.assertEqual(len(yaws), 2)
        self.assertNotIn(6, model.active)                         # the noisy iris feature is dropped
        self.assertEqual(len(cal.steps), 36)                      # 2 x (15 grid + 3 check)
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "calibration_samples.npz")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "calibration_report.json")))
        self.assertFalse(eng.calibrating)

    def test_cancel_stops_cleanly_and_saves_nothing(self):
        eng, results, _ = self.run_flow(cancel_after=5)
        self.assertEqual(results[0][0], False)
        self.assertIn("cancelled", results[0][1])
        self.assertIsNone(eng.saved)
        self.assertFalse(eng.calibrating)

    def test_preview_is_counted_and_released(self):
        eng = FakeEngine()
        eng.cfg["calibration_preview"] = True
        results = []
        cal = es.QuickCalibration(eng, lambda ok, msg: results.append((ok, msg)))
        self.assertEqual(eng.preview_wanted, 1)
        cal.cancel()
        t0 = time.time()
        while not results and time.time() - t0 < 10:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertEqual(eng.preview_wanted, 0)


if __name__ == "__main__":
    unittest.main()
