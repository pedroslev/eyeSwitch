"""Face feature geometry on synthetic landmarks. Run: python -m unittest discover tests"""
import os
import sys
import time
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tracker import HAND_MODEL_PATH, LEFT_EYE, NOSE_TIP, RIGHT_EYE, HandProcess, _face  # noqa: E402

W, H = 640, 480


def landmarks(iris_dx=0.0, iris_dy=0.0, nose=(0.5, 0.58)):
    """478 landmarks; both eyes 0.06 wide, irises offset from the eye centers."""
    lm = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]

    def put(i, x, y):
        lm[i] = SimpleNamespace(x=x, y=y, z=0.0)

    # right eye is on the image's left: outer corner farther left than inner
    o, n, iris = RIGHT_EYE
    put(o, 0.40, 0.50), put(n, 0.46, 0.50), put(iris, 0.43 + iris_dx, 0.50 + iris_dy)
    o, n, iris = LEFT_EYE
    put(o, 0.60, 0.50), put(n, 0.54, 0.50), put(iris, 0.57 + iris_dx, 0.50 + iris_dy)
    put(NOSE_TIP, *nose)
    return lm


def matrix(yaw_deg=0.0, t=(1.0, 2.0, 40.0)):
    a = np.radians(yaw_deg)
    m = np.eye(4)
    m[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    m[:3, :3] *= 1.3                           # MediaPipe's matrix carries a scale
    m[:3, 3] = t
    return m


class FaceTest(unittest.TestCase):
    def test_centered_irises(self):
        f = _face(matrix(), landmarks(), W, H)
        self.assertAlmostEqual(f.yaw, 0, places=3)
        self.assertAlmostEqual(f.pitch, 0, places=3)
        self.assertAlmostEqual(f.h, 0, places=3)
        self.assertAlmostEqual(f.v, 0, places=3)
        self.assertEqual((f.tx, f.ty, f.tz), (1.0, 2.0, 40.0))

    def test_both_eyes_agree_on_direction(self):
        right = _face(matrix(), landmarks(iris_dx=+0.01), W, H)     # iris towards image right
        self.assertAlmostEqual(right.h, 0.01 * W / (0.06 * W), places=3)
        left = _face(matrix(), landmarks(iris_dx=-0.01), W, H)
        self.assertAlmostEqual(left.h, -right.h, places=3)

    def test_looking_down_is_positive_v(self):
        f = _face(matrix(), landmarks(iris_dy=+0.005), W, H)
        self.assertGreater(f.v, 0)

    def test_nose_offset_is_in_eye_distances_and_ignores_distance_and_roll(self):
        f = _face(matrix(), landmarks(nose=(0.50, 0.58)), W, H)
        self.assertAlmostEqual(f.nx, 0.0, places=3)
        self.assertGreater(f.ny, 0)                                       # nose below the eyes
        turned = _face(matrix(), landmarks(nose=(0.53, 0.58)), W, H)      # nose towards image right
        self.assertAlmostEqual(turned.nx - f.nx, 0.03 * W / (0.20 * W), places=3)
        # the same face twice as far away (everything scaled about the image centre): same nx/ny
        far = landmarks(nose=(0.53, 0.58))
        for p in far:
            p.x, p.y = 0.5 + (p.x - 0.5) / 2, 0.5 + (p.y - 0.5) / 2
        again = _face(matrix(), far, W, H)
        self.assertAlmostEqual(again.nx, turned.nx, places=3)
        self.assertAlmostEqual(again.ny, turned.ny, places=3)

    def test_yaw_and_scale(self):
        self.assertAlmostEqual(_face(matrix(25), landmarks(), W, H).yaw, 25, places=3)
        self.assertAlmostEqual(_face(matrix(-10), landmarks(), W, H).yaw, -10, places=3)


@unittest.skipUnless(os.path.exists(HAND_MODEL_PATH), "needs the hand model downloaded")
class HandProcessTest(unittest.TestCase):
    """The recognizer lives in its own process: a crash there must not take the app down."""

    def ask(self, hand, frames=40):
        verdicts = []
        for i in range(frames):
            v = hand.poll()
            if v is not HandProcess.NOTHING_NEW:
                verdicts.append(v)
            if not hand.busy:
                hand.submit(np.zeros((90, 160, 3), np.uint8), i * 100)
            time.sleep(0.05)
        return verdicts

    def test_answers_and_survives_being_killed(self):
        hand = HandProcess(HAND_MODEL_PATH)
        try:
            self.assertGreater(len(self.ask(hand)), 3)
            hand.proc.kill()
            time.sleep(0.3)
            self.ask(hand, 20)
            self.assertEqual(hand.restarts, 1)
            self.assertTrue(hand.alive)
            self.assertGreater(len(self.ask(hand)), 3)
        finally:
            hand.close()


if __name__ == "__main__":
    unittest.main()
