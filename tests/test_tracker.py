"""Face feature geometry on synthetic landmarks. Run: python -m unittest discover tests"""
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tracker import LEFT_EYE, RIGHT_EYE, _face  # noqa: E402

W, H = 640, 480


def landmarks(iris_dx=0.0, iris_dy=0.0):
    """478 landmarks; both eyes 0.06 wide, irises offset from the eye centers."""
    lm = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]

    def put(i, x, y):
        lm[i] = SimpleNamespace(x=x, y=y, z=0.0)

    # right eye is on the image's left: outer corner farther left than inner
    o, n, iris = RIGHT_EYE
    put(o, 0.40, 0.50), put(n, 0.46, 0.50), put(iris, 0.43 + iris_dx, 0.50 + iris_dy)
    o, n, iris = LEFT_EYE
    put(o, 0.60, 0.50), put(n, 0.54, 0.50), put(iris, 0.57 + iris_dx, 0.50 + iris_dy)
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

    def test_yaw_and_scale(self):
        self.assertAlmostEqual(_face(matrix(25), landmarks(), W, H).yaw, 25, places=3)
        self.assertAlmostEqual(_face(matrix(-10), landmarks(), W, H).yaw, -10, places=3)


if __name__ == "__main__":
    unittest.main()
