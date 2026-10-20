"""Run: python -m unittest discover tests"""
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preview import render  # noqa: E402


class PreviewTest(unittest.TestCase):
    def test_renders_with_and_without_a_face(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        self.assertEqual(render(frame, None, ["no face"]).shape, (270, 480, 3))
        lm = [SimpleNamespace(x=0.5, y=0.5, z=0) for _ in range(478)]
        img = render(frame, lm, ["a", "b", "c"], width=320)
        self.assertEqual(img.shape, (180, 320, 3))
        self.assertGreater(img.sum(), 0)                    # something was drawn

    def test_is_mirrored(self):
        frame = np.zeros((100, 200, 3), np.uint8)
        frame[:, :20] = 255                                  # a bright strip on the camera's left
        img = render(frame, None, [], width=200)
        self.assertGreater(img[:, -20:].mean(), img[:, :20].mean())   # appears on the right, like a mirror


if __name__ == "__main__":
    unittest.main()
