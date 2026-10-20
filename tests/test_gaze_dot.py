"""The test dot's motion. Run: python -m unittest discover tests"""
import math
import os
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PySide6.QtWidgets import QApplication  # noqa: E402

import eyeswitch as es  # noqa: E402
from backends.base import Monitor  # noqa: E402

MONITORS = [Monitor(0, 0, 1920, 1080), Monitor(1920, 0, 1920, 1080)]


class GlideTest(unittest.TestCase):
    def test_approaches_the_target_without_overshooting(self):
        pos, path = (0.0, 0.0), []
        for _ in range(60):
            pos = es.glide(pos, (1000.0, 500.0), 1 / 60)
            path.append(pos[0])
        self.assertTrue(all(b >= a for a, b in zip(path, path[1:])))          # monotonic: no overshoot
        self.assertLess(1000 - path[-1], 5)                                    # arrives within a second
        self.assertLess(path[0], 300)                                          # but not in one jump

    def test_a_screen_wide_jump_is_drawn_as_a_run_not_a_tug(self):
        pos, steps = (0.0, 0.0), []
        for _ in range(120):
            new = es.glide(pos, (3000.0, 600.0), 1 / 60)
            steps.append(math.hypot(new[0] - pos[0], new[1] - pos[1]))
            pos = new
        self.assertLessEqual(max(steps), es.MAX_DOT_SPEED / 60 + 1e-6)         # never faster than the cap
        self.assertGreater(sum(1 for s in steps if s > 5), 15)                  # many visible frames of travel
        self.assertAlmostEqual(pos[0], 3000.0, delta=2)                         # and it still arrives

    def test_small_moves_are_not_slowed_by_the_cap(self):
        slow = fast = (0.0, 0.0)
        for _ in range(10):
            slow = es.glide(slow, (60.0, 0.0), 1 / 60)
            fast = es.glide(fast, (60.0, 0.0), 1 / 60, vmax=1e9)
        self.assertAlmostEqual(slow[0], fast[0], places=6)

    def test_the_same_speed_at_any_frame_rate(self):
        slow = fast = (0.0, 0.0)
        for _ in range(10):
            slow = es.glide(slow, (1000.0, 0.0), 1 / 30)
        for _ in range(20):
            fast = es.glide(fast, (1000.0, 0.0), 1 / 60)
        self.assertAlmostEqual(slow[0], fast[0], delta=1.0)


class GazeDotTest(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])

    def test_glides_to_a_new_target_and_hides_without_one(self):
        dot = es.GazeDot("live")
        dot.set_target((480.0, 270.0), MONITORS)
        dot.tick(1 / 60)
        first = (dot.pos_f[0], dot.pos_f[1])
        dot.set_target((1440.0, 810.0), MONITORS)                      # a big jump of the estimate
        dot.tick(1 / 60)
        self.assertGreater(dot.pos_f[0], first[0])
        self.assertLess(dot.pos_f[0], dot.target[0])                   # moving, not teleported
        for _ in range(120):
            dot.tick(1 / 60)
        self.assertAlmostEqual(dot.pos_f[0], dot.target[0], delta=2)
        dot.set_target(None, MONITORS)
        self.assertIsNone(dot.pos_f)
        self.assertFalse(dot.isVisible())

    def test_ring_and_dot_have_different_sizes(self):
        self.assertGreater(es.GazeDot("held").size_px, es.GazeDot("live").size_px)


if __name__ == "__main__":
    unittest.main()
