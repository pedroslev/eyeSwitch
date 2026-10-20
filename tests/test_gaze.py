"""Run: python -m unittest discover tests"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gaze import GazeModel, Smoother, fit  # noqa: E402

GRID = [(fx, fy) for fy in (0.08, 0.5, 0.92) for fx in (0.08, 0.5, 0.92)]
SCREEN = (3840, 1080)


def synthetic(noise, seed=0, per_point=25):
    """Fake face features whose true screen point is a mild nonlinear function of them."""
    rng = np.random.default_rng(seed)
    feats, targets, groups = [], [], []
    for m in range(2):                                   # two monitors side by side
        for k, (fx, fy) in enumerate(GRID):
            tx, ty = (m * 0.5 + fx * 0.5) * SCREEN[0], fy * SCREEN[1]
            yaw, pitch = (tx / SCREEN[0] - 0.5) * 60, (ty / SCREEN[1] - 0.5) * 20
            for _ in range(per_point):
                f = [yaw + rng.normal(0, noise * 2), pitch + rng.normal(0, noise),
                     rng.normal(0, 1), rng.normal(0, 1), 50 + rng.normal(0, 1),
                     (tx / SCREEN[0] - 0.5) * 0.3 + rng.normal(0, noise * 0.02),
                     (ty / SCREEN[1] - 0.5) * 0.2 + rng.normal(0, noise * 0.02),
                     0.5 + rng.normal(0, noise * 0.02)]
                feats.append(f)
                targets.append((tx, ty))
                groups.append(m * 9 + k)
    return np.array(feats), np.array(targets), np.array(groups)


class GazeTest(unittest.TestCase):
    def test_recovers_clean_mapping(self):
        x, y, g = synthetic(noise=0.0)
        model = fit(x, y, g)
        for f, t in list(zip(x, y))[::17]:
            px, py = model.predict(f)
            self.assertLess(abs(px - t[0]) + abs(py - t[1]), 40)

    def test_noise_gives_bounded_error_and_honest_estimate(self):
        x, y, g = synthetic(noise=0.5, seed=1)
        model = fit(x, y, g)
        xt, yt, _ = synthetic(noise=0.5, seed=2)          # fresh frames, same mapping
        err = np.linalg.norm(np.array([model.predict(f) for f in xt]) - yt, axis=1)
        self.assertLess(np.sqrt((err ** 2).mean()), 400)
        self.assertGreater(model.error_px, 0)

    def test_outlier_frames_are_trimmed(self):
        x, y, g = synthetic(noise=0.1, seed=3)
        clean = fit(x, y, g)
        x2 = x.copy()
        x2[::10] += 50                                    # blink / tracking glitch frames
        glitchy = fit(x2, y, g)
        probe = x[100]
        self.assertLess(np.hypot(*(np.array(glitchy.predict(probe)) - clean.predict(probe))), 250)

    def test_prefers_the_linear_model_when_it_is_enough(self):
        rng = np.random.default_rng(7)
        groups = np.repeat(np.arange(18), 25)
        x = rng.normal(0, 1, (len(groups), 7))
        centers = rng.normal(0, 1, (18, 7))
        x = centers[groups] + rng.normal(0, 0.05, (len(groups), 7))
        y = np.stack([960 + 300 * x[:, 0] + 100 * x[:, 5], 540 + 200 * x[:, 1]], axis=1)
        y += rng.normal(0, 5, y.shape)
        self.assertEqual(fit(x, y, groups).degree, 1)

    def test_reports_the_error_of_every_calibration_point(self):
        x, y, g = synthetic(noise=0.3, seed=4)
        model = fit(x, y, g)
        self.assertEqual(sorted(model.point_errors), list(range(18)))
        self.assertTrue(all(e > 0 for e in model.point_errors.values()))

    def test_roundtrip_dict(self):
        x, y, g = synthetic(noise=0.2)
        model = fit(x, y, g)
        again = GazeModel.from_dict(model.to_dict())
        self.assertEqual(model.predict(x[5]), again.predict(x[5]))

    def test_smoother(self):
        s = Smoother(0.5)
        self.assertEqual(s.update((0.0, 0.0)), (0.0, 0.0))
        self.assertEqual(s.update((10.0, 20.0)), (5.0, 10.0))
        self.assertIsNone(s.update(None))
        self.assertEqual(s.update((8.0, 8.0)), (8.0, 8.0))


if __name__ == "__main__":
    unittest.main()
