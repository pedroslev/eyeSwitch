"""Run: python -m unittest discover tests"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gaze import FixationFilter, GazeModel, OneEuro, SeatCompensation, Smoother, fit  # noqa: E402

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

    def test_drops_features_that_are_only_noise(self):
        x, y, g = synthetic(noise=0.3, seed=5)               # columns 2, 3, 4 are pure noise
        model = fit(x, y, g)
        self.assertTrue({0, 1, 5}.issubset(model.active))
        self.assertFalse({2, 3}.intersection(model.active))

    def test_a_spiky_frame_cannot_throw_the_estimate_off_screen(self):
        rng = np.random.default_rng(11)
        groups = np.repeat(np.arange(18), 25)
        centers = rng.normal(0, 1, (18, 4))
        x = centers[groups] + rng.normal(0, 0.05, (len(groups), 4))
        y = np.stack([960 + 400 * x[:, 0] + 150 * x[:, 1] ** 2, 540 + 250 * x[:, 2]], axis=1)
        model = fit(x, y, groups)
        spike = x[0].copy()
        spike[:] = 12                                         # a 12-sigma glitch on every feature
        px, py = model.predict(spike)
        for value, col in ((px, y[:, 0]), (py, y[:, 1])):     # unclamped this is ~15x the data range
            beyond = max(0.0, value - col.max(), col.min() - value)
            self.assertLess(beyond, 0.5 * np.ptp(col))

    def test_chooses_among_feature_sets_and_never_uses_noise_columns(self):
        x, y, g = synthetic(noise=0.3, seed=6)                 # columns 2, 3, 4 are pure noise
        model = fit(x, y, g, column_sets=[[0, 1, 5, 6], [0, 1, 2, 3, 4, 5, 6]])
        self.assertTrue(set(model.active) <= {0, 1, 5, 6})

    def test_reports_the_measured_noise_and_how_far_out_of_range_a_pose_is(self):
        x, y, g = synthetic(noise=0.3, seed=2)
        model = fit(x, y, g)
        self.assertGreater(model.noise_px, 0)
        self.assertEqual(model.range_excess(x[0]), 0.0)
        self.assertGreater(model.range_excess(x[0] + 100), 0.5)

    def test_roundtrip_dict(self):
        x, y, g = synthetic(noise=0.2)
        model = fit(x, y, g)
        again = GazeModel.from_dict(model.to_dict())
        self.assertEqual(model.predict(x[5]), again.predict(x[5]))

    def test_median_window_ignores_a_one_frame_spike(self):
        s = Smoother(factor=1.0, window=5)
        for _ in range(4):
            s.update((100.0, 100.0))
        self.assertEqual(s.update((5000.0, -4000.0)), (100.0, 100.0))        # the spike is outvoted
        for _ in range(4):
            out = s.update((200.0, 200.0))
        self.assertEqual(out, (200.0, 200.0))                                # a real move gets through

    def test_smoother(self):
        s = Smoother(0.5)
        self.assertEqual(s.update((0.0, 0.0)), (0.0, 0.0))
        self.assertEqual(s.update((10.0, 20.0)), (5.0, 10.0))
        self.assertIsNone(s.update(None))
        self.assertEqual(s.update((8.0, 8.0)), (8.0, 8.0))


class FixationFilterTest(unittest.TestCase):
    def test_holds_still_while_the_estimate_only_jitters(self):
        f = FixationFilter(radius=100, confirm=3)
        rng = np.random.default_rng(1)
        out = np.array([f.update((500 + rng.normal(0, 30), 400 + rng.normal(0, 30))) for _ in range(60)])
        self.assertLess(out[5:].std(axis=0).max(), 12)                       # raw scatter is 30

    def test_moves_once_enough_points_agree_on_a_new_place(self):
        f = FixationFilter(radius=100, confirm=3)
        for _ in range(5):
            f.update((0, 0))
        out = [f.update((1000, 0)) for _ in range(3)]
        self.assertEqual(out[:2], [(0.0, 0.0), (0.0, 0.0)])
        self.assertAlmostEqual(out[2][0], 1000)

    def test_ignores_a_glitch_and_a_wandering_burst(self):
        f = FixationFilter(radius=100, confirm=3)
        for _ in range(5):
            f.update((0, 0))
        self.assertEqual(f.update((2000, 900)), (0.0, 0.0))                  # one wild point
        self.assertEqual(f.update((0, 0)), (0.0, 0.0))
        for p in ((900, 0), (0, 900), (-900, 0)):                            # points that disagree
            self.assertEqual(f.update(p), (0.0, 0.0))

    def test_resets_after_a_gap(self):
        f = FixationFilter()
        f.update((10, 10))
        self.assertIsNone(f.update(None))
        self.assertEqual(f.update((700, 300)), (700.0, 300.0))


class OneEuroTest(unittest.TestCase):
    def run_filter(self, points, fps=15.0, **kw):
        f, out = OneEuro(**kw), []
        for i, p in enumerate(points):
            out.append(f.update(p, i / fps))
        return np.array(out)

    def test_a_still_point_stays_put(self):
        out = self.run_filter([(500.0, 300.0)] * 30)
        self.assertTrue(np.allclose(out, [500.0, 300.0]))

    def test_smooths_jitter_when_still(self):
        rng = np.random.default_rng(2)
        raw = np.array([(500 + rng.normal(0, 60), 300 + rng.normal(0, 60)) for _ in range(120)])
        out = self.run_filter(raw)
        self.assertLess(out[20:].std(axis=0).max(), 0.6 * raw[20:].std(axis=0).max())

    def test_follows_a_fast_move_with_little_lag_unlike_a_fixed_smoother(self):
        step = [(0.0, 0.0)] * 10 + [(1500.0, 0.0)] * 10
        euro = self.run_filter(step)[:, 0]
        ema = np.array([Smoother(factor=0.2, window=1).update(p)[0] if False else 0 for p in step])
        s, ema = Smoother(factor=0.2, window=1), []
        for p in step:
            ema.append(s.update(p)[0])
        reach = lambda xs: next((i for i, v in enumerate(xs[10:]) if v > 1200), 99)   # frames to cover 80%
        self.assertLess(reach(euro), reach(ema))
        self.assertLessEqual(reach(euro), 4)

    def test_resets_after_a_gap(self):
        f = OneEuro()
        f.update((10, 10), 0.0)
        self.assertIsNone(f.update(None, 0.1))
        self.assertEqual(f.update((900, 400), 5.0), (900.0, 400.0))


def vec(tx=0.0, ty=0.0, tz=-40.0):
    return np.array([0, 0, tx, ty, tz, 0, 0, 0, 0], float)


class SeatCompensationTest(unittest.TestCase):
    def setUp(self):
        self.seat = SeatCompensation(ref=[0, 0, -40], px_per_cm=30, eye_px=[1000, 500])

    def test_nothing_changes_where_you_calibrated(self):
        self.assertEqual(self.seat.apply((1500, 700), vec()), (1500, 700))

    def test_sitting_further_to_your_right_moves_the_point_right(self):
        # the camera sees you on the left of its picture (x decreases) when you move to your right
        x, y = self.seat.apply((1500, 700), vec(tx=-2))
        self.assertAlmostEqual(x, 1560)
        self.assertAlmostEqual(y, 700)

    def test_sitting_higher_moves_the_point_up_the_screen(self):
        x, y = self.seat.apply((1500, 700), vec(ty=1))
        self.assertAlmostEqual(y, 670)                       # smaller y is higher on the desktop

    def test_sitting_further_back_scales_the_point_away_from_the_eye(self):
        x, y = self.seat.apply((1500, 700), vec(tz=-48))     # 20% farther
        self.assertAlmostEqual(x, 1000 + 1.2 * 500)
        self.assertAlmostEqual(y, 500 + 1.2 * 200)

    def test_the_strength_scales_the_shift_and_survives_the_roundtrip(self):
        half = self.seat.with_gain(0.5)
        self.assertAlmostEqual(half.apply((1500, 700), vec(tx=-2))[0], 1530)
        self.assertEqual(SeatCompensation.from_dict(half.to_dict()).gain, 0.5)
        self.assertEqual(SeatCompensation.from_dict({"ref": [0, 0, -40], "px_per_cm": 30, "eye_px": [0, 0]}).gain, 1.0)

    def test_an_absurd_distance_is_limited(self):
        x, _ = self.seat.apply((1500, 700), vec(tz=-400))
        self.assertAlmostEqual(x, 1000 + 1.6 * 500)

    def test_the_offset_is_in_cm_as_the_user_sees_it(self):
        right, up, farther = self.seat.offset_cm(vec(tx=-2, ty=1, tz=-45))
        self.assertEqual((right, up, farther), (2, -1, 5))

    def test_it_is_made_from_the_calibration_rows_and_survives_the_model_roundtrip(self):
        model = fit(*synthetic(10)[:3])
        rows = np.tile(vec(tx=-1.5, ty=1.5, tz=-42.5)[:len(model.mean)], (20, 1))
        model.seat = SeatCompensation.make(model, rows, 32.0)
        self.assertEqual(model.seat.ref, [-1.5, 1.5, -42.5])
        again = GazeModel.from_dict(model.to_dict())
        self.assertEqual(again.seat.ref, model.seat.ref)
        self.assertEqual(again.seat.eye_px, model.seat.eye_px)

    def test_a_calibration_saved_before_this_has_no_seat(self):
        model = fit(*synthetic(10)[:3])
        d = model.to_dict()
        del d["seat"]
        self.assertIsNone(GazeModel.from_dict(d).seat)


if __name__ == "__main__":
    unittest.main()
