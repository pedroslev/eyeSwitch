"""Run: python -m unittest discover tests"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gestures import Detector, GestureModel, HandPoses, HandToggle, hand_features, train  # noqa: E402

NAMES = ["_neutral", "browInnerUp", "browOuterUpLeft", "mouthRollLower", "mouthPressLeft", "jawOpen",
         "eyeBlinkLeft", "mouthSmileLeft"]
# a resting face that is not zero: asymmetric brows, lids half closed because of the camera angle
REST = {"_neutral": 0.0, "browInnerUp": 0.001, "browOuterUpLeft": 0.28, "mouthRollLower": 0.09,
        "mouthPressLeft": 0.18, "jawOpen": 0.01, "eyeBlinkLeft": 0.63, "mouthSmileLeft": 0.02}


def frames(n, seed, **changes):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        f = {k: max(0.0, v + rng.normal(0, 0.02)) for k, v in REST.items()}
        for k, v in changes.items():
            f[k] = max(0.0, v + rng.normal(0, 0.05))
        out.append(f)
    return out


class TrainTest(unittest.TestCase):
    def test_picks_the_blendshapes_that_change(self):
        model = train(frames(40, 1), frames(40, 2, browInnerUp=0.7, browOuterUpLeft=0.8))
        self.assertEqual(set(model.names), {"browInnerUp", "browOuterUpLeft"})     # not the noise
        self.assertTrue(model.reliable)

    def test_scores_relaxed_near_zero_and_gesture_near_one(self):
        model = train(frames(40, 1), frames(40, 2, browInnerUp=0.7))
        relaxed = np.mean([model.score(f) for f in frames(30, 3)])
        gesture = np.mean([model.score(f) for f in frames(30, 4, browInnerUp=0.7)])
        self.assertLess(abs(relaxed), 0.15)
        self.assertLess(abs(gesture - 1), 0.15)

    def test_a_subtle_gesture_is_flagged_as_unreliable(self):
        model = train(frames(40, 1), frames(40, 2, mouthRollLower=0.12))     # barely different
        self.assertFalse(model.reliable)

    def test_roundtrip_dict(self):
        model = train(frames(40, 1), frames(40, 2, browInnerUp=0.7))
        again = GestureModel.from_dict(model.to_dict())
        f = frames(1, 5, browInnerUp=0.5)[0]
        self.assertAlmostEqual(model.score(f), again.score(f))

    def test_a_different_gesture_does_not_trigger(self):
        brows = train(frames(40, 1), frames(40, 2, browInnerUp=0.7, browOuterUpLeft=0.8))
        yawn = frames(30, 6, jawOpen=0.8)
        self.assertLess(max(brows.score(f) for f in yawn), 0.6)


class DetectorTest(unittest.TestCase):
    def feed(self, det, scores, step=0.05, start=0.0):
        fired = []
        for i, s in enumerate(scores):
            if det.update(s, start + i * step):
                fired.append(round(start + i * step, 2))
        return fired

    def test_needs_to_be_held(self):
        det = Detector(hold_s=0.25)
        self.assertEqual(self.feed(det, [1.0] * 3 + [0.0] * 10), [])            # 0.1 s: a twitch
        self.assertEqual(len(self.feed(Detector(hold_s=0.25), [1.0] * 10)), 1)  # held

    def test_fires_once_per_gesture_and_rearms_on_release(self):
        det = Detector(hold_s=0.1, cooldown_s=0.3)
        fired = self.feed(det, [1.0] * 20 + [0.0] * 10 + [1.0] * 20)
        self.assertEqual(len(fired), 2)

    def test_hovering_between_the_thresholds_does_not_rearm(self):
        det = Detector(hold_s=0.1, cooldown_s=0.3)
        fired = self.feed(det, [1.0] * 10 + [0.45] * 20 + [1.0] * 10)           # never below OFF
        self.assertEqual(len(fired), 1)

    def test_losing_the_face_resets_the_hold(self):
        det = Detector(hold_s=0.2)
        self.assertEqual(self.feed(det, [1.0, 1.0, 1.0, None, 1.0, 1.0, 1.0]), [])


def run_hand(steps, dt=0.05):
    """steps: [(hand verdict, seconds)] -> times at which HandToggle fired"""
    det, t, fired = HandToggle(), 0.0, []
    for hand, seconds in steps:
        for _ in range(round(seconds / dt)):
            if det.update(hand, t):
                fired.append(round(t, 2))
            t += dt
    return fired


PALM, FIST = ("Open_Palm", 0.95), ("Closed_Fist", 0.95)


class HandToggleTest(unittest.TestCase):
    def test_fires_on_an_open_palm_closing_into_a_fist(self):
        self.assertEqual(len(run_hand([(None, 0.5), (PALM, 0.4), (FIST, 0.3), (None, 0.5)])), 1)

    def test_a_palm_or_a_fist_alone_never_fires(self):
        self.assertEqual(run_hand([(PALM, 3.0), (None, 0.5)]), [])
        self.assertEqual(run_hand([(FIST, 3.0), (None, 0.5)]), [])

    def test_a_flicker_of_a_palm_is_not_enough(self):
        self.assertEqual(run_hand([(PALM, 0.1), (FIST, 0.5)]), [])

    def test_the_fist_must_follow_soon(self):
        self.assertEqual(run_hand([(PALM, 0.4), (None, 2.4), (FIST, 0.5)]), [])
        self.assertEqual(len(run_hand([(PALM, 0.4), (None, 1.0), (FIST, 0.5)])), 1)

    def test_a_doubtful_recognition_is_ignored(self):
        self.assertEqual(run_hand([(("Open_Palm", 0.4), 0.5), (("Closed_Fist", 0.4), 0.5)]), [])

    def test_closing_the_hand_may_pass_through_other_poses(self):
        # half-closed fingers read as other canned gestures (or none) for an instant
        steps = [(PALM, 0.4), (("Pointing_Up", 0.8), 0.15), (None, 0.15), (FIST, 0.3)]
        self.assertEqual(len(run_hand(steps)), 1)

    def test_it_waits_before_firing_again(self):
        fired = run_hand([(PALM, 0.4), (FIST, 0.3), (PALM, 0.4), (FIST, 0.3), (None, 0.3),
                          (None, 1.5), (PALM, 0.4), (FIST, 0.3)])
        self.assertEqual(len(fired), 2)                        # the second try came too soon, the third did not


def hand(curl, scale=1.0, angle=0.0, seed=0):
    """21 landmarks of a hand: wrist at the origin, fingers pointing up; `curl` 0 = open, 1 = fist."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((21, 3))
    pts[9] = (0, 100, 0)
    for k, (knuckle, x) in enumerate(zip((2, 5, 9, 13, 17), (-60, -30, 0, 30, 60))):
        base = np.array([x, 100 if k else 40, 0.0])
        pts[knuckle] = base
        length = 80 if k else 50
        for j in range(1, 4):
            ext = 1 - curl * (0.2 * j + 0.4)                       # curled fingers fold back toward the palm
            pts[knuckle + j] = base + (0, length * j / 3 * ext * 1.0, -curl * 30 * j / 3)
    pts += rng.normal(0, 1.0, pts.shape)
    c, s_ = np.cos(angle), np.sin(angle)
    pts[:, :2] = pts[:, :2] @ np.array([[c, -s_], [s_, c]])
    return pts * scale


class HandPosesTest(unittest.TestCase):
    def poses(self, seed):
        return ([hand_features(hand(0.0, seed=seed + i)) for i in range(30)],
                [hand_features(hand(1.0, seed=seed + 50 + i)) for i in range(30)],
                [hand_features(hand(0.45, seed=seed + 90 + i)) for i in range(30)])

    def test_features_do_not_depend_on_distance_or_direction(self):
        near, far = hand_features(hand(0.0, scale=2.0)), hand_features(hand(0.0, scale=0.5, angle=1.1))
        for k in near:
            self.assertAlmostEqual(near[k], far[k], delta=0.08)

    def test_tells_the_open_palm_the_fist_and_a_resting_hand_apart(self):
        poses = HandPoses.train(*self.poses(1))
        self.assertTrue(poses.reliable)
        opened, fist, rest = self.poses(500)
        self.assertGreater(np.mean([poses.verdict(f) is not None and poses.verdict(f)[0] == "Open_Palm" for f in opened]), 0.9)
        self.assertGreater(np.mean([poses.verdict(f) is not None and poses.verdict(f)[0] == "Closed_Fist" for f in fist]), 0.9)
        self.assertLess(np.mean([poses.verdict(f) is not None for f in rest]), 0.2)       # neither

    def test_roundtrip_and_works_with_hand_toggle(self):
        poses = HandPoses.from_dict(HandPoses.train(*self.poses(1)).to_dict())
        opened, fist, _ = self.poses(700)
        det, t, fired = HandToggle(), 0.0, 0
        for f in [opened[0]] * 8 + [fist[0]] * 4:
            fired += det.update(poses.verdict(f), t)
            t += 0.1
        self.assertEqual(fired, 1)


if __name__ == "__main__":
    unittest.main()
