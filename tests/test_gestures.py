"""Run: python -m unittest discover tests"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gestures import Detector, GestureModel, train  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
