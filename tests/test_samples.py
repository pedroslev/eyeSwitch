"""Run: python -m unittest discover tests"""
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from samples import (PAD_AFTER, PAD_BEFORE, BlinkDetector, clean_capture,  # noqa: E402
                     clean_indices)


def feed(scores):
    d = BlinkDetector()
    return [d.update(s) for s in scores]


class BlinkDetectorTest(unittest.TestCase):
    def test_flags_closing_lids_against_the_persons_own_level(self):
        flags = feed([0.6] * 20 + [0.8, 0.85, 0.8] + [0.6] * 5)             # a squinty resting face at 0.6
        self.assertEqual(flags, [False] * 20 + [True] * 3 + [False] * 5)

    def test_a_high_resting_level_is_not_a_blink(self):
        flags = feed([0.2] * 20 + [0.5] * 40)                               # looking down: lids settle lower
        self.assertLessEqual(sum(flags), 7)                                 # about MAX_BLINK_S at 15 fps
        self.assertFalse(flags[-1])

    def test_blink_frames_do_not_drag_the_baseline_up(self):
        flags = feed([0.3] * 20 + [0.7] * 3 + [0.3] * 20 + [0.7] * 3)
        self.assertEqual(sum(flags), 6)                                     # the second blink is still caught

    def test_a_blink_is_seen_the_same_at_15_and_at_30_fps(self):
        for fps in (15, 30):
            d, flags = BlinkDetector(), []
            n = int(3 * fps)
            for i in range(n + int(0.3 * fps) + 5):                         # 3 s open, a 0.3 s closure, open
                closed = n <= i < n + int(0.3 * fps)
                flags.append(d.update(0.8 if closed else 0.6, i / fps))
            self.assertEqual(sum(flags), int(0.3 * fps), f"{fps} fps")

    def test_a_long_closure_becomes_the_new_level_at_any_frame_rate(self):
        for fps in (15, 30):
            d, flags = BlinkDetector(), []
            for i in range(int(3 * fps) + int(3 * fps)):                    # lids settle lower for good
                flags.append(d.update(0.2 if i < 3 * fps else 0.5, i / fps))
            self.assertFalse(flags[-1])
            self.assertLessEqual(sum(flags), int(0.4 * fps) + 1)            # at most MAX_BLINK_S, inclusive

    def test_missing_scores_are_ignored(self):
        self.assertEqual(feed([None, None, 0.5]), [False, False, False])


def records(n, dt=0.07, blink_at=(), yaw_step=0.0):
    return [(i * dt, SimpleNamespace(yaw=yaw_step * i, pitch=0.0), i in blink_at) for i in range(n)]


class CleanCaptureTest(unittest.TestCase):
    def test_keeps_everything_when_nothing_is_wrong(self):
        faces, stats = clean_capture(records(14))
        self.assertEqual((len(faces), stats["blink"], stats["moving"]), (14, 0, 0))

    def test_drops_a_blink_with_padding_before_and_after(self):
        faces, stats = clean_capture(records(20, blink_at={10}))
        before, after = int(PAD_BEFORE / 0.07 + 1e-9), int(PAD_AFTER / 0.07 + 1e-9)
        self.assertEqual(stats["blink"], before + 1 + after)
        self.assertEqual(len(faces), 20 - (before + 1 + after))

    def test_drops_frames_while_the_head_is_still_travelling(self):
        faces, stats = clean_capture(records(14, yaw_step=2.0))             # 2 deg per 70 ms = 28 deg/s
        self.assertEqual(len(faces), 0)
        self.assertEqual(stats["moving"], 14)
        slow, _ = clean_capture(records(14, yaw_step=0.5))                  # 7 deg/s: a slow, steady turn
        self.assertEqual(len(slow), 14)

    def test_indices_line_up_with_the_faces_that_survive(self):
        recs = records(20, blink_at={10})
        kept, _ = clean_indices(recs)
        faces, _ = clean_capture(recs)
        self.assertEqual([recs[i][1] for i in kept], faces)
        self.assertNotIn(10, kept)

    def test_empty_capture(self):
        self.assertEqual(clean_capture([]), ([], {"total": 0, "blink": 0, "moving": 0}))


if __name__ == "__main__":
    unittest.main()
