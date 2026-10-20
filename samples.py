"""Cleaning what the camera sees: blinks and fast movements must not become calibration data.

Real data: a blink pushed the gaze estimate 230-390 px away, and 11% of the frames of a 20 s
recording were affected. Eyelid closure comes from the face tracker's blink blendshapes, judged
against the person's own open-eye level (a resting face can sit at 0.3-0.6 depending on the camera
angle), the way eye-tracking pipelines null blink samples with padding, since the artifacts start
before and end after the closure itself.
"""
from collections import deque

import numpy as np

BLINK_DELTA = 0.10          # score above the person's own baseline that counts as lids closing
BASELINE_S = 3.0            # how much history the person's own eyelid level is taken from
MAX_BLINK_S = 0.4           # longer than this is a new resting level (looking down), not a blink
PAD_BEFORE, PAD_AFTER = 0.10, 0.40      # seconds around a blink that are discarded too (the features
                                        # stay off for ~0.4 s after the lids reopen, measured on real data)
MAX_HEAD_SPEED = 25.0       # deg/s: faster is a head still travelling between dots, not a fixation
MIN_BASELINE_SAMPLES = 8
FRAME_S = 1 / 15            # time per frame when the caller gives no timestamps


class BlinkDetector:
    """Is this frame (part of) a blink? Fed one blink score (mean of both eyes) per frame, with its time,
    so it behaves the same at 15 and 30 frames per second."""

    def __init__(self):
        self.history, self.run_start, self._clock = deque(), None, 0.0

    def update(self, score, t=None):
        if score is None:
            return False
        if t is None:                                  # no timestamps: pretend 15 frames per second
            self._clock += FRAME_S
            t = self._clock
        while self.history and t - self.history[0][0] > BASELINE_S:
            self.history.popleft()
        if len(self.history) < MIN_BASELINE_SAMPLES:
            self.history.append((t, score))
            return False
        if score - float(np.median([s for _, s in self.history])) > BLINK_DELTA:
            if self.run_start is None:
                self.run_start = t
            if t - self.run_start <= MAX_BLINK_S:
                return True                            # blink frames never move the baseline
        else:
            self.run_start = None
        self.history.append((t, score))
        return False


def clean_indices(records):
    """records: [(t, face, blinking, ...)] -> (indices worth keeping, stats).

    Drops blink frames with padding on both sides, and frames where the head is still moving.
    """
    if not records:
        return [], {"total": 0, "blink": 0, "moving": 0}
    t = np.array([r[0] for r in records], float)
    blink = np.zeros(len(t), bool)
    for ti, flag in zip(t, (r[2] for r in records)):
        if flag:
            blink |= (t >= ti - PAD_BEFORE) & (t <= ti + PAD_AFTER)
    moving = np.zeros(len(t), bool)
    if len(t) >= 3:
        yaw, pitch = np.array([r[1].yaw for r in records]), np.array([r[1].pitch for r in records])
        speed = np.hypot(np.gradient(yaw, t), np.gradient(pitch, t))
        moving = speed > MAX_HEAD_SPEED
    keep = ~(blink | moving)
    return [i for i, k in enumerate(keep) if k], {
        "total": len(t), "blink": int(blink.sum()), "moving": int((moving & ~blink).sum())}


def clean_capture(records):
    """Like clean_indices, but returns the faces themselves."""
    kept, stats = clean_indices(records)
    return [records[i][1] for i in kept], stats
