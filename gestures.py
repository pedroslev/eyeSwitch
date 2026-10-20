"""Facial gestures from MediaPipe blendshapes: train one from examples, detect it live.

Every face rests differently (camera angle, asymmetry), so there are no fixed thresholds. Training
compares a relaxed face with the gesture, keeps the few blendshapes that separate them best and
combines them with a linear discriminant: score 0 is "relaxed", 1 is "gesture".
"""
import numpy as np

TOP_FEATURES = 3
RELEVANT = 0.3                      # a blendshape must separate at least this much of the best one
ON, OFF = 0.6, 0.3                  # score thresholds to trigger and to re-arm
MIN_QUALITY = 1.5                   # below this the gesture is too close to a relaxed face
IGNORED = {"_neutral"}


def _matrix(samples, names):
    return np.array([[s[n] for n in names] for s in samples], float)


class GestureModel:
    def __init__(self, names, weights, bias, quality=None):
        self.names, self.weights, self.bias, self.quality = list(names), np.asarray(weights, float), bias, quality

    def score(self, shapes):
        """0 for a relaxed face, 1 for the trained gesture. `shapes` is {blendshape: value}."""
        return float(sum(w * shapes.get(n, 0.0) for n, w in zip(self.names, self.weights)) + self.bias)

    @property
    def reliable(self):
        return self.quality is not None and self.quality >= MIN_QUALITY

    def to_dict(self):
        return {"names": self.names, "weights": self.weights.tolist(), "bias": self.bias,
                "quality": self.quality}

    @classmethod
    def from_dict(cls, d):
        return cls(d["names"], d["weights"], d["bias"], d.get("quality"))


def train(relaxed, gesture):
    """Model from two lists of {blendshape: value} dicts (frames of a relaxed face / of the gesture)."""
    names = [n for n in relaxed[0] if n not in IGNORED]
    xr, xg = _matrix(relaxed, names), _matrix(gesture, names)
    mr, mg = xr.mean(axis=0), xg.mean(axis=0)
    spread = np.sqrt((xr.var(axis=0) + xg.var(axis=0)) / 2 + 1e-4)
    separation = np.abs(mg - mr) / spread
    order = np.argsort(-separation)[:TOP_FEATURES]
    best = [i for i in order if separation[i] >= RELEVANT * separation[order[0]]]   # drop noise

    pooled = (np.cov(xr[:, best].T, ddof=0) + np.cov(xg[:, best].T, ddof=0)) / 2
    pooled = np.atleast_2d(pooled) + 1e-4 * np.eye(len(best))
    w = np.linalg.solve(pooled, mg[best] - mr[best])
    pr, pg = xr[:, best] @ w, xg[:, best] @ w
    gap = pg.mean() - pr.mean()
    w, offset = w / gap, pr.mean() / gap                 # relaxed -> 0, gesture -> 1
    pr, pg = pr / gap - offset, pg / gap - offset
    quality = 1.0 / np.sqrt((pr.var() + pg.var()) / 2 + 1e-9)
    return GestureModel([names[i] for i in best], w, -float(offset), float(quality))


class Detector:
    """Fires once per gesture: held for `hold_s`, re-armed after release and a short cooldown."""

    def __init__(self, hold_s=0.25, cooldown_s=0.8):
        self.hold_s, self.cooldown_s = hold_s, cooldown_s
        self.armed, self.above_since, self.last_fire = True, None, float("-inf")

    def update(self, score, now):
        """score None = no face. True on the frame that triggers."""
        if score is None:
            self.above_since = None
            return False
        if not self.armed:
            if score < OFF and now - self.last_fire >= self.cooldown_s:
                self.armed = True
            return False
        if score < ON:
            self.above_since = None
            return False
        if self.above_since is None:
            self.above_since = now
        if now - self.above_since >= self.hold_s:
            self.armed, self.last_fire, self.above_since = False, now, None
            return True
        return False


class HandToggle:
    """A deliberate hand signal: an open palm, then closing it into a fist.

    Resting a hand on the keyboard is rarely an "open palm" facing the camera, and the closing makes it a
    sequence rather than a pose, so it does not fire by accident. Fed the recognizer's verdict per sample
    ((name, score) or None) with its time. Fires once, then waits a moment (COOLDOWN_S).
    """
    OPEN, FIST = "Open_Palm", "Closed_Fist"
    MIN_SCORE = 0.6
    OPEN_S, FIST_S = 0.25, 0.10         # how long each pose must be held
    WINDOW_S = 2.0                      # the fist must come this soon after the palm was last seen
    COOLDOWN_S = 1.5

    def __init__(self):
        self.open_since = self.palm_seen = self.fist_since = None
        self.last_fire = float("-inf")

    def update(self, hand, now):
        name = hand[0] if hand and hand[1] >= self.MIN_SCORE else None
        if now - self.last_fire < self.COOLDOWN_S:
            self.open_since = self.palm_seen = self.fist_since = None
            return False
        if name == self.OPEN:
            self.open_since = self.open_since or now
            if now - self.open_since >= self.OPEN_S:
                self.palm_seen = now                      # a held palm: the fist may follow
            self.fist_since = None
        else:
            self.open_since = None
            if name == self.FIST and self.palm_seen is not None and now - self.palm_seen <= self.WINDOW_S:
                self.fist_since = self.fist_since or now
                if now - self.fist_since >= self.FIST_S:
                    self.last_fire, self.open_since, self.palm_seen, self.fist_since = now, None, None, None
                    return True
            else:
                self.fist_since = None
                if self.palm_seen is not None and now - self.palm_seen > self.WINDOW_S:
                    self.palm_seen = None
        return False
