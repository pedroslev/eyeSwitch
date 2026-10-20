"""Gaze model: face features -> point on the desktop, fitted from calibration samples.

Ridge regression on standardized features, linear or quadratic. The degree and the
regularization are picked by leave-one-point-out cross-validation, which also gives an
honest error estimate (it is pessimistic for the edge points, since they get extrapolated).

Lessons from real calibration data, each one now enforced here:
- Features that carry no information are dropped, and the caller chooses among candidate feature
  sets by cross-validation. Head *position* in the camera frame is not a gaze cue: its fitted
  coefficients exploded (1 cm of seat shift moved the estimate ~1000 px), so callers leave it out.
- Live inputs are clamped to the range seen during calibration, and `range_excess` tells when the
  pose is far outside it (camera or chair moved), so nobody acts on a meaningless estimate.
- The estimate is steadied with a fixation filter whose radius comes from the measured noise.
"""
import numpy as np

DEGREES = (1,)       # quadratic added nothing on two real calibrations (166 px either way), and choosing between
                     # the two by cross-validation was worse than either (177 px): with ~30 dots that choice is noise
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0)         # scaled by the number of samples
SIMPLICITY = 1.05                       # accept a simpler model this much worse than the best
MIN_SNR = 2.0                            # a feature must vary this much more between points than within one
CLAMP_MARGIN = 0.25                      # live inputs may exceed the calibrated range by this fraction
TRIM = 3.0                               # drop samples farther than TRIM x the median distance


def _expand(z, degree):
    n = z.shape[1]
    cols = [np.ones(len(z))] + [z[:, i] for i in range(n)]
    if degree >= 2:
        cols += [z[:, i] * z[:, j] for i in range(n) for j in range(i, n)]
    return np.stack(cols, axis=1)


def _ridge(a, y, alpha):
    reg = alpha * len(a) * np.eye(a.shape[1])
    reg[0, 0] = 0                        # do not penalize the bias
    return np.linalg.solve(a.T @ a + reg, a.T @ y)


class GazeModel:
    def __init__(self, degree, mean, std, weights, error_px=None, active=None, lo=None, hi=None,
                 noise_px=None):
        self.degree, self.mean, self.std = degree, np.asarray(mean, float), np.asarray(std, float)
        self.weights, self.error_px = np.asarray(weights), error_px
        self.active = list(range(len(self.mean))) if active is None else [int(i) for i in active]
        self.lo = None if lo is None else np.asarray(lo, float)      # clamp range of the active features
        self.hi = None if hi is None else np.asarray(hi, float)
        self.noise_px = noise_px            # scatter of the estimate while looking at one point
        self.point_errors = {}

    def _z(self, vec):
        return ((np.asarray(vec, float) - self.mean) / self.std)[self.active]

    def predict(self, vec):
        z = self._z(vec)
        if self.lo is not None:
            z = np.clip(z, self.lo, self.hi)
        x, y = (_expand(z[None], self.degree) @ self.weights)[0]
        return float(x), float(y)

    def range_excess(self, vec):
        """How far the pose is outside the calibrated range, as a fraction of that range (0 = inside)."""
        if self.lo is None:
            return 0.0
        z, span = self._z(vec), np.maximum(self.hi - self.lo, 1e-9)
        return float(np.max(np.maximum(np.maximum(z - self.hi, self.lo - z), 0) / span))

    def to_dict(self):
        return {"degree": self.degree, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "weights": self.weights.tolist(), "error_px": self.error_px, "active": self.active,
                "lo": None if self.lo is None else self.lo.tolist(),
                "hi": None if self.hi is None else self.hi.tolist(), "noise_px": self.noise_px}

    @classmethod
    def from_dict(cls, d):
        return cls(d["degree"], d["mean"], d["std"], d["weights"], d.get("error_px"),
                   d.get("active"), d.get("lo"), d.get("hi"), d.get("noise_px"))


def _informative(z, groups, cols):
    """The columns among `cols` that change between calibration points more than within one."""
    ids = np.unique(groups)
    between = np.array([z[groups == g][:, cols].mean(axis=0) for g in ids]).std(axis=0)
    within = np.array([z[groups == g][:, cols].std(axis=0) for g in ids]).mean(axis=0)
    snr = between / np.maximum(within, 1e-9)
    keep = [cols[i] for i in np.where(snr >= MIN_SNR)[0]]
    return keep if len(keep) >= 2 else sorted(cols[i] for i in np.argsort(-snr)[:2])


def _trim_outliers(z, groups):
    """Mask of samples to keep: per calibration point, drop frames far from the point's median."""
    keep = np.ones(len(z), bool)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        dist = np.linalg.norm(z[idx] - np.median(z[idx], axis=0), axis=1)
        keep[idx] = dist <= max(TRIM * np.median(dist), 1e-6)
    return keep


def _fit_columns(z_all, y, groups, cols, mean, std):
    """Best model that uses only (informative) `cols`."""
    active = _informative(z_all, groups, cols)
    z = z_all[:, active]
    keep = _trim_outliers(z, groups)
    z, y, groups = z[keep], y[keep], groups[keep]

    candidates = []                                 # (cv rms, degree, alpha, per-point mean errors)
    for degree in DEGREES:
        a = _expand(z, degree)
        for alpha in ALPHAS:
            sq, per_point = [], {}
            for g in np.unique(groups):
                test = groups == g
                w = _ridge(a[~test], y[~test], alpha)
                err = np.sum((a[test] @ w - y[test]) ** 2, axis=1)
                sq.append(err)
                per_point[int(g)] = float(np.sqrt(err).mean())
            candidates.append((float(np.sqrt(np.concatenate(sq).mean())), degree, alpha, per_point))
    # A complex model that is only slightly better in cross-validation overshoots at the edges
    # of the screen: take the simplest (low degree, strong regularization) within 5% of the best.
    limit = min(c[0] for c in candidates) * SIMPLICITY
    rms, degree, alpha, per_point = min((c for c in candidates if c[0] <= limit),
                                        key=lambda c: (c[1], -c[2]))
    weights = _ridge(_expand(z, degree), y, alpha)
    lo, hi = np.percentile(z, 1, axis=0), np.percentile(z, 99, axis=0)
    span = hi - lo
    model = GazeModel(degree, mean, std, weights, rms, active,
                      lo - CLAMP_MARGIN * span, hi + CLAMP_MARGIN * span)
    model.point_errors = per_point                  # leave-one-point-out error per calibration point, px

    guess = _expand(z, degree) @ weights            # how much the estimate wobbles while looking at one point
    spread = [guess[groups == g] - np.median(guess[groups == g], axis=0) for g in np.unique(groups)]
    model.noise_px = float(np.sqrt((np.concatenate(spread) ** 2).sum(axis=1).mean()))
    return model


def fit(features, targets, groups, column_sets=None):
    """features (N, F), targets (N, 2) desktop pixels, groups (N,) calibration point ids.

    `column_sets` are candidate feature subsets (lists of column indices); the one that
    cross-validates best wins, with a preference for fewer columns when it is a near tie.
    """
    x, y, groups = np.asarray(features, float), np.asarray(targets, float), np.asarray(groups)
    column_sets = column_sets or [list(range(x.shape[1]))]
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-9] = 1.0
    z_all = (x - mean) / std
    models = [_fit_columns(z_all, y, groups, list(cols), mean, std) for cols in column_sets]
    limit = min(m.error_px for m in models) * SIMPLICITY
    return min((m for m in models if m.error_px <= limit), key=lambda m: (len(m.active), m.error_px))


class Smoother:
    """Median of the last `window` points (kills one-frame spikes), then exponential smoothing.
    Resets after a gap in the data."""

    def __init__(self, factor=0.35, window=1):
        self.factor, self.window, self.recent, self.point = factor, window, [], None

    def update(self, point):
        if point is None:
            self.recent, self.point = [], None
            return None
        self.recent = (self.recent + [point])[-self.window:]
        med = tuple(float(np.median([p[i] for p in self.recent])) for i in (0, 1))
        if self.point is None:
            self.point = med
        else:
            self.point = tuple(p + self.factor * (n - p) for p, n in zip(self.point, med))
        return self.point


class FixationFilter:
    """Holds the estimate still while you fixate; moves only when you look somewhere else.

    The output is the median of the recent points inside `radius` of the current fixation. A point
    outside the radius does not move it until `confirm` consecutive points agree on a new place
    (so a blink or a glitch cannot). On real data this beat a median+EMA filter on both counts:
    less jitter (44 vs 52 px) and less lag (276 vs 414 ms). Resets after a gap in the data.
    """

    def __init__(self, radius=200.0, confirm=4, keep=8):
        self.radius, self.confirm, self.keep = radius, confirm, keep
        self.reset()

    def reset(self):
        self.center, self.buffer, self.pending = None, [], []

    def update(self, point):
        if point is None:
            self.reset()
            return None
        p = np.asarray(point, float)
        if self.center is None:
            self.center, self.buffer = p, [p]
        elif np.hypot(*(p - self.center)) <= self.radius:
            self.buffer = (self.buffer + [p])[-self.keep:]
            self.pending = []
            self.center = np.median(self.buffer, axis=0)
        else:
            self.pending.append(p)
            if len(self.pending) >= self.confirm:
                new = np.median(self.pending, axis=0)
                if all(np.hypot(*(q - new)) <= self.radius for q in self.pending):   # a consistent new place
                    self.center, self.buffer, self.pending = new, list(self.pending), []
                else:
                    self.pending = self.pending[1:]
        return float(self.center[0]), float(self.center[1])


class OneEuro:
    """Speed-adaptive low-pass filter for a 2D point (Casiez et al., "1 euro filter", CHI 2012).

    Slow movement is smoothed hard (cutoff near `min_cutoff` Hz); fast movement raises the cutoff by
    `beta` per px/s, so it follows you with little lag. Used for what is *shown*; decisions use the
    steadier FixationFilter. Defaults tuned on real calibration data: with beta 0.004 the jitter itself
    looks like speed and almost nothing gets smoothed (87 px of jitter vs 73 px here, at 68 vs 137 ms
    of lag). Resets after a gap in the data.
    """

    def __init__(self, min_cutoff=0.5, beta=0.001, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x = self.raw = self.dx = self.t = None

    @staticmethod
    def _alpha(dt, cutoff):
        return 1.0 / (1.0 + 1.0 / (2 * np.pi * cutoff) / dt)

    def update(self, point, t):
        if point is None:
            self.reset()
            return None
        p = np.asarray(point, float)
        if self.x is None:
            self.x, self.raw, self.dx, self.t = p, p, np.zeros(2), t
            return float(p[0]), float(p[1])
        dt = max(t - self.t, 1e-3)
        self.dx = self.dx + self._alpha(dt, self.d_cutoff) * ((p - self.raw) / dt - self.dx)
        cutoff = self.min_cutoff + self.beta * float(np.hypot(*self.dx))
        self.x = self.x + self._alpha(dt, cutoff) * (p - self.x)
        self.raw, self.t = p, t
        return float(self.x[0]), float(self.x[1])
