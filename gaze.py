"""Gaze model: face features -> point on the desktop, fitted from calibration samples.

Ridge regression on standardized features, linear or quadratic. The degree and the
regularization are picked by leave-one-point-out cross-validation, which also gives an
honest error estimate (it is pessimistic for the edge points, since they get extrapolated).

Two protections against wild estimates, both found the hard way with real calibration data:
features that carry no information (the vertical iris offset is mostly noise at webcam
resolution) are dropped, and live inputs are clamped to the range seen during calibration, since a
single glitchy frame fed to a quadratic can otherwise land a screen away.
"""
import numpy as np

DEGREES = (1, 2)
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
    def __init__(self, degree, mean, std, weights, error_px=None, active=None, lo=None, hi=None):
        self.degree, self.mean, self.std = degree, np.asarray(mean, float), np.asarray(std, float)
        self.weights, self.error_px = np.asarray(weights), error_px
        self.active = list(range(len(self.mean))) if active is None else [int(i) for i in active]
        self.lo = None if lo is None else np.asarray(lo, float)      # clamp range of the active features
        self.hi = None if hi is None else np.asarray(hi, float)
        self.point_errors = {}

    def predict(self, vec):
        z = ((np.asarray(vec, float) - self.mean) / self.std)[self.active]
        if self.lo is not None:
            z = np.clip(z, self.lo, self.hi)
        x, y = (_expand(z[None], self.degree) @ self.weights)[0]
        return float(x), float(y)

    def to_dict(self):
        return {"degree": self.degree, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "weights": self.weights.tolist(), "error_px": self.error_px, "active": self.active,
                "lo": None if self.lo is None else self.lo.tolist(),
                "hi": None if self.hi is None else self.hi.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["degree"], d["mean"], d["std"], d["weights"], d.get("error_px"),
                   d.get("active"), d.get("lo"), d.get("hi"))


def _informative(z, groups):
    """Indices of the features that change between calibration points more than within one."""
    ids = np.unique(groups)
    between = np.array([z[groups == g].mean(axis=0) for g in ids]).std(axis=0)
    within = np.array([z[groups == g].std(axis=0) for g in ids]).mean(axis=0)
    snr = between / np.maximum(within, 1e-9)
    active = [int(i) for i in np.where(snr >= MIN_SNR)[0]]
    return active if len(active) >= 2 else sorted(int(i) for i in np.argsort(-snr)[:2])


def _trim_outliers(z, groups):
    """Mask of samples to keep: per calibration point, drop frames far from the point's median."""
    keep = np.ones(len(z), bool)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        dist = np.linalg.norm(z[idx] - np.median(z[idx], axis=0), axis=1)
        keep[idx] = dist <= max(TRIM * np.median(dist), 1e-6)
    return keep


def fit(features, targets, groups):
    """features (N, F), targets (N, 2) desktop pixels, groups (N,) calibration point ids."""
    x, y, groups = np.asarray(features, float), np.asarray(targets, float), np.asarray(groups)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-9] = 1.0
    z_all = (x - mean) / std
    active = _informative(z_all, groups)
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
    return model


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
