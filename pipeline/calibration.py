"""Turn a similarity score into a confidence percentage - honestly.

A cosine score of 0.81 means nothing to an audience, and "81%" would be a
lie: cosine similarity is not a probability. This module learns the actual
relationship from labelled pairs of faces taken with the real cameras:

* **genuine pairs** - two photos of the same person
* **impostor pairs** - photos of two different people

and fits a logistic curve P(same person | score). The classes are weighted
equally, so the percentage answers "if a same-person pair and a
different-person pair were equally likely, how likely is this one to be the
same person?" - the base rate of who walks past is unknown, so this is the
honest prior to assume, and it is stated in the UI.

With no calibration file, :meth:`ScoreCalibrator.probability` returns None
and the UI shows the raw score with an "uncalibrated" label. A percentage is
never shown unless it was measured. Too little data to measure is refused
rather than fitted.

Pure numpy; no OpenCV. Used by both the Nano pipeline and the hub.
"""

import json
import math
import time
from pathlib import Path

import numpy as np

import config

MIN_GENUINE = 10
MIN_IMPOSTOR = 10
MIN_IDENTITIES = 2


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def fit_logistic(genuine, impostor, l2=1e-3, iters=100):
    """Class-balanced 1-D logistic regression, by Newton's method.

    Returns (a, b) with P(same | s) = sigmoid(a*s + b).
    """
    g = np.asarray(genuine, dtype=np.float64).reshape(-1)
    i = np.asarray(impostor, dtype=np.float64).reshape(-1)
    s = np.concatenate([g, i])
    y = np.concatenate([np.ones(len(g)), np.zeros(len(i))])
    w = np.concatenate([np.full(len(g), 0.5 / len(g)), np.full(len(i), 0.5 / len(i))])
    X = np.stack([s, np.ones_like(s)], axis=1)

    theta = np.array([10.0, -5.0])          # a sensible start for cosine in [-1, 1]
    reg = np.diag([2.0 * l2, 0.0])
    for _ in range(iters):
        p = _sigmoid(X.dot(theta))
        grad = X.T.dot(w * (p - y)) + reg.dot(theta)
        hess = (X * (w * p * (1.0 - p))[:, None]).T.dot(X) + reg + np.eye(2) * 1e-9
        step = np.linalg.solve(hess, grad)
        theta = theta - step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(theta[0]), float(theta[1])


def error_rates(genuine, impostor):
    """Equal error rate and true-accept rate at 1% false-accept.

    These are the numbers that say whether the thresholds in config.py are
    sensible for *these* cameras, so calibrate.py prints them.
    """
    g = np.sort(np.asarray(genuine, dtype=np.float64))
    i = np.sort(np.asarray(impostor, dtype=np.float64))
    thresholds = np.unique(np.concatenate([g, i]))
    best = None
    for thr in thresholds:
        far = float(np.mean(i >= thr))
        frr = float(np.mean(g < thr))
        gap = abs(far - frr)
        if best is None or gap < best[0]:
            best = (gap, (far + frr) / 2.0, float(thr))
    far_target = float(np.quantile(i, 0.99))
    tar = float(np.mean(g > far_target))
    return {
        "eer": round(best[1], 4),
        "eer_threshold": round(best[2], 4),
        "tar_at_far_1pct": round(tar, 4),
        "far_1pct_threshold": round(far_target, 4),
    }


def build_calibration(genuine, impostor, n_identities, source):
    """Fit, validate and describe. Raises ValueError when the data cannot
    support an honest percentage."""
    genuine = [float(v) for v in genuine]
    impostor = [float(v) for v in impostor]
    if n_identities < MIN_IDENTITIES:
        raise ValueError("need photos of at least %d different people, got %d"
                         % (MIN_IDENTITIES, n_identities))
    if len(genuine) < MIN_GENUINE:
        raise ValueError("need at least %d same-person pairs, got %d - add more "
                         "photos per person" % (MIN_GENUINE, len(genuine)))
    if len(impostor) < MIN_IMPOSTOR:
        raise ValueError("need at least %d different-person pairs, got %d"
                         % (MIN_IMPOSTOR, len(impostor)))
    if np.mean(genuine) <= np.mean(impostor):
        raise ValueError("same-person scores are not higher than different-person "
                         "scores (%.3f vs %.3f) - check the folders are labelled "
                         "correctly" % (np.mean(genuine), np.mean(impostor)))

    a, b = fit_logistic(genuine, impostor)
    if a <= 0:
        raise ValueError("fitted curve is not increasing (a=%.3f); the data do "
                         "not separate same from different people" % a)

    rates = error_rates(genuine, impostor)
    return {
        "a": a,
        "b": b,
        "n_genuine": len(genuine),
        "n_impostor": len(impostor),
        "n_identities": int(n_identities),
        "genuine_mean": round(float(np.mean(genuine)), 4),
        "genuine_std": round(float(np.std(genuine)), 4),
        "impostor_mean": round(float(np.mean(impostor)), 4),
        "impostor_std": round(float(np.std(impostor)), 4),
        "score_at_50pct": round(-b / a, 4),
        "source": source,
        "fitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prior": "equal",
        **rates,
    }


class ScoreCalibrator:
    """Loads calibration.json; maps a cosine score to P(same person)."""

    # Never claim certainty. A model fitted on a few dozen pairs cannot
    # support "100%", and showing it would undo the point of calibrating.
    FLOOR, CEIL = 0.01, 0.99

    def __init__(self, path=None, data=None):
        self.path = Path(path or getattr(config, "CALIBRATION_FILE",
                                         config.DB_DIR / "calibration.json"))
        self.data = data
        if self.data is None:
            self.reload()

    def reload(self):
        self.data = None
        try:
            if self.path.exists():
                with open(str(self.path), "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if float(data.get("a", 0)) > 0:
                    self.data = data
        except Exception as exc:                           # corrupt file
            print("[calibration] ignored %s: %s" % (self.path, exc))
            self.data = None
        return self.calibrated

    @property
    def calibrated(self):
        return self.data is not None

    def probability(self, score):
        """P(same person | score), or None when uncalibrated or no score."""
        if self.data is None or score is None:
            return None
        try:
            z = float(self.data["a"]) * float(score) + float(self.data["b"])
        except (TypeError, ValueError):
            return None
        p = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))
        return round(min(self.CEIL, max(self.FLOOR, p)), 4)

    def save(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(self.path), "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        self.data = data

    def clear(self):
        if self.path.exists():
            self.path.unlink()
        self.data = None

    def summary(self):
        if self.data is None:
            return {"calibrated": False}
        d = self.data
        return {
            "calibrated": True,
            "n_genuine": d.get("n_genuine"),
            "n_impostor": d.get("n_impostor"),
            "n_identities": d.get("n_identities"),
            "eer": d.get("eer"),
            "score_at_50pct": d.get("score_at_50pct"),
            "source": d.get("source"),
            "fitted_at": d.get("fitted_at"),
        }
