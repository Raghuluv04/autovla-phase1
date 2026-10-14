"""Calibration utilities for the confidence score (Milestone 8).

``confidence`` is read as the probability that a plan succeeds (ADE below the
failure threshold). Temperature scaling then fits ``p = sigmoid(logit(conf)/T + b)``
by minimising the negative log-likelihood on a held-out calibration split;
``b`` is fixed at 0 (pure temperature scaling) unless ``fit_bias=True``.

All functions are NumPy-only. ``label`` conventions: ``success`` is 1 when the
plan was good; ``failure = 1 - success`` is the positive class for AUROC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .config import ECE_BINS

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def nll(prob, success) -> float:
    """Mean binary negative log-likelihood."""
    p = np.clip(np.asarray(prob, float), EPS, 1 - EPS)
    y = np.asarray(success, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def reliability_bins(conf, success, n_bins: int = ECE_BINS):
    """[(lo, hi, count, mean_confidence, accuracy)] over equal-width bins."""
    conf, success = np.asarray(conf, float), np.asarray(success, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        out.append((edges[b], edges[b + 1], int(m.sum()),
                    float(conf[m].mean()) if m.any() else float("nan"),
                    float(success[m].mean()) if m.any() else float("nan")))
    return out


def compute_ece(conf, success, n_bins: int = ECE_BINS) -> float:
    """Expected calibration error: sum_b (n_b / N) * |acc_b - conf_b|."""
    n = len(conf)
    return float(sum(c / n * abs(a - m) for _, _, c, m, a in reliability_bins(conf, success, n_bins) if c))


def compute_auroc(score, label) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie), via rank statistics (no sklearn).

    ``label`` 1 marks the positive class. Returns NaN if one class is empty.
    """
    score, label = np.asarray(score, float), np.asarray(label).astype(bool)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_scores = score[order]
    i = 0
    while i < len(score):                         # average ranks across ties
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@dataclass
class TemperatureScaler:
    """Maps a raw confidence to a calibrated success probability.

    kind="logit"     p = sigmoid(logit(conf) / T + b)      (b = 0: classic temperature scaling)
    kind="variance"  p = 1 / (1 + u / T), u = 1/conf - 1    (T rescales the *uncertainty* u =
                     sigma^2_ale + sigma^2_epi, the regression-style temperature of Laves et al.)

    The logit form cannot move predictions whose confidence is saturated near 0
    (large-variance regimes: all logits ~ -5, so dividing by T only squeezes them
    towards 0.5 from one side); the variance form rescales exactly that quantity.
    """

    temperature: float = 1.0
    bias: float = 0.0
    kind: str = "logit"

    def __call__(self, conf):
        c = np.clip(np.asarray(conf, float), EPS, 1.0)
        if self.kind == "variance":
            return 1.0 / (1.0 + (1.0 / c - 1.0) / self.temperature)
        return _sigmoid(_logit(c) / self.temperature + self.bias)

    @property
    def name(self) -> str:
        return "variance_T" if self.kind == "variance" else ("logit_T_bias" if self.bias else "logit_T")


def temperature_scaling(conf, success, fit_bias: bool = False, kind: str = "logit") -> TemperatureScaler:
    """Fit T (and optionally b) minimising NLL of ``success`` on a calibration set."""
    y = np.asarray(success, float)
    if kind == "variance":
        u = 1.0 / np.clip(np.asarray(conf, float), EPS, 1.0) - 1.0
        r = minimize_scalar(lambda lt: nll(1.0 / (1.0 + u / np.exp(lt)), y), bounds=(-10.0, 14.0), method="bounded")
        return TemperatureScaler(float(np.exp(r.x)), 0.0, "variance")
    z = _logit(conf)

    def loss(t, b=0.0):
        return nll(_sigmoid(z / t + b), y)

    if not fit_bias:
        r = minimize_scalar(lambda lt: loss(np.exp(lt)), bounds=(-6.0, 6.0), method="bounded")
        return TemperatureScaler(float(np.exp(r.x)), 0.0, "logit")
    r = minimize(lambda w: loss(np.exp(w[0]), w[1]), x0=[0.0, 0.0], method="Nelder-Mead")
    return TemperatureScaler(float(np.exp(r.x[0])), float(r.x[1]), "logit")


def fit_calibrators(conf, success) -> dict:
    """All candidate calibrators fitted on the same calibration split."""
    return {"logit_T": temperature_scaling(conf, success),
            "variance_T": temperature_scaling(conf, success, kind="variance"),
            "logit_T_bias": temperature_scaling(conf, success, fit_bias=True)}


def select_calibrator(conf, success, candidates: dict | None = None) -> TemperatureScaler:
    """The candidate with the lowest NLL on the (calibration) data it is given."""
    candidates = candidates or fit_calibrators(conf, success)
    return min(candidates.values(), key=lambda sc: nll(sc(conf), success))


def bootstrap_ci(fn, *arrays, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI of ``fn(*arrays)`` under resampling of rows."""
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    N = len(arrays[0])
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        v = fn(*[a[idx] for a in arrays])
        if np.isfinite(v):
            vals.append(v)
    return (float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))) \
        if vals else (float("nan"), float("nan"))
