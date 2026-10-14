"""Open-loop planning metrics (UniAD protocol) and latency measurement (M7).

Trajectories are ``(T+1, 3)`` ego-frame poses ``(x, y, yaw)`` at 2 Hz with row 0
the current pose, so horizon ``h`` seconds is row ``h / DT`` (1 s -> 2, 3 s -> 6).

* **L2 @ h** -- Euclidean error of the predicted position at exactly ``h``
  seconds (the VAD/UniAD per-horizon convention). ``l2_cumulative`` gives the
  ST-P3 "mean over [0, h]" variant for comparison with older tables.
* **Collision @ h** -- the planned ego box (4.084 x 1.730 m) overlaps an agent
  box at *any* step up to ``h`` seconds (cumulative "has collided by h").
  Agent boxes are ``(x, y, yaw, length, width)`` per future step.
* **Route completion** -- open-loop proxy: arc-length progress of the predicted
  end point along the ground-truth path, over the path's length, clipped to [0, 1].
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import COLLISION_SAFETY_MARGIN_M, DT, EGO_LENGTH, EGO_WIDTH, EVAL_HORIZONS_S


def step_of(h: float) -> int:
    return int(round(h / DT))


def l2_at_horizons(pred, gt, horizons=EVAL_HORIZONS_S) -> dict:
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    return {h: float(np.linalg.norm(pred[step_of(h), :2] - gt[step_of(h), :2])) for h in horizons}


def l2_cumulative(pred, gt, horizons=EVAL_HORIZONS_S) -> dict:
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    d = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1)
    return {h: float(d[1:step_of(h) + 1].mean()) for h in horizons}


def ade(pred, gt) -> float:
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    return float(np.linalg.norm(pred[1:, :2] - gt[1:, :2], axis=1).mean())


def box_corners(x, y, yaw, length, width, margin=0.0) -> np.ndarray:
    """(4, 2) corners of an oriented box."""
    l, w = length / 2 + margin, width / 2 + margin
    local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]])
    c, s = np.cos(yaw), np.sin(yaw)
    return local @ np.array([[c, s], [-s, c]]) + np.array([x, y])


def obb_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating-axis test for two (4, 2) convex quads."""
    for quad in (a, b):
        for i in range(4):
            edge = quad[(i + 1) % 4] - quad[i]
            axis = np.array([-edge[1], edge[0]])
            n = np.linalg.norm(axis)
            if n < 1e-12:
                continue
            axis /= n
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def collision_by_horizon(pred, obstacles, horizons=EVAL_HORIZONS_S,
                         ego_length=EGO_LENGTH, ego_width=EGO_WIDTH,
                         margin=COLLISION_SAFETY_MARGIN_M) -> dict:
    """{h: bool} -- collided at some step in (0, h]. ``obstacles``: (T+1, N, 5) or None."""
    pred = np.asarray(pred, float)
    out = {h: False for h in horizons}
    if obstacles is None or np.size(obstacles) == 0:
        return out
    obstacles = np.asarray(obstacles, float)
    first_hit = None
    for t in range(1, min(len(pred), len(obstacles))):
        ego = box_corners(pred[t, 0], pred[t, 1], pred[t, 2], ego_length, ego_width)
        for ox, oy, oyaw, ol, ow in obstacles[t]:
            if obb_overlap(ego, box_corners(ox, oy, oyaw, ol, ow, margin)):
                first_hit = t
                break
        if first_hit is not None:
            break
    if first_hit is not None:
        for h in horizons:
            out[h] = first_hit <= step_of(h)
    return out


def route_completion(pred, gt) -> float:
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    seg = np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1)
    total = float(seg.sum())
    if total < 0.5:                                         # stationary route
        return 1.0 if np.linalg.norm(pred[-1, :2] - pred[0, :2]) < 0.5 else 0.0
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    end = pred[-1, :2]
    best, best_d = 0.0, np.inf
    for i in range(len(seg)):                               # project onto each GT segment
        a, b = gt[i, :2], gt[i + 1, :2]
        ab = b - a
        t = np.clip(np.dot(end - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
        d = np.linalg.norm(end - (a + t * ab))
        if d < best_d:
            best_d, best = d, cum[i] + t * seg[i]
    return float(np.clip(best / total, 0.0, 1.0))


# ---------------------------------------------------------------------------
# latency
# ---------------------------------------------------------------------------
class LatencyMeter:
    """Collect per-call wall times (seconds); ``measure()`` is a timing context."""

    def __init__(self):
        self.samples: list[float] = []

    def add(self, seconds: float):
        self.samples.append(float(seconds))

    def measure(self):
        meter = self

        class _Ctx:
            def __enter__(self_):
                self_.t0 = time.perf_counter()
            def __exit__(self_, *exc):
                meter.add(time.perf_counter() - self_.t0)
        return _Ctx()

    def summary(self) -> dict:
        a = np.asarray(self.samples, float)
        if a.size == 0:
            return dict(n=0, mean=float("nan"), median=float("nan"), p95=float("nan"))
        return dict(n=int(a.size), mean=float(a.mean()), median=float(np.median(a)),
                    p95=float(np.percentile(a, 95)))


def latency_reduction(fast_mean: float, slow_mean: float) -> float:
    """Fraction of slow-mode latency saved by fast mode (paper Table 2: 66.8 %)."""
    return 1.0 - fast_mean / slow_mean if slow_mean > 0 else float("nan")


# ---------------------------------------------------------------------------
# accumulator
# ---------------------------------------------------------------------------
@dataclass
class OpenLoopAccumulator:
    horizons: tuple = EVAL_HORIZONS_S
    l2: dict = field(default_factory=dict)
    col: dict = field(default_factory=dict)
    ade_: list = field(default_factory=list)
    route: list = field(default_factory=list)
    latency: LatencyMeter = field(default_factory=LatencyMeter)
    cot_tokens: list = field(default_factory=list)

    def add(self, pred, gt, obstacles=None, latency_s=None, cot_tokens=0):
        for h, v in l2_at_horizons(pred, gt, self.horizons).items():
            self.l2.setdefault(h, []).append(v)
        for h, v in collision_by_horizon(pred, obstacles, self.horizons).items():
            self.col.setdefault(h, []).append(float(v))
        self.ade_.append(ade(pred, gt))
        self.route.append(route_completion(pred, gt))
        self.cot_tokens.append(cot_tokens)
        if latency_s is not None:
            self.latency.add(latency_s)

    def __len__(self):
        return len(self.ade_)

    def summary(self) -> dict:
        out = {"n": len(self)}
        for h in self.horizons:
            out[f"L2@{h:g}s"] = float(np.mean(self.l2[h])) if self.l2 else float("nan")
        for h in self.horizons:
            out[f"Col@{h:g}s"] = float(np.mean(self.col[h])) if self.col else float("nan")
        out["ADE"] = float(np.mean(self.ade_)) if self.ade_ else float("nan")
        out["route_completion"] = float(np.mean(self.route)) if self.route else float("nan")
        out["latency_s"] = self.latency.summary()["mean"]
        out["cot_tokens"] = float(np.mean(self.cot_tokens)) if self.cot_tokens else float("nan")
        return out
