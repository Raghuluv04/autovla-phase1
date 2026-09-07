"""Kinematic bicycle model: feasibility checking and trajectory rollout.

AutoVLA claims its action tokens are *physical* — every token is a motion
segment actually observed in a real driving log, so feasibility is inherited
from the data. We verify that claim explicitly here by inverting each token
back to the (acceleration, steering) pair that would produce it and checking
the result against vehicle limits. This is Objective 4's "bicycle-model
dynamics check" in the project spec.
"""

from __future__ import annotations

import numpy as np

from .config import (DT, MAX_ACCEL, MAX_CURVATURE, MAX_DECEL, MAX_SPEED,
                     MAX_STEER, WHEELBASE)
from .geometry import compose, wrap_angle


def rollout(v0, accel, steer, dt=0.1, wheelbase=WHEELBASE, start=(0.0, 0.0, 0.0)):
    """Integrate the kinematic bicycle model.

    accel, steer: (T,) control sequences held over each dt.
    Returns poses (T+1, 3) and speeds (T+1,).
    """
    accel = np.asarray(accel, dtype=float)
    steer = np.asarray(steer, dtype=float)
    T = len(accel)
    poses = np.zeros((T + 1, 3))
    speeds = np.zeros(T + 1)
    poses[0] = start
    speeds[0] = v0
    for t in range(T):
        v = speeds[t]
        x, y, th = poses[t]
        x += v * np.cos(th) * dt
        y += v * np.sin(th) * dt
        th = wrap_angle(th + v / wheelbase * np.tan(steer[t]) * dt)
        v = np.clip(v + accel[t] * dt, 0.0, MAX_SPEED)
        poses[t + 1] = (x, y, th)
        speeds[t + 1] = v
    return poses, speeds


def invert_segment(delta, v0, dt=DT, wheelbase=WHEELBASE):
    """Recover the (mean speed, acceleration, curvature, steer) of one segment.

    `delta` is a body-frame (dx, dy, dtheta) covering `dt` seconds starting at
    speed `v0`. We fit a constant-curvature arc, which is the standard
    closed-form inverse of the bicycle model over a short horizon.
    """
    delta = np.asarray(delta, dtype=float)
    dx, dy, dth = delta[..., 0], delta[..., 1], delta[..., 2]
    chord = np.hypot(dx, dy)
    # Arc length from chord and subtended angle: s = chord * (a/2) / sin(a/2).
    half = np.abs(dth) / 2.0
    scale = np.where(half > 1e-6, half / np.maximum(np.sin(half), 1e-9), 1.0)
    arc = chord * scale
    curvature = np.where(arc > 1e-6, dth / np.maximum(arc, 1e-9), 0.0)
    v_mean = arc / dt
    v1 = 2.0 * v_mean - v0            # constant-acceleration end speed
    accel = (v1 - v0) / dt
    steer = np.arctan(curvature * wheelbase)
    return dict(arc=arc, v_mean=v_mean, v_end=v1, accel=accel,
                curvature=curvature, steer=steer)


def is_feasible(delta, v0=5.0, dt=DT):
    """Boolean feasibility mask for motion segments under vehicle limits."""
    k = invert_segment(delta, v0=v0, dt=dt)
    ok = (
        (np.abs(k["curvature"]) <= MAX_CURVATURE)
        & (np.abs(k["steer"]) <= MAX_STEER)
        & (k["accel"] <= MAX_ACCEL + 1e-6)
        & (k["accel"] >= MAX_DECEL - 1e-6)
        & (k["v_mean"] >= -1e-6)
        & (k["v_mean"] <= MAX_SPEED)
    )
    return ok, k


def feasibility_report(deltas, v0=5.0, dt=DT):
    """Summarise what fraction of a set of segments is physically realisable."""
    ok, k = is_feasible(deltas, v0=v0, dt=dt)
    return {
        "n": int(np.size(ok)),
        "feasible_frac": float(np.mean(ok)),
        "max_abs_curvature": float(np.max(np.abs(k["curvature"]))),
        "max_abs_steer_deg": float(np.rad2deg(np.max(np.abs(k["steer"])))),
        "speed_range": (float(np.min(k["v_mean"])), float(np.max(k["v_mean"]))),
        "accel_range": (float(np.min(k["accel"])), float(np.max(k["accel"]))),
    }
