"""Procedural driving corpus — a stand-in until nuScenes is downloaded.

This is NOT a model of driving; it is a source of kinematically valid ego
trajectories so that the codebook, tokenizer, statistics and visualisations can
be built, tested and demonstrated before the real dataset lands. Every
trajectory is produced by integrating the kinematic bicycle model, so the
segments it yields are physically feasible by construction — the same property
the paper relies on when it clusters real WOMD logs.

Manoeuvre mix and speed distribution are set to resemble urban nuScenes-style
driving (heavy stop-and-go, ~5 m/s mean speed, occasional intersection turns).
Swap `--source synthetic` for `--source nuscenes` once the real data is in
place and nothing downstream changes.
"""

from __future__ import annotations

import numpy as np

from ..config import MAX_SPEED, WHEELBASE
from ..kinematics import rollout

SIM_DT = 0.1                      # integration step; downsampled to 2 Hz after

# (name, probability) — urban driving is dominated by straight stop-and-go.
MANEUVERS = [
    ("cruise", 0.30),
    ("stop_and_go", 0.16),
    ("decelerate", 0.12),
    ("accelerate", 0.12),
    ("left_turn", 0.08),
    ("right_turn", 0.08),
    ("lane_change", 0.07),
    ("curve", 0.07),
]


def _smooth(rng, n, sigma, corr=12):
    """Temporally correlated noise.

    White noise on steering or acceleration is unphysical — it implies infinite
    jerk and steering rate, and it makes every 0.5 s segment unique, which
    artificially inflates tokenization error. Real control inputs are smooth, so
    we low-pass the noise over ~`corr` simulation steps.
    """
    w = rng.normal(0.0, 1.0, n + corr)
    kern = np.hanning(corr + 2)[1:-1]
    kern /= kern.sum()
    out = np.convolve(w, kern, mode="same")[:n]
    sd = out.std()
    return out * (sigma / sd) if sd > 1e-9 else out


def _speed_profile(rng, kind, n):
    """Sample an acceleration sequence for the manoeuvre."""
    a = np.zeros(n)
    if kind == "accelerate":
        a[:] = rng.uniform(0.6, 2.2)
    elif kind == "decelerate":
        a[:] = rng.uniform(-2.5, -0.8)
    elif kind == "stop_and_go":
        half = n // 2
        a[:half] = rng.uniform(-3.0, -1.5)
        a[half:] = rng.uniform(0.5, 2.0)
    else:
        a[:] = _smooth(rng, n, 0.25)             # gentle, smooth speed noise
    return a


def _steer_profile(rng, kind, n, v0=6.0):
    """Sample a steering sequence for the manoeuvre."""
    s = np.zeros(n)
    if kind == "left_turn":
        peak = rng.uniform(0.20, 0.42)
        s = peak * np.sin(np.linspace(0, np.pi, n)) ** 2
    elif kind == "right_turn":
        peak = rng.uniform(0.20, 0.42)
        s = -peak * np.sin(np.linspace(0, np.pi, n)) ** 2
    elif kind == "lane_change":
        # One lane width over T_lc seconds, then hold the new lane. For a
        # sinusoidal steer of amplitude A the lateral offset integrates to
        # y = A v^2 T^2 / (2 pi L), so A must scale as 1/v^2 to keep the
        # manoeuvre one lane wide regardless of speed.
        T_lc = 4.0
        width = rng.uniform(3.0, 3.9) * rng.choice([-1, 1])
        v = max(v0, 2.0)
        peak = float(np.clip(width * 2 * np.pi * WHEELBASE / (v ** 2 * T_lc ** 2),
                             -0.35, 0.35))
        n_lc = min(n, int(T_lc / SIM_DT))
        s = np.zeros(n)
        s[:n_lc] = peak * np.sin(np.linspace(0, 2 * np.pi, n_lc))
    elif kind == "curve":
        peak = rng.uniform(0.02, 0.10) * rng.choice([-1, 1])
        s = peak * np.ones(n) * np.linspace(0.5, 1.0, n)
    return s + _smooth(rng, n, 0.004)             # smooth steering jitter


def sample_trajectory(rng, duration=12.0):
    """One continuous ego trajectory, returned at the simulation rate."""
    kinds, probs = zip(*MANEUVERS)
    kind = rng.choice(kinds, p=np.array(probs) / np.sum(probs))
    n = int(duration / SIM_DT)

    if kind in ("left_turn", "right_turn"):
        v0 = rng.uniform(2.0, 8.0)
    elif kind == "cruise":
        v0 = rng.uniform(0.0, 16.0)
    elif kind == "accelerate":
        v0 = rng.uniform(0.0, 4.0)
    else:
        v0 = rng.uniform(1.0, 14.0)

    # A slice of urban logs is simply stationary (queued at a light).
    if rng.random() < 0.10:
        poses = np.zeros((n + 1, 3))
        return poses, np.zeros(n + 1), "stationary"

    accel = _speed_profile(rng, kind, n)
    steer = _steer_profile(rng, kind, n, v0=v0)
    poses, speeds = rollout(np.clip(v0, 0, MAX_SPEED), accel, steer, dt=SIM_DT)
    return poses, speeds, kind


def build_synthetic_corpus(n_traj=4000, duration=12.0, seed=0, rate_hz=2.0):
    """Return a list of dicts with 2 Hz poses, speeds and the manoeuvre label."""
    rng = np.random.default_rng(seed)
    step = int(round((1.0 / rate_hz) / SIM_DT))
    corpus = []
    for i in range(n_traj):
        poses, speeds, kind = sample_trajectory(rng, duration)
        corpus.append(dict(
            traj_id=f"syn_{i:06d}",
            poses=poses[::step],           # (T, 3) at 2 Hz, world frame
            speeds=speeds[::step],
            maneuver=kind,
        ))
    return corpus
