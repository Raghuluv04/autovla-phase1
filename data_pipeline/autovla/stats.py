"""Corpus statistics used for the data-processing report."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .geometry import to_deltas
from .kinematics import invert_segment


def segment_bank(corpus, dt=0.5, max_segments=None):
    """Cut every trajectory in the corpus into 0.5 s body-frame segments.

    This is the sample pool that K-disk clustering runs over.
    """
    segs, meta = [], []
    for tr in corpus:
        poses = np.asarray(tr["poses"], dtype=float)
        if len(poses) < 2:
            continue
        d = to_deltas(poses)
        segs.append(d)
        meta.extend([tr.get("maneuver", "?")] * len(d))
    segs = np.concatenate(segs) if segs else np.zeros((0, 3))
    if max_segments is not None and len(segs) > max_segments:
        idx = np.random.default_rng(0).choice(len(segs), max_segments, replace=False)
        segs, meta = segs[idx], [meta[i] for i in idx]
    return segs, meta


def windows(corpus, n_future=10, stride=1):
    """Sliding 5 s future windows, in the ego frame of each window start."""
    from .geometry import relative
    out = []
    for tr in corpus:
        poses = np.asarray(tr["poses"], dtype=float)
        for i in range(0, len(poses) - n_future, stride):
            cur = poses[i]
            out.append(relative(cur, poses[i:i + n_future + 1]))
    return out


def describe_segments(segments, dt=0.5, v0=5.0):
    """Speed / curvature / heading summary of a segment bank."""
    k = invert_segment(segments, v0=v0, dt=dt)
    speed = np.linalg.norm(segments[:, :2], axis=1) / dt
    return dict(
        n=len(segments),
        speed_mps=speed,
        curvature=k["curvature"],
        dtheta_deg=np.rad2deg(segments[:, 2]),
        lateral=segments[:, 1],
        longitudinal=segments[:, 0],
        stationary_frac=float(np.mean(speed < 0.5)),
        speed_mean=float(speed.mean()),
        speed_p95=float(np.percentile(speed, 95)),
    )


def describe_samples(samples):
    """Distribution summary over built DrivingSamples."""
    return dict(
        n=len(samples),
        commands=Counter(s.command for s in samples),
        modes=Counter(s.thinking_mode for s in samples),
        datasets=Counter(s.dataset for s in samples),
        velocity=np.array([s.velocity for s in samples]),
        acceleration=np.array([s.acceleration for s in samples]),
        endpoints=np.array([np.asarray(s.future_traj)[-1, :2] for s in samples]),
    )


def format_report(seg_desc, sample_desc, tok_eval, codebook):
    L = []
    L.append("=" * 66)
    L.append("AutoVLA data pipeline — Phase I report")
    L.append("=" * 66)
    L.append("")
    L.append(f"Motion segments (0.5 s)        : {seg_desc['n']:,}")
    L.append(f"  mean speed                   : {seg_desc['speed_mean']:.2f} m/s")
    L.append(f"  95th pct speed               : {seg_desc['speed_p95']:.2f} m/s")
    L.append(f"  stationary fraction          : {seg_desc['stationary_frac']:.1%}")
    L.append("")
    L.append(f"Action codebook                : K = {len(codebook)}")
    L.append(f"  covering radius              : {codebook.covering_radius:.4f} m")
    L.append(f"  clustering mode              : {codebook.meta['mode']}")
    L.append(f"  pool size                    : {codebook.meta['n_samples']:,}")
    L.append("")
    L.append(f"Tokenizer reconstruction (5 s, {tok_eval['n_traj']:,} trajectories)")
    L.append(f"  ADE                          : {tok_eval['ade']:.4f} m")
    L.append(f"  FDE                          : {tok_eval['fde']:.4f} m")
    L.append(f"  heading MAE                  : {tok_eval['heading_mae_deg']:.3f} deg")
    L.append(f"  codebook usage               : {tok_eval['codebook_usage']:.1%}")
    L.append("")
    L.append(f"Training samples               : {sample_desc['n']:,}")
    L.append(f"  thinking modes               : {dict(sample_desc['modes'])}")
    L.append(f"  commands                     : {dict(sample_desc['commands'])}")
    L.append(f"  mean |velocity|              : {np.abs(sample_desc['velocity']).mean():.2f} m/s")
    L.append("=" * 66)
    return "\n".join(L)
