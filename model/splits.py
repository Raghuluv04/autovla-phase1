"""Scene-difficulty labels and synthetic agent boxes.

``is_hard`` implements the Milestone-8 definition of the hard split: sharp
manoeuvre OR stationary-to-moving transition OR high occlusion. Occlusion needs
perception data; it is read from ``DrivingSample.extra["occlusion"]`` when an
adapter provides it and ignored otherwise (the synthetic corpus has none).

``synthetic_obstacles`` fabricates agent boxes so the collision metric can be
exercised when no annotated agents exist. They are placed relative to the
*ground-truth* path -- a stopped vehicle beyond its end and parked/side cars
beside it -- so the GT plan never collides but an overshooting or swerving plan
can. Numbers computed with them test the pipeline, not real-world safety.
"""

from __future__ import annotations

import hashlib

import numpy as np

from autovla.schema import DrivingSample

from .config import (DT, EGO_LENGTH, EGO_WIDTH, HARD_END_SPEED, HARD_HEADING_RAD,
                     HARD_LATERAL_M, HARD_OCCLUSION, HARD_START_SPEED)
from .metrics import box_corners, obb_overlap

CAR_LENGTH, CAR_WIDTH = 4.5, 1.9
SIDE_OFFSET_M = 3.2           # lateral distance of side traffic from the GT path
LEAD_GAP_M = (3.0, 8.0)       # gap between GT end-point and the stopped lead car


def difficulty_reasons(s: DrivingSample) -> list[str]:
    traj = np.asarray(s.future_traj, float)
    reasons = []
    if abs(traj[-1, 2] - traj[0, 2]) > HARD_HEADING_RAD or abs(traj[-1, 1] - traj[0, 1]) > HARD_LATERAL_M:
        reasons.append("sharp_manoeuvre")
    v_end = np.linalg.norm(np.diff(traj[-3:, :2], axis=0), axis=1).mean() / DT
    if s.velocity < HARD_START_SPEED and v_end > HARD_END_SPEED:
        reasons.append("stationary_to_moving")
    if float(s.extra.get("occlusion", 0.0)) > HARD_OCCLUSION:
        reasons.append("high_occlusion")
    return reasons


def is_hard(s: DrivingSample) -> bool:
    return bool(difficulty_reasons(s))


def _rng(sample_id: str, salt: str = "obs") -> np.random.Generator:
    return np.random.default_rng(int(hashlib.md5(f"{salt}:{sample_id}".encode()).hexdigest()[:12], 16))


def synthetic_obstacles(s: DrivingSample, seed: int = 0) -> np.ndarray:
    """(T+1, N, 5) static boxes (x, y, yaw, length, width), repeated over time."""
    gt = np.asarray(s.future_traj, float)
    rng = _rng(f"{seed}:{s.sample_id}")
    boxes = []
    end, yaw = gt[-1, :2], gt[-1, 2]
    fwd = np.array([np.cos(yaw), np.sin(yaw)])
    left = np.array([-np.sin(yaw), np.cos(yaw)])
    gap = rng.uniform(*LEAD_GAP_M)
    boxes.append((*(end + fwd * (EGO_LENGTH / 2 + CAR_LENGTH / 2 + gap)), yaw, CAR_LENGTH, CAR_WIDTH))
    for t_idx in rng.choice(np.arange(2, len(gt)), size=2, replace=False):
        side = rng.choice([-1.0, 1.0])
        yaw_t = gt[t_idx, 2]
        pos = gt[t_idx, :2] + side * SIDE_OFFSET_M * np.array([-np.sin(yaw_t), np.cos(yaw_t)])
        boxes.append((*pos, yaw_t, CAR_LENGTH, CAR_WIDTH))
    keep = []
    for b in boxes:                                  # never collide with the GT plan itself
        quad = box_corners(*b[:3], b[3], b[4])
        if not any(obb_overlap(box_corners(*g, EGO_LENGTH, EGO_WIDTH), quad) for g in gt):
            keep.append(b)
    arr = np.array(keep, float).reshape(-1, 5)
    return np.repeat(arr[None], len(gt), axis=0)
