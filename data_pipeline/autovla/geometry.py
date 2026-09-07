"""SE(2) pose algebra and vehicle contour utilities.

The action codebook in AutoVLA is defined over *relative* motion segments
(dx, dy, dtheta) expressed in the ego frame at the start of the segment, and
the distance between two segments is the mean distance between the vehicle
footprint contours they induce (Appendix A). Everything needed for that lives
here.
"""

from __future__ import annotations

import numpy as np

from .config import CONTOUR_POINTS, EGO_LENGTH, EGO_WIDTH


# --------------------------------------------------------------------------
# SE(2) algebra. A pose is (x, y, theta) with theta in radians, x forward.
# --------------------------------------------------------------------------
def wrap_angle(theta):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi


def rot(theta):
    """2x2 rotation matrix, or (...,2,2) stack for array input."""
    theta = np.asarray(theta, dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


def compose(pose, delta):
    """Apply a body-frame delta to a world pose.

    pose:  (..., 3) world pose (x, y, theta)
    delta: (..., 3) body-frame motion (dx, dy, dtheta)
    returns (..., 3) new world pose. This is the operator used to decode a
    sequence of action tokens back into a trajectory.
    """
    pose = np.asarray(pose, dtype=float)
    delta = np.asarray(delta, dtype=float)
    th = pose[..., 2]
    c, s = np.cos(th), np.sin(th)
    x = pose[..., 0] + c * delta[..., 0] - s * delta[..., 1]
    y = pose[..., 1] + s * delta[..., 0] + c * delta[..., 1]
    return np.stack([x, y, wrap_angle(th + delta[..., 2])], -1)


def relative(pose_from, pose_to):
    """Express pose_to in the frame of pose_from. Inverse of `compose`."""
    pose_from = np.asarray(pose_from, dtype=float)
    pose_to = np.asarray(pose_to, dtype=float)
    th = pose_from[..., 2]
    c, s = np.cos(th), np.sin(th)
    dx = pose_to[..., 0] - pose_from[..., 0]
    dy = pose_to[..., 1] - pose_from[..., 1]
    return np.stack([
        c * dx + s * dy,
        -s * dx + c * dy,
        wrap_angle(pose_to[..., 2] - th),
    ], -1)


def to_deltas(traj):
    """Absolute trajectory (T+1, 3) including the start pose -> (T, 3) deltas."""
    traj = np.asarray(traj, dtype=float)
    return relative(traj[:-1], traj[1:])


def from_deltas(deltas, start=(0.0, 0.0, 0.0)):
    """(T, 3) body-frame deltas -> (T+1, 3) absolute trajectory."""
    deltas = np.asarray(deltas, dtype=float)
    poses = [np.asarray(start, dtype=float)]
    for d in deltas:
        poses.append(compose(poses[-1], d))
    return np.stack(poses)


# --------------------------------------------------------------------------
# Vehicle footprint contours
# --------------------------------------------------------------------------
def canonical_contour(length=EGO_LENGTH, width=EGO_WIDTH, n=CONTOUR_POINTS):
    """`n` points spread evenly along the perimeter of the ego bounding box.

    Origin is the rear axle centre (the reference point trajectories use), so
    the box extends from -overhang to length-overhang along x.
    """
    rear_overhang = 0.9
    x0, x1 = -rear_overhang, length - rear_overhang
    y0, y1 = -width / 2, width / 2
    corners = np.array([[x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=float)
    # Resample the closed polygon at n equally spaced arc-length positions.
    loop = np.vstack([corners, corners[:1]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, cum[-1], n, endpoint=False)
    idx = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, len(seg) - 1)
    t = ((targets - cum[idx]) / seg[idx])[:, None]
    return loop[idx] + t * (loop[idx + 1] - loop[idx])


def contours_from_poses(poses, contour=None):
    """(..., 3) poses -> (..., n, 2) footprint contour points."""
    poses = np.asarray(poses, dtype=float)
    if contour is None:
        contour = canonical_contour()
    R = rot(poses[..., 2])                       # (..., 2, 2)
    pts = np.einsum("...ij,nj->...ni", R, contour)
    return pts + poses[..., None, :2]


def contour_distance(a, b, contour=None):
    """Mean per-point contour distance between pose sets.

    a: (..., 3), b: (..., 3) broadcastable. Returns (...,).
    This is the metric K-disk clustering thresholds on (delta = 0.05 m).
    """
    ca = contours_from_poses(a, contour)
    cb = contours_from_poses(b, contour)
    return np.linalg.norm(ca - cb, axis=-1).mean(-1)
