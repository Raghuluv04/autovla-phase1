"""Physical action codebook via K-disk clustering (AutoVLA Appendix A).

Procedure from the paper:
  1. Sample 0.5 s motion segments from a large corpus of real vehicle
     trajectories (WOMD in the paper; nuScenes / synthetic here).
  2. Characterise each segment by the vehicle footprint contour at its final
     frame, in the coordinate frame of its first frame.
  3. Run K-disk clustering: iteratively select representative segments such
     that no two are within delta = 0.05 m of each other under mean contour
     distance.
  4. Store each selected segment's (dx, dy, dtheta) as action token a_k.

Selection strategy. Plain thresholded greedy selection gives no control over
the final K, so we default to a farthest-point (max-min) variant: at each step
take the candidate furthest from everything already selected. This yields the
same covering property, produces exactly K tokens, and reports the achieved
covering radius so it can be compared against the paper's delta. The
thresholded variant is kept available for faithfulness.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from .config import CODEBOOK_SIZE, CONTOUR_POINTS, EGO_LENGTH, EGO_WIDTH, KDISK_DELTA
from .geometry import canonical_contour, contours_from_poses


@dataclass
class Codebook:
    """K action tokens, each a body-frame (dx, dy, dtheta) over one dt."""

    actions: np.ndarray          # (K, 3)
    dt: float
    covering_radius: float       # achieved max distance from any sample to its token
    meta: dict

    def __len__(self) -> int:
        return len(self.actions)

    def save(self, path):
        np.savez(path, actions=self.actions, dt=self.dt,
                 covering_radius=self.covering_radius,
                 meta=json.dumps(self.meta))
        return path

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(actions=z["actions"], dt=float(z["dt"]),
                   covering_radius=float(z["covering_radius"]),
                   meta=json.loads(str(z["meta"])))


def _contour_points(segments, contour):
    """(N, 3) segments -> (N, P*2) flattened contour coordinates.

    Flattening lets us use plain Euclidean machinery: the mean per-point
    contour distance is ||ca - cb||_F / sqrt(P) up to a constant, so we scale
    so that Euclidean distance in this space *is* the mean contour distance.
    """
    pts = contours_from_poses(segments, contour)          # (N, P, 2)
    P = pts.shape[1]
    return pts.reshape(len(pts), -1) / np.sqrt(P), P


def _mean_contour_dist(flat_a, flat_b_row):
    """Distance from every row of flat_a to a single flat_b row.

    Euclidean distance in the scaled flattened space equals the RMS per-point
    contour distance, which upper-bounds and closely tracks the mean; for the
    rigid-transform contours used here the two agree to within a few percent
    and induce the same clustering.
    """
    return np.linalg.norm(flat_a - flat_b_row, axis=1)


def build_codebook(segments, k=CODEBOOK_SIZE, dt=0.5, delta=KDISK_DELTA,
                   mode="fps", ego_length=EGO_LENGTH, ego_width=EGO_WIDTH,
                   contour_points=CONTOUR_POINTS, seed=0, verbose=True):
    """Run K-disk clustering over motion segments.

    segments: (N, 3) array of body-frame (dx, dy, dtheta) motion segments.
    mode: "fps" (exactly k tokens, reports covering radius) or
          "threshold" (accept candidates greedily while >= delta apart).
    """
    segments = np.asarray(segments, dtype=float)
    rng = np.random.default_rng(seed)
    contour = canonical_contour(ego_length, ego_width, contour_points)
    flat, P = _contour_points(segments, contour)
    n = len(flat)

    # Seed with the stationary-most segment so "hold still" is always token 0.
    first = int(np.argmin(np.linalg.norm(segments[:, :2], axis=1)))
    selected = [first]
    min_dist = _mean_contour_dist(flat, flat[first])

    if mode == "fps":
        target = min(k, n)
        while len(selected) < target:
            nxt = int(np.argmax(min_dist))
            if min_dist[nxt] <= 0.0:
                break                      # corpus exhausted: duplicates only
            selected.append(nxt)
            np.minimum(min_dist, _mean_contour_dist(flat, flat[nxt]), out=min_dist)
            if verbose and len(selected) % 256 == 0:
                print(f"  [{len(selected):5d}/{target}] covering radius "
                      f"{min_dist.max():.4f} m")
    elif mode == "threshold":
        order = rng.permutation(n)
        for i in order:
            if len(selected) >= k:
                break
            if min_dist[i] >= delta:
                selected.append(int(i))
                np.minimum(min_dist, _mean_contour_dist(flat, flat[i]), out=min_dist)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    actions = segments[selected]
    covering = float(min_dist.max())
    meta = dict(mode=mode, delta=delta, n_samples=int(n), contour_points=int(P),
                ego_length=ego_length, ego_width=ego_width, seed=seed)
    if verbose:
        print(f"  built {len(actions)} tokens; covering radius {covering:.4f} m")
    return Codebook(actions=actions, dt=dt, covering_radius=covering, meta=meta)
