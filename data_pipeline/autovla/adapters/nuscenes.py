"""nuScenes -> unified DrivingSample adapter.

Reads the nuScenes JSON tables directly rather than depending on
nuscenes-devkit, which keeps the pipeline installable on any Python version.
Works identically on v1.0-mini (10 scenes, ~4 GB) and v1.0-trainval.

Per the paper's preprocessing (Appendix E.1) each keyframe becomes one sample:
  * future trajectory: next 10 keyframes (5 s at 2 Hz), ego frame, (x, y, yaw)
  * history: previous 3 keyframes (2 s of context), ego frame
  * images: CAM_FRONT_LEFT / CAM_FRONT / CAM_FRONT_RIGHT over 4 frames
  * state: speed and acceleration by finite differences of the ego pose
  * command: inferred from the future path
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

from ..config import CAMERAS, HISTORY_FRAMES, N_ACTION_TOKENS
from ..geometry import relative, wrap_angle
from ..schema import DrivingSample, infer_command


def _load(root, version, name):
    with open(os.path.join(root, version, f"{name}.json")) as f:
        return json.load(f)


def quat_to_rot(q):
    """(w, x, y, z) quaternion -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def project_to_image(points_ego, calib, z_ego=0.0, min_depth=1.0):
    """Ego-frame ground points (N, 2 or 3) -> pixel coordinates (M, 2).

    nuScenes camera axes are x-right, y-down, z-forward, and the calibrated
    sensor rotation maps camera -> ego, so we invert it. Points behind the
    image plane are dropped.
    """
    pts = np.asarray(points_ego, dtype=float)
    if pts.shape[1] == 2:
        pts = np.hstack([pts, np.full((len(pts), 1), z_ego)])
    R = quat_to_rot(calib["rotation"])
    cam = (pts - calib["translation"]) @ R          # == R^T (p - t)
    keep = cam[:, 2] > min_depth
    cam = cam[keep]
    if not len(cam):
        return np.zeros((0, 2)), keep
    uv = cam @ calib["intrinsic"].T
    return uv[:, :2] / uv[:, 2:3], keep


def quat_to_yaw(q):
    """nuScenes stores rotations as (w, x, y, z); we only need the yaw."""
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class NuScenesReader:
    def __init__(self, root="data/nuscenes", version="v1.0-mini"):
        self.root, self.version = root, version
        self.sample = {s["token"]: s for s in _load(root, version, "sample")}
        self.ego_pose = {e["token"]: e for e in _load(root, version, "ego_pose")}
        self.scene = {s["token"]: s for s in _load(root, version, "scene")}
        sensor = {s["token"]: s for s in _load(root, version, "sensor")}
        calib = {c["token"]: c for c in _load(root, version, "calibrated_sensor")}
        self.calib = calib
        self.channel_of_calib = {
            t: sensor[c["sensor_token"]]["channel"] for t, c in calib.items()}

        # keyframe sample_data, indexed by (sample_token, channel)
        self.sd_by_sample = defaultdict(dict)
        for sd in _load(root, version, "sample_data"):
            if not sd["is_key_frame"]:
                continue
            ch = self.channel_of_calib.get(sd["calibrated_sensor_token"])
            if ch in CAMERAS:
                self.sd_by_sample[sd["sample_token"]][ch] = sd

    # -- scene traversal --------------------------------------------------
    def scene_samples(self, scene_token):
        """Ordered list of sample tokens for one scene."""
        tok = self.scene[scene_token]["first_sample_token"]
        out = []
        while tok:
            out.append(tok)
            tok = self.sample[tok]["next"]
        return out

    def pose_of(self, sample_token):
        """World-frame (x, y, yaw) of the ego at a keyframe."""
        sd = self.sd_by_sample[sample_token].get("CAM_FRONT")
        if sd is None:
            return None
        e = self.ego_pose[sd["ego_pose_token"]]
        return np.array([e["translation"][0], e["translation"][1],
                         quat_to_yaw(e["rotation"])])

    def camera_calibration(self, sample_token, channel="CAM_FRONT"):
        """Intrinsics and extrinsics, for projecting trajectories onto images."""
        sd = self.sd_by_sample[sample_token].get(channel)
        if sd is None:
            return None
        c = self.calib[sd["calibrated_sensor_token"]]
        return dict(intrinsic=np.array(c["camera_intrinsic"], dtype=float),
                    translation=np.array(c["translation"], dtype=float),
                    rotation=np.array(c["rotation"], dtype=float))

    # -- sample construction ----------------------------------------------
    def build_samples(self, n_future=N_ACTION_TOKENS, n_history=HISTORY_FRAMES,
                      dt=0.5):
        samples = []
        for scene_token in self.scene:
            toks = self.scene_samples(scene_token)
            poses = [self.pose_of(t) for t in toks]
            if any(p is None for p in poses):
                continue
            poses = np.stack(poses)

            for i in range(len(toks)):
                if i + n_future >= len(poses):
                    break                      # not enough future for 5 s
                if i < n_history - 1:
                    continue                   # not enough history for 2 s

                cur = poses[i]
                fut = relative(cur, poses[i:i + n_future + 1])
                hist = relative(cur, poses[i - n_history + 1:i + 1])

                # Speed / acceleration by finite differences of the ego pose.
                v = np.linalg.norm(poses[i, :2] - poses[i - 1, :2]) / dt
                v_prev = np.linalg.norm(poses[i - 1, :2] - poses[i - 2, :2]) / dt \
                    if i >= 2 else v
                a = (v - v_prev) / dt

                images = {}
                for ch in CAMERAS:
                    seq = []
                    for j in range(i - n_history + 1, i + 1):
                        sd = self.sd_by_sample[toks[j]].get(ch)
                        seq.append(sd["filename"] if sd else "")
                    images[ch] = seq

                samples.append(DrivingSample(
                    sample_id=toks[i],
                    dataset="nuscenes",
                    future_traj=fut,
                    history_traj=hist,
                    images=images,
                    velocity=float(v),
                    acceleration=float(a),
                    command=infer_command(fut),
                    extra=dict(scene=self.scene[scene_token]["name"],
                               frame_idx=i),
                ))
        return samples

    def trajectory_corpus(self):
        """Continuous per-scene ego trajectories, for codebook construction."""
        corpus = []
        for scene_token in self.scene:
            toks = self.scene_samples(scene_token)
            poses = [self.pose_of(t) for t in toks]
            if any(p is None for p in poses) or len(poses) < 3:
                continue
            poses = np.stack(poses)
            speeds = np.concatenate([
                [0.0], np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1) / 0.5])
            corpus.append(dict(traj_id=self.scene[scene_token]["name"],
                               poses=poses, speeds=speeds, maneuver="real"))
        return corpus


def available(root="data/nuscenes", version="v1.0-mini"):
    return os.path.isdir(os.path.join(root, version))
