"""Unified driving-sample schema (AutoVLA Appendix E.1).

The paper standardises every dataset (nuPlan, nuScenes, Waymo, CARLA) into one
format before training. Each sample carries:
  1) ground-truth future trajectory in the ego frame at 2 Hz,
  2) image paths for multi-view camera sequences (4 frames at 2 Hz = 2 s history),
  3) CoT reasoning annotation (optional -> fast vs slow thinking sample),
  4) vehicle state (velocity, acceleration),
  5) high-level driving instruction.

Every adapter in autovla/adapters/ emits this record, so the tokenizer,
statistics and dataset builder are dataset-agnostic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from .config import COMMANDS


@dataclass
class Reasoning:
    """Four-part structured CoT (paper Sec. 3.2 annotation pipeline)."""

    scene_description: str = ""
    critical_objects: str = ""
    agent_intentions: str = ""
    driving_decision: str = ""

    def to_text(self):
        return (
            f"Scene: {self.scene_description}\n"
            f"Critical objects: {self.critical_objects}\n"
            f"Predicted intentions: {self.agent_intentions}\n"
            f"Driving decision: {self.driving_decision}"
        )

    def is_empty(self):
        return not any(vars(self).values())


@dataclass
class DrivingSample:
    sample_id: str
    dataset: str
    # (T+1, 3) ego-frame poses, index 0 = current pose (0,0,0), 2 Hz.
    future_traj: np.ndarray
    # (H, 3) ego-frame history poses ending at the current pose.
    history_traj: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # {camera_name: [path_t-3, path_t-2, path_t-1, path_t]}
    images: dict = field(default_factory=dict)
    velocity: float = 0.0            # m/s
    acceleration: float = 0.0        # m/s^2
    command: str = "Go Straight"
    reasoning: Reasoning = field(default_factory=Reasoning)
    extra: dict = field(default_factory=dict)

    @property
    def has_reasoning(self):
        return not self.reasoning.is_empty()

    @property
    def thinking_mode(self):
        return "slow" if self.has_reasoning else "fast"

    def to_json(self):
        d = asdict(self)
        d["future_traj"] = np.asarray(self.future_traj).round(4).tolist()
        d["history_traj"] = np.asarray(self.history_traj).round(4).tolist()
        return d

    @classmethod
    def from_json(cls, d):
        d = dict(d)
        d["future_traj"] = np.asarray(d["future_traj"], dtype=float)
        d["history_traj"] = np.asarray(d["history_traj"], dtype=float)
        d["reasoning"] = Reasoning(**d.get("reasoning", {}))
        return cls(**d)


def infer_command(future_traj, lateral_thresh=4.0, stop_thresh=1.0):
    """Derive the high-level navigation instruction from the GT trajectory.

    nuScenes/nuPlan do not ship an explicit command for every frame, so the
    standard practice (UniAD, VAD, and AutoVLA's preprocessing) is to read it
    off the future path: net lateral displacement decides left/right, near-zero
    travel means stop.
    """
    traj = np.asarray(future_traj, dtype=float)
    dist = np.linalg.norm(traj[-1, :2] - traj[0, :2])
    if dist < stop_thresh:
        return "Stop"
    lateral = traj[-1, 1] - traj[0, 1]
    if lateral > lateral_thresh:
        return "Turn Left"
    if lateral < -lateral_thresh:
        return "Turn Right"
    return "Go Straight"


def write_samples(samples, path):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s.to_json()) + "\n")
    return path


def read_samples(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(DrivingSample.from_json(json.loads(line)))
    return out
