"""Milestone 5: populate ``DrivingSample.reasoning`` with real four-field CoT.

Two sources, in priority order:

1. **DriveLM-nuScenes** QA (``v1_1_train_nus.json``): perception / prediction /
   planning / behavior question-answer pairs and ``key_object_infos`` are
   re-cast into scene_description / critical_objects / agent_intentions /
   driving_decision. Joined to samples on the nuScenes ``sample_token``.
2. **Trajectory-statistics template** (fallback when DriveLM is absent, as with
   the synthetic corpus): the command plus speed / acceleration / curvature of
   the ground-truth plan are verbalised. This is *ego-motion-derived* text, not
   perception -- it never claims to see anything -- and is tagged
   ``source="trajectory_template"`` so downstream code can tell the two apart.

Placeholder reasoning left by Phase I (``[pending]``) counts as *no* CoT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from autovla.config import DT
from autovla.schema import DrivingSample, Reasoning, read_samples

from .config import (COT_JSONL, COT_MIN_FRACTION, MAX_SENTENCES_PER_FIELD,
                     UNIFIED_JSONL)

PLACEHOLDER_RE = re.compile(r"\[(pending|to be annotated[^\]]*)\]", re.I)
_OBJ_TAG = re.compile(r"<(c\d+),(CAM_[A-Z_]+),[\d.]+,[\d.]+>")
CAM_WORDS = {"CAM_FRONT": "front", "CAM_FRONT_LEFT": "front-left", "CAM_FRONT_RIGHT": "front-right",
             "CAM_BACK": "rear", "CAM_BACK_LEFT": "rear-left", "CAM_BACK_RIGHT": "rear-right"}

# Thresholds for verbalising kinematics (m/s, m/s^2, m, rad); named, not inlined.
STATIONARY_SPEED = 0.3
SLOW_SPEED = 3.0
FAST_SPEED = 10.0
ACCEL_THRESH = 0.7
LATERAL_THRESH = 1.5
HEADING_THRESH = 0.35


def is_placeholder(r: Reasoning) -> bool:
    return r.is_empty() or any(PLACEHOLDER_RE.search(t) for t in vars(r).values())


# ---------------------------------------------------------------------------
# DriveLM -> four fields
# ---------------------------------------------------------------------------
def _clean_tags(text: str) -> str:
    return _OBJ_TAG.sub(lambda m: f"object {m.group(1)} ({CAM_WORDS.get(m.group(2), m.group(2))} camera)", text).strip()


def _answers(qa: dict, key: str, limit: int) -> list[str]:
    out = []
    for item in qa.get(key, []) or []:
        a = _clean_tags(str(item.get("A", "")))
        if a and a not in out:
            out.append(a if a.endswith((".", "!", "?")) else a + ".")
        if len(out) >= limit:
            break
    return out


def reformat_drivelm_frame(frame: dict, limit: int = MAX_SENTENCES_PER_FIELD) -> Reasoning | None:
    """One DriveLM key-frame dict -> Reasoning (None if too sparse to be useful)."""
    qa = frame.get("QA", {}) or {}
    scene = _answers(qa, "perception", limit)

    objects = []
    for tag, info in (frame.get("key_object_infos") or {}).items():
        desc = info.get("Visual_description") or info.get("Category") or ""
        if desc:
            status = f" ({info['Status'].lower()})" if info.get("Status") else ""
            objects.append(f"{_clean_tags(tag)}: {desc.rstrip('.')}{status}.")
        if len(objects) >= limit:
            break
    if not objects:
        objects = [a for a in _answers(qa, "perception", limit * 2) if a.lower().startswith("there")][:limit]

    intent = _answers(qa, "prediction", limit)
    decision = _answers(qa, "behavior", 2) + _answers(qa, "planning", 1)
    fields = [" ".join(scene), " ".join(objects), " ".join(intent), " ".join(decision)]
    if sum(bool(f) for f in fields) < 3 or not fields[3]:
        return None
    return Reasoning(*fields)


def load_drivelm(path) -> dict[str, dict]:
    """DriveLM json -> {frame_token: frame_dict}."""
    with open(path) as f:
        data = json.load(f)
    frames = {}
    for scene in data.values():
        for token, frame in (scene.get("key_frames") or {}).items():
            frames[token] = frame
    return frames


# ---------------------------------------------------------------------------
# trajectory-statistics template
# ---------------------------------------------------------------------------
@dataclass
class MotionStats:
    v0: float
    v_end: float
    accel: float            # mean longitudinal acceleration over the horizon
    distance: float
    lateral: float          # signed, + = left
    heading: float          # net yaw change, rad
    stop_go: bool           # starts (nearly) stationary and ends moving
    stops: bool             # ends (nearly) stationary


def motion_stats(s: DrivingSample) -> MotionStats:
    traj = np.asarray(s.future_traj, dtype=float)
    seg = np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1) / DT
    v0 = float(s.velocity)
    v_end = float(seg[-2:].mean()) if len(seg) else v0
    T = DT * len(seg)
    return MotionStats(
        v0=v0, v_end=v_end, accel=(v_end - v0) / max(T, DT),
        distance=float(seg.sum() * DT), lateral=float(traj[-1, 1] - traj[0, 1]),
        heading=float(traj[-1, 2] - traj[0, 2]),
        stop_go=v0 < STATIONARY_SPEED and v_end > SLOW_SPEED / 2,
        stops=v_end < STATIONARY_SPEED)


def _pick(sample_id: str, options: list[str], salt: str) -> str:
    h = int(hashlib.md5(f"{sample_id}:{salt}".encode()).hexdigest(), 16)
    return options[h % len(options)]


def synthesize_cot(s: DrivingSample) -> Reasoning:
    """Verbalise command + trajectory statistics into the four fields."""
    m = motion_stats(s)
    pick = lambda salt, opts: _pick(s.sample_id, opts, salt)
    speed_word = ("stationary" if m.v0 < STATIONARY_SPEED else "moving slowly" if m.v0 < SLOW_SPEED
                  else "travelling at moderate speed" if m.v0 < FAST_SPEED else "travelling at high speed")

    if abs(m.heading) > HEADING_THRESH or abs(m.lateral) > LATERAL_THRESH * 2:
        turn = "left" if m.lateral > 0 or m.heading > 0 else "right"
        action = f"turn {turn}"
        road = f"The planned path curves to the {turn}, suggesting a junction or bend ahead."
    elif abs(m.lateral) > LATERAL_THRESH:
        side = "left" if m.lateral > 0 else "right"
        action = f"shift towards the {side} lane"
        road = f"The planned path drifts {abs(m.lateral):.1f} m to the {side}, suggesting a lane change or lane-offset."
    else:
        action = None
        road = "The planned path runs straight along the current lane."

    if m.stops and m.v0 < STATIONARY_SPEED:
        longi = "stay stopped"
    elif m.stops:
        longi = "brake to a stop"
    elif m.stop_go:
        longi = "pull away from rest"
    elif m.accel > ACCEL_THRESH:
        longi = "accelerate"
    elif m.accel < -ACCEL_THRESH:
        longi = "decelerate"
    else:
        longi = "hold the current speed"
    meta = f"{action} and {longi}" if action else longi

    scene = pick("scene", [
        f"The ego vehicle is {speed_word} ({m.v0:.1f} m/s). {road}",
        f"Ego motion shows the vehicle {speed_word} at {m.v0:.1f} m/s. {road}",
    ]) + f" The navigation instruction is '{s.command}'. No camera imagery is available, so this description is inferred from ego motion only."

    if longi in ("brake to a stop", "decelerate"):
        crit = "The braking profile implies a constraint ahead, such as a lead vehicle, a stop line or a signal; no tracked agents are provided."
    elif longi == "pull away from rest":
        crit = "The vehicle leaves a standstill, so the space immediately ahead must be clear; no tracked agents are provided."
    elif longi == "stay stopped":
        crit = "Nothing in the ego motion indicates a free path, so the vehicle should keep waiting; no tracked agents are provided."
    elif action:
        crit = "The lane geometry of the manoeuvre and any agents sharing the target lane or junction are critical; no tracked agents are provided."
    else:
        crit = "The lane ahead and the gap to any lead vehicle are critical; no tracked agents are provided."

    intent = pick("intent", [
        "Surrounding road users are assumed to keep their current motion and right-of-way, so no conflicting manoeuvre is expected within the horizon.",
        "Other agents are expected to continue as they are; the plan leaves room for them to behave normally.",
    ])

    decision = (f"Plan to {meta}: {m.distance:.1f} m over {len(np.asarray(s.future_traj)) - 1} steps, "
                f"speed {m.v0:.1f} -> {m.v_end:.1f} m/s, consistent with the instruction '{s.command}'.")
    return Reasoning(scene, crit, intent, decision)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
@dataclass
class AnnotationStats:
    n: int = 0
    drivelm: int = 0
    template: int = 0
    kept_existing: int = 0
    placeholders_left: int = 0
    fraction: float = 0.0
    sources: dict = field(default_factory=dict)


def annotate_samples(samples, drivelm: dict[str, dict] | None = None,
                     min_fraction: float = COT_MIN_FRACTION):
    """Return ``({sample_id: (source, Reasoning)}, AnnotationStats)``.

    * every sample with a usable DriveLM frame gets DriveLM CoT;
    * every sample Phase I already marked slow (placeholder reasoning) gets a
      template CoT unless DriveLM covered it, so its thinking mode is unchanged;
    * if coverage is still below ``min_fraction``, further samples are promoted
      to slow-thinking in a deterministic (hash) order until it is met.
    """
    drivelm = drivelm or {}
    ann: dict[str, tuple[str, Reasoning]] = {}
    st = AnnotationStats(n=len(samples))
    for s in samples:
        if s.sample_id in drivelm:
            r = reformat_drivelm_frame(drivelm[s.sample_id])
            if r is not None:
                ann[s.sample_id] = ("drivelm", r)
                continue
        if s.has_reasoning and not is_placeholder(s.reasoning):
            ann[s.sample_id] = ("existing", s.reasoning)
        elif s.has_reasoning:
            ann[s.sample_id] = ("trajectory_template", synthesize_cot(s))

    need = int(np.ceil(min_fraction * len(samples))) - len(ann)
    if need > 0:
        rest = sorted((s for s in samples if s.sample_id not in ann),
                      key=lambda s: hashlib.md5(s.sample_id.encode()).hexdigest())
        for s in rest[:need]:
            ann[s.sample_id] = ("trajectory_template", synthesize_cot(s))

    for src, _ in ann.values():
        st.sources[src] = st.sources.get(src, 0) + 1
    st.drivelm, st.template = st.sources.get("drivelm", 0), st.sources.get("trajectory_template", 0)
    st.kept_existing = st.sources.get("existing", 0)
    st.fraction = len(ann) / max(len(samples), 1)
    return ann, st


def write_annotations(ann, path=COT_JSONL):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sid, (src, r) in ann.items():
            f.write(json.dumps(dict(sample_id=sid, source=src, reasoning=vars(r))) + "\n")
    return path


def read_annotations(path=COT_JSONL) -> dict[str, tuple[str, Reasoning]]:
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d["sample_id"]] = (d["source"], Reasoning(**d["reasoning"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--samples", default=str(UNIFIED_JSONL))
    ap.add_argument("--drivelm", default=None, help="DriveLM v1_1_train_nus.json (optional)")
    ap.add_argument("--out", default=str(COT_JSONL))
    ap.add_argument("--min-fraction", type=float, default=COT_MIN_FRACTION)
    args = ap.parse_args()

    samples = read_samples(args.samples)
    drivelm = load_drivelm(args.drivelm) if args.drivelm else None
    ann, st = annotate_samples(samples, drivelm, args.min_fraction)
    write_annotations(ann, args.out)
    print(f"{len(ann)}/{st.n} samples carry real CoT ({st.fraction:.1%}); sources {st.sources}")
    if drivelm is None:
        print("DriveLM not provided: CoT is trajectory-derived (no perception content).")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
