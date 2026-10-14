"""Milestone 9: uncertainty-gated adaptive reasoning.

Replaces AutoVLA's learned (implicit) fast/slow choice with an explicit rule on
the calibrated confidence of :mod:`model.uncertainty`:

    confidence >= HIGH            -> fast thinking   (actions immediately)
    LOW <= confidence < HIGH      -> slow thinking   (CoT, then actions)
    confidence < LOW              -> minimal-risk manoeuvre (brake to a stop)

The minimal-risk plan is a comfortable straight-line stop (``MRM_DECEL``) encoded
into the same action-token vocabulary, so it is still a drivable token sequence.

Thresholds are calibrated on a validation set to maximise

    J = mean(driving score) - w * mean(latency / mean slow latency)

with driving score = route_completion * (1 - collided) * clip(1 - ADE / GATE_ADE_SCALE, 0, 1)
(CARLA-style: progress times infraction penalty, then accuracy). Without the progress
factor "always brake to a stop" scores well on any weak policy, because stopping never
collides and its ADE is no worse than a bad plan's -- the gate then degenerates.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import (GATE_ADE_SCALE, GATE_DEFAULT_HIGH, GATE_DEFAULT_LOW, GATE_GRID_STEPS,
                     GATE_LATENCY_WEIGHT, MRM_DECEL, N_ACTION_TOKENS, DT, PHASE2_DIR, RESULTS_DIR)
from .metrics import OpenLoopAccumulator, collision_by_horizon, route_completion

FAST, SLOW, MRM = "fast", "slow", "minimal_risk"
MODES = (FAST, SLOW, MRM)


# ---------------------------------------------------------------------------
# minimal-risk manoeuvre
# ---------------------------------------------------------------------------
def minimal_risk_trajectory(v0: float, n_steps: int = N_ACTION_TOKENS, decel: float = MRM_DECEL,
                            dt: float = DT) -> np.ndarray:
    """(n_steps+1, 3) straight-ahead constant-deceleration stop from ``v0`` m/s."""
    t = np.arange(n_steps + 1) * dt
    t_stop = max(v0, 0.0) / decel
    tc = np.minimum(t, t_stop)
    x = max(v0, 0.0) * tc - 0.5 * decel * tc ** 2
    return np.stack([x, np.zeros_like(x), np.zeros_like(x)], 1)


def minimal_risk_plan(v0: float, codec) -> tuple[list[int], np.ndarray]:
    """Action ids and the *decoded* (quantised) trajectory of the stop manoeuvre."""
    ids = codec.encode(minimal_risk_trajectory(v0)).tolist()
    return ids, codec.decode(ids)


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------
@dataclass
class ConfidenceGate:
    high: float = GATE_DEFAULT_HIGH
    low: float = GATE_DEFAULT_LOW

    def __post_init__(self):
        if not 0.0 <= self.low <= self.high <= 1.0:
            raise ValueError(f"need 0 <= low <= high <= 1, got low={self.low}, high={self.high}")

    def decide(self, confidence: float) -> str:
        if confidence >= self.high:
            return FAST
        if confidence >= self.low:
            return SLOW
        return MRM

    def decide_many(self, confidence) -> np.ndarray:
        c = np.asarray(confidence, float)
        return np.where(c >= self.high, FAST, np.where(c >= self.low, SLOW, MRM))

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(dict(high=self.high, low=self.low)))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    @classmethod
    def calibrate(cls, confidence, outcomes: "ModeOutcomes", latency_weight: float = GATE_LATENCY_WEIGHT,
                  overhead_s=None, steps: int = GATE_GRID_STEPS, min_slow_fraction: float = 0.0):
        """Grid-search (low, high) over confidence quantiles to maximise J.

        ``min_slow_fraction`` > 0 forces a non-empty slow band (at least that share
        of the calibration scenes). The unconstrained optimum can legitimately have
        *no* slow band when slow thinking does not beat fast on the validation set;
        the constraint exists to exercise the three-way gate, not because it is optimal.
        """
        conf = np.asarray(confidence, float)
        qs = np.unique(np.quantile(conf, np.linspace(0.0, 1.0, steps)))
        cands = np.unique(np.concatenate([[0.0, 1.0 + 1e-9], qs]))
        score, lat = outcomes.score_matrix(), outcomes.latency
        overhead = np.zeros(len(conf)) if overhead_s is None else np.asarray(overhead_s, float)
        norm = max(float(lat[:, 1].mean()), 1e-9)
        rows = np.arange(len(conf))
        best = (-np.inf, 0.0, 0.0)
        for lo in cands:
            for hi in cands:
                if hi < lo:
                    continue
                choice = np.where(conf >= hi, 0, np.where(conf >= lo, 1, 2))
                if min_slow_fraction and (choice == 1).mean() < min_slow_fraction:
                    continue
                J = score[rows, choice].mean() - latency_weight * ((lat[rows, choice] + overhead) / norm).mean()
                if J > best[0]:
                    best = (J, lo, hi)
        gate = cls(high=float(min(best[2], 1.0)), low=float(min(best[1], 1.0)))
        gate.objective = float(best[0])
        return gate


# ---------------------------------------------------------------------------
# per-mode outcomes (computed once, then any gate is just indexing)
# ---------------------------------------------------------------------------
@dataclass
class ModeOutcomes:
    ids: list
    trajs: np.ndarray            # (N, 3, steps+1, 3)   modes ordered (fast, slow, minimal_risk)
    gt: np.ndarray               # (N, steps+1, 3)
    obstacles: list              # per-item (steps+1, M, 5) or None
    latency: np.ndarray          # (N, 3) seconds
    cot_tokens: np.ndarray       # (N, 3)
    hard: np.ndarray             # (N,)
    ade_scale: float = GATE_ADE_SCALE   # metres at which the accuracy factor reaches 0

    def __len__(self):
        return len(self.ids)

    def ade(self) -> np.ndarray:
        return np.linalg.norm(self.trajs[..., 1:, :2] - self.gt[:, None, 1:, :2], axis=-1).mean(-1)

    def collided(self) -> np.ndarray:
        out = np.zeros((len(self), 3), bool)
        for i in range(len(self)):
            for m in range(3):
                out[i, m] = collision_by_horizon(self.trajs[i, m], self.obstacles[i])[3.0]
        return out

    def route(self) -> np.ndarray:
        return np.array([[route_completion(self.trajs[i, m], self.gt[i]) for m in range(3)]
                         for i in range(len(self))])

    def score_matrix(self) -> np.ndarray:
        return (self.route() * (~self.collided())
                * np.clip(1.0 - self.ade() / self.ade_scale, 0.0, 1.0))


def collect_mode_outcomes(bb, items, codec, cot_max_tokens: int | None = None,
                          ade_scale: float = GATE_ADE_SCALE) -> ModeOutcomes:
    from .inference import generate_plan

    kw = {} if cot_max_tokens is None else dict(cot_max_tokens=cot_max_tokens)
    trajs, lat, cot = [], [], []
    for it in items:
        row_t, row_l, row_c = [], [], []
        for mode in (FAST, SLOW):
            plan = generate_plan(bb, it.messages, mode=mode, **kw)
            row_t.append(codec.decode(plan.action_ids)); row_l.append(plan.latency_s); row_c.append(plan.n_cot_tokens)
        _, mrm = minimal_risk_plan(it.sample.velocity, codec)
        row_t.append(mrm); row_l.append(0.0); row_c.append(0)
        trajs.append(np.stack(row_t)); lat.append(row_l); cot.append(row_c)
    return ModeOutcomes([it.sample.sample_id for it in items], np.stack(trajs),
                        np.stack([np.asarray(it.sample.future_traj, float) for it in items]),
                        [it.obstacles for it in items], np.array(lat), np.array(cot),
                        np.array([it.hard for it in items]), ade_scale)


# ---------------------------------------------------------------------------
# evaluation of policies over outcomes
# ---------------------------------------------------------------------------
def policy_report(out: ModeOutcomes, choice: np.ndarray, overhead_s=None) -> dict:
    """Metrics for a per-scene mode ``choice`` (indices 0/1/2), split all/easy/hard."""
    overhead = np.zeros(len(out)) if overhead_s is None else np.asarray(overhead_s, float)
    rows = np.arange(len(out))
    res = {}
    for name, mask in (("all", np.ones(len(out), bool)), ("easy", ~out.hard), ("hard", out.hard)):
        if not mask.any():
            continue
        acc = OpenLoopAccumulator()
        for i in np.where(mask)[0]:
            acc.add(out.trajs[i, choice[i]], out.gt[i], out.obstacles[i],
                    out.latency[i, choice[i]] + overhead[i], out.cot_tokens[i, choice[i]])
        s = acc.summary()
        c = choice[mask]
        s["frac_fast"], s["frac_slow"], s["frac_mrm"] = (float((c == k).mean()) for k in range(3))
        s["driving_score"] = float(out.score_matrix()[rows[mask], c].mean())
        res[name] = s
    return res


def evaluate_policies(out: ModeOutcomes, conf_full, conf_ale, gate: ConfidenceGate,
                      est_overhead, feat_overhead, auto_choice=None, seed: int = 0) -> dict:
    """Always-fast / always-slow / random / learned / gated (+ aleatoric-only gate)."""
    n = len(out)
    rng = np.random.default_rng(seed)
    idx = {FAST: 0, SLOW: 1, MRM: 2}
    pols = {
        "always_fast": np.zeros(n, int),
        "always_slow": np.ones(n, int),
        "random": rng.integers(0, 2, n),
    }
    if auto_choice is not None:
        pols["learned_switch"] = np.asarray(auto_choice, int)
    res = {k: policy_report(out, v) for k, v in pols.items()}
    g_full = np.array([idx[m] for m in gate.decide_many(conf_full)])
    res["gated_full_conf"] = policy_report(out, g_full, est_overhead)
    g_ale = np.array([idx[m] for m in gate.decide_many(conf_ale)])
    res["gated_aleatoric_only"] = policy_report(out, g_ale, feat_overhead)
    return res


def latency_cut(report: dict, split: str = "easy", ref: str = "always_slow", pol: str = "gated_full_conf") -> float:
    a, b = report[pol][split]["latency_s"], report[ref][split]["latency_s"]
    return 1.0 - a / b if b > 0 else float("nan")


def epistemic_quantile_modes(epi, conf, gate: ConfidenceGate, q: float = 0.75) -> dict:
    """Mode mix on epistemic-high (top quartile by default) vs the rest."""
    epi, modes = np.asarray(epi, float), gate.decide_many(conf)
    hi = epi >= np.quantile(epi, q)
    mix = lambda m: {k: float((m == k).mean()) for k in MODES} if len(m) else {}
    return dict(epistemic_high=mix(modes[hi]), epistemic_low=mix(modes[~hi]), threshold=float(np.quantile(epi, q)))


def format_report(report: dict, split: str) -> str:
    cols = ["L2@1s", "L2@2s", "L2@3s", "Col@3s", "driving_score", "frac_fast", "frac_slow", "frac_mrm", "latency_s"]
    head = "| policy | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1)
    lines = []
    for p, r in report.items():
        if not p.startswith("_") and split in r:
            s = r[split]
            lines.append(f"| {p} | " + " | ".join(
                f"{s[c] * 1000:.1f} ms" if c == "latency_s" else f"{s[c]:.3f}" for c in cols) + " |")
    return head + "\n" + "\n".join(lines)


def main():
    from .config import BackboneConfig
    from .evaluate import build_eval_items
    from .runtime import load_action_codec, make_backbone
    from .uncertainty import (SceneOutputs, UncertaintyModel, collect_scene_outputs, fuse_confidence)

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--standin", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--holdout", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=250)
    ap.add_argument("--n-test", type=int, default=250)
    ap.add_argument("--cot-max-tokens", type=int, default=None)
    ap.add_argument("--latency-weight", type=float, default=GATE_LATENCY_WEIGHT)
    ap.add_argument("--ade-scale", type=float, default=GATE_ADE_SCALE,
                    help="metres at which the driving-score accuracy factor hits 0 (rescale for weak stand-ins)")
    ap.add_argument("--min-slow-fraction", type=float, default=0.15,
                    help="also report a gate constrained to send at least this share to slow thinking")
    args = ap.parse_args()
    tag = args.tag or ("standin" if args.standin else "model")
    bb = make_backbone(args.standin, BackboneConfig(), args.checkpoint)
    codec = load_action_codec()
    unc = UncertaintyModel.load(PHASE2_DIR / f"unc_head_{tag}.pt")
    n_all = args.n_val + args.n_test
    # val/test scenes are the *tail* of the hold-out, disjoint from the first 2500
    # scenes that `model.uncertainty` uses to train/calibrate/test its head.
    items = build_eval_items(None, holdout=args.holdout)[-n_all:]
    val_items, test_items = items[:args.n_val], items[args.n_val:]
    sc = {}
    for name, its in (("val", val_items), ("test", test_items)):
        scene = collect_scene_outputs(bb, its, codec, unc.T)
        out = collect_mode_outcomes(bb, its, codec, args.cot_max_tokens, args.ade_scale)
        ale = unc.head.sigma2(__import__("torch").as_tensor(scene.features, dtype=__import__("torch").float32)).numpy()
        sc[name] = (scene, out, ale)
    sv, ov, av = sc["val"]
    conf_val = unc.confidence(av, sv.epi)
    gate = ConfidenceGate.calibrate(conf_val, ov, args.latency_weight, overhead_s=sv.est_s)
    st, ot, at = sc["test"]
    rep = evaluate_policies(ot, unc.confidence(at, st.epi), unc.confidence(at, 0.0), gate, st.est_s, st.feat_s)
    rep["_gate"] = dict(high=gate.high, low=gate.low, objective=getattr(gate, "objective", None))
    rep["_epistemic_modes"] = epistemic_quantile_modes(st.epi, unc.confidence(at, st.epi), gate)
    # constrained variant: forces a slow band so the three-way behaviour is visible
    gate_c = ConfidenceGate.calibrate(conf_val, ov, args.latency_weight, overhead_s=sv.est_s,
                                      min_slow_fraction=args.min_slow_fraction)
    rep_c = evaluate_policies(ot, unc.confidence(at, st.epi), unc.confidence(at, 0.0), gate_c, st.est_s, st.feat_s)
    rep["gated_min_slow_band"] = rep_c["gated_full_conf"]
    rep["_gate_min_slow"] = dict(high=gate_c.high, low=gate_c.low, objective=getattr(gate_c, "objective", None))
    rep["_epistemic_modes_min_slow"] = epistemic_quantile_modes(st.epi, unc.confidence(at, st.epi), gate_c)
    rep["_latency_cut_easy_vs_slow"] = latency_cut(rep, "easy", "always_slow")
    rep["_latency_cut_easy_vs_random"] = latency_cut(rep, "easy", "random")
    gate.save(PHASE2_DIR / f"gate_{tag}.json")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"gating_{tag}.json").write_text(json.dumps(rep, indent=1, default=float))
    for split in ("easy", "hard"):
        print(f"\n[{split}]\n" + format_report(rep, split))
    print("\ngate (unconstrained optimum):", rep["_gate"], "| with forced slow band:", rep["_gate_min_slow"])
    print("epistemic-high vs low mode mix (forced slow band):", json.dumps(rep["_epistemic_modes_min_slow"], default=float))
    print("epistemic-high vs low mode mix:", json.dumps(rep["_epistemic_modes"], default=float))
    print(f"easy-split latency cut vs always-slow {rep['_latency_cut_easy_vs_slow']:.1%}, vs random {rep['_latency_cut_easy_vs_random']:.1%}")


if __name__ == "__main__":
    main()
