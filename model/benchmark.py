"""Milestone 11: full evaluation + the four-row ablation table.

Rows (each is a different policy and/or deployment stack):

  1. SFT only                           SFT policy, learned fast/slow switch
  2. + GRPO (no uncertainty head)       GRPO policy (no calibration reward), learned switch
  3. + uncertainty head (no calib rwd)  that GRPO policy + Bayesian head + explicit gate
  4. + calibration reward (FULL MODEL)  GRPO-with-r_calibration policy + head + gate

Shared protocol: scenes come from the SFT hold-out (never trained on by SFT or
GRPO), split into disjoint head-train / calibration / gate-validation / test
partitions; every row is scored on the same test scenes with the same failure
threshold tau. ECE/AUROC describe confidence in the direct (fast-mode) plan:
rows 1-2 use the policy's own action-token probability (uncalibrated -- the only
confidence they have), rows 3-4 the calibrated fused confidence of the head.

``--standin`` runs everything on the tiny CPU stand-in with synthetic agents; the
CSV's ``provenance`` column says so. Numbers from it validate the pipeline only.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autovla.schema import DrivingSample

from .calibration import bootstrap_ci, compute_auroc, compute_ece
from .config import (BackboneConfig, EVAL_HORIZONS_S, GATE_ADE_SCALE, MC_DROPOUT_T, PHASE2_DIR, RESULTS_DIR)
from .evaluate import EvalItem, build_eval_items
from .gating import (ConfidenceGate, ModeOutcomes, collect_mode_outcomes, epistemic_quantile_modes,
                     policy_report)
from .grpo import GRPOConfig, sample_rollout
from .inference import PromptEncoder
from .metrics import collision_by_horizon, l2_at_horizons
from .runtime import load_action_codec, make_backbone
from .uncertainty import (SceneOutputs, collect_scene_outputs, fit_and_evaluate)

H3 = max(EVAL_HORIZONS_S)


@dataclass
class Variant:
    name: str
    checkpoint: str
    head: bool                    # attach the uncertainty head + explicit gate
    calibration_reward: bool      # (informational) policy was trained with r_calibration


@dataclass
class Partitions:
    head_train: list
    calib: list
    gate_val: list
    test: list


def make_partitions(holdout: int, n_head: int, n_calib: int, n_val: int, n_test: int, seed: int = 0) -> Partitions:
    pool = build_eval_items(None, holdout=holdout)
    need = n_head + n_calib + n_val + n_test
    if len(pool) < need:
        raise SystemExit(f"hold-out has {len(pool)} scenes but {need} are needed; raise --holdout")
    perm = np.random.default_rng(seed).permutation(len(pool))
    take = lambda a, b: [pool[i] for i in perm[a:b]]
    a, b, c = n_head, n_head + n_calib, n_head + n_calib + n_val
    return Partitions(take(0, a), take(a, b), take(b, c), take(c, c + n_test))


def _per_scene_plan_metrics(trajs, gt, obstacles) -> dict:
    l2 = np.array([l2_at_horizons(t, g)[H3] for t, g in zip(trajs, gt)])
    col = np.array([collision_by_horizon(t, o)[H3] for t, o in zip(trajs, obstacles)], float)
    return dict(l2=l2, col=col)


def learned_switch_plans(bb, items, codec, cot_max_tokens: int | None):
    """Greedy plans where the *policy* picks fast/slow (AutoVLA's implicit gate)."""
    enc = PromptEncoder(bb.tokenizer, bb.processor)
    cfg = GRPOConfig(**({} if cot_max_tokens is None else dict(cot_max_tokens=cot_max_tokens)))
    from .inference import generate_plan

    trajs, lat, cot, modes = [], [], [], []
    for it in items:
        plan = generate_plan(bb, it.messages, mode="auto", cot_max_tokens=cfg.cot_max_tokens, encoder=enc)
        trajs.append(codec.decode(plan.action_ids)); lat.append(plan.latency_s)
        cot.append(plan.n_cot_tokens); modes.append(plan.mode)
    return np.stack(trajs), np.array(lat), np.array(cot), np.array(modes)


def evaluate_variant(v: Variant, parts: Partitions, tau, standin: bool, T: int,
                     cot_max_tokens: int | None, log=print, ade_scale: float = GATE_ADE_SCALE) -> dict:
    bb = make_backbone(standin, BackboneConfig(), v.checkpoint)
    codec = load_action_codec()
    test = parts.test
    gt = np.stack([np.asarray(i.sample.future_traj, float) for i in test])
    obstacles = [i.obstacles for i in test]
    row = dict(variant=v.name, uncertainty_head=v.head, calibration_reward=v.calibration_reward, n_test=len(test))

    log(f"[{v.name}] scoring test scenes ...")
    sc_test = collect_scene_outputs(bb, test, codec, T if v.head else 0)
    sc_calib = collect_scene_outputs(bb, parts.calib, codec, T if v.head else 0)
    tau_val = float(np.median(sc_calib.ade)) if tau == "auto" else float(tau)
    row["tau"] = tau_val
    success = (sc_test.ade <= tau_val).astype(float)
    row["failure_rate"] = float(1 - success.mean())

    if not v.head:
        trajs, lat, cot, modes = learned_switch_plans(bb, test, codec, cot_max_tokens)
        conf = sc_test.pconf
        row.update(confidence_source="policy_action_token_prob", temperature=1.0)
        row.update(frac_fast=float((modes == "fast").mean()), frac_slow=float((modes == "slow").mean()),
                   frac_mrm=0.0, latency_s=float(lat.mean()), cot_tokens=float(cot.mean()))
        chosen = trajs
    else:
        import torch

        log(f"[{v.name}] fitting head on {len(parts.head_train)} scenes ...")
        sc_head = collect_scene_outputs(bb, parts.head_train, codec, 0)
        sc_val = collect_scene_outputs(bb, parts.gate_val, codec, T)
        # the head is fit on head_train, temperature-scaled on calib; fit_and_evaluate also scores
        # `sc_test`, which is the table's test partition.
        unc, rep = fit_and_evaluate(sc_head, sc_calib, sc_test, tau_val)
        ale = lambda s: unc.head.sigma2(torch.as_tensor(s.features, dtype=torch.float32)).numpy()
        conf = unc.confidence(ale(sc_test), sc_test.epi)
        row.update(confidence_source="fused_head_calibrated", temperature=unc.scaler.temperature,
                   ece_uncalibrated=rep["uncalibrated"]["ece"], auroc_aleatoric=rep["auroc_failure"]["aleatoric_only"],
                   auroc_epistemic=rep["auroc_failure"]["epistemic_only"])
        log(f"[{v.name}] collecting per-mode outcomes ...")
        out_val = collect_mode_outcomes(bb, parts.gate_val, codec, cot_max_tokens, ade_scale)
        out_test = collect_mode_outcomes(bb, test, codec, cot_max_tokens, ade_scale)
        gate = ConfidenceGate.calibrate(unc.confidence(ale(sc_val), sc_val.epi), out_val, overhead_s=sc_val.est_s)
        choice = np.array([{"fast": 0, "slow": 1, "minimal_risk": 2}[m] for m in gate.decide_many(conf)])
        rep_g = policy_report(out_test, choice, sc_test.est_s)["all"]
        row.update(frac_fast=rep_g["frac_fast"], frac_slow=rep_g["frac_slow"], frac_mrm=rep_g["frac_mrm"],
                   latency_s=rep_g["latency_s"], cot_tokens=rep_g["cot_tokens"], gate_high=gate.high, gate_low=gate.low)
        chosen = out_test.trajs[np.arange(len(test)), choice]
        row["hard_vs_easy_unc"] = [rep["hard_vs_easy"]["mean_unc_hard"], rep["hard_vs_easy"]["mean_unc_easy"]]
        row["epistemic_modes"] = epistemic_quantile_modes(sc_test.epi, conf, gate)

    per = _per_scene_plan_metrics(chosen, gt, obstacles)
    for h in EVAL_HORIZONS_S:
        row[f"L2@{h:g}s"] = float(np.mean([l2_at_horizons(t, g)[h] for t, g in zip(chosen, gt)]))
        row[f"Col@{h:g}s"] = float(np.mean([collision_by_horizon(t, o)[h] for t, o in zip(chosen, obstacles)]))
    row["L2@3s_ci"] = bootstrap_ci(np.mean, per["l2"], n=500)
    row["Col@3s_ci"] = bootstrap_ci(np.mean, per["col"], n=500)
    from .metrics import route_completion
    row["route_completion"] = float(np.mean([route_completion(t, g) for t, g in zip(chosen, gt)]))
    row["ECE"] = compute_ece(conf, success)
    row["AUROC"] = compute_auroc(1 - np.asarray(conf), 1 - success)
    row["latency_ms"] = row.pop("latency_s") * 1000.0
    return row


CSV_COLS = ["variant", "L2@1s", "L2@2s", "L2@3s", "Col@1s", "Col@2s", "Col@3s", "route_completion",
            "ECE", "AUROC", "latency_ms", "frac_fast", "frac_slow", "frac_mrm", "cot_tokens",
            "uncertainty_head", "calibration_reward", "confidence_source", "tau", "failure_rate", "n_test",
            "L2@3s_ci", "Col@3s_ci", "provenance"]


def write_table(rows: list[dict], provenance: str, out_dir=RESULTS_DIR) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ablation_table.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for r in rows:
            r = dict(r, provenance=provenance)
            w.writerow([("%.5f" % r[c] if isinstance(r[c], float) else
                         ("[%.4f, %.4f]" % tuple(r[c]) if isinstance(r[c], (tuple, list)) else r[c]))
                        for c in CSV_COLS])
    (out_dir / "ablation_table.json").write_text(json.dumps(rows, indent=1, default=float))
    md = ["| Model variant | L2@3s (m) ↓ | Collision@3s ↓ | ECE ↓ | Latency (ms) ↓ | AUROC ↑ |", "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['variant']} | {r['L2@3s']:.3f} | {r['Col@3s']:.3f} | {r['ECE']:.3f} | "
                  f"{r['latency_ms']:.1f} | {r['AUROC']:.3f} |")
    (out_dir / "ablation_table.md").write_text("\n".join(md) + f"\n\nprovenance: {provenance}\n")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--standin", action="store_true")
    ap.add_argument("--sft", required=True)
    ap.add_argument("--grpo-nocal", required=True)
    ap.add_argument("--grpo-cal", required=True)
    ap.add_argument("--holdout", type=int, default=3000)
    ap.add_argument("--n-head", type=int, default=1000)
    ap.add_argument("--n-calib", type=int, default=400)
    ap.add_argument("--n-val", type=int, default=250)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--T", type=int, default=MC_DROPOUT_T)
    ap.add_argument("--tau", default="auto")
    ap.add_argument("--cot-max-tokens", type=int, default=None)
    ap.add_argument("--ade-scale", type=float, default=GATE_ADE_SCALE)
    args = ap.parse_args()

    parts = make_partitions(args.holdout, args.n_head, args.n_calib, args.n_val, args.n_test)
    tau = args.tau if args.tau == "auto" else float(args.tau)
    variants = [
        Variant("SFT only (no GRPO)", args.sft, False, False),
        Variant("+ GRPO (no uncertainty head)", args.grpo_nocal, False, False),
        Variant("+ uncertainty head (no calib reward)", args.grpo_nocal, True, False),
        Variant("+ calibration reward (FULL MODEL)", args.grpo_cal, True, True),
    ]
    # one tau for every row: the SFT policy's median calibration ADE
    if tau == "auto":
        bb0 = make_backbone(args.standin, BackboneConfig(), args.sft)
        tau = float(np.median(collect_scene_outputs(bb0, parts.calib, load_action_codec(), 0).ade))
        print(f"failure threshold tau = {tau:.3f} m (median ADE of the SFT policy on the calibration split)")
    rows = [evaluate_variant(v, parts, tau, args.standin, args.T, args.cot_max_tokens, ade_scale=args.ade_scale)
            for v in variants]
    prov = ("tiny CPU stand-in model on the Phase-I synthetic corpus with synthetic agent boxes (pipeline "
            "validation, NOT Qwen2.5-VL-3B / nuScenes results)" if args.standin else "qwen2.5-vl-3b")
    print("wrote", write_table(rows, prov))
    print(Path(RESULTS_DIR / "ablation_table.md").read_text())


if __name__ == "__main__":
    main()
