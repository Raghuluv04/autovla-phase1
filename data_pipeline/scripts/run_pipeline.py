#!/usr/bin/env python
"""End-to-end Phase I data pipeline: corpus -> codebook -> tokenizer -> SFT set.

    python scripts/run_pipeline.py --source synthetic --n-traj 4000
    python scripts/run_pipeline.py --source nuscenes --nuscenes-root data/nuscenes

Everything downstream of the adapter is dataset-agnostic, so switching sources
changes nothing but the first stage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autovla import stats, viz
from autovla.build_dataset import build_sft_records, write_action_vocab, write_records
from autovla.codebook import Codebook, build_codebook
from autovla.config import CODEBOOK_SIZE, DT, N_ACTION_TOKENS
from autovla.geometry import relative
from autovla.kinematics import feasibility_report
from autovla.schema import DrivingSample, infer_command, write_samples
from autovla.tokenizer import ActionTokenizer, evaluate_tokenizer


def load_corpus(args):
    if args.source == "nuscenes":
        from autovla.adapters.nuscenes import NuScenesReader, available
        if not available(args.nuscenes_root, args.nuscenes_version):
            sys.exit(f"nuScenes not found at {args.nuscenes_root}/"
                     f"{args.nuscenes_version}. Run scripts/download_nuscenes.sh")
        reader = NuScenesReader(args.nuscenes_root, args.nuscenes_version)
        return reader.trajectory_corpus(), reader
    from autovla.adapters.synthetic import build_synthetic_corpus
    return build_synthetic_corpus(args.n_traj, seed=args.seed), None


def corpus_to_samples(corpus, n_future=N_ACTION_TOKENS, stride=2):
    """Sliding-window DrivingSamples from raw trajectories (synthetic path)."""
    out = []
    for tr in corpus:
        poses = np.asarray(tr["poses"], dtype=float)
        speeds = np.asarray(tr["speeds"], dtype=float)
        for i in range(3, len(poses) - n_future, stride):
            cur = poses[i]
            fut = relative(cur, poses[i:i + n_future + 1])
            hist = relative(cur, poses[i - 3:i + 1])
            v = float(speeds[i])
            a = float((speeds[i] - speeds[i - 1]) / DT)
            out.append(DrivingSample(
                sample_id=f"{tr['traj_id']}_{i:03d}",
                dataset="synthetic",
                future_traj=fut, history_traj=hist,
                images={}, velocity=v, acceleration=a,
                command=infer_command(fut),
                extra=dict(maneuver=tr.get("maneuver", "?")),
            ))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="synthetic", choices=["synthetic", "nuscenes"])
    p.add_argument("--n-traj", type=int, default=4000)
    p.add_argument("--codebook-size", type=int, default=CODEBOOK_SIZE)
    p.add_argument("--max-segments", type=int, default=120000)
    p.add_argument("--mode", default="fps", choices=["fps", "threshold"])
    p.add_argument("--eval-traj", type=int, default=2000)
    p.add_argument("--ablation", action="store_true",
                   help="sweep K to reproduce the paper's Table 4 trend")
    p.add_argument("--cot-frac", type=float, default=0.27,
                   help="fraction of samples given CoT (paper: 45.6k/166.3k on nuPlan)")
    p.add_argument("--nuscenes-root", default="data/nuscenes")
    p.add_argument("--nuscenes-version", default="v1.0-mini")
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    figdir = os.path.join(args.outdir, "figures")
    artdir = os.path.join(args.outdir, "artifacts")
    os.makedirs(artdir, exist_ok=True)
    t0 = time.time()

    # -- 1. corpus ---------------------------------------------------------
    print("[1/6] loading corpus ...")
    corpus, reader = load_corpus(args)
    print(f"  {len(corpus)} trajectories from {args.source}")

    # -- 2. unified samples ------------------------------------------------
    print("[2/6] building unified samples ...")
    if reader is not None:
        samples = reader.build_samples()
    else:
        samples = corpus_to_samples(corpus)
    print(f"  {len(samples)} samples")

    # Mark a subset as carrying CoT so the fast/slow mix is visible. Real CoT
    # text is generated in Milestone 4 by the Qwen2.5-VL-72B annotation stage.
    rng = np.random.default_rng(args.seed)
    from autovla.schema import Reasoning
    for s in samples:
        if rng.random() < args.cot_frac:
            s.reasoning = Reasoning(
                scene_description="[to be annotated by the Qwen2.5-VL-72B pipeline]",
                critical_objects="[pending]", agent_intentions="[pending]",
                driving_decision=f"{s.command} at {s.velocity:.1f} m/s.")

    # -- 3. segment bank ---------------------------------------------------
    print("[3/6] cutting 0.5 s motion segments ...")
    segments, seg_labels = stats.segment_bank(corpus, dt=DT,
                                              max_segments=args.max_segments)
    print(f"  {len(segments):,} segments")

    # -- 4. codebook -------------------------------------------------------
    print(f"[4/6] K-disk clustering (K={args.codebook_size}, mode={args.mode}) ...")
    cb = build_codebook(segments, k=args.codebook_size, dt=DT, mode=args.mode,
                        seed=args.seed)
    cb.save(os.path.join(artdir, "action_codebook.npz"))
    feas = feasibility_report(cb.actions,
                              v0=np.linalg.norm(cb.actions[:, :2], axis=1) / DT)
    print(f"  feasible tokens: {feas['feasible_frac']:.1%}  "
          f"max |steer| {feas['max_abs_steer_deg']:.1f} deg")

    # -- 5. tokenizer evaluation ------------------------------------------
    print("[5/6] evaluating tokenizer ...")
    tok = ActionTokenizer(cb)
    eval_trajs = [np.asarray(s.future_traj) for s in samples[: args.eval_traj]]
    ev = evaluate_tokenizer(tok, eval_trajs)
    print(f"  ADE {ev['ade']*100:.2f} cm  FDE {ev['fde']*100:.2f} cm  "
          f"usage {ev['codebook_usage']:.1%}")

    ablation = []
    if args.ablation:
        for k in [128, 256, 512, 1024, 2048, 4096]:
            if k > len(segments):
                break
            print(f"  ablation K={k} ...")
            cb_k = build_codebook(segments, k=k, dt=DT, mode=args.mode,
                                  seed=args.seed, verbose=False)
            tok_k = ActionTokenizer(cb_k)
            ev_k = evaluate_tokenizer(tok_k, eval_trajs)
            ol_k = evaluate_tokenizer(tok_k, eval_trajs, mode="open_loop")
            ablation.append(dict(k=k, ade=ev_k["ade"], fde=ev_k["fde"],
                                 ade_open_loop=ol_k["ade"],
                                 fde_open_loop=ol_k["fde"],
                                 heading_mae_deg=ev_k["heading_mae_deg"],
                                 codebook_usage=ev_k["codebook_usage"],
                                 covering_radius=cb_k.covering_radius))

    # -- 6. SFT records ----------------------------------------------------
    print("[6/6] building SFT records ...")
    records, rec_stats = build_sft_records(samples, tok)
    write_records(records, os.path.join(artdir, "sft_train.jsonl"))
    write_action_vocab(tok, os.path.join(artdir, "action_vocab.json"))
    write_samples(samples, os.path.join(artdir, "unified_samples.jsonl"))
    print(f"  {len(records)} records  ({rec_stats['fast']} fast / "
          f"{rec_stats['slow']} slow)")

    # -- report + figures --------------------------------------------------
    seg_desc = stats.describe_segments(segments, dt=DT)
    samp_desc = stats.describe_samples(samples)
    report = stats.format_report(seg_desc, samp_desc, ev, cb)
    print("\n" + report)
    with open(os.path.join(artdir, "report.txt"), "w") as f:
        f.write(report + "\n")

    print("\nrendering figures ...")
    viz.fig_pipeline_overview(dict(
        raw=f"{len(corpus):,} trajectories", samples=f"{len(samples):,} samples",
        segments=f"{len(segments):,} segments", codebook=f"K = {len(cb)}",
        tokens=f"{N_ACTION_TOKENS} per sample",
        records=f"{len(records):,} records"), figdir)
    viz.fig_codebook(cb, figdir)
    viz.fig_feasibility(cb, figdir)

    labels, trajs = [], []
    for want in ("left_turn", "right_turn", "cruise", "stop_and_go",
                 "lane_change", "curve", "accelerate", "decelerate"):
        for s in samples:
            if s.extra.get("maneuver") == want:
                labels.append(want.replace("_", " "))
                trajs.append(np.asarray(s.future_traj))
                break
    if len(trajs) < 8:
        for s in samples[: 8 - len(trajs)]:
            labels.append(s.command); trajs.append(np.asarray(s.future_traj))
    viz.fig_reconstruction(tok, trajs, figdir, labels=labels)

    if ablation:
        viz.fig_codebook_ablation(ablation, figdir)
        with open(os.path.join(artdir, "ablation.json"), "w") as f:
            json.dump(ablation, f, indent=2)
    viz.fig_token_usage(ev["token_ids"], len(cb), figdir)
    viz.fig_dataset_stats(seg_desc, samp_desc, figdir)

    card = next((s for s in samples if s.command == "Turn Left"), samples[0])
    viz.fig_sample_card(card, tok, figdir)

    if reader is not None:
        viz.fig_camera_overlay(reader, samples, tok, figdir, args.nuscenes_root)

    summary = dict(source=args.source, n_trajectories=len(corpus),
                   n_samples=len(samples), n_segments=int(len(segments)),
                   codebook_size=len(cb), covering_radius=cb.covering_radius,
                   feasibility=feas, tokenizer={k: v for k, v in ev.items()
                                                if k != "token_ids"},
                   sft={k: v for k, v in rec_stats.items()},
                   runtime_s=round(time.time() - t0, 1))
    with open(os.path.join(artdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\ndone in {summary['runtime_s']}s -> {args.outdir}/")


if __name__ == "__main__":
    main()
