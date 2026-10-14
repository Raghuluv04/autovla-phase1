"""Milestone 7: open-loop evaluation of fast / slow / random / auto thinking.

For every held-out scene the policy plans in a given mode, the 10 action tokens
are decoded to a trajectory, and L2 / collision are computed per horizon against
the ground-truth path (UniAD protocol, see :mod:`model.metrics`).

Modes
  fast    force the no-reasoning prefix            (always the cheap path)
  slow    force the reasoning prefix + CoT         (always the expensive path)
  random  coin-flip fast/slow per scene            (baseline for adaptive switching)
  auto    the model's own learned switch           (AutoVLA's implicit gate)

``--standin`` uses a tiny CPU model on the synthetic corpus with synthetic agent
boxes: it validates the pipeline and reports *its* numbers, not Qwen2.5-VL's.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autovla.codebook import Codebook
from autovla.schema import DrivingSample, read_samples
from autovla.tokenizer import ActionTokenizer

from .backbone import Backbone
from .config import (CODEBOOK_NPZ, RESULTS_DIR, SFT_JSONL, UNIFIED_JSONL, EVAL_HORIZONS_S)
from .dataset import holdout_ids
from .inference import generate_plan
from .metrics import OpenLoopAccumulator, latency_reduction
from .splits import is_hard, synthetic_obstacles

MODES = ("fast", "slow", "random", "auto")


@dataclass
class EvalItem:
    sample: DrivingSample
    messages: list                 # system + user turns
    obstacles: np.ndarray | None
    hard: bool


def build_eval_items(n: int | None = None, holdout: int = 200, seed: int = 0,
                     sft_path=SFT_JSONL, unified_path=UNIFIED_JSONL, obstacles: str = "synthetic"):
    held = set(holdout_ids(sft_path, holdout, seed))
    by_id = {s.sample_id: s for s in read_samples(unified_path) if s.sample_id in held}
    items = []
    with open(sft_path) as f:
        for line in f:
            rec = json.loads(line)
            s = by_id.get(rec["id"])
            if s is None:
                continue
            obs = synthetic_obstacles(s, seed) if obstacles == "synthetic" else None
            items.append(EvalItem(s, rec["messages"][:2], obs, is_hard(s)))
    return items[:n] if n else items


def resolve_mode(mode: str, rng: np.random.Generator) -> str:
    if mode == "random":
        return "fast" if rng.random() < 0.5 else "slow"
    return mode


def run_mode(bb: Backbone, items: list[EvalItem], mode: str, tok: ActionTokenizer,
             seed: int = 0, cot_max_tokens: int | None = None) -> tuple[dict, list[dict]]:
    """Evaluate one mode on all items -> ({split: summary}, per-item rows)."""
    rng = np.random.default_rng(seed)
    accs = {"all": OpenLoopAccumulator(), "easy": OpenLoopAccumulator(), "hard": OpenLoopAccumulator()}
    rows = []
    kw = {} if cot_max_tokens is None else dict(cot_max_tokens=cot_max_tokens)
    for it in items:
        used = resolve_mode(mode, rng)
        plan = generate_plan(bb, it.messages, mode=used, **kw)
        pred = tok.decode(plan.action_ids)
        gt = np.asarray(it.sample.future_traj, float)
        for key in ("all", "hard" if it.hard else "easy"):
            accs[key].add(pred, gt, it.obstacles, plan.latency_s, plan.n_cot_tokens)
        rows.append(dict(id=it.sample.sample_id, hard=it.hard, mode=plan.mode,
                         latency_s=plan.latency_s, cot_tokens=plan.n_cot_tokens,
                         ade=float(np.linalg.norm(pred[1:, :2] - gt[1:, :2], axis=1).mean()),
                         action_ids=plan.action_ids))
    return {k: a.summary() for k, a in accs.items()}, rows


def evaluate_modes(bb: Backbone, items: list[EvalItem], modes=MODES, seed: int = 0, **kw) -> dict:
    tok = ActionTokenizer(Codebook.load(CODEBOOK_NPZ))
    out = {}
    for m in modes:
        summ, rows = run_mode(bb, items, m, tok, seed, **kw)
        out[m] = dict(summary=summ, rows=rows)
    if "fast" in out and "slow" in out:
        out["latency_reduction"] = {
            split: latency_reduction(out["fast"]["summary"][split]["latency_s"],
                                     out["slow"]["summary"][split]["latency_s"])
            for split in ("all", "easy", "hard") if out["fast"]["summary"][split]["n"]}
    return out


COLUMNS = [f"L2@{h:g}s" for h in EVAL_HORIZONS_S] + [f"Col@{h:g}s" for h in EVAL_HORIZONS_S] \
    + ["ADE", "latency_s", "cot_tokens"]


def format_table(result: dict, split: str = "all") -> str:
    head = "| mode | n | " + " | ".join(COLUMNS) + " |\n|" + "---|" * (len(COLUMNS) + 2) + "\n"
    lines = []
    for m, r in result.items():
        if m == "latency_reduction":
            continue
        s = r["summary"][split]
        if not s["n"]:
            continue
        cells = [f"{s[c]:.4f}" if c.startswith(("Col", "L2", "ADE")) else
                 (f"{s[c] * 1000:.1f} ms" if c == "latency_s" else f"{s[c]:.1f}") for c in COLUMNS]
        lines.append(f"| {m} | {s['n']} | " + " | ".join(cells) + " |")
    return head + "\n".join(lines)


def save_results(result: dict, tag: str, out_dir=RESULTS_DIR) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"openloop_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "split", "n"] + COLUMNS + ["route_completion"])
        for m, r in result.items():
            if m == "latency_reduction":
                continue
            for split, s in r["summary"].items():
                if s["n"]:
                    w.writerow([m, split, s["n"]] + [f"{s[c]:.6f}" for c in COLUMNS] + [f"{s['route_completion']:.6f}"])
    (out_dir / f"openloop_{tag}.json").write_text(json.dumps(
        {m: (r["summary"] if m != "latency_reduction" else r) for m, r in result.items()}, indent=1))
    return csv_path


def main():
    from .config import BackboneConfig
    from .runtime import make_backbone

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--standin", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--cot-max-tokens", type=int, default=None)
    args = ap.parse_args()
    bb = make_backbone(args.standin, BackboneConfig(), args.checkpoint)
    items = build_eval_items(args.n)
    res = evaluate_modes(bb, items, args.modes, cot_max_tokens=args.cot_max_tokens)
    for split in ("all", "easy", "hard"):
        print(f"\n[{split}]\n" + format_table(res, split))
    print("\nlatency reduction fast vs slow:", {k: f"{v:.1%}" for k, v in res.get("latency_reduction", {}).items()})
    print("wrote", save_results(res, args.tag or ("standin" if args.standin else "model")))


if __name__ == "__main__":
    main()
