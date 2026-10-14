"""Milestone 4 smoke test: load -> forward pass -> decode 10 action tokens.

    python -m model.forward_test            # tiny CPU stand-in (< 5 s)
    python -m model.forward_test --real     # Qwen2.5-VL-3B (downloads ~7 GB)

The stand-in is a random-weight Qwen2.5-VL built by :mod:`model.standin`; it runs
the identical extend-tokenizer / resize / freeze / LoRA code as the real model.
With random weights the emitted tokens are arbitrary -- the check is that the
plumbing yields exactly ``N_ACTION_TOKENS`` valid ids that decode to a trajectory.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from autovla.codebook import Codebook
from autovla.tokenizer import ActionTokenizer

from .backbone import Backbone, assert_lora_only, load_model, trainable_summary
from .config import (CODEBOOK_NPZ, CODEBOOK_SIZE, N_ACTION_TOKENS, SFT_JSONL,
                     BackboneConfig)
from .inference import PromptEncoder, generate_plan
from .prompting import extract_action_ids, render_training_text


def first_record(path=SFT_JSONL) -> dict:
    with open(path) as f:
        return json.loads(f.readline())


def teacher_forced_loss(bb: Backbone, record: dict) -> float:
    """One labelled forward pass over the record's own response."""
    prompt, full, _ = render_training_text(record["messages"], record["messages"][2]["content"])
    tok = bb.tokenizer
    ids = tok(full, return_tensors="pt", add_special_tokens=False)["input_ids"].to(bb.device)
    n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
    labels = ids.clone()
    labels[:, :n_prompt] = -100
    out = bb.model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
    return float(out.loss.detach())


def run(bb: Backbone, record: dict | None = None) -> dict:
    record = record or first_record()
    summary = assert_lora_only(bb.model)
    loss = teacher_forced_loss(bb, record)
    plan = generate_plan(bb, record["messages"][:2], mode="fast")
    codebook = Codebook.load(CODEBOOK_NPZ)
    traj = ActionTokenizer(codebook).decode(plan.action_ids)
    gold = extract_action_ids(record["messages"][2]["content"])

    assert plan.valid, f"expected {N_ACTION_TOKENS} action tokens, got {len(plan.action_ids)}"
    assert all(0 <= i < CODEBOOK_SIZE for i in plan.action_ids)
    assert traj.shape == (N_ACTION_TOKENS + 1, 3) and np.isfinite(traj).all()
    assert np.isfinite(loss)
    return dict(trainable=summary, loss=loss, action_ids=plan.action_ids, gold_ids=gold,
                trajectory_end=traj[-1].round(3).tolist())


def build_standin() -> Backbone:
    from .standin import build_tiny_backbone

    model, tok = build_tiny_backbone()
    return load_model(BackboneConfig(load_in_4bit=False), model=model, tokenizer=tok)


def test_forward_smoke():
    result = run(build_standin())
    assert len(result["action_ids"]) == N_ACTION_TOKENS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="use Qwen2.5-VL-3B (downloads weights)")
    args = ap.parse_args()
    bb = load_model(BackboneConfig()) if args.real else build_standin()
    r = run(bb)
    s = r["trainable"]
    print(f"trainable params: {s['trainable']:,} / {s['total']:,} "
          f"(lora {s['lora']:,}, new-token rows {s['new_tokens']:,}, vision {s['vision']})")
    print(f"teacher-forced loss on record 0: {r['loss']:.3f}")
    print(f"emitted {len(r['action_ids'])} action tokens: {r['action_ids']}")
    print(f"gold tokens                    : {r['gold_ids']}")
    print(f"decoded trajectory end pose    : {r['trajectory_end']}")
    print("PASS" + ("" if args.real else "  (stand-in model: random weights, plumbing only)"))


if __name__ == "__main__":
    main()
