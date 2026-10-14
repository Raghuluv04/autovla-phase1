"""M6: Eq.-2 loss, masks, 3:1 mixing, and a few real optimizer steps on the stand-in."""

import math

import pytest
import torch

from autovla.schema import Reasoning
from model.backbone import load_model
from model.config import SFT_JSONL, BackboneConfig
from model.dataset import MixedModeSampler, SFTCollator, SFTExample, load_examples, split_holdout
from model.prompting import extract_action_ids
from model.standin import build_tiny_backbone
from model.trainer import SFTConfig, SFTTrainer, evaluate_holdout, loss_trend, sft_loss

COT = Reasoning("Straight road.", "Lead car ahead.", "It keeps moving.", "Keep lane.")


def make_examples(n=16):
    import json
    ex = []
    with open(SFT_JSONL) as f:
        for i, line in enumerate(f):
            if len(ex) >= n:
                break
            rec = json.loads(line)
            ids = extract_action_ids(rec["messages"][2]["content"])
            slow = len(ex) % 4 == 3
            ex.append(SFTExample(rec["id"], "slow" if slow else "fast", rec["messages"][:2], ids,
                                 COT if slow else None))
    return ex


@pytest.fixture(scope="module")
def tiny():
    model, tok = build_tiny_backbone()
    return load_model(BackboneConfig(load_in_4bit=False), model=model, tokenizer=tok)


def test_loss_matches_eq2_by_hand():
    V, L = 11, 6
    logits = torch.zeros(2, L, V)                      # uniform -> CE = log V everywhere
    ids = torch.randint(0, V, (2, L))
    am = torch.zeros(2, L, dtype=torch.bool); am[:, 4:] = True
    lm = torch.zeros(2, L, dtype=torch.bool); lm[:, 2:4] = True
    loss, parts = sft_loss(logits, ids, am, lm, torch.tensor([1.0, 0.5]), lambda_action=2.0)
    ce = math.log(V)
    assert float(loss) == pytest.approx(((1.0 + 0.5) / 2) * (ce + 2.0 * ce), rel=1e-5)
    assert float(parts["l_action"]) == pytest.approx(ce, rel=1e-5)


def test_loss_separates_regions():
    V, L = 7, 5
    ids = torch.tensor([[0, 1, 2, 3, 4]])
    logits = torch.zeros(1, L, V)
    logits[0, 2, 3] = 20.0                              # predicts ids[3] at pos 2 perfectly
    am = torch.zeros(1, L, dtype=torch.bool); am[0, 3] = True    # action token = ids[3]
    lm = torch.zeros(1, L, dtype=torch.bool); lm[0, 4] = True
    _, p = sft_loss(logits, ids, am, lm, torch.ones(1))
    assert float(p["l_action"]) < 1e-3 and float(p["l_lm"]) > 1.0


def test_collator_masks_and_weights(tiny):
    col = SFTCollator(tiny.tokenizer)
    ex = make_examples(4)
    b = col(ex)
    assert b["action_mask"].sum(1).tolist() == [11] * 4          # 10 actions + closing <|im_end|>
    assert b["weight"].tolist() == [1.0, 1.0, 1.0, 0.5]
    assert b["lm_mask"].sum(1)[3] > b["lm_mask"].sum(1)[0]           # CoT adds language tokens
    first_act = b["action_mask"][0].nonzero()[0, 0]
    assert not b["lm_mask"][0, first_act:].any()                       # nothing language-side after the plan starts
    assert not (b["action_mask"] & b["lm_mask"]).any()


def test_mixed_sampler_is_three_to_one_without_repeats():
    modes = ["fast"] * 30 + ["slow"] * 9
    idx = list(MixedModeSampler(modes, 3, seed=1))
    assert len(idx) == len(set(idx)) == 36
    assert sum(modes[i] == "slow" for i in idx) == 9
    assert sum(modes[i] == "fast" for i in idx) == 27


def test_load_and_split_real_file():
    ex, dropped = load_examples(SFT_JSONL, cot=None, limit=300)
    assert dropped > 0 and all(e.mode == "fast" for e in ex)        # slow dropped without CoT
    assert all(len(e.action_ids) == 10 for e in ex)
    train, held = split_holdout(ex, 20)
    assert len(held) == 20 and len(train) == len(ex) - 20
    assert {e.sample_id for e in held} == {e.sample_id for e in split_holdout(ex, 20)[1]}


def test_sft_reduces_loss_and_only_touches_trainable_params(tiny):
    ex = make_examples(16)
    base = {n: p.detach().clone() for n, p in tiny.model.named_parameters() if not p.requires_grad}
    cfg = SFTConfig(lr=3e-3, warmup_steps=2, grad_accum=2, epochs=3, max_steps=24, log_every=1000,
                    backbone=BackboneConfig(load_in_4bit=False))
    tr = SFTTrainer(tiny, ex, cfg)
    hist = tr.train()
    first, last = loss_trend(hist)
    assert len(hist) == 24 and last < first
    assert all(torch.equal(p, base[n]) for n, p in tiny.model.named_parameters() if n in base)
    assert {h["slow"] > 0 and h["fast"] > 0 for h in hist[:1]} == {True}      # both modes in each step


def test_evaluate_holdout_reports_both_validity_notions(tiny):
    res = evaluate_holdout(tiny, make_examples(4), n=3)
    assert res["n"] == 3 and res["valid_constrained"] == 1.0
    assert 0.0 <= res["valid_free"] <= 1.0 and 0.0 <= res["token_acc"] <= 1.0


def test_token_level_lm_norm_restores_opener_weight():
    """With per-sample means the opener token of a long record is nearly ignored."""
    V, L = 9, 40
    ids = torch.randint(0, V, (2, L))
    lm = torch.zeros(2, L, dtype=torch.bool); lm[0, 2:10] = True; lm[1, 2:38] = True    # 8 vs 36 LM tokens
    am = torch.zeros(2, L, dtype=torch.bool)
    logits = torch.zeros(2, L, V, requires_grad=True)
    loss, _ = sft_loss(logits, ids, am, lm, torch.ones(2), 0.0, lm_norm=None)
    loss.backward()
    g_mean = logits.grad[:, 1].abs().sum(-1)                  # grad at the position predicting the opener (idx 2)
    logits.grad = None
    loss, _ = sft_loss(logits, ids, am, lm, torch.ones(2), 0.0, lm_norm=22.0)
    loss.backward()
    g_tok = logits.grad[:, 1].abs().sum(-1)
    assert g_mean[0] / g_mean[1] == pytest.approx(36 / 8, rel=1e-3)      # per-sample: short record dominates
    assert g_tok[0] / g_tok[1] == pytest.approx(1.0, rel=1e-3)           # token-level: equal per token
