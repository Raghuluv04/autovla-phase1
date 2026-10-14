"""M10: roles, loss, reference swap, log-prob consistency and one real GRPO step (stand-in)."""

import numpy as np
import pytest
import torch

from autovla.schema import DrivingSample
from model.backbone import load_model, trainable_state_dict
from model.config import BackboneConfig
from model.evaluate import EvalItem
from model.grpo import (ACTION, FREE, OPENER, SKIP, GRPOConfig, GRPOTrainer, build_roles, grpo_loss,
                        policy_confidence, reference_weights, sample_rollout, token_logprobs)
from model.inference import PromptEncoder
from model.reward import RewardConfig
from model.runtime import load_action_codec
from model.standin import build_tiny_backbone

MSGS = [{"role": "system", "content": "planner"},
        {"role": "user", "content": [{"type": "text", "text": "Ego state: speed 5.00 m/s. Stop."}]}]


@pytest.fixture()
def tiny():
    model, tok = build_tiny_backbone()
    bb = load_model(BackboneConfig(load_in_4bit=False), model=model, tokenizer=tok)
    with torch.no_grad():
        for n, p in bb.model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.3)
    bb.model.eval()
    return bb


def item():
    traj = np.stack([np.arange(11) * 1.0, np.zeros(11), np.zeros(11)], 1)
    return EvalItem(DrivingSample("t0", "synthetic", traj, velocity=2.0), MSGS, None, False)


def test_build_roles_layout():
    r = build_roles(n_prefix=3, n_cot=2, n_actions=10, capped=False)
    assert list(r[:3]) == [OPENER, SKIP, SKIP] and list(r[3:5]) == [FREE, FREE]
    assert r[5] == FREE and (r[6:] == ACTION).all() and len(r) == 15
    assert build_roles(3, 0, 10, capped=True)[3] == ACTION
    assert (build_roles(3, 0, 10, True) == ACTION).sum() == 10


def test_loss_gradient_direction_and_clipping():
    roles = np.array([ACTION] * 4)
    for adv, sign in ((1.0, +1), (-1.0, -1)):
        lp = torch.zeros(4, requires_grad=True)
        loss, kl = grpo_loss(lp, lp.detach(), lp.detach(), adv, roles, 0.2, 0.0)
        loss.backward()
        assert float(loss) == pytest.approx(-adv) and (lp.grad * sign < 0).all() and float(kl) == 0
    # ratio far above 1+eps with positive advantage -> clipped -> no gradient
    lp = torch.full((4,), 1.0, requires_grad=True)
    loss, _ = grpo_loss(lp, torch.zeros(4), torch.zeros(4), 1.0, roles, 0.2, 0.0)
    loss.backward()
    assert lp.grad.abs().sum() == 0


def test_kl_is_nonnegative_and_pulls_toward_reference():
    roles = np.array([ACTION] * 3 + [SKIP])
    new = torch.tensor([-1.0, -2.0, -0.5, -9.0], requires_grad=True)
    ref = torch.tensor([-1.0, -1.0, -1.0, 0.0])
    loss, kl = grpo_loss(new, new.detach(), ref, 0.0, roles, 0.2, 1.0)
    assert float(kl) > 0
    loss.backward()
    assert new.grad[3] == 0                                    # SKIP positions get no gradient
    assert new.grad[1] < 0 < new.grad[2]                       # moves new[1] up and new[2] down toward ref


def test_reference_weights_swap_and_restore(tiny):
    ref = trainable_state_dict(tiny.model)
    with torch.no_grad():
        for n, p in tiny.model.named_parameters():
            if p.requires_grad:
                p.add_(0.5)
    cur = trainable_state_dict(tiny.model)
    with reference_weights(tiny.model, ref):
        inside = trainable_state_dict(tiny.model)
        assert all(torch.equal(inside[k], ref[k]) for k in ref)
    after = trainable_state_dict(tiny.model)
    assert all(torch.equal(after[k], cur[k]) for k in cur)


def test_rollout_structure_and_logprobs(tiny):
    codec, cfg = load_action_codec(), GRPOConfig(cot_max_tokens=6)
    enc = PromptEncoder(tiny.tokenizer)
    it = item()
    ro = sample_rollout(tiny, enc, it.messages, codec, it.sample.future_traj, cfg)
    assert ro.valid and len(ro.roles) == len(ro.input_ids) - ro.n_prompt
    assert (ro.roles == ACTION).sum() + (ro.roles == FREE).sum() >= 10 - 1 + (ro.mode == "slow") * 0
    prompt = dict(enc(it.messages, None))
    with torch.no_grad():
        lp = token_logprobs(tiny, prompt, ro)
    active = torch.as_tensor(ro.roles != SKIP)
    assert (lp[active] <= 1e-6).all() and torch.isfinite(lp[active]).all()
    assert (lp[~active] == 0).all()
    # action log-probs equal a hand-computed softmax over the action-id slice
    ids = torch.tensor([ro.input_ids])
    with torch.no_grad():
        logits = tiny.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0].float()
    pos = int(np.where(ro.roles == ACTION)[0][-1])
    lo = tiny.first_action_id
    sl = logits[ro.n_prompt + pos - 1, lo:lo + 2048]
    tok = ro.input_ids[ro.n_prompt + pos] - lo
    assert float(torch.log_softmax(sl, -1)[tok]) == pytest.approx(float(lp[pos]), abs=1e-4)
    assert 0 < policy_confidence(lp, ro.roles) <= 1


def test_one_grpo_step_updates_only_lora_and_starts_at_zero_kl(tiny):
    codec = load_action_codec()
    cfg = GRPOConfig(group_size=3, scenes_per_step=1, steps=1, lr=5e-3, cot_max_tokens=4,
                     reward=RewardConfig(l_tol=2.0, gamma=0.5), log_every=100)
    base = {n: p.detach().clone() for n, p in tiny.model.named_parameters() if not p.requires_grad}
    before = trainable_state_dict(tiny.model)
    tr = GRPOTrainer(tiny, [item()], cfg, codec)
    hist = tr.train()
    after = trainable_state_dict(tiny.model)
    assert hist[0]["kl"] == pytest.approx(0.0, abs=1e-5)             # policy == reference at step 0
    assert any(not torch.equal(before[k], after[k]) for k in before)
    assert all(torch.equal(p, base[n]) for n, p in tiny.model.named_parameters() if n in base)
    assert np.isfinite(hist[0]["reward"]) and hist[0]["grad_norm"] >= 0
    ev = tr.evaluate([item()])
    assert ev["all"]["n"] == 1 and ev["easy"]["n"] == 1 and not ev["hard"]
