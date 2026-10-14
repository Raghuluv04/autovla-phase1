"""Milestone 10: Group-Relative Policy Optimisation on top of the SFT policy.

Per scene: sample G plans from the policy (mode chosen by the policy itself, CoT
sampled freely, actions sampled from the 2048 action tokens), score each with the
driving reward minus the CoT-length penalty (plus the team's calibration term),
standardise within the group to get advantages, and take a clipped policy-gradient
step with a KL penalty to the frozen SFT reference. Only LoRA and the new-token
rows move; the reference is a snapshot of those same tensors, so it costs a few MB.

Log-probabilities are taken under exactly the distribution the sampler used
(``PlanFormatProcessor``): the mode opener is a 2-way choice, CoT tokens come from
the softmax with EOS removed, action tokens from the softmax restricted to the
2048 action ids. Forced template tokens carry no policy gradient.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .backbone import Backbone, assert_lora_only, save_trainable, trainable_state_dict
from .config import (CHECKPOINT_DIR, COT_MAX_TOKENS, GRPO_GROUP_SIZE, GRPO_KL_BETA, N_ACTION_TOKENS)
from .inference import PromptEncoder, generate_plan
from .prompting import assistant_prefix
from .reward import RewardConfig, group_advantages, total_reward

SKIP, OPENER, FREE, ACTION = 0, 1, 2, 3


@dataclass
class GRPOConfig:
    group_size: int = GRPO_GROUP_SIZE
    scenes_per_step: int = 2
    steps: int = 100
    lr: float = 1e-5
    kl_beta: float = GRPO_KL_BETA
    clip_eps: float = 0.2
    num_iterations: int = 1
    temperature: float = 1.0
    cot_max_tokens: int = COT_MAX_TOKENS
    max_grad_norm: float = 1.0
    seed: int = 0
    reward: RewardConfig = field(default_factory=RewardConfig)
    eval_every: int = 25
    n_eval_scenes: int = 40
    log_every: int = 5
    out_dir: str = str(CHECKPOINT_DIR / "grpo")


@dataclass
class Rollout:
    mode: str
    input_ids: list[int]            # prompt + forced prefix + generated tokens (no trailing EOS)
    n_prompt: int                   # tokens before the assistant response (incl. its header)
    roles: np.ndarray               # (R,) role of each response position
    action_ids: list[int]
    n_cot: int
    traj: np.ndarray
    ade: float = 0.0
    confidence: float = 1.0
    logp_old: torch.Tensor | None = None
    reward: dict = field(default_factory=dict)
    advantage: float = 0.0

    @property
    def valid(self) -> bool:
        return len(self.action_ids) == N_ACTION_TOKENS


def build_roles(n_prefix: int, n_cot: int, n_actions: int, capped: bool) -> np.ndarray:
    """Role per response position: [opener, forced..., CoT..., actions...]."""
    roles = [OPENER] + [SKIP] * (n_prefix - 1) + [FREE] * n_cot
    if n_actions:                       # the first action is the CoT -> actions transition
        roles.append(ACTION if capped else FREE)
        roles += [ACTION] * (n_actions - 1)
    return np.array(roles, dtype=np.int64)


def sample_rollout(bb: Backbone, enc: PromptEncoder, messages, codec, gt, cfg: GRPOConfig,
                   greedy: bool = False) -> Rollout:
    plan = generate_plan(bb, messages, mode="auto", do_sample=not greedy, temperature=cfg.temperature,
                         cot_max_tokens=cfg.cot_max_tokens, encoder=enc)
    tok = bb.tokenizer
    prompt_ids = enc(messages, None)["input_ids"][0].tolist()
    full_ids = enc(messages, assistant_prefix(plan.mode))["input_ids"][0].tolist()
    gen = [t for t in plan.gen_ids]
    for stop in (bb.eos_id, tok.pad_token_id):
        if stop in gen:
            gen = gen[:gen.index(stop)]
    n_prefix = len(full_ids) - len(prompt_ids)
    lo, hi = bb.first_action_id, bb.first_action_id + len(bb.action_token_ids)
    first_act = next((i for i, t in enumerate(gen) if lo <= t < hi), len(gen))
    n_cot = first_act
    n_act = len(gen) - first_act
    capped = plan.mode == "fast" or n_cot >= cfg.cot_max_tokens
    roles = build_roles(n_prefix, n_cot, n_act, capped)
    ids = full_ids + gen
    traj = codec.decode(plan.action_ids) if plan.valid else np.zeros((N_ACTION_TOKENS + 1, 3))
    ade = float(np.linalg.norm(traj[1:, :2] - np.asarray(gt)[1:, :2], axis=1).mean())
    return Rollout(plan.mode, ids, len(prompt_ids), roles, plan.action_ids, n_cot, traj, ade)


def token_logprobs(bb: Backbone, prompt_inputs: dict, ro: Rollout, temperature: float = 1.0) -> torch.Tensor:
    """(R,) log pi of every response token under the sampling distribution (0 where SKIP)."""
    dev = bb.device
    ids = torch.tensor([ro.input_ids], device=dev)
    kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
    for k in ("pixel_values", "image_grid_thw"):
        if k in prompt_inputs:
            kw[k] = prompt_inputs[k].to(dev)
    use_amp = dev.type == "cuda"
    with torch.autocast(dev.type, dtype=getattr(torch, bb.cfg.compute_dtype), enabled=use_amp):
        logits = bb.model(**kw).logits[0]
    R = len(ro.roles)
    start = ro.n_prompt - 1
    lg = logits[start:start + R].float() / temperature
    tgt = ids[0, ro.n_prompt:ro.n_prompt + R]
    roles = torch.as_tensor(ro.roles, device=dev)
    out = torch.zeros(R, device=dev)

    openers = torch.tensor([bb.tokenizer(assistant_prefix(m), add_special_tokens=False)["input_ids"][0]
                            for m in ("fast", "slow")], device=dev)
    idx = (roles == OPENER).nonzero(as_tuple=True)[0]
    if len(idx):
        lp = torch.log_softmax(lg[idx][:, openers], -1)
        which = (tgt[idx].unsqueeze(1) == openers.unsqueeze(0)).float().argmax(1)
        out = out.index_put((idx,), lp.gather(1, which[:, None])[:, 0])
    idx = (roles == FREE).nonzero(as_tuple=True)[0]
    if len(idx):
        masked = lg[idx].clone()
        masked[:, bb.eos_id] = float("-inf")
        lp = torch.log_softmax(masked, -1).gather(1, tgt[idx][:, None])[:, 0]
        out = out.index_put((idx,), lp)
    idx = (roles == ACTION).nonzero(as_tuple=True)[0]
    if len(idx):
        lo, n = bb.first_action_id, len(bb.action_token_ids)
        lp = torch.log_softmax(lg[idx][:, lo:lo + n], -1).gather(1, (tgt[idx] - lo)[:, None])[:, 0]
        out = out.index_put((idx,), lp)
    return out


def policy_confidence(logp: torch.Tensor, roles: np.ndarray) -> float:
    """Geometric-mean probability of the emitted action tokens, in (0, 1]."""
    m = torch.as_tensor(roles == ACTION, device=logp.device)
    if not m.any():
        return 0.0
    return float(torch.exp(logp[m].mean()))


@contextmanager
def grpo_train_mode(model):
    """Train mode (so gradient checkpointing is active) but with dropout off."""
    was = model.training
    model.train()
    drops = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    for m in drops:
        m.eval()
    try:
        yield
    finally:
        model.train(was)


@contextmanager
def reference_weights(model, ref_state: dict):
    """Temporarily load the SFT snapshot of the trainable tensors."""
    params = dict(model.named_parameters())
    cur = {n: params[n].detach().clone() for n in ref_state}
    with torch.no_grad():
        for n, v in ref_state.items():
            params[n].copy_(v.to(params[n].device, params[n].dtype))
    try:
        yield
    finally:
        with torch.no_grad():
            for n, v in cur.items():
                params[n].copy_(v)


def grpo_loss(logp_new, logp_old, logp_ref, advantage, roles, clip_eps: float, kl_beta: float):
    """Per-sample clipped PG + KL, averaged over active tokens. Returns (loss, kl)."""
    active = torch.as_tensor(roles != SKIP, device=logp_new.device)
    ratio = torch.exp(logp_new - logp_old)[active]
    adv = torch.as_tensor(advantage, dtype=ratio.dtype, device=ratio.device)
    pg = -torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    d = (logp_ref - logp_new)[active]
    kl = torch.exp(d) - d - 1.0                               # k3 estimator, >= 0
    return (pg + kl_beta * kl).mean(), kl.mean().detach()


class GRPOTrainer:
    def __init__(self, bb: Backbone, scenes: list, cfg: GRPOConfig, codec, confidence_fn=None):
        self.bb, self.scenes, self.cfg, self.codec = bb, scenes, cfg, codec
        self.confidence_fn = confidence_fn
        assert_lora_only(bb.model)
        self.params = [p for p in bb.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=cfg.lr, weight_decay=0.0)
        self.ref_state = trainable_state_dict(bb.model)       # clones: the SFT snapshot
        self.enc = PromptEncoder(bb.tokenizer, bb.processor)
        self.rng = np.random.default_rng(cfg.seed)
        self.history: list[dict] = []

    # -- scoring ----------------------------------------------------------
    def _rewards(self, item, rollouts):
        cfg = self.cfg
        ades = np.array([r.ade for r in rollouts])
        lens = np.array([r.n_cot for r in rollouts], float)
        if self.confidence_fn is not None:
            conf = np.array([self.confidence_fn(item, r) for r in rollouts])
        else:
            conf = np.array([r.confidence for r in rollouts])
        rw = total_reward(ades, lens, conf, cfg.reward)
        adv = group_advantages(rw["total"])
        for i, r in enumerate(rollouts):
            r.reward = {k: float(v[i]) for k, v in rw.items()}
            r.advantage = float(adv[i]); r.confidence = float(conf[i])
        return rw

    def _scene_step(self, item):
        """Sample, score, and back-propagate one group. Returns stats."""
        cfg, bb = self.cfg, self.bb
        gt = np.asarray(item.sample.future_traj, float)
        prompt_inputs = {k: v for k, v in self.enc(item.messages, None).items()}
        bb.model.eval()
        with torch.no_grad():
            rollouts = [sample_rollout(bb, self.enc, item.messages, self.codec, gt, cfg)
                        for _ in range(cfg.group_size)]
            for r in rollouts:
                r.logp_old = token_logprobs(bb, prompt_inputs, r, cfg.temperature)
                r.confidence = policy_confidence(r.logp_old, r.roles)
        rw = self._rewards(item, rollouts)
        kls = []
        for _ in range(cfg.num_iterations):
            for r in rollouts:
                with torch.no_grad(), reference_weights(bb.model, self.ref_state):
                    logp_ref = token_logprobs(bb, prompt_inputs, r, cfg.temperature)
                with grpo_train_mode(bb.model):
                    logp_new = token_logprobs(bb, prompt_inputs, r, cfg.temperature)
                    loss, kl = grpo_loss(logp_new, r.logp_old, logp_ref, r.advantage, r.roles,
                                         cfg.clip_eps, cfg.kl_beta)
                    (loss / (cfg.group_size * cfg.scenes_per_step * cfg.num_iterations)).backward()
                kls.append(float(kl))
        bb.model.eval()
        return dict(
            reward=float(rw["total"].mean()), r_drive=float(rw["driving"].mean()),
            r_cot=float(rw["cot"].mean()), r_cal=float(rw["calibration"].mean()),
            ade=float(np.mean([r.ade for r in rollouts])), cot=float(np.mean([r.n_cot for r in rollouts])),
            frac_slow=float(np.mean([r.mode == "slow" for r in rollouts])),
            conf=float(np.mean([r.confidence for r in rollouts])), kl=float(np.mean(kls)) if kls else 0.0,
            hard=bool(item.hard))

    @torch.no_grad()
    def evaluate(self, items, n_samples: int = 1) -> dict:
        """Mean reward / CoT length over fixed scenes (policy sampling, no learning)."""
        cfg, bb = self.cfg, self.bb
        bb.model.eval()
        rows = []
        for item in items:
            gt = np.asarray(item.sample.future_traj, float)
            prompt_inputs = dict(self.enc(item.messages, None))
            for _ in range(n_samples):
                r = sample_rollout(bb, self.enc, item.messages, self.codec, gt, cfg)
                r.confidence = policy_confidence(token_logprobs(bb, prompt_inputs, r, cfg.temperature), r.roles)
                rw = total_reward(r.ade, r.n_cot, r.confidence, cfg.reward)
                rows.append(dict(hard=bool(item.hard), reward=float(rw["total"]), r_drive=float(rw["driving"]),
                                 cot=r.n_cot, slow=r.mode == "slow", ade=r.ade, conf=r.confidence))
        def agg(sel):
            sub = [x for x in rows if sel(x)]
            if not sub:
                return {}
            return {k: float(np.mean([x[k] for x in sub])) for k in ("reward", "r_drive", "cot", "slow", "ade", "conf")} | {"n": len(sub)}
        return dict(all=agg(lambda x: True), easy=agg(lambda x: not x["hard"]), hard=agg(lambda x: x["hard"]))

    def train(self, eval_items=None) -> list[dict]:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        t0 = time.time()
        self.evals = []
        if eval_items:
            self.evals.append(dict(step=0, **self.evaluate(eval_items)))
            print(f"[eval step 0] {json.dumps({k: round(v, 4) for k, v in self.evals[-1]['all'].items()})}", flush=True)
        for step in range(cfg.steps):
            picks = self.rng.choice(len(self.scenes), cfg.scenes_per_step, replace=False)
            stats = [self._scene_step(self.scenes[i]) for i in picks]
            gnorm = torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)
            self.opt.step(); self.opt.zero_grad(set_to_none=True)
            rec = dict(step=step, grad_norm=float(gnorm), elapsed_s=time.time() - t0,
                       **{k: float(np.mean([s[k] for s in stats])) for k in stats[0] if k != "hard"})
            easy = [s["cot"] for s in stats if not s["hard"]]
            rec["cot_easy"] = float(np.mean(easy)) if easy else float("nan")
            self.history.append(rec)
            if step % cfg.log_every == 0:
                print(f"step {step:4d} reward {rec['reward']:+.4f} (drive {rec['r_drive']:+.3f}, "
                      f"cot {rec['r_cot']:.3f}, cal {rec['r_cal']:+.3f}) ade {rec['ade']:.2f} "
                      f"cot_len {rec['cot']:.1f} slow {rec['frac_slow']:.2f} kl {rec['kl']:.4f}", flush=True)
            if eval_items and cfg.eval_every and (step + 1) % cfg.eval_every == 0:
                self.evals.append(dict(step=step + 1, **self.evaluate(eval_items)))
                print(f"[eval step {step + 1}] {json.dumps({k: round(v, 4) for k, v in self.evals[-1]['all'].items()})}", flush=True)
        return self.history

    def save(self, name="adapter.pt"):
        path = Path(self.cfg.out_dir) / name
        save_trainable(self.bb.model, path)
        (Path(self.cfg.out_dir) / "history.json").write_text(json.dumps(
            dict(history=self.history, evals=getattr(self, "evals", []))))
        return path


def smooth_trend(values, frac: float = 0.25) -> tuple[float, float]:
    k = max(1, int(len(values) * frac))
    v = np.asarray(values, float)
    return float(np.nanmean(v[:k])), float(np.nanmean(v[-k:]))


def build_scene_pool(n: int | None = None, holdout: int = 3000, seed: int = 0, train: bool = True):
    """Scenes for GRPO: training-side (not in the SFT hold-out) or the hold-out itself."""
    from .evaluate import EvalItem, build_eval_items
    from .dataset import holdout_ids
    from .config import SFT_JSONL, UNIFIED_JSONL
    from .splits import is_hard
    from autovla.schema import read_samples

    if not train:
        return build_eval_items(n, holdout=holdout, seed=seed, obstacles=None)
    held = set(holdout_ids(SFT_JSONL, holdout, seed))
    msgs = {}
    with open(SFT_JSONL) as f:
        for line in f:
            rec = json.loads(line)
            if rec["id"] not in held:
                msgs[rec["id"]] = rec["messages"][:2]
    rng = np.random.default_rng(seed)
    samples = [s for s in read_samples(UNIFIED_JSONL) if s.sample_id in msgs]
    order = rng.permutation(len(samples))[: n or len(samples)]
    return [EvalItem(samples[i], msgs[samples[i].sample_id], None, is_hard(samples[i])) for i in order]


def main():
    from .config import BackboneConfig
    from .runtime import load_action_codec, make_backbone

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--standin", action="store_true")
    ap.add_argument("--sft-checkpoint", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--scenes-per-step", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=GRPO_GROUP_SIZE)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--no-calibration", action="store_true", help="drop r_calibration (ablation)")
    ap.add_argument("--cot-max-tokens", type=int, default=COT_MAX_TOKENS)
    ap.add_argument("--l-tol", type=float, default=None, help="CoT length tolerance (tokens)")
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--tau-fail", type=float, default=None, help="failure threshold of r_calibration (m)")
    ap.add_argument("--n-scenes", type=int, default=400)
    ap.add_argument("--holdout", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--out", required=True, help="output dir for the GRPO checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rc = RewardConfig(use_calibration=not args.no_calibration)
    if args.l_tol:
        rc = RewardConfig(**{**rc.__dict__, "l_tol": args.l_tol})
    if args.gamma:
        rc = RewardConfig(**{**rc.__dict__, "gamma": args.gamma})
    if args.tau_fail:
        rc = RewardConfig(**{**rc.__dict__, "tau_fail": args.tau_fail})
    cfg = GRPOConfig(group_size=args.group_size, scenes_per_step=args.scenes_per_step, steps=args.steps,
                     lr=args.lr or (1e-3 if args.standin else 1e-5), cot_max_tokens=args.cot_max_tokens,
                     reward=rc, eval_every=args.eval_every, n_eval_scenes=args.n_eval, out_dir=args.out,
                     seed=args.seed)
    bb = make_backbone(args.standin, BackboneConfig(), args.sft_checkpoint)
    scenes = build_scene_pool(args.n_scenes, args.holdout, args.seed, train=True)
    eval_items = build_scene_pool(args.n_eval, args.holdout, args.seed, train=False)
    tr = GRPOTrainer(bb, scenes, cfg, load_action_codec())
    hist = tr.train(eval_items)
    path = tr.save()
    from .runtime import write_meta, read_meta
    meta = read_meta(args.sft_checkpoint)
    if meta:
        write_meta(args.out, meta)
    f, l = smooth_trend([h["reward"] for h in hist])
    fc, lc = smooth_trend([h["cot_easy"] for h in hist])
    print(f"reward first quarter {f:+.4f} -> last quarter {l:+.4f}; "
          f"CoT length on easy scenes {fc:.1f} -> {lc:.1f}\nsaved {path}")


if __name__ == "__main__":
    main()
