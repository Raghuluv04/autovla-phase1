"""Milestone 6: SFT with the combined loss of the paper's Eq. 2.

    L = w * (L_LM + lambda_action * L_action)

``L_LM`` is the mean token cross-entropy over the assistant's language tokens
(mode prefix, CoT, end-of-turn), ``L_action`` the mean over the 10 action tokens,
``w = lambda_cot`` for slow-thinking records and ``1`` for fast ones. Fast and
slow records share one DataLoader through :class:`MixedModeSampler` (3:1).
LoRA + the new-token rows are the only trainable tensors (asserted).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .backbone import Backbone, assert_lora_only, save_trainable
from .config import (CHECKPOINT_DIR, COT_JSONL, LAMBDA_ACTION, LAMBDA_COT, REPO_ROOT,
                     SFT_GRAD_ACCUM, SFT_JSONL, BackboneConfig, LoraSettings)
from .dataset import (MixedModeSampler, SFTCollator, SFTDataset, SFTExample, holdout_ids,
                      load_examples, split_holdout)
from .inference import generate_plan
from .prompting import extract_action_ids


@dataclass
class SFTConfig:
    lr: float = 2e-4
    weight_decay: float = 0.0
    warmup_steps: int = 20
    max_grad_norm: float = 1.0
    epochs: int = 1
    batch_size: int = 1
    grad_accum: int = SFT_GRAD_ACCUM
    max_steps: int | None = None
    seed: int = 0                      # fixes the held-out split: never change it between runs
    order_seed: int | None = None      # record order; None = seed. A resumed run needs a different one
    log_every: int = 10
    save_every: int = 50               # write adapter.pt every N optimizer steps (0 = only at the end)
    out_dir: str = str(CHECKPOINT_DIR / "sft")
    lambda_action: float = LAMBDA_ACTION
    lambda_cot: float = LAMBDA_COT
    lm_norm: str = "token"             # 'token' (token-level mean) or 'sample' (per-sample mean)
    fast_per_slow: int = 3
    max_length: int | None = 2048
    holdout: int = 200
    n_eval: int = 20
    sft_jsonl: str = str(SFT_JSONL)
    cot_jsonl: str = str(COT_JSONL)
    limit: int | None = None
    backbone: BackboneConfig = field(default_factory=BackboneConfig)

    @classmethod
    def from_yaml(cls, path) -> "SFTConfig":
        import yaml

        raw = yaml.safe_load(open(path))
        m, lo, d, ls, op, tr, ev = (raw.get(k, {}) for k in
                                    ("model", "lora", "data", "loss", "optim", "train", "eval"))
        names = {f.name for f in fields(BackboneConfig)}
        bb = BackboneConfig(**{k: v for k, v in m.items() if k in names},
                            lora=LoraSettings(**{k: v for k, v in lo.items()}))
        def rel(p):
            return str(p if Path(p).is_absolute() else REPO_ROOT / p)
        return cls(
            lr=op.get("lr", 2e-4), weight_decay=op.get("weight_decay", 0.0),
            warmup_steps=op.get("warmup_steps", 20), max_grad_norm=op.get("max_grad_norm", 1.0),
            epochs=tr.get("epochs", 1), batch_size=tr.get("batch_size", 1),
            grad_accum=tr.get("grad_accum", SFT_GRAD_ACCUM), max_steps=tr.get("max_steps"),
            seed=tr.get("seed", 0), order_seed=tr.get("order_seed"), log_every=tr.get("log_every", 10),
            save_every=tr.get("save_every", 50),
            out_dir=rel(tr.get("out_dir", "checkpoints/sft")),
            lambda_action=ls.get("lambda_action", LAMBDA_ACTION), lambda_cot=ls.get("lambda_cot", LAMBDA_COT),
            lm_norm=ls.get("lm_norm", "token"),
            fast_per_slow=d.get("fast_per_slow", 3), max_length=d.get("max_length"),
            holdout=d.get("holdout", 200), n_eval=ev.get("n_samples", 20),
            sft_jsonl=rel(d.get("sft_jsonl", SFT_JSONL)), cot_jsonl=rel(d.get("cot_jsonl", COT_JSONL)),
            limit=d.get("limit"), backbone=bb)


def sft_loss(logits, input_ids, action_mask, lm_mask, weight, lambda_action=LAMBDA_ACTION,
             lm_norm: float | None = None):
    """Eq. 2 per sample, averaged over the batch. Returns ``(loss, parts)``.

    ``L_action`` is always the mean over the action tokens. ``L_LM`` is the mean
    over each sample's language tokens when ``lm_norm`` is None, otherwise the
    *sum* divided by the constant ``lm_norm`` (the corpus-mean number of language
    tokens per sample), i.e. a token-level mean as in ordinary LM training. The
    per-sample mean gives the one mode-opener token of a 500-token CoT record
    1/500 of the weight it has in an 8-token fast record, which biases the
    fast/slow switch towards "fast"; the token-level form does not.
    """
    logits = logits[:, :-1].float()
    target = input_ids[:, 1:]
    ce = F.cross_entropy(logits.transpose(1, 2), target, reduction="none")
    am, lm = action_mask[:, 1:].float(), lm_mask[:, 1:].float()
    l_act = (ce * am).sum(1) / am.sum(1).clamp(min=1)
    l_lm = (ce * lm).sum(1) / (lm.sum(1).clamp(min=1) if lm_norm is None else lm_norm)
    per_sample = weight * (l_lm + lambda_action * l_act)
    acc = (((logits.argmax(-1) == target).float() * am).sum(1) / am.sum(1).clamp(min=1)).mean()
    return per_sample.mean(), dict(l_lm=l_lm.mean().detach(), l_action=l_act.mean().detach(),
                                   action_acc=acc.detach())


def _lr_at(step, cfg: SFTConfig, total):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    prog = (step - cfg.warmup_steps) / max(total - cfg.warmup_steps, 1)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))


class SFTTrainer:
    def __init__(self, bb: Backbone, train_examples: list[SFTExample], cfg: SFTConfig):
        self.bb, self.cfg = bb, cfg
        assert_lora_only(bb.model)
        self.dataset = SFTDataset(train_examples)
        self.collate = SFTCollator(bb.tokenizer, bb.processor, cfg.max_length, cfg.lambda_cot)
        self.params = [p for p in bb.model.parameters() if p.requires_grad]
        # fp16 (T4 has no bf16) needs loss scaling, and GradScaler only unscales fp32 params
        self.scaler = None
        if cfg.backbone.compute_dtype == "float16" and bb.device.type == "cuda":
            for p in self.params:
                p.data = p.data.float()
            self.scaler = torch.amp.GradScaler("cuda")
        self.opt = torch.optim.AdamW(self.params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.history: list[dict] = []
        self.prior_history: list[dict] = []        # earlier runs this one resumed from
        self.save_every = cfg.save_every
        self.lm_norm_const = self._mean_lm_tokens() if cfg.lm_norm == "token" else None

    def _mean_lm_tokens(self, n: int = 256) -> float:
        """Corpus-mean count of language (non-action) response tokens per sample."""
        rng = random.Random(self.cfg.seed)
        picks = rng.sample(self.dataset.examples, min(n, len(self.dataset)))
        counts = [int(self.collate.encode(e)["lm_mask"].sum()) for e in picks]
        return float(sum(counts) / len(counts))

    def _loader(self, epoch):
        order = self.cfg.seed if self.cfg.order_seed is None else self.cfg.order_seed
        sampler = MixedModeSampler(self.dataset.modes(), self.cfg.fast_per_slow, order)
        sampler.set_epoch(epoch)
        return DataLoader(self.dataset, batch_size=self.cfg.batch_size, sampler=sampler,
                          collate_fn=self.collate)

    def _micro_step(self, batch):
        dev = self.bb.device
        kw = {k: batch[k].to(dev) for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
              if k in batch}
        use_amp = dev.type == "cuda"
        with torch.autocast(dev.type, dtype=getattr(torch, self.cfg.backbone.compute_dtype), enabled=use_amp):
            logits = self.bb.model(**kw).logits
        return sft_loss(logits, kw["input_ids"], batch["action_mask"].to(dev), batch["lm_mask"].to(dev),
                        batch["weight"].to(dev), self.cfg.lambda_action, self.lm_norm_const)

    def train(self) -> list[dict]:
        cfg = self.cfg
        random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        micro_per_epoch = len(MixedModeSampler(self.dataset.modes(), cfg.fast_per_slow)) // cfg.batch_size
        total_steps = cfg.max_steps or (cfg.epochs * micro_per_epoch // cfg.grad_accum)
        self.bb.model.train()
        step, micro, t0 = 0, 0, time.time()
        acc = dict(loss=0.0, l_lm=0.0, l_action=0.0, action_acc=0.0, n=0, fast=0, slow=0)
        self.opt.zero_grad(set_to_none=True)
        for epoch in range(cfg.epochs):
            for batch in self._loader(epoch):
                loss, parts = self._micro_step(batch)
                scaled = loss / cfg.grad_accum
                (self.scaler.scale(scaled) if self.scaler else scaled).backward()
                acc["loss"] += float(loss.detach()); acc["n"] += 1
                for k in ("l_lm", "l_action", "action_acc"):
                    acc[k] += float(parts[k])
                for m in batch["modes"]:
                    acc[m] += 1
                micro += 1
                if micro % cfg.grad_accum:
                    continue
                for g in self.opt.param_groups:
                    g["lr"] = _lr_at(step, cfg, total_steps)
                if self.scaler:
                    self.scaler.unscale_(self.opt)
                gnorm = torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)
                if self.scaler:
                    self.scaler.step(self.opt); self.scaler.update()    # skips the step on inf/nan grads
                else:
                    self.opt.step()
                self.opt.zero_grad(set_to_none=True)
                rec = dict(step=step, lr=self.opt.param_groups[0]["lr"], grad_norm=float(gnorm),
                           fast=acc["fast"], slow=acc["slow"], elapsed_s=time.time() - t0,
                           **{k: acc[k] / acc["n"] for k in ("loss", "l_lm", "l_action", "action_acc")})
                self.history.append(rec)
                if step % cfg.log_every == 0:
                    print(f"step {step:5d}/{total_steps} loss {rec['loss']:.4f} "
                          f"(lm {rec['l_lm']:.3f}, action {rec['l_action']:.3f}, "
                          f"act-acc {rec['action_acc']:.2f}) lr {rec['lr']:.2e} "
                          f"{rec['elapsed_s'] / (step + 1):.1f}s/step", flush=True)
                acc = dict(loss=0.0, l_lm=0.0, l_action=0.0, action_acc=0.0, n=0, fast=0, slow=0)
                step += 1
                if self.save_every and step % self.save_every == 0:
                    self.save()                      # resumable-by-eye insurance against session loss
                if step >= total_steps:
                    return self.history
        return self.history

    def save(self, name="adapter.pt"):
        path = Path(self.cfg.out_dir) / name
        save_trainable(self.bb.model, path)
        (Path(self.cfg.out_dir) / "history.json").write_text(json.dumps(self.prior_history + self.history))
        return path


def loss_trend(history, frac=0.25) -> tuple[float, float]:
    """Mean loss over the first vs last ``frac`` of optimizer steps."""
    k = max(1, int(len(history) * frac))
    return (sum(h["loss"] for h in history[:k]) / k, sum(h["loss"] for h in history[-k:]) / k)


@torch.no_grad()
def evaluate_holdout(bb: Backbone, examples: list[SFTExample], n: int = 20) -> dict:
    """Format and accuracy checks on held-out records.

    ``valid_constrained`` is true by construction (the logits processor forces 10
    action tokens); ``valid_free`` generates with the processor *off* and so
    measures whether the model itself learned to finish with exactly 10 actions.
    """
    bb.model.eval()
    rows = []
    for ex in examples[:n]:
        gold = extract_action_ids(ex.response)
        con = generate_plan(bb, ex.messages, mode=ex.mode, constrained=True)
        free = generate_plan(bb, ex.messages, mode=ex.mode, constrained=False)
        valid_free = len(free.action_ids) == len(gold) and all(
            0 <= i < len(bb.action_token_ids) for i in free.action_ids)
        rows.append(dict(
            id=ex.sample_id, mode=ex.mode, valid_constrained=con.valid, valid_free=valid_free,
            token_acc=sum(a == b for a, b in zip(con.action_ids, gold)) / len(gold)))
    avg = lambda k: sum(r[k] for r in rows) / max(len(rows), 1)
    return dict(n=len(rows), valid_constrained=avg("valid_constrained"), valid_free=avg("valid_free"),
                token_acc=avg("token_acc"), rows=rows)


def unseen_ids(cfg: SFTConfig, steps: int) -> set[str]:
    """Ids never drawn by a single-epoch SFT run of ``steps`` optimizer steps with ``cfg``.

    Includes the explicit hold-out. Valid only for a run that was not resumed, since a
    resumed run reshuffles (``order_seed``) and so consumes a different set of records.
    """
    from .cot_annotator import read_annotations
    from .dataset import consumed_ids

    cot = read_annotations(cfg.cot_jsonl) if Path(cfg.cot_jsonl).exists() else None
    examples, _ = load_examples(cfg.sft_jsonl, cot, cfg.limit)
    train, _ = split_holdout(examples, cfg.holdout, cfg.seed,
                             held_ids=holdout_ids(cfg.sft_jsonl, cfg.holdout, cfg.seed))
    seen = consumed_ids(train, steps, cfg.grad_accum, cfg.batch_size, cfg.fast_per_slow,
                        cfg.seed if cfg.order_seed is None else cfg.order_seed)
    with open(cfg.sft_jsonl) as f:
        every = [json.loads(line)["id"] for line in f]
    return {i for i in every if i not in seen}


def run_sft(bb: Backbone, cfg: SFTConfig, examples=None, prior_history=None) -> dict:
    """Train + evaluate; returns a summary dict."""
    if examples is None:
        from .cot_annotator import read_annotations

        cot = read_annotations(cfg.cot_jsonl) if Path(cfg.cot_jsonl).exists() else None
        examples, dropped = load_examples(cfg.sft_jsonl, cot, cfg.limit)
        if dropped:
            print(f"dropped {dropped} slow records without real CoT")
    train, held = split_holdout(examples, cfg.holdout, cfg.seed,
                                held_ids=holdout_ids(cfg.sft_jsonl, cfg.holdout, cfg.seed))
    trainer = SFTTrainer(bb, train, cfg)
    trainer.prior_history = prior_history or []
    hist = trainer.train()
    path = trainer.save()
    first, last = loss_trend(hist)
    ev = evaluate_holdout(bb, held, cfg.n_eval)
    return dict(first_quarter_loss=first, last_quarter_loss=last, steps=len(hist),
                holdout=ev, checkpoint=str(path), history=hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "sft.yaml"))
    ap.add_argument("--standin", action="store_true", help="tiny CPU stand-in model instead of Qwen2.5-VL-3B")
    ap.add_argument("--steps", type=int, default=None, help="override optimizer steps")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--resume", default=None, help="adapter.pt from an earlier run to continue from")
    ap.add_argument("--order-seed", type=int, default=None,
                    help="record-order seed (default: seed+1 when resuming, so records are reshuffled; "
                         "the held-out split always uses the config seed)")
    ap.add_argument("--n-eval", type=int, default=None, help="held-out records to generate from after training")
    ap.add_argument("--holdout", type=int, default=None, help="records held out of SFT")
    ap.add_argument("--out-dir", default=None, help="checkpoint directory (default checkpoints/sft_standin)")
    ap.add_argument("--standin-hidden", type=int, default=64)
    ap.add_argument("--standin-layers", type=int, default=2)
    ap.add_argument("--lora-rank", type=int, default=None, help="override LoRA rank (stand-in only; 3B uses 8)")
    args = ap.parse_args()
    from .backbone import load_model

    cfg = SFTConfig.from_yaml(args.config)
    if args.steps:
        cfg.max_steps = args.steps
    if args.limit:
        cfg.limit = args.limit
    if args.lr:
        cfg.lr = args.lr
    if args.holdout:
        cfg.holdout = args.holdout
    if args.n_eval is not None:
        cfg.n_eval = args.n_eval
    if args.resume:
        cfg.order_seed = args.order_seed if args.order_seed is not None else cfg.seed + 1
        if not args.out_dir:               # never overwrite the adapter being resumed from
            cfg.out_dir = str(Path(args.resume).resolve().parent) + "_resumed"
    elif args.order_seed is not None:
        cfg.order_seed = args.order_seed
    if args.standin:
        from .runtime import make_backbone, write_meta

        if args.lora_rank:
            cfg.backbone.lora.rank = cfg.backbone.lora.alpha = args.lora_rank
        cfg.out_dir = args.out_dir or (cfg.out_dir if args.resume else str(CHECKPOINT_DIR / "sft_standin"))
        spec = dict(standin=True, hidden=args.standin_hidden, layers=args.standin_layers,
                    lora_rank=cfg.backbone.lora.rank)
        bb = make_backbone(spec=spec, cfg=cfg.backbone)
        write_meta(cfg.out_dir, spec)
    else:
        if args.out_dir:
            cfg.out_dir = args.out_dir
        bb = load_model(cfg.backbone)
    prior = []
    if args.resume:
        from .backbone import load_trainable

        n = load_trainable(bb.model, args.resume)
        hp = Path(args.resume).parent / "history.json"
        prior = json.loads(hp.read_text()) if hp.exists() else []
        print(f"resumed {n} trainable tensors from {args.resume} ({len(prior)} earlier steps); "
              f"order_seed={cfg.order_seed}, writing to {cfg.out_dir}")
    res = run_sft(bb, cfg, prior_history=prior)
    ev = res["holdout"]
    print(f"loss: first quarter {res['first_quarter_loss']:.4f} -> last quarter {res['last_quarter_loss']:.4f}")
    print(f"held-out ({ev['n']}): valid(constrained) {ev['valid_constrained']:.0%}, "
          f"valid(free-running) {ev['valid_free']:.0%}, token acc {ev['token_acc']:.1%}")
    print(f"saved {res['checkpoint']}")


if __name__ == "__main__":
    main()
