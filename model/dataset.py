"""Milestone 6 data: ``sft_train.jsonl`` -> tokenised, region-labelled examples.

Phase I wrote placeholder reasoning into its slow records and used its own fast
template, so responses are *re-rendered* here from structured parts (gold action
ids parsed from the record, CoT from the Milestone-5 annotation file) with the
shared templates in :mod:`model.prompting`. A slow record without real CoT is
dropped rather than trained on placeholder text.

Each example carries two token masks over the assistant turn, which the loss in
:mod:`model.trainer` needs for Eq. 2:
  ``action_mask``  the 10 ``<action_i>`` tokens and the closing ``<|im_end|>`` -> L_action
  ``lm_mask``      every other assistant token (mode prefix, CoT)  -> L_LM

The closing token belongs to the action term on purpose: "exactly 10 actions"
is a counting task, and giving the stop token the action weight (lambda_action)
is what teaches the model to stop after the tenth.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, Sampler

from autovla.schema import Reasoning

from .config import FAST_PREFIX, LAMBDA_COT, SFT_JSONL, COT_JSONL
from .prompting import (IM_END, build_response, extract_action_ids, render_chat)


@dataclass
class SFTExample:
    sample_id: str
    mode: str                              # 'fast' | 'slow'
    messages: list                         # system + user turns only
    action_ids: list[int]
    reasoning: Reasoning | None = None
    meta: dict = field(default_factory=dict)

    @property
    def response(self) -> str:
        return build_response(self.mode, self.action_ids, self.reasoning)


def load_examples(path=SFT_JSONL, cot: dict | None = None, limit: int | None = None):
    """Parse the JSONL; returns ``(examples, n_dropped_slow_without_cot)``."""
    examples, dropped = [], 0
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            ids = extract_action_ids(rec["messages"][2]["content"])
            mode = rec["thinking_mode"]
            reasoning = None
            if mode == "slow":
                if not cot or rec["id"] not in cot:
                    dropped += 1
                    continue
                reasoning = cot[rec["id"]][1]
            examples.append(SFTExample(rec["id"], mode, rec["messages"][:2], ids, reasoning,
                                       meta=dict(dataset=rec.get("dataset"))))
            if limit and len(examples) >= limit:
                break
    return examples, dropped


def _hash_key(sample_id: str, seed: int) -> str:
    return hashlib.md5(f"{seed}:{sample_id}".encode()).hexdigest()


def holdout_ids(path=SFT_JSONL, n_holdout: int = 200, seed: int = 0) -> list[str]:
    """The ``n`` ids with smallest hash among *all* records in the file.

    Computed over the whole file (not just the trainable subset) so the held-out
    set is identical for SFT, evaluation and GRPO whatever CoT coverage is.
    """
    with open(path) as f:
        ids = [json.loads(line)["id"] for line in f]
    return sorted(ids, key=lambda i: _hash_key(i, seed))[:n_holdout]


def split_holdout(examples, n_holdout: int, seed: int = 0, held_ids=None):
    """(train, held) with held-out chosen by id hash, or by explicit ``held_ids``."""
    if held_ids is None:
        held_ids = sorted((e.sample_id for e in examples), key=lambda i: _hash_key(i, seed))[:n_holdout]
    held_ids = set(held_ids)
    return ([e for e in examples if e.sample_id not in held_ids],
            [e for e in examples if e.sample_id in held_ids])


class SFTDataset(Dataset):
    def __init__(self, examples: list[SFTExample]):
        self.examples = examples

    @classmethod
    def from_jsonl(cls, path=SFT_JSONL, cot_path=COT_JSONL, limit=None):
        from .cot_annotator import read_annotations

        cot = read_annotations(cot_path) if cot_path and _exists(cot_path) else None
        examples, _ = load_examples(path, cot, limit)
        return cls(examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> SFTExample:
        return self.examples[i]

    def modes(self) -> list[str]:
        return [e.mode for e in self.examples]


def _exists(p):
    from pathlib import Path
    return Path(p).exists()


class MixedModeSampler(Sampler):
    """Yields indices in repeating [fast x k, slow] blocks (paper: ~3:1).

    Both pools are shuffled each epoch; the epoch ends when either pool cannot
    fill another block, so no record repeats within an epoch.
    """

    def __init__(self, modes: list[str], fast_per_slow: int = 3, seed: int = 0):
        self.fast = [i for i, m in enumerate(modes) if m == "fast"]
        self.slow = [i for i, m in enumerate(modes) if m == "slow"]
        self.k, self.seed, self.epoch = fast_per_slow, seed, 0
        if not self.slow or not self.fast:
            raise ValueError("MixedModeSampler needs both fast and slow examples")

    def __len__(self):
        return min(len(self.fast) // self.k, len(self.slow)) * (self.k + 1)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed * 1000003 + self.epoch)
        fast, slow = self.fast[:], self.slow[:]
        rng.shuffle(fast); rng.shuffle(slow)
        for b in range(len(self) // (self.k + 1)):
            block = fast[b * self.k:(b + 1) * self.k] + [slow[b]]
            rng.shuffle(block)
            yield from block


def consumed_ids(train: list[SFTExample], steps: int, grad_accum: int, batch_size: int = 1,
                 fast_per_slow: int = 3, order_seed: int = 0, epoch: int = 0) -> set[str]:
    """Ids of the records the first ``steps`` optimizer steps of an SFT run drew.

    The sampler is a seeded shuffle, so a finished run's training set can be replayed
    exactly. Everything *not* returned was never seen, which is what the uncertainty
    head needs (it has to learn the policy's error on scenes it did not fit).
    """
    sampler = MixedModeSampler([e.mode for e in train], fast_per_slow, order_seed)
    sampler.set_epoch(epoch)
    n = steps * grad_accum * batch_size
    order = list(sampler)
    if n > len(order):
        raise ValueError(f"{steps} steps need {n} records but one epoch only yields {len(order)}")
    return {train[i].sample_id for i in order[:n]}


class SFTCollator:
    """Encode examples into padded tensors with Eq.-2 masks and per-sample w."""

    def __init__(self, tokenizer, processor=None, max_length: int | None = None,
                 lambda_cot: float = LAMBDA_COT):
        from .inference import PromptEncoder

        self.tok, self.enc, self.max_length = tokenizer, PromptEncoder(tokenizer, processor), max_length
        self.lambda_cot = lambda_cot
        self.pad_id = tokenizer.pad_token_id
        ids = tokenizer.convert_tokens_to_ids(["<action_0>", "<action_2047>"])
        self.act_lo, self.act_hi = ids[0], ids[1]

    def encode(self, ex: SFTExample) -> dict:
        prompt = self.enc(ex.messages, None)
        p_ids = prompt["input_ids"][0]
        resp_ids = self.tok(ex.response + IM_END, add_special_tokens=False,
                            return_tensors="pt")["input_ids"][0]
        ids = torch.cat([p_ids, resp_ids])
        if self.max_length and len(ids) > self.max_length:
            raise ValueError(f"{ex.sample_id}: {len(ids)} tokens exceeds max_length={self.max_length}")
        is_action = (resp_ids >= self.act_lo) & (resp_ids <= self.act_hi)
        is_action[-1] = True                                   # closing <|im_end|> -> L_action
        n_p = len(p_ids)
        action_mask = torch.cat([torch.zeros(n_p, dtype=torch.bool), is_action])
        lm_mask = torch.cat([torch.zeros(n_p, dtype=torch.bool), ~is_action])
        extra = {k: v for k, v in prompt.items() if k not in ("input_ids", "attention_mask")}
        return dict(input_ids=ids, action_mask=action_mask, lm_mask=lm_mask,
                    weight=self.lambda_cot if ex.mode == "slow" else 1.0,   # Eq. 2's w
                    extra=extra, n_prompt=n_p)

    def __call__(self, batch: list[SFTExample]) -> dict:
        items = [self.encode(e) for e in batch]
        L = max(len(it["input_ids"]) for it in items)
        B = len(items)
        input_ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        attn = torch.zeros(B, L, dtype=torch.long)
        act = torch.zeros(B, L, dtype=torch.bool)
        lm = torch.zeros(B, L, dtype=torch.bool)
        for b, it in enumerate(items):
            n = len(it["input_ids"])
            input_ids[b, :n] = it["input_ids"]; attn[b, :n] = 1
            act[b, :n] = it["action_mask"]; lm[b, :n] = it["lm_mask"]
        out = dict(input_ids=input_ids, attention_mask=attn, action_mask=act, lm_mask=lm,
                   weight=torch.tensor([it["weight"] for it in items]),
                   modes=[e.mode for e in batch], ids=[e.sample_id for e in batch])
        for key in ("pixel_values", "image_grid_thw"):
            parts = [it["extra"][key] for it in items if key in it["extra"]]
            if parts:
                out[key] = torch.cat(parts)
        return out
