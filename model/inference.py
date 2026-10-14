"""Prompt encoding and constrained plan generation.

A plan is ``[mode prefix][CoT?]<action_i> x 10``. Generation enforces the format:
once the first action token appears (or the CoT budget is spent) the logits are
masked to the 2048 action tokens until exactly ``N_ACTION_TOKENS`` have been
emitted, then EOS is forced. Hence every plan decodes to a drivable trajectory,
even from an untrained model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from autovla.tokenizer import ActionTokenizer

from .backbone import Backbone
from .config import COT_MAX_TOKENS, N_ACTION_TOKENS
from .prompting import assistant_prefix, parse_response, render_chat


class PromptEncoder:
    """messages (+ assistant prefix) -> model inputs. Uses the processor for images."""

    def __init__(self, tokenizer, processor=None):
        self.tokenizer, self.processor = tokenizer, processor

    def __call__(self, messages, assistant_text: str | None = None) -> dict:
        text, image_paths = render_chat(messages, assistant_text)
        if image_paths:
            if self.processor is None:
                raise ValueError("record has images but no processor was provided")
            from PIL import Image

            imgs = [Image.open(p).convert("RGB") for p in image_paths]
            return dict(self.processor(text=[text], images=imgs, return_tensors="pt"))
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return dict(enc)


class PlanFormatProcessor(LogitsProcessor):
    """Stateless format enforcement; state is read off ``input_ids``."""

    def __init__(self, prompt_len, first_action, n_codes, eos_id, cot_max,
                 n_actions=N_ACTION_TOKENS):
        self.prompt_len, self.first, self.n_codes = prompt_len, first_action, n_codes
        self.eos, self.cot_max, self.n_actions = eos_id, cot_max, n_actions

    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.prompt_len:]
        n_act = ((gen >= self.first) & (gen < self.first + self.n_codes)).sum(1)
        mask = torch.zeros_like(scores, dtype=torch.bool)       # True = allowed
        for b in range(scores.shape[0]):
            k = int(n_act[b])
            if k >= self.n_actions:
                mask[b, self.eos] = True
            elif k > 0 or gen.shape[1] >= self.cot_max:
                mask[b, self.first:self.first + self.n_codes] = True
            else:
                mask[b, :] = True
                mask[b, self.eos] = False                        # no EOS before actions
        return scores.masked_fill(~mask, float("-inf"))


@torch.no_grad()
def mode_logits(bb: Backbone, messages, encoder: PromptEncoder | None = None):
    """Next-token logits of the two mode openers, ordered [fast, slow].

    This is AutoVLA's *learned* fast/slow decision: the first token of the
    assistant turn is either the no-reasoning or the reasoning prefix's opener.
    """
    enc = encoder or PromptEncoder(bb.tokenizer, bb.processor)
    openers = [bb.tokenizer(assistant_prefix(m), add_special_tokens=False)["input_ids"][0]
               for m in ("fast", "slow")]
    if openers[0] == openers[1]:
        raise RuntimeError("fast/slow prefixes share their first token; cannot gate on it")
    inputs = {k: v.to(bb.device) for k, v in enc(messages, None).items()}
    logits = bb.model(**inputs).logits[0, -1]
    return logits[openers].float()


@dataclass
class PlanOutput:
    mode: str                       # mode actually used ('auto' resolves to fast/slow)
    text: str
    cot_text: str
    action_ids: list[int]
    n_cot_tokens: int
    latency_s: float
    gen_ids: list[int] = field(default_factory=list)
    prompt_len: int = 0

    @property
    def valid(self) -> bool:
        return len(self.action_ids) == N_ACTION_TOKENS


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def generate_plan(bb: Backbone, messages, mode: str = "fast", *, do_sample: bool = False,
                  temperature: float = 1.0, cot_max_tokens: int = COT_MAX_TOKENS,
                  encoder: PromptEncoder | None = None, generator=None,
                  constrained: bool = True) -> PlanOutput:
    """Generate one plan.

    mode='fast'  force the no-reasoning prefix, emit actions immediately
    mode='slow'  force the reasoning prefix, free CoT (capped), then actions
    constrained=False disables the format-enforcing logits processor, so the
    output shows whether the model itself learned to end with 10 action tokens.

    mode='auto'  the model picks between the two prefixes itself (AutoVLA's
                 learned switch): argmax/sample over the two opener logits
    """
    enc = encoder or PromptEncoder(bb.tokenizer, bb.processor)
    t_start = time.perf_counter()
    if mode == "auto":
        lg = mode_logits(bb, messages, enc)
        if do_sample:
            idx = int(torch.multinomial(torch.softmax(lg / temperature, -1), 1, generator=generator))
        else:
            idx = int(lg.argmax())
        mode = ("fast", "slow")[idx]
    prefix = assistant_prefix(mode) if mode in ("fast", "slow") else None
    inputs = {k: v.to(bb.device) for k, v in enc(messages, prefix).items()}
    prompt_len = inputs["input_ids"].shape[1]

    cot_cap = 0 if mode == "fast" else cot_max_tokens
    proc = PlanFormatProcessor(prompt_len, bb.first_action_id, len(bb.action_token_ids),
                               bb.eos_id, cot_cap)
    gen_kwargs = dict(
        max_new_tokens=cot_cap + N_ACTION_TOKENS + 2,
        logits_processor=LogitsProcessorList([proc] if constrained else []),
        do_sample=do_sample, repetition_penalty=1.0,
        eos_token_id=bb.eos_id, pad_token_id=bb.tokenizer.pad_token_id)
    if do_sample:
        # Qwen's shipped generation_config sets top_k/top_p/repetition_penalty;
        # all three would distort an action-token distribution, so override.
        gen_kwargs.update(temperature=temperature, top_k=0, top_p=1.0)
        if generator is not None:
            gen_kwargs["generator"] = generator
    else:                                    # silence sampling-only warnings
        gen_kwargs.update(temperature=None)
    _sync(bb.device)
    out = bb.model.generate(**inputs, **gen_kwargs)
    _sync(bb.device)
    latency = time.perf_counter() - t_start   # includes the auto-mode decision

    gen_ids = out[0, prompt_len:].tolist()
    text = (prefix or "") + bb.tokenizer.decode(gen_ids, skip_special_tokens=False)
    for tok in ("<|im_end|>", "<|endoftext|>"):
        text = text.replace(tok, "")
    parsed = parse_response(text)
    n_cot = sum(1 for t in gen_ids if not (bb.first_action_id <= t < bb.first_action_id + len(bb.action_token_ids))
                and t not in (bb.eos_id, bb.tokenizer.pad_token_id))
    return PlanOutput(mode=mode, text=text,
                      cot_text=parsed.cot_text, action_ids=parsed.action_ids,
                      n_cot_tokens=n_cot, latency_s=latency, gen_ids=gen_ids,
                      prompt_len=prompt_len)


@torch.no_grad()
def generate_action_batch(bb: Backbone, messages, n: int, *, encoder: PromptEncoder | None = None) -> list[list[int]]:
    """``n`` greedy fast-mode plans for one prompt in a single batched ``generate``.

    Used for MC-dropout: with dropout switched on, each batch row draws its own
    mask, so the rows are n independent stochastic forward passes -- for roughly
    the price of one on a GPU, instead of n sequential generations.
    """
    enc = encoder or PromptEncoder(bb.tokenizer, bb.processor)
    inputs = {k: v.to(bb.device) for k, v in enc(messages, assistant_prefix("fast")).items()}
    prompt_len = inputs["input_ids"].shape[1]
    for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
        if k in inputs:
            inputs[k] = inputs[k].repeat(n, *([1] * (inputs[k].dim() - 1)))
    proc = PlanFormatProcessor(prompt_len, bb.first_action_id, len(bb.action_token_ids), bb.eos_id, 0)
    out = bb.model.generate(
        **inputs, max_new_tokens=N_ACTION_TOKENS + 2, logits_processor=LogitsProcessorList([proc]),
        do_sample=False, repetition_penalty=1.0, eos_token_id=bb.eos_id,
        pad_token_id=bb.tokenizer.pad_token_id, temperature=None)
    lo, hi = bb.first_action_id, bb.first_action_id + len(bb.action_token_ids)
    return [[t - lo for t in row[prompt_len:].tolist() if lo <= t < hi] for row in out]   # token id -> codebook index


def decode_plan(tok: ActionTokenizer, action_ids):
    """Action ids -> (11, 3) ego-frame trajectory."""
    return tok.decode(action_ids)
