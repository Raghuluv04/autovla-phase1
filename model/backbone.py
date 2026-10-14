"""Milestone 4: Qwen2.5-VL backbone, action-token vocabulary, LoRA.

Pipeline (each step is a separate function so each can be tested alone):

  load_base()         Qwen2.5-VL + processor, NF4 4-bit when CUDA is present
  extend_tokenizer()  append <action_0>..<action_2047> as special tokens
  resize embeddings   grow the embedding (and tied/untied head) to fit them
  freeze_vision()     vision tower + merger get requires_grad=False
  attach_lora()       LoRA on every language-model projection; the *rows* of the
                      embedding that belong to the new tokens are the only base
                      weights that train (PEFT ``trainable_token_indices``)

``load_model`` runs all of it. Passing ``model``/``tokenizer`` explicitly (e.g.
from :mod:`model.standin`) skips the download and reuses the identical code path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import torch

from .config import (ACTION_TOKEN_PREFIX, CODEBOOK_SIZE, EMBED_PAD_MULTIPLE,
                     BackboneConfig)


@dataclass
class Backbone:
    """Everything downstream stages need, bundled."""

    model: torch.nn.Module
    tokenizer: object
    processor: object | None
    action_token_ids: list[int]
    cfg: BackboneConfig = field(default_factory=BackboneConfig)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def first_action_id(self) -> int:
        return self.action_token_ids[0]

    @property
    def eos_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("<|im_end|>")


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------
def load_action_tokens(path) -> list[str]:
    with open(path) as f:
        tokens = json.load(f)["additional_special_tokens"]
    expected = [f"{ACTION_TOKEN_PREFIX}{i}>" for i in range(CODEBOOK_SIZE)]
    if tokens != expected:
        raise ValueError(
            f"{path}: expected {CODEBOOK_SIZE} tokens <action_0>..<action_{CODEBOOK_SIZE - 1}> "
            f"in order, got {len(tokens)}")
    return tokens


def extend_tokenizer(tokenizer, vocab_path) -> list[int]:
    """Add the action tokens; return their ids (contiguous, in codebook order).

    Idempotent: a tokenizer that already contains them is left untouched.
    """
    tokens = load_action_tokens(vocab_path)
    # add_tokens(special_tokens=True) appends without clobbering Qwen's existing
    # additional specials (<|im_start|>, <|vision_start|>, ...), unlike
    # add_special_tokens, whose replace semantics differ across versions.
    tokenizer.add_tokens(tokens, special_tokens=True)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if ids != list(range(ids[0], ids[0] + len(ids))):
        raise RuntimeError("action tokens did not receive contiguous ids")
    if tokenizer.convert_ids_to_tokens(ids[0]) != tokens[0]:
        raise RuntimeError("action token id round-trip failed")
    return ids


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------
def load_base(cfg: BackboneConfig):
    """Download/load Qwen2.5-VL and its processor. Needs network + (ideally) CUDA."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    import transformers
    from packaging.version import Version

    dtype = getattr(torch, cfg.compute_dtype)
    # `torch_dtype` was renamed `dtype` in transformers 4.56
    dtype_key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    kwargs = {dtype_key: dtype, "attn_implementation": cfg.attn_implementation}
    if cfg.load_in_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        kwargs["device_map"] = {"": torch.cuda.current_device()}
    elif cfg.load_in_4bit:
        print("[backbone] CUDA unavailable: bitsandbytes 4-bit skipped, loading "
              f"{cfg.compute_dtype} weights instead.")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg.model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(cfg.model_id, max_pixels=cfg.max_pixels)
    return model, processor


def resize_for_new_tokens(model, tokenizer) -> int:
    """Grow embedding (and untied head) to cover the enlarged vocabulary."""
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=EMBED_PAD_MULTIPLE)
    return model.get_input_embeddings().weight.shape[0]


def vision_tower(model):
    """The vision encoder module, wherever this transformers version nests it."""
    for path in ("model.visual", "visual"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("model has no vision tower")


def freeze_vision(model) -> int:
    n = 0
    for p in vision_tower(model).parameters():
        p.requires_grad = False
        n += p.numel()
    return n


def attach_lora(model, cfg: BackboneConfig, new_token_ids):
    """LoRA on the language model + trainable rows for the new tokens only."""
    from peft import LoraConfig, get_peft_model

    tied = bool(getattr(model.config.get_text_config(), "tie_word_embeddings", False))
    token_modules = {"embed_tokens": list(new_token_ids)}
    if not tied:                       # untied head needs its own new rows trained
        token_modules["lm_head"] = list(new_token_ids)
    lcfg = LoraConfig(
        r=cfg.lora.rank, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_regex, bias="none",
        trainable_token_indices=token_modules)
    return get_peft_model(model, lcfg)


def trainable_summary(model) -> dict:
    """Parameter counts by group; used to assert the LoRA-only invariant."""
    groups = dict(lora=0, new_tokens=0, vision=0, other=0, total=0)
    for name, p in model.named_parameters():
        groups["total"] += p.numel()
        if not p.requires_grad:
            continue
        if "lora_" in name:
            groups["lora"] += p.numel()
        elif "trainable_tokens" in name:
            groups["new_tokens"] += p.numel()
        elif "visual" in name:
            groups["vision"] += p.numel()
        else:
            groups["other"] += p.numel()
    groups["trainable"] = groups["lora"] + groups["new_tokens"] + groups["vision"] + groups["other"]
    return groups


def assert_lora_only(model) -> dict:
    s = trainable_summary(model)
    if s["vision"] or s["other"]:
        bad = [n for n, p in model.named_parameters()
               if p.requires_grad and "lora_" not in n and "trainable_tokens" not in n]
        raise AssertionError(f"non-LoRA parameters are trainable: {bad[:5]}")
    return s


def build_backbone(model, tokenizer, processor=None, cfg: BackboneConfig | None = None) -> Backbone:
    """Steps shared by the real model and the stand-in."""
    cfg = cfg or BackboneConfig()
    ids = extend_tokenizer(tokenizer, cfg.action_vocab_path)
    resize_for_new_tokens(model, tokenizer)
    freeze_vision(model)
    for p in model.parameters():            # base is frozen; LoRA re-enables its own
        p.requires_grad = False
    if cfg.load_in_4bit and torch.cuda.is_available():
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_model = attach_lora(model, cfg, ids)
    if cfg.gradient_checkpointing:
        peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        peft_model.enable_input_require_grads()
    assert_lora_only(peft_model)
    return Backbone(peft_model, tokenizer, processor, ids, cfg)


def load_model(cfg: BackboneConfig | None = None, model=None, tokenizer=None) -> Backbone:
    """Full M4 setup. Pass ``model``+``tokenizer`` to bypass the download."""
    cfg = cfg or BackboneConfig()
    processor = None
    if model is None:
        model, processor = load_base(cfg)
        tokenizer = processor.tokenizer
    return build_backbone(model, tokenizer, processor, cfg)


def trainable_state_dict(model) -> dict:
    """Only the parameters that train (LoRA + new-token rows); a few MB, not GB."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def save_trainable(model, path) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(trainable_state_dict(model), path)


def load_trainable(model, path) -> int:
    state = torch.load(path, map_location="cpu")
    params = dict(model.named_parameters())
    missing = [n for n in state if n not in params]
    if missing:
        raise KeyError(f"checkpoint has parameters this model lacks: {missing[:3]}")
    with torch.no_grad():
        for n, v in state.items():
            params[n].copy_(v.to(params[n].dtype))
    return len(state)
