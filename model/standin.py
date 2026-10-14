"""Tiny, hermetic stand-ins so every module can be smoke-tested on CPU in seconds.

``build_tiny_backbone`` returns a *real* ``Qwen2_5_VLForConditionalGeneration``
with ~100k parameters and a byte-level BPE tokenizer trained on the project's own
prompts, so the exact same ``backbone.py`` code path (tokenizer extension,
embedding resize, LoRA, vision freeze) is exercised as for the 3B model. Nothing
here touches the network or the Hugging Face cache.
"""

from __future__ import annotations

import functools

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from autovla.prompts import SYSTEM_PROMPT

from .config import COT_FIELDS_TEXT, FAST_PREFIX, SLOW_PREFIX

QWEN_SPECIALS = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
    "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",
]
_CORPUS = [
    SYSTEM_PROMPT, FAST_PREFIX, SLOW_PREFIX, COT_FIELDS_TEXT,
    "Camera observations: none available for this sample.",
    "Ego state: speed 0.00 m/s, acceleration 0.00 m/s^2.",
    "Navigation instruction: Go Straight. Turn Left. Turn Right. Stop.",
    "Plan the next 5 seconds. 0123456789 . - ,",
    "the ego vehicle is approaching an intersection with a pedestrian on the left; "
    "a lead car ahead brakes; the road is wet and dark; keep lane and slow down.",
]


@functools.lru_cache(maxsize=1)
def _base_tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400, special_tokens=QWEN_SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(_CORPUS * 4, trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>",
        bos_token=None, unk_token=None)


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    """Fresh copy (callers add tokens to it, so never share the cached one)."""
    import copy
    return copy.deepcopy(_base_tokenizer())


def _mrope_sections(hidden: int, heads: int) -> list[int]:
    half = hidden // heads // 2          # must sum to head_dim / 2
    return [half // 4, half // 4, half - 2 * (half // 4)]


def tiny_config(tok, tie_word_embeddings: bool = True, hidden: int = 32, layers: int = 2) -> Qwen2_5_VLConfig:
    v = len(tok)
    ids = {t: tok.convert_tokens_to_ids(t) for t in QWEN_SPECIALS}
    return Qwen2_5_VLConfig(
        text_config=dict(
            vocab_size=v, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
            rope_scaling={"type": "mrope", "mrope_section": _mrope_sections(hidden, 4)},
            tie_word_embeddings=tie_word_embeddings, bos_token_id=ids["<|endoftext|>"],
            eos_token_id=ids["<|im_end|>"], pad_token_id=ids["<|endoftext|>"]),
        vision_config=dict(
            depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=hidden,
            patch_size=14, spatial_merge_size=2, temporal_patch_size=2,
            window_size=56, fullatt_block_indexes=[1]),
        image_token_id=ids["<|image_pad|>"], video_token_id=ids["<|video_pad|>"],
        vision_start_token_id=ids["<|vision_start|>"],
        vision_end_token_id=ids["<|vision_end|>"],
        tie_word_embeddings=tie_word_embeddings, bos_token_id=ids["<|endoftext|>"],
        eos_token_id=ids["<|im_end|>"], pad_token_id=ids["<|endoftext|>"])


def build_tiny_backbone(seed: int = 0, tie_word_embeddings: bool = True, hidden: int = 32, layers: int = 2):
    """(model, tokenizer): random-weight Qwen2.5-VL + matching tokenizer."""
    torch.manual_seed(seed)
    tok = tiny_tokenizer()
    model = Qwen2_5_VLForConditionalGeneration(tiny_config(tok, tie_word_embeddings, hidden, layers))
    # A random frozen embedding table at the default std (0.02) caps every logit
    # gap near 1, so LoRA could never fit even a fixed string. Real checkpoints
    # have trained, well-scaled embeddings; mimic that scale here.
    with torch.no_grad():
        model.get_input_embeddings().weight.normal_(0.0, 0.3)
        if not tie_word_embeddings:
            model.lm_head.weight.normal_(0.0, 0.3)
    model.eval()
    return model, tok
