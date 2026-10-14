"""Build a ready-to-use backbone (real or stand-in) and restore a checkpoint.

A stand-in checkpoint directory holds ``meta.json`` (hidden size, layers, LoRA
rank) so later stages rebuild the identical architecture before loading weights.
"""

from __future__ import annotations

import json
from pathlib import Path

from .backbone import Backbone, load_model, load_trainable
from .config import BackboneConfig

META = "meta.json"


def write_meta(out_dir, spec: dict) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / META).write_text(json.dumps(spec))


def read_meta(checkpoint) -> dict:
    p = Path(checkpoint).parent / META
    return json.loads(p.read_text()) if p.exists() else {}


def make_backbone(standin: bool = False, cfg: BackboneConfig | None = None,
                  checkpoint: str | None = None, seed: int = 0, spec: dict | None = None) -> Backbone:
    """``spec`` (``standin``/``hidden``/``layers``/``lora_rank``) overrides the flags;
    when a checkpoint is given its sibling ``meta.json`` supplies the spec."""
    cfg = cfg or BackboneConfig()
    spec = spec or (read_meta(checkpoint) if checkpoint else {}) or {}
    standin = spec.get("standin", standin)
    if standin:
        from .standin import build_tiny_backbone

        cfg.load_in_4bit = False
        if "lora_rank" in spec:
            cfg.lora.rank = cfg.lora.alpha = spec["lora_rank"]
        model, tok = build_tiny_backbone(seed=seed, hidden=spec.get("hidden", 32),
                                         layers=spec.get("layers", 2))
        bb = load_model(cfg, model=model, tokenizer=tok)
    else:
        bb = load_model(cfg)
    if checkpoint:
        load_trainable(bb.model, checkpoint)
    bb.model.eval()
    return bb


_CODEC = None


def load_action_codec():
    """The Phase-I ActionTokenizer (codebook lookup / trajectory decoding), cached."""
    global _CODEC
    if _CODEC is None:
        from autovla.codebook import Codebook
        from autovla.tokenizer import ActionTokenizer

        from .config import CODEBOOK_NPZ

        _CODEC = ActionTokenizer(Codebook.load(CODEBOOK_NPZ))
    return _CODEC
