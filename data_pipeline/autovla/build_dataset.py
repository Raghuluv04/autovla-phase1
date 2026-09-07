"""Turn unified DrivingSamples into the SFT training corpus.

This is the final data-processing stage: each sample's ground-truth trajectory
is tokenized into <action_i> tokens and packed into a chat record whose
assistant turn contains (optionally) CoT reasoning followed by the action
sequence. Fast- and slow-thinking records are mixed, matching the SFT recipe in
Sec. 3.3.
"""

from __future__ import annotations

import json

import numpy as np

from .config import N_ACTION_TOKENS
from .prompts import build_conversation
from .tokenizer import ActionTokenizer


def build_sft_records(samples, tokenizer: ActionTokenizer,
                      n_tokens=N_ACTION_TOKENS, max_images=None):
    records, stats = [], dict(fast=0, slow=0, dropped=0, ade=[])
    for s in samples:
        traj = np.asarray(s.future_traj, dtype=float)[: n_tokens + 1]
        if len(traj) < n_tokens + 1:
            stats["dropped"] += 1
            continue
        ids = tokenizer.encode(traj)
        stats["ade"].append(tokenizer.reconstruction_error(traj)["ade"])

        paths = []
        for cam, seq in s.images.items():
            paths.extend([p for p in seq if p])
        if max_images is not None:
            paths = paths[:max_images]

        records.append(build_conversation(s, tokenizer.ids_to_text(ids), paths))
        stats[s.thinking_mode] += 1
    stats["ade"] = float(np.mean(stats["ade"])) if stats["ade"] else None
    return records, stats


def write_records(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def write_action_vocab(tokenizer: ActionTokenizer, path):
    """The special tokens that must be added to the VLM tokenizer."""
    with open(path, "w") as f:
        json.dump({"additional_special_tokens": tokenizer.vocab}, f, indent=2)
    return path
