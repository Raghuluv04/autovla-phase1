"""Chat rendering and the dual-mode response format shared by every stage.

SFT targets, GRPO rollouts and evaluation must agree byte-for-byte on how a
response is laid out, so the layout lives here once:

  fast : ``FAST_PREFIX``\\n``<action_i>`` x 10
  slow : ``SLOW_PREFIX``\\n``Scene: ..``\\n``Critical objects: ..``\\n
         ``Predicted intentions: ..``\\n``Driving decision: ..``\\n``<action_i>`` x 10

Rendering is hand-rolled ChatML (what Qwen's template emits) so it needs no
processor and works with any tokenizer that has the Qwen special tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from autovla.schema import Reasoning

from .config import ACTION_TOKEN_PREFIX, FAST_PREFIX, N_ACTION_TOKENS, SLOW_PREFIX

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
ACTION_RE = re.compile(r"<action_(\d+)>")
COT_FIELDS = (
    ("scene_description", "Scene:"),
    ("critical_objects", "Critical objects:"),
    ("agent_intentions", "Predicted intentions:"),
    ("driving_decision", "Driving decision:"),
)
_COT_RE = re.compile(
    r"Scene:\s*(?P<s>.*?)\s*Critical objects:\s*(?P<c>.*?)\s*"
    r"Predicted intentions:\s*(?P<i>.*?)\s*Driving decision:\s*(?P<d>.*)", re.S)


def action_text(ids) -> str:
    return "".join(f"{ACTION_TOKEN_PREFIX}{int(i)}>" for i in ids)


def extract_action_ids(text: str) -> list[int]:
    return [int(m) for m in ACTION_RE.findall(text)]


def assistant_prefix(mode: str) -> str:
    """Text the assistant turn starts with for a given thinking mode."""
    if mode == "fast":
        return FAST_PREFIX + "\n"
    if mode == "slow":
        return SLOW_PREFIX + "\n" + COT_FIELDS[0][1]
    raise ValueError(f"mode must be 'fast' or 'slow', got {mode!r}")


def build_response(mode: str, action_ids, reasoning: Reasoning | None = None) -> str:
    """Full assistant turn for a training target."""
    acts = action_text(action_ids)
    if mode == "fast":
        return f"{FAST_PREFIX}\n{acts}"
    if reasoning is None or reasoning.is_empty():
        raise ValueError("slow-thinking response needs a non-empty Reasoning")
    return f"{SLOW_PREFIX}\n{reasoning.to_text()}\n{acts}"


@dataclass
class ParsedResponse:
    mode: str                       # 'fast' | 'slow' | 'unknown'
    cot_text: str                   # text between the mode prefix and first action
    reasoning: Reasoning | None
    action_ids: list[int]

    @property
    def n_actions(self) -> int:
        return len(self.action_ids)

    @property
    def valid(self) -> bool:
        return self.n_actions == N_ACTION_TOKENS


def parse_response(text: str) -> ParsedResponse:
    """Split a (generated) assistant turn into mode, CoT and action ids."""
    head = text.split(ACTION_TOKEN_PREFIX, 1)[0]
    ids = extract_action_ids(text)
    if head.strip().startswith(SLOW_PREFIX):
        mode = "slow"
    elif head.strip().startswith(FAST_PREFIX):
        mode = "fast"
    else:
        mode = "unknown"
    m = _COT_RE.search(head)
    reasoning = None
    if m:
        reasoning = Reasoning(scene_description=m["s"], critical_objects=m["c"],
                              agent_intentions=m["i"], driving_decision=m["d"].strip())
    cot = head.replace(SLOW_PREFIX, "", 1).replace(FAST_PREFIX, "", 1).strip()
    return ParsedResponse(mode=mode, cot_text=cot, reasoning=reasoning, action_ids=ids)


def render_chat(messages, assistant_text: str | None = None):
    """Render system+user (and optionally a partial/complete assistant) turns.

    Returns ``(text, image_paths)``. With ``assistant_text=None`` the string ends
    at the assistant header, ready for generation; otherwise the assistant turn
    is appended *without* a closing ``<|im_end|>`` when ``assistant_text`` is a
    generation prefix — use :func:`render_training_text` for a closed turn.
    """
    parts, images = [], []
    for msg in messages:
        if msg["role"] == "assistant":
            continue
        content = msg["content"]
        if isinstance(content, str):
            body = content
        else:
            chunks = []
            for item in content:
                if item["type"] == "image":
                    chunks.append(IMAGE_PLACEHOLDER)
                    images.append(item["image"])
                else:
                    chunks.append(item["text"])
            body = "".join(chunks) if len(chunks) == 1 else "\n".join(chunks)
        parts.append(f"{IM_START}{msg['role']}\n{body}{IM_END}\n")
    parts.append(f"{IM_START}assistant\n")
    if assistant_text:
        parts.append(assistant_text)
    return "".join(parts), images


def render_training_text(messages, response: str):
    """(prompt_text, full_text, images); full_text = prompt + response + <|im_end|>."""
    prompt, images = render_chat(messages)
    return prompt, prompt + response + IM_END, images
