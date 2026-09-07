"""System / user prompt construction and the dual-mode response format.

The paper renders its exact prompt as an image (Fig. S5), so the text below is
a reconstruction that follows the structure it describes (Appendix C):
  * system prompt: role, task, expected four-step CoT format,
  * user message: multi-view camera observations, ego state (speed,
    acceleration), high-level driving instruction,
  * response: reasoning tokens followed by action tokens, with a short fixed
    template standing in for the reasoning in fast-thinking mode.
"""

from __future__ import annotations

from .config import CAMERAS, HORIZON_S, N_ACTION_TOKENS

SYSTEM_PROMPT = f"""You are an autonomous driving planner. You are given \
surround-view camera streams from the front-left, front, and front-right \
cameras of a vehicle, the vehicle's current state, and a high-level navigation \
instruction. Your task is to plan a safe, comfortable and rule-compliant \
{HORIZON_S:.0f}-second trajectory for the ego vehicle.

You operate in two modes:
- Fast thinking: when the scene is straightforward, output the planning \
trajectory directly without reasoning.
- Slow thinking: when the scene is complex or ambiguous, reason step by step \
before planning, using exactly these four steps:
  1. Scene: describe the road layout, traffic control, and conditions.
  2. Critical objects: identify the agents and elements that constrain the \
ego vehicle's motion.
  3. Predicted intentions: state what those agents are likely to do next.
  4. Driving decision: state the meta-action and justify it from steps 1-3.

Always end your response with exactly {N_ACTION_TOKENS} action tokens of the \
form <action_i>, which decode into the planned trajectory. Each action token \
represents 0.5 seconds of vehicle motion."""

FAST_TEMPLATE = "This is a straightforward scenario; no reasoning is needed."
SLOW_TEMPLATE = "This scenario requires careful reasoning."


def build_user_message(sample):
    """User turn: camera observations, ego state, navigation instruction."""
    n_frames = len(next(iter(sample.images.values()))) if sample.images else 0
    views = ", ".join(c.replace("CAM_", "").replace("_", "-").lower()
                      for c in CAMERAS if c in sample.images)
    if n_frames:
        obs = (f"Camera observations: {n_frames} consecutive frames at 2 Hz "
               f"from the {views} cameras.")
    else:
        # Trajectory-only corpus (e.g. the synthetic stand-in): the model is
        # conditioned on state and instruction alone.
        obs = "Camera observations: none available for this sample."
    return (
        f"{obs}\n"
        f"Ego state: speed {sample.velocity:.2f} m/s, "
        f"acceleration {sample.acceleration:.2f} m/s^2.\n"
        f"Navigation instruction: {sample.command}.\n"
        f"Plan the next {HORIZON_S:.0f} seconds."
    )


def build_response(sample, action_text):
    """Assistant turn: reasoning template (+ CoT if present) then actions."""
    if sample.has_reasoning:
        return f"{SLOW_TEMPLATE}\n{sample.reasoning.to_text()}\n{action_text}"
    return f"{FAST_TEMPLATE}\n{action_text}"


def build_conversation(sample, action_text, image_paths=None):
    """One training record in the standard multimodal chat format."""
    content = []
    for p in (image_paths or []):
        content.append({"type": "image", "image": p})
    content.append({"type": "text", "text": build_user_message(sample)})
    return {
        "id": sample.sample_id,
        "dataset": sample.dataset,
        "thinking_mode": sample.thinking_mode,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
            {"role": "assistant", "content": build_response(sample, action_text)},
        ],
    }
