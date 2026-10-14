"""Phase II constants. Values traceable to the AutoVLA paper are cited inline.

Phase I constants (codebook size, dt, horizon, ...) are re-exported from
``autovla.config`` so there is a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from autovla.config import (CODEBOOK_SIZE, DT, EGO_LENGTH, EGO_WIDTH,  # noqa: F401
                            HORIZON_S, N_ACTION_TOKENS)

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "data_pipeline" / "outputs" / "artifacts"
SFT_JSONL = ARTIFACTS / "sft_train.jsonl"
UNIFIED_JSONL = ARTIFACTS / "unified_samples.jsonl"
CODEBOOK_NPZ = ARTIFACTS / "action_codebook.npz"
ACTION_VOCAB_JSON = ARTIFACTS / "action_vocab.json"
PHASE2_DIR = REPO_ROOT / "outputs" / "phase2"          # Phase II artifacts
COT_JSONL = PHASE2_DIR / "cot_annotations.jsonl"
RESULTS_DIR = REPO_ROOT / "results"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

# --- Backbone (paper Sec. 3.1 / App. B) -----------------------------------
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ACTION_TOKEN_PREFIX = "<action_"
LORA_RANK = 8                # paper's RFT setting
LORA_ALPHA = 8
LORA_DROPOUT = 0.1
# Every attention/MLP projection of the *language* model; the negative lookahead
# keeps the vision tower (whose MLP reuses the gate/up/down_proj names) frozen
# and works across transformers versions with different module nesting.
LORA_TARGET_REGEX = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
NEW_TOKEN_INIT_STD = 0.02    # noise added to the mean embedding for new tokens
EMBED_PAD_MULTIPLE = 64

# --- Response templates (paper Sec. 3.2/3.3) -------------------------------
FAST_PREFIX = "No additional reasoning required."
SLOW_PREFIX = "This scenario requires careful reasoning."
COT_FIELDS_TEXT = "Scene: Critical objects: Predicted intentions: Driving decision:"
COT_MAX_TOKENS = 512         # hard cap on generated CoT; paper's L_tol is 400

# --- CoT annotation (Milestone 5) -----------------------------------------
COT_MIN_FRACTION = 0.15      # at least this share of samples must carry real CoT
MAX_SENTENCES_PER_FIELD = 3

# --- SFT loss, Eq. 2 -------------------------------------------------------
LAMBDA_ACTION = 2.0          # paper Sec. 3.3
LAMBDA_COT = 0.5
SFT_GRAD_ACCUM = 8

# --- Open-loop evaluation (UniAD protocol) ---------------------------------
EVAL_HORIZONS_S = (1.0, 2.0, 3.0)

# Scene difficulty (hard split = any of: sharp manoeuvre, stationary->moving
# transition, high occlusion). Thresholds are in metres / radians / m/s.
HARD_HEADING_RAD = 0.5       # net yaw change over the horizon
HARD_LATERAL_M = 3.0         # net lateral displacement
HARD_START_SPEED = 0.5       # "stationary" at t=0 ...
HARD_END_SPEED = 2.0         # ... and moving by the end of the horizon
HARD_OCCLUSION = 0.5         # DrivingSample.extra["occlusion"] fraction, if present
COLLISION_SAFETY_MARGIN_M = 0.0   # inflate agent boxes by this much (UniAD: none)

# --- GRPO (paper Sec. 3.4 + App. D) ----------------------------------------
GRPO_GROUP_SIZE = 8
REWARD_DELTA = 2.0           # (delta - ADE) / kappa, Waymo/nuScenes variant
REWARD_KAPPA = 10.0
COT_L_TOL = 400
COT_GAMMA = 2e-3
LAMBDA_R = 0.3
# Team extension: r_calibration = -beta * confidence * 1[ADE > tau_fail]
CALIB_BETA = 0.5
CALIB_TAU_FAIL = 0.5         # metres
GRPO_KL_BETA = 0.04

# --- Uncertainty (Milestone 8) ---------------------------------------------
MC_DROPOUT_T = 10
UNC_DROPOUT_P = 0.2
N_ENSEMBLE = 3
ECE_BINS = 10
ECE_TARGET = 0.05
AUROC_TARGET = 0.6

# --- Gating (Milestone 9) --------------------------------------------------
GATE_DEFAULT_HIGH = 0.75
GATE_DEFAULT_LOW = 0.35
GATE_ADE_SCALE = 5.0         # driving score = route_completion * (1 - collided) * clip(1 - ADE / scale, 0, 1)
GATE_LATENCY_WEIGHT = 0.1    # J = mean(driving score) - weight * mean(latency / mean slow latency)
GATE_GRID_STEPS = 21         # candidate thresholds = confidence quantiles on a grid this fine
MRM_DECEL = 3.0              # m/s^2, comfortable braking for the minimal-risk manoeuvre


@dataclass
class LoraSettings:
    rank: int = LORA_RANK
    alpha: int = LORA_ALPHA
    dropout: float = LORA_DROPOUT
    target_regex: str = LORA_TARGET_REGEX


@dataclass
class BackboneConfig:
    model_id: str = MODEL_ID
    load_in_4bit: bool = True            # bitsandbytes NF4; ignored without CUDA
    compute_dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    attn_implementation: str = "sdpa"
    action_vocab_path: str = str(ACTION_VOCAB_JSON)
    lora: LoraSettings = field(default_factory=LoraSettings)
    max_pixels: int = 224 * 224 * 4      # keeps visual tokens bounded on a T4
