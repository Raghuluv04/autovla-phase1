"""Milestone 10 rewards (paper Sec. 3.4 / App. D, plus the team's calibration term).

    r_Driving     = (delta - ADE) / kappa              delta = 2, kappa = 10
    r_CoT         = 1 / (1 + exp(-(L - L_tol) * gamma)) L_tol = 400 tokens, gamma = 2e-3
    r_calibration = -beta * confidence * 1[ADE > tau_fail]     beta = 0.5, tau = 0.5 m
    r             = r_Driving - lambda_r * r_CoT + r_calibration          lambda_r = 0.3

``r_calibration`` is the novel term: a *confident* wrong plan costs more than a
hesitant wrong one, which removes the incentive to fake certainty (and so skip
slow thinking) when the driving reward alone would not punish it.

Group-relative advantage: A_i = (r_i - mean(r)) / (std(r) + eps) within the G
samples drawn for one scene; a group with identical rewards yields zero advantage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (CALIB_BETA, CALIB_TAU_FAIL, COT_GAMMA, COT_L_TOL, LAMBDA_R, REWARD_DELTA,
                     REWARD_KAPPA)

ADV_EPS = 1e-6


@dataclass(frozen=True)
class RewardConfig:
    delta: float = REWARD_DELTA
    kappa: float = REWARD_KAPPA
    l_tol: float = COT_L_TOL
    gamma: float = COT_GAMMA
    lambda_r: float = LAMBDA_R
    beta: float = CALIB_BETA
    tau_fail: float = CALIB_TAU_FAIL
    use_calibration: bool = True


def driving_reward(ade, delta: float = REWARD_DELTA, kappa: float = REWARD_KAPPA):
    return (delta - np.asarray(ade, float)) / kappa


def cot_length_penalty(length, l_tol: float = COT_L_TOL, gamma: float = COT_GAMMA):
    """Sigmoid of the excess CoT length; 0.5 at ``l_tol`` tokens."""
    z = np.clip((np.asarray(length, float) - l_tol) * gamma, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def calibration_penalty(confidence, ade, beta: float = CALIB_BETA, tau_fail: float = CALIB_TAU_FAIL):
    """<= 0: ``-beta * confidence`` when the plan failed (ADE > tau_fail), else 0."""
    return -beta * np.asarray(confidence, float) * (np.asarray(ade, float) > tau_fail)


def total_reward(ade, cot_len, confidence=None, cfg: RewardConfig = RewardConfig()) -> dict:
    """Reward components for one or many samples (scalars or equal-length arrays)."""
    r_drive = driving_reward(ade, cfg.delta, cfg.kappa)
    r_cot = cot_length_penalty(cot_len, cfg.l_tol, cfg.gamma)
    r_cal = (calibration_penalty(confidence, ade, cfg.beta, cfg.tau_fail)
             if (cfg.use_calibration and confidence is not None) else np.zeros_like(r_drive))
    return dict(driving=r_drive, cot=r_cot, calibration=r_cal,
                total=r_drive - cfg.lambda_r * r_cot + r_cal)


def group_advantages(rewards, eps: float = ADV_EPS) -> np.ndarray:
    r = np.asarray(rewards, float)
    sd = r.std()
    return (r - r.mean()) / (sd + eps) if sd > eps else np.zeros_like(r)
