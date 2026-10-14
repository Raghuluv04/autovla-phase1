"""M10 rewards: the formulas, their limits, and group-relative advantages."""

import numpy as np
import pytest

from model.reward import (RewardConfig, calibration_penalty, cot_length_penalty, driving_reward,
                          group_advantages, total_reward)


def test_driving_reward_matches_paper_variant():
    assert driving_reward(0.0) == pytest.approx(0.2)
    assert driving_reward(2.0) == pytest.approx(0.0)
    assert driving_reward(12.0) == pytest.approx(-1.0)
    assert np.all(np.diff(driving_reward(np.linspace(0, 10, 20))) < 0)


def test_cot_penalty_is_a_sigmoid_centered_on_l_tol():
    assert cot_length_penalty(400) == pytest.approx(0.5)
    assert cot_length_penalty(0) == pytest.approx(1 / (1 + np.exp(0.8)))
    assert cot_length_penalty(1200) == pytest.approx(1 / (1 + np.exp(-1.6)))
    p = cot_length_penalty(np.arange(0, 2000, 50))
    assert np.all(np.diff(p) > 0) and p.min() > 0 and p.max() < 1
    assert cot_length_penalty(10 ** 6) == pytest.approx(1.0)       # no overflow


def test_calibration_penalty_only_on_confident_failures():
    assert calibration_penalty(1.0, 0.6) == pytest.approx(-0.5)
    assert calibration_penalty(0.2, 0.6) == pytest.approx(-0.1)
    assert calibration_penalty(1.0, 0.5) == 0.0                    # ADE must exceed tau strictly
    assert calibration_penalty(1.0, 0.1) == 0.0
    # confident-but-wrong is punished more than hesitant-but-wrong
    assert calibration_penalty(0.9, 3.0) < calibration_penalty(0.1, 3.0) < 0


def test_total_reward_composition_and_ablation_switch():
    full = total_reward(ade=1.0, cot_len=400, confidence=0.8)
    assert full["total"] == pytest.approx(0.1 - 0.3 * 0.5 - 0.5 * 0.8)
    off = total_reward(1.0, 400, 0.8, RewardConfig(use_calibration=False))
    assert off["calibration"] == 0 and off["total"] == pytest.approx(0.1 - 0.15)
    assert total_reward(1.0, 400)["calibration"] == 0               # no confidence supplied
    v = total_reward(np.array([0.1, 3.0]), np.array([0, 600]), np.array([0.9, 0.9]))
    assert v["total"].shape == (2,) and v["total"][0] > v["total"][1]


def test_group_advantage_is_standardised_and_safe():
    a = group_advantages([1.0, 2.0, 3.0, 4.0])
    assert a.mean() == pytest.approx(0, abs=1e-9) and a.std() == pytest.approx(1, abs=1e-3)
    assert np.all(group_advantages([0.3] * 8) == 0)
    assert np.argmax(a) == 3
