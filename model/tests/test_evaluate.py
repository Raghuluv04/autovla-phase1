"""M7: metric definitions, split rules, synthetic agents, and the evaluation loop."""

import numpy as np
import pytest

from autovla.schema import DrivingSample
from model.backbone import load_model
from model.config import BackboneConfig
from model.evaluate import build_eval_items, evaluate_modes, format_table, resolve_mode
from model.metrics import (LatencyMeter, OpenLoopAccumulator, box_corners, collision_by_horizon,
                           l2_at_horizons, l2_cumulative, latency_reduction, obb_overlap,
                           route_completion)
from model.splits import difficulty_reasons, is_hard, synthetic_obstacles
from model.standin import build_tiny_backbone


def straight(v=4.0, n=11):
    x = np.arange(n) * 0.5 * v
    return np.stack([x, np.zeros(n), np.zeros(n)], 1)


def test_l2_uses_exact_horizon_rows():
    gt, pred = straight(), straight()
    pred[2, 1] = 1.0; pred[4, 1] = 2.0; pred[6, 1] = 3.0     # 1 s, 2 s, 3 s rows
    assert l2_at_horizons(pred, gt) == {1.0: 1.0, 2.0: 2.0, 3.0: 3.0}
    cum = l2_cumulative(pred, gt)
    assert cum[1.0] == pytest.approx(0.5) and cum[3.0] < 3.0


def test_obb_overlap_cases():
    a = box_corners(0, 0, 0, 4, 2)
    assert obb_overlap(a, box_corners(3.5, 0, 0, 4, 2))          # 0.5 m overlap in x
    assert not obb_overlap(a, box_corners(4.5, 0, 0, 4, 2))
    assert not obb_overlap(a, box_corners(0, 2.5, 0, 4, 2))
    assert obb_overlap(a, box_corners(0, 2.2, np.pi / 4, 4, 2))  # rotated corner pokes in


def test_collision_is_cumulative_over_horizons():
    pred = straight(v=4.0)                                        # ego centre x = 2t
    obs = np.zeros((11, 1, 5)); obs[:, 0] = [12.0, 0.0, 0.0, 4.0, 2.0]   # spans x 10..14
    # step 3 (1.5 s): ego spans 3.96..8.04 -> clear; step 4 (2 s): 5.96..10.04 -> hit
    assert collision_by_horizon(pred, obs) == {1.0: False, 2.0: True, 3.0: True}
    assert collision_by_horizon(pred, None) == {1.0: False, 2.0: False, 3.0: False}
    far = obs.copy(); far[:, 0, 1] = 30.0
    assert not any(collision_by_horizon(pred, far).values())


def test_collision_monotone_in_horizon():
    pred = straight(v=6.0)
    for x in (4.0, 8.0, 14.0):
        obs = np.zeros((11, 1, 5)); obs[:, 0] = [x, 0, 0, 4.0, 2.0]
        c = collision_by_horizon(pred, obs)
        assert c[1.0] <= c[2.0] <= c[3.0]


def test_route_completion():
    gt = straight()
    assert route_completion(gt, gt) == pytest.approx(1.0)
    half = gt.copy(); half[:, 0] *= 0.5
    assert route_completion(half, gt) == pytest.approx(0.5, abs=0.01)
    still = np.zeros((11, 3))
    assert route_completion(still, still) == 1.0 and route_completion(gt, still) == 0.0


def test_latency_utilities():
    m = LatencyMeter()
    for x in (0.1, 0.2, 0.3):
        m.add(x)
    assert m.summary()["mean"] == pytest.approx(0.2) and m.summary()["n"] == 3
    assert latency_reduction(0.1, 0.3) == pytest.approx(2 / 3)


def test_accumulator_summary_keys():
    acc = OpenLoopAccumulator()
    acc.add(straight(), straight(), None, 0.01, 5)
    s = acc.summary()
    assert s["L2@1s"] == 0 and s["Col@3s"] == 0 and s["n"] == 1 and s["cot_tokens"] == 5


def sample(v, traj, sid="x"):
    return DrivingSample(sid, "synthetic", traj, velocity=v)


def test_hard_split_rules():
    turn = straight(); turn[:, 2] = np.linspace(0, 1.2, 11)
    assert difficulty_reasons(sample(5, turn)) == ["sharp_manoeuvre"]
    go = straight(v=0.0); go[5:, 0] = np.arange(6) * 2.0 + go[4, 0]
    assert "stationary_to_moving" in difficulty_reasons(sample(0.0, go))
    assert not is_hard(sample(5.0, straight()))
    s = sample(5.0, straight()); s.extra["occlusion"] = 0.8
    assert difficulty_reasons(s) == ["high_occlusion"]


def test_synthetic_obstacles_never_hit_ground_truth():
    for i in range(30):
        s = sample(4.0, straight(v=2.0 + i % 5), f"id{i}")
        obs = synthetic_obstacles(s)
        assert not any(collision_by_horizon(s.future_traj, obs).values())
        assert obs.shape[0] == 11 and obs.shape[2] == 5
    a, b = synthetic_obstacles(sample(4, straight(), "q")), synthetic_obstacles(sample(4, straight(), "q"))
    assert np.array_equal(a, b)                                  # deterministic per id


def test_evaluate_modes_end_to_end_on_standin():
    model, tok = build_tiny_backbone()
    bb = load_model(BackboneConfig(load_in_4bit=False), model=model, tokenizer=tok)
    items = build_eval_items(n=4)
    res = evaluate_modes(bb, items, modes=("fast", "slow", "random"), cot_max_tokens=6)
    assert set(res) >= {"fast", "slow", "random", "latency_reduction"}
    assert res["fast"]["summary"]["all"]["n"] == 4
    assert res["fast"]["summary"]["all"]["cot_tokens"] == 0 and res["slow"]["summary"]["all"]["cot_tokens"] > 0
    assert {r["mode"] for r in res["random"]["rows"]} <= {"fast", "slow"}
    assert "L2@3s" in format_table(res)
    assert resolve_mode("fast", np.random.default_rng(0)) == "fast"
