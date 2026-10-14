"""M9: gate semantics, minimal-risk manoeuvre, threshold calibration, policy comparison."""

import numpy as np
import pytest

from model.gating import (FAST, MRM, SLOW, ConfidenceGate, ModeOutcomes, epistemic_quantile_modes,
                          evaluate_policies, latency_cut, minimal_risk_plan, minimal_risk_trajectory)
from model.runtime import load_action_codec


def test_gate_regions_and_boundaries():
    g = ConfidenceGate(high=0.8, low=0.4)
    assert [g.decide(c) for c in (0.95, 0.8, 0.79, 0.4, 0.39, 0.0)] == [FAST, FAST, SLOW, SLOW, MRM, MRM]
    assert list(g.decide_many([0.9, 0.5, 0.1])) == [FAST, SLOW, MRM]
    with pytest.raises(ValueError):
        ConfidenceGate(high=0.3, low=0.6)


def test_gate_roundtrip(tmp_path):
    ConfidenceGate(0.7, 0.2).save(tmp_path / "g.json")
    assert ConfidenceGate.load(tmp_path / "g.json") == ConfidenceGate(0.7, 0.2)


def test_minimal_risk_trajectory_is_a_straight_stop():
    t = minimal_risk_trajectory(9.0)                      # stops after 3 s at 3 m/s^2
    x = t[:, 0]
    assert x[0] == 0 and np.all(np.diff(x) >= -1e-9) and np.all(t[:, 1:] == 0)
    assert x[-1] == pytest.approx(9.0 ** 2 / (2 * 3.0))   # v^2 / 2a
    assert np.allclose(x[7:], x[-1])                      # held after stopping
    assert np.all(minimal_risk_trajectory(0.0) == 0)


def test_minimal_risk_plan_uses_ten_valid_tokens():
    ids, traj = minimal_risk_plan(6.0, load_action_codec())
    assert len(ids) == 10 and all(0 <= i < 2048 for i in ids)
    assert np.linalg.norm(traj[-1, :2] - minimal_risk_trajectory(6.0)[-1, :2]) < 0.5


def synthetic_outcomes(n=600, seed=0):
    rng = np.random.default_rng(seed)
    conf = rng.uniform(0, 1, n)
    moving = conf >= 0.3                                   # region C (<0.3) is a stationary scene
    gt = np.zeros((n, 11, 3)); gt[moving, :, 0] = np.arange(11) * 2.0
    def offset(err):
        t = gt.copy(); t[:, :, 1] += err[:, None]; return t
    e_fast = np.where(conf > 0.7, 0.2, 6.0); e_slow = np.where(conf >= 0.3, 0.3, 8.0)
    mrm = np.zeros((n, 11, 3))
    mrm[moving] = gt[moving] + np.array([0, 9.0, 0])        # MRM is wrong when the scene is moving
    trajs = np.stack([offset(e_fast), offset(e_slow), mrm], 1)
    lat = np.stack([np.full(n, 0.01), np.full(n, 0.5), np.zeros(n)], 1)
    out = ModeOutcomes([str(i) for i in range(n)], trajs, gt, [None] * n, lat,
                       np.stack([np.zeros(n), np.full(n, 100), np.zeros(n)], 1), conf < 0.7)
    return conf, out


def test_calibration_recovers_the_three_regimes():
    conf, out = synthetic_outcomes()
    g = ConfidenceGate.calibrate(conf, out, latency_weight=0.1)
    assert 0.6 <= g.high <= 0.8 and 0.2 <= g.low <= 0.4
    modes = g.decide_many(conf)
    assert (modes[conf > 0.8] == FAST).all() and (modes[(conf > 0.4) & (conf < 0.6)] == SLOW).all()
    assert (modes[conf < 0.2] == MRM).all()


def test_latency_weight_discourages_slow_thinking():
    conf, out = synthetic_outcomes()
    cheap = ConfidenceGate.calibrate(conf, out, latency_weight=0.0)
    costly = ConfidenceGate.calibrate(conf, out, latency_weight=50.0)
    frac_slow = lambda g: (g.decide_many(conf) == SLOW).mean()
    assert frac_slow(costly) <= frac_slow(cheap)
    assert frac_slow(costly) < 0.05


def test_policy_comparison_shows_latency_cut_and_mode_change_on_hard():
    conf, out = synthetic_outcomes()
    g = ConfidenceGate.calibrate(conf, out)
    n = len(conf)
    rep = evaluate_policies(out, conf, conf, g, np.full(n, 0.002), np.full(n, 0.0005))
    assert set(rep) >= {"always_fast", "always_slow", "random", "gated_full_conf", "gated_aleatoric_only"}
    assert latency_cut(rep, "easy") > 0.8                  # easy (conf>=.7) scenes mostly fast
    assert rep["gated_full_conf"]["hard"]["frac_slow"] > rep["gated_full_conf"]["easy"]["frac_slow"]
    assert rep["gated_full_conf"]["all"]["driving_score"] > rep["always_fast"]["all"]["driving_score"]
    assert rep["always_slow"]["all"]["frac_slow"] == 1.0


def test_epistemic_quantile_modes_shift_with_uncertainty():
    rng = np.random.default_rng(0)
    epi = rng.uniform(0, 10, 1000)
    conf = 1 / (1 + epi)
    mix = epistemic_quantile_modes(epi, conf, ConfidenceGate(0.5, 0.15))
    assert mix["epistemic_high"][FAST] < mix["epistemic_low"][FAST]
    assert mix["epistemic_high"][SLOW] + mix["epistemic_high"][MRM] > 0.9


def test_min_slow_fraction_forces_a_slow_band():
    conf, out = synthetic_outcomes()
    # make slow thinking useless so the unconstrained optimum has no slow band
    out.trajs[:, 1] = out.trajs[:, 0]
    free = ConfidenceGate.calibrate(conf, out, latency_weight=1.0)
    forced = ConfidenceGate.calibrate(conf, out, latency_weight=1.0, min_slow_fraction=0.2)
    assert (free.decide_many(conf) == SLOW).mean() == 0
    assert (forced.decide_many(conf) == SLOW).mean() >= 0.2


def test_driving_score_needs_progress_so_always_stopping_is_not_free():
    n = 50
    gt = np.zeros((n, 11, 3)); gt[:, :, 0] = np.arange(11) * 2.0          # everyone drives 20 m
    fast = gt.copy(); fast[:, :, 1] += 3.0                                 # 3 m lateral error, full progress
    stop = np.zeros((n, 11, 3))                                            # braked to a standstill
    out = ModeOutcomes([str(i) for i in range(n)], np.stack([fast, fast, stop], 1), gt, [None] * n,
                       np.ones((n, 3)), np.zeros((n, 3)), np.zeros(n, bool))
    sc = out.score_matrix()
    assert out.route()[0, 0] > 0.95 and out.route()[0, 2] == pytest.approx(0.0, abs=0.01)
    assert sc[:, 0].mean() > sc[:, 2].mean()                               # fast beats a pointless stop
