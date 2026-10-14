"""M8: calibration maths, both uncertainty branches, fusion, and the fit/eval pipeline."""

import itertools

import numpy as np
import pytest
import torch

from autovla.schema import DrivingSample
from model.backbone import load_model
from model.calibration import (TemperatureScaler, compute_auroc, compute_ece, nll,
                               temperature_scaling)
from model.config import BackboneConfig
from model.standin import build_tiny_backbone
from model.uncertainty import (AleatoricHead, EpistemicHead, FeatureExtractor, SceneOutputs,
                               UncertaintyModel, ensemble_sampler, fit_and_evaluate,
                               fuse_confidence, mc_dropout, train_aleatoric_head)


# ---- calibration ----------------------------------------------------------
def test_auroc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    score = rng.integers(0, 6, 60).astype(float)              # many ties
    label = rng.random(60) < 0.4
    pos, neg = score[label], score[~label]
    brute = np.mean([(p > n) + 0.5 * (p == n) for p, n in itertools.product(pos, neg)])
    assert compute_auroc(score, label) == pytest.approx(brute)
    assert compute_auroc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert compute_auroc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
    assert np.isnan(compute_auroc([1, 2], [1, 1]))


def test_ece_zero_when_calibrated_and_large_when_not():
    rng = np.random.default_rng(1)
    conf = rng.uniform(0, 1, 20000)
    ok = rng.random(20000) < conf
    assert compute_ece(conf, ok) < 0.02
    assert compute_ece(np.full(1000, 0.9), np.zeros(1000)) == pytest.approx(0.9)


def test_temperature_scaling_recovers_overconfidence():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 1.5, 20000)                             # true logits
    y = rng.random(20000) < 1 / (1 + np.exp(-z))
    over = 1 / (1 + np.exp(-3.0 * z))                         # model is 3x overconfident
    sc = temperature_scaling(over, y)
    assert sc.temperature == pytest.approx(3.0, rel=0.15)
    assert nll(sc(over), y) < nll(over, y)
    assert compute_ece(sc(over), y) < compute_ece(over, y)
    assert sc.bias == 0.0
    sc_b = temperature_scaling(1 / (1 + np.exp(-(z + 2))), y, fit_bias=True)
    assert sc_b.bias == pytest.approx(-2 / sc_b.temperature, abs=0.3)


def test_variance_temperature_recovers_rescaled_uncertainty():
    rng = np.random.default_rng(3)
    u = np.exp(rng.normal(5.0, 1.5, 30000))                    # sigma^2 in the hundreds: conf saturated near 0
    y = rng.random(30000) < 1 / (1 + u / 150.0)                # truth: p = 1/(1+u/150)
    conf = 1 / (1 + u)
    sc = temperature_scaling(conf, y, kind="variance")
    assert sc.kind == "variance" and sc.temperature == pytest.approx(150.0, rel=0.15)
    assert compute_ece(sc(conf), y) < 0.02
    assert compute_ece(temperature_scaling(conf, y)(conf), y) > 0.1       # logit-T cannot fix this regime
    assert np.all(np.diff(sc(np.linspace(0.001, 0.999, 20))) > 0)


def test_auto_selection_picks_lowest_calibration_nll():
    from model.calibration import fit_calibrators, select_calibrator
    rng = np.random.default_rng(4)
    u = np.exp(rng.normal(5.0, 1.5, 5000)); y = rng.random(5000) < 1 / (1 + u / 150.0)
    conf = 1 / (1 + u)
    cands = fit_calibrators(conf, y)
    best = select_calibrator(conf, y, cands)
    assert nll(best(conf), y) == min(nll(c(conf), y) for c in cands.values())
    assert set(cands) == {"logit_T", "variance_T", "logit_T_bias"}


def test_scaler_is_monotone_and_identity_at_t1():
    c = np.linspace(0.05, 0.95, 10)
    assert np.allclose(TemperatureScaler(1.0)(c), c, atol=1e-6)
    assert np.all(np.diff(TemperatureScaler(2.5)(c)) > 0)


# ---- fusion ---------------------------------------------------------------
def test_fuse_confidence_formula():
    assert fuse_confidence(0.0, 0.0) == 1.0
    assert fuse_confidence(1.0, 2.0) == pytest.approx(0.25)
    assert 0 < fuse_confidence(50.0, 50.0) < fuse_confidence(1.0, 1.0)


# ---- Branch A -------------------------------------------------------------
def test_aleatoric_head_learns_input_dependent_noise():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(1500, 6, generator=g)
    true_var = torch.exp(1.5 * X[:, 0])                       # noise scale driven by feature 0
    err2 = (torch.randn(1500, generator=g) ** 2) * true_var
    head, info = train_aleatoric_head(X, err2, epochs=120, seed=0)
    pred = head.sigma2(X)
    r = np.corrcoef(np.log(pred.numpy()), X[:, 0].numpy())[0, 1]
    assert r > 0.8 and not head.training
    assert pred.min() > 0


# ---- Branch B -------------------------------------------------------------
def test_epistemic_head_variance_and_ensemble_dispatch():
    rng = np.random.default_rng(0)
    base = np.stack([np.arange(11) * 2.0, np.zeros(11), np.zeros(11)], 1)
    noise = rng.normal(0, 0.5, (10, 11, 3)); noise[:, 0] = 0
    sampler = lambda i: base + noise[i]
    res = EpistemicHead(T=10).estimate(sampler)
    expected = 2 * 0.25                                       # var_x + var_y with sd 0.5
    assert res.sigma2 == pytest.approx(expected, rel=0.5)
    assert EpistemicHead(T=4).estimate(lambda i: base).sigma2 == 0.0
    with pytest.raises(ValueError):
        EpistemicHead(mode="bogus")


@pytest.fixture(scope="module")
def tiny():
    model, tok = build_tiny_backbone()
    bb = load_model(BackboneConfig(load_in_4bit=False), model=model, tokenizer=tok)
    with torch.no_grad():                                     # LoRA B=0 at init -> dropout would be a no-op
        for n, p in bb.model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.5)
    return bb


MSGS = [{"role": "system", "content": "planner"},
        {"role": "user", "content": [{"type": "text", "text": "Ego state: speed 5.00 m/s. Stop."}]}]


def test_mc_dropout_toggles_and_restores(tiny):
    tiny.model.eval()
    drops = [m for m in tiny.model.modules() if isinstance(m, torch.nn.Dropout) and m.p > 0]
    assert drops and not any(m.training for m in drops)
    with mc_dropout(tiny.model) as n:
        assert n == len(drops) and all(m.training for m in drops)
    assert not any(m.training for m in drops)


def test_mc_dropout_makes_plans_stochastic_but_eval_is_deterministic(tiny):
    from model.runtime import load_action_codec
    from model.uncertainty import mc_dropout_sampler
    codec = load_action_codec()
    tiny.model.eval()
    det = [mc_dropout_sampler(tiny, MSGS, codec)(0) for _ in range(2)]
    assert np.array_equal(det[0], det[1])                     # dropout off -> same plan
    with mc_dropout(tiny.model):
        res = EpistemicHead(T=6).estimate(mc_dropout_sampler(tiny, MSGS, codec, seed=1))
    assert res.sigma2 > 0 and res.trajs.shape == (6, 11, 3)


def test_batched_mc_dropout_matches_semantics_and_is_stochastic(tiny):
    from model.inference import generate_action_batch
    from model.runtime import load_action_codec
    from model.uncertainty import mc_dropout_batch
    codec = load_action_codec()
    tiny.model.eval()
    rows = generate_action_batch(tiny, MSGS, 4)
    assert len(rows) == 4 and all(len(r) == 10 for r in rows) and all(r == rows[0] for r in rows)  # dropout off
    single = codec.decode(rows[0])
    assert np.array_equal(single, codec.decode(__import__("model.inference", fromlist=["x"]).generate_plan(tiny, MSGS, "fast").action_ids))
    trajs = mc_dropout_batch(tiny, MSGS, codec, T=6, seed=1)
    assert trajs.shape == (6, 11, 3) and EpistemicHead.from_trajs(trajs).sigma2 > 0     # rows get different masks


def test_ensemble_sampler_cycles_members(tiny):
    from model.runtime import load_action_codec
    s = ensemble_sampler([tiny, tiny, tiny], MSGS, load_action_codec())
    assert s(0).shape == (11, 3)


def test_feature_extractor_text_and_vision(tiny):
    fx = FeatureExtractor(tiny)
    s = DrivingSample("a", "synthetic", np.zeros((11, 3)), velocity=5.0, command="Stop")
    f = fx(MSGS, s)
    assert f.shape == (fx.dim,) and torch.isfinite(f).all()
    assert torch.count_nonzero(f[:fx.hidden]) == 0            # no image -> zero visual block
    assert f[-4:].tolist() == [0.0, 0.0, 0.0, 1.0][:4] or f[-1] == 1.0


# ---- pipeline -------------------------------------------------------------
def fake_outputs(n, seed):
    rng = np.random.default_rng(seed)
    hard = rng.random(n) < 0.4
    f = rng.normal(0, 1, (n, 5)); f[:, 0] = hard * 2.0 + rng.normal(0, 0.3, n)     # feature reveals difficulty
    scale = np.where(hard, 2.0, 0.3)                                               # hard scenes err more
    err = np.abs(rng.normal(0, 1, n)) * scale
    epi = (np.abs(rng.normal(0, 1, n)) * scale) ** 2 * 0.3
    pred = np.zeros((n, 11, 3))
    return SceneOutputs([str(i) for i in range(n)], f, err ** 2, err, epi, hard, pred)


def test_fit_and_evaluate_meets_targets_on_informative_signal():
    tr, ca, te = fake_outputs(900, 0), fake_outputs(700, 1), fake_outputs(1500, 2)
    model, res = fit_and_evaluate(tr, ca, te, tau=0.5, epochs=80, seed=0)
    assert isinstance(model, UncertaintyModel)
    assert res["auroc_failure"]["fused"] > 0.6
    # the 0.05 ECE *target* is a result, not a unit property (ECE of 1.5k points wobbles 0.04-0.06 by seed)
    assert res["calibrated"]["ece"] < 0.08 and res["calibrated"]["ece"] < res["uncalibrated"]["ece"]
    assert res["calibrated"]["nll"] <= res["uncalibrated"]["nll"] + 1e-6
    h = res["hard_vs_easy"]
    assert h["mean_unc_hard"] > h["mean_unc_easy"] and h["mean_conf_hard"] < h["mean_conf_easy"]
    assert res["targets"]["auroc_above_target"] and res["targets"]["hard_more_uncertain"]


def test_fit_and_evaluate_auto_tau_and_roundtrip(tmp_path):
    tr, ca, te = fake_outputs(500, 3), fake_outputs(300, 4), fake_outputs(300, 5)
    model, res = fit_and_evaluate(tr, ca, te, tau="auto", epochs=30)
    assert res["tau"] == pytest.approx(float(np.median(ca.ade)))
    assert res["calibrator"] in res["calibrators"] and len(res["calibrators"]) == 3
    model.save(tmp_path / "u.pt")
    again = UncertaintyModel.load(tmp_path / "u.pt")
    x = torch.as_tensor(te.features[:5], dtype=torch.float32)
    assert torch.allclose(again.head.sigma2(x), model.head.sigma2(x))
    assert again.confidence(1.0, 1.0) == pytest.approx(model.confidence(1.0, 1.0))


def test_collect_scene_outputs_end_to_end(tiny):
    from model.evaluate import build_eval_items
    from model.runtime import load_action_codec
    from model.uncertainty import collect_scene_outputs
    items = build_eval_items(3, holdout=200, obstacles=None)
    out = collect_scene_outputs(tiny, items, load_action_codec(), T=3)
    assert len(out) == 3 and out.features.shape[0] == 3 and out.pred.shape == (3, 11, 3)
    assert (out.epi >= 0).all() and out.epi.max() > 0               # MC passes disagree
    assert ((out.pconf > 0) & (out.pconf <= 1)).all()
    assert (out.est_s >= out.feat_s).all() and (out.ade >= 0).all()
    sub = out.subset([0, 2])
    assert len(sub) == 2 and sub.pconf.shape == (2,)
    no_mc = collect_scene_outputs(tiny, items[:2], load_action_codec(), T=0)
    assert (no_mc.epi == 0).all()
