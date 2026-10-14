"""M5: DriveLM reformatting, template CoT, coverage, and the explanation metrics."""

import numpy as np
import pytest

from autovla.schema import DrivingSample, Reasoning
from model.cot_annotator import (annotate_samples, is_placeholder, motion_stats,
                                 reformat_drivelm_frame, synthesize_cot)
from model.explanation_metrics import (bleu, cider, load_bddx_references, meteor,
                                       score_explanations)

FRAME = {  # DriveLM-nuScenes key-frame layout
    "key_object_infos": {"<c1,CAM_FRONT,1088.3,497.5>": {
        "Category": "Vehicle", "Status": "Moving", "Visual_description": "Black sedan."}},
    "QA": {
        "perception": [{"Q": "What are objects to the front of the ego car?",
                        "A": "There is a black sedan to the front of the ego car."}],
        "prediction": [{"Q": "What is the future state of <c1,CAM_FRONT,1088.3,497.5>?",
                        "A": "Going ahead."}],
        "planning": [{"Q": "What actions could the ego vehicle take?", "A": "Keep going at the same speed."}],
        "behavior": [{"Q": "Predict the behavior of the ego vehicle.",
                      "A": "The ego vehicle is going straight. The ego vehicle is driving slowly."}],
    }}


def sample(i, v=5.0, dy=0.0, cmd="Go Straight", reasoning=None):
    xs = np.linspace(0, v * 5, 11)
    traj = np.stack([xs, np.linspace(0, dy, 11), np.zeros(11)], 1)
    return DrivingSample(f"s{i}", "synthetic", traj, velocity=v, command=cmd,
                         reasoning=reasoning or Reasoning())


def test_drivelm_reformat_fills_all_four_fields():
    r = reformat_drivelm_frame(FRAME)
    assert all(vars(r).values())
    assert "black sedan" in r.critical_objects.lower() and "object c1" in r.critical_objects
    assert r.agent_intentions == "Going ahead."
    assert "going straight" in r.driving_decision.lower()


def test_drivelm_sparse_frame_rejected():
    assert reformat_drivelm_frame({"QA": {"perception": [{"A": "A car."}]}}) is None


def test_template_cot_reflects_motion():
    stop = synthesize_cot(sample(0, v=0.0, cmd="Stop"))
    assert "stay stopped" in stop.driving_decision
    left = synthesize_cot(sample(1, v=5.0, dy=8.0, cmd="Turn Left"))
    assert "left" in left.driving_decision
    assert motion_stats(sample(2, v=5.0, dy=8.0)).lateral == pytest.approx(8.0)
    assert not is_placeholder(left)


def test_placeholder_detection():
    assert is_placeholder(Reasoning("[to be annotated by the Qwen2.5-VL-72B pipeline]", "[pending]", "[pending]", "Stop."))
    assert is_placeholder(Reasoning())


def test_annotate_meets_coverage_and_prefers_drivelm():
    ph = Reasoning("[pending]", "[pending]", "[pending]", "x")
    samples = [sample(i, reasoning=ph if i < 3 else None) for i in range(40)]
    ann, st = annotate_samples(samples, drivelm={"s39": FRAME}, min_fraction=0.15)
    assert st.fraction >= 0.15 and len(ann) == 6
    assert ann["s39"][0] == "drivelm"
    assert all(ann[f"s{i}"][0] == "trajectory_template" for i in range(3))
    assert not any(is_placeholder(r) for _, r in ann.values())


# ---- metrics --------------------------------------------------------------
HYPS = ["the car slows down because the traffic ahead is stopped", "the vehicle turns left at the light",
        "a pedestrian is crossing so the car stops", "the car goes straight on an empty road",
        "it stops for a red light"]
REFS = [["the car slows down since traffic in front has stopped", "traffic ahead stopped so the car slows"],
        ["the car turns left because the light is green", "the vehicle is turning left"],
        ["the car stops as pedestrians cross the road", "pedestrian crossing so car stops"],
        ["the car drives straight because the road is clear", "the car moves forward on a clear road"],
        ["the car stops at the red light", "stopped because the light is red"]]


def test_metrics_match_reference_implementations():
    # Values cross-checked once against sacrebleu 2.6 and pycocoevalcap 1.2.
    assert bleu(HYPS, REFS) == pytest.approx(0.1801475563, abs=1e-9)
    assert cider(HYPS, REFS) == pytest.approx(1.7093469525, abs=1e-9)


def test_metrics_identity_and_disjoint():
    ident = score_explanations(HYPS, [[h] for h in HYPS])
    assert ident.bleu4 == pytest.approx(1.0) and ident.meteor > 0.99 and ident.cider > 5
    none = score_explanations(["zzz qqq www vvv"], [["the car stops"]])
    assert none.bleu4 < 0.05 and none.meteor == 0 and none.cider == 0
    assert 0 < meteor(HYPS, REFS) < 1


def test_meteor_penalises_fragmentation():
    ordered = meteor(["a b c d e f"], [["a b c d e f"]])
    shuffled = meteor(["f e d c b a"], [["a b c d e f"]])
    assert ordered > shuffled > 0


def test_bddx_parser(tmp_path):
    p = tmp_path / "bddx.csv"
    p.write_text("Input.Video,Answer.1action,Answer.1justification,Answer.2action,Answer.2justification\n"
                 "v1,the car stops,because the light is red,the car goes,as the light turns green\n")
    refs = load_bddx_references(p)
    assert refs == {"v1#1": ["because the light is red"], "v1#2": ["as the light turns green"]}
