"""M11: partitions are disjoint, every ablation row is produced, and the CSV is well-formed."""

import csv

import pytest

from model.benchmark import CSV_COLS, Variant, evaluate_variant, make_partitions, write_table


@pytest.fixture(scope="module")
def parts():
    return make_partitions(holdout=200, n_head=8, n_calib=6, n_val=3, n_test=3)


def test_partitions_are_disjoint_and_sized(parts):
    groups = [parts.head_train, parts.calib, parts.gate_val, parts.test]
    assert [len(g) for g in groups] == [8, 6, 3, 3]
    ids = [i.sample.sample_id for g in groups for i in g]
    assert len(ids) == len(set(ids))


def test_partitions_reject_oversized_request():
    with pytest.raises(SystemExit):
        make_partitions(holdout=200, n_head=150, n_calib=50, n_val=10, n_test=10)


@pytest.mark.parametrize("head", [False, True])
def test_variant_row_has_every_metric(parts, head):
    v = Variant("v", None, head, False)
    row = evaluate_variant(v, parts, tau=5.0, standin=True, T=2, cot_max_tokens=3, log=lambda *_: None)
    for key in ("L2@1s", "L2@2s", "L2@3s", "Col@1s", "Col@2s", "Col@3s", "ECE", "AUROC", "latency_ms",
                "route_completion", "frac_fast", "frac_slow", "frac_mrm", "confidence_source"):
        assert key in row, key
    assert 0 <= row["ECE"] <= 1 and 0 <= row["Col@3s"] <= 1 and row["latency_ms"] > 0
    assert row["Col@1s"] <= row["Col@2s"] <= row["Col@3s"]
    assert abs(row["frac_fast"] + row["frac_slow"] + row["frac_mrm"] - 1) < 1e-9
    assert row["confidence_source"] == ("fused_head_calibrated" if head else "policy_action_token_prob")


def test_write_table_has_four_ablation_rows(tmp_path):
    base = {c: 0.1 for c in CSV_COLS}
    names = ["SFT only (no GRPO)", "+ GRPO (no uncertainty head)",
             "+ uncertainty head (no calib reward)", "+ calibration reward (FULL MODEL)"]
    rows = [dict(base, variant=n, uncertainty_head=False, calibration_reward=False,
                 confidence_source="x", n_test=4, **{"L2@3s_ci": (0.0, 1.0), "Col@3s_ci": (0.0, 1.0)}) for n in names]
    path = write_table(rows, "unit-test", tmp_path)
    got = list(csv.DictReader(open(path)))
    assert [r["variant"] for r in got] == names and all(r["provenance"] == "unit-test" for r in got)
    assert (tmp_path / "ablation_table.md").read_text().count("\n| ") >= 4
