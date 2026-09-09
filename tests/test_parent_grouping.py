import numpy as np

from ospedit.data import PairRecord, StructurePair
from ospedit.experiment import evaluate_parent_workloads, group_records_by_parent, parent_workloads
from ospedit.models import CopyParentEditor


def test_group_records_by_parent_preserves_candidate_order_and_split():
    coords = np.zeros((2, 1, 3))
    records = [
        PairRecord(StructurePair("a", "AA", "AY", coords, coords, (1,)), "parent-1", "f", "dev"),
        PairRecord(StructurePair("b", "AA", "AG", coords, coords, (1,)), "parent-1", "f", "dev"),
        PairRecord(StructurePair("c", "AA", "AF", coords, coords, (1,)), "parent-2", "g", "test"),
    ]
    groups = group_records_by_parent(records, split="dev")
    assert list(groups) == ["parent-1"]
    assert [record.pair.pair_id for record in groups["parent-1"]] == ["a", "b"]


def test_parent_workloads_are_deterministic_and_can_require_full_counts():
    coords = np.zeros((2, 1, 3))
    records = [
        PairRecord(StructurePair(f"a{i}", "AA", "AY", coords, coords, (1,)), "parent", "f", "dev")
        for i in range(3)
    ]
    first = parent_workloads(records, (1, 2), split="dev", seed=7)
    second = parent_workloads(records, (1, 2), split="dev", seed=7)
    first_ids = [[row.pair.pair_id for row in first["parent"][count]] for count in (1, 2)]
    second_ids = [[row.pair.pair_id for row in second["parent"][count]] for count in (1, 2)]
    assert first_ids == second_ids
    assert parent_workloads(records, (4,), require_full=True) == {}


def test_parent_workloads_reject_duplicate_candidate_counts():
    try:
        parent_workloads([], (1, 1))
    except ValueError as error:
        assert "duplicates" in str(error)
    else:
        raise AssertionError("expected duplicate candidate count validation")


def test_evaluate_parent_workloads_returns_each_candidate_count():
    coords = np.zeros((2, 1, 3))
    records = [
        PairRecord(StructurePair(f"a{i}", "AA", "AY", coords, coords, (1,)), "parent", "f", "dev")
        for i in range(2)
    ]
    workloads = parent_workloads(records, (1, 2), require_full=True)
    reports = evaluate_parent_workloads(workloads, CopyParentEditor(), batch_size=2, method="copy")
    assert set(reports["parent"]) == {1, 2}
    assert reports["parent"][2].runtime_summary["records"] == 2.0
