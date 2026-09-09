import numpy as np

from ospedit.data import PairRecord, StructurePair, assign_group_splits


def _record(pair_id: str, family: str, parent: str) -> PairRecord:
    coords = np.zeros((2, 4, 3), dtype=float)
    pair = StructurePair(pair_id, "AA", "AB", coords, coords.copy(), (1,), ("N", "CA", "C", "O"))
    return PairRecord(pair, parent, family, "dev")


def test_group_split_is_deterministic_and_keeps_families_together():
    records = [_record("a", "family-a", "parent-a"), _record("b", "family-a", "parent-b"), _record("c", "family-b", "parent-c")]
    first = assign_group_splits(records, seed=17, train_fraction=0.5, dev_fraction=0.25)
    second = assign_group_splits(records, seed=17, train_fraction=0.5, dev_fraction=0.25)
    assert [row.split for row in first] == [row.split for row in second]
    assert first[0].split == first[1].split


def test_parent_grouping_keeps_same_parent_together():
    records = [_record("a", "family-a", "parent-a"), _record("b", "family-b", "parent-a")]
    split = assign_group_splits(records, seed=3, group_by="parent")
    assert split[0].split == split[1].split


def test_group_split_keeps_all_splits_nonempty_when_capacity_allows():
    records = [_record(f"pair-{index}", f"family-{index}", f"parent-{index}") for index in range(4)]
    split = assign_group_splits(records, seed=11)
    counts = {name: sum(row.split == name for row in split) for name in ("train", "dev", "test")}
    assert counts == {"train": 2, "dev": 1, "test": 1}


def test_group_split_rejects_duplicate_pair_ids():
    records = [_record("duplicate", "family-a", "parent-a"), _record("duplicate", "family-b", "parent-b")]
    try:
        assign_group_splits(records)
    except ValueError as error:
        assert "duplicate pair_id" in str(error)
    else:
        raise AssertionError("expected duplicate pair_id validation")
