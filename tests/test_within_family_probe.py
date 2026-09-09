import numpy as np
import pytest

from ospedit.data import PairRecord, StructurePair
from ospedit.endpoint_groups import endpoint_group_key
from scripts.build_within_family_probe import build_within_family_probe


def _record(pair_id: str, family: str, source: str, target: str) -> PairRecord:
    pair = StructurePair(
        pair_id=pair_id,
        parent_sequence="AA",
        mutant_sequence="AC",
        parent_coords=np.zeros((2, 1, 3)),
        mutant_coords=np.ones((2, 1, 3)),
        mutation_indices=(1,),
        atom_names=("CA",),
    )
    return PairRecord(
        pair=pair,
        parent_id=source,
        family_id=family,
        split="train",
        source_file=source,
        target_file=target,
        source_chain="A",
        target_chain="A",
    )


def test_within_family_probe_keeps_endpoint_groups_together_and_families_seen():
    records = [
        _record("a", "family", "one", "two"),
        _record("a_duplicate", "family", "one", "two"),
        _record("b", "family", "three", "four"),
        _record("c", "family", "five", "six"),
        _record("excluded", "singleton", "seven", "eight"),
    ]

    probe, report = build_within_family_probe(records, dev_fraction=0.34, seed=4)

    assert report["summary"]["eligible_families"] == 1
    assert report["summary"]["excluded_records"] == 1
    assert {record.family_id for record in probe if record.split == "train"} == {"family"}
    assert {record.family_id for record in probe if record.split == "dev"} == {"family"}
    assignments = {}
    for record in probe:
        assignments.setdefault(endpoint_group_key(record), set()).add(record.split)
    assert all(len(splits) == 1 for splits in assignments.values())


def test_within_family_probe_is_deterministic():
    records = [
        _record("a", "family", "one", "two"),
        _record("b", "family", "three", "four"),
    ]

    first, _ = build_within_family_probe(records, seed=9)
    second, _ = build_within_family_probe(reversed(records), seed=9)

    assert [(r.pair.pair_id, r.split) for r in first] == [
        (r.pair.pair_id, r.split) for r in second
    ]


def test_within_family_probe_rejects_non_train_input():
    record = _record("a", "family", "one", "two")

    with pytest.raises(ValueError, match="only train"):
        build_within_family_probe([PairRecord.from_dict({**record.to_dict(), "split": "dev"})])
