import numpy as np

from ospedit.data import PairRecord, StructurePair
from scripts.audit_manifest_endpoint_groups import audit_manifest_endpoint_groups


def _record(
    pair_id,
    parent,
    mutant,
    source_checksum,
    target_checksum,
    *,
    source_chain="A",
    target_chain="A",
):
    coords = np.zeros((3, 1, 3))
    pair = StructurePair(pair_id, parent, mutant, coords, coords.copy(), (1,))
    return PairRecord(
        pair,
        "parent",
        "family",
        "train",
        source_chain=source_chain,
        target_chain=target_chain,
        source_checksum=source_checksum * 64,
        target_checksum=target_checksum * 64,
    )


def test_endpoint_group_audit_separates_same_and_reverse_duplicates():
    records = [
        _record("a", "AAA", "AYA", "a", "b"),
        _record("b", "AAA", "AYA", "a", "b"),
        _record("unique", "AAA", "ACA", "a", "c"),
        _record("reverse-a", "AAA", "AGA", "d", "e"),
        _record("reverse-b", "AGA", "AAA", "e", "d"),
    ]

    report = audit_manifest_endpoint_groups(records)

    assert report["summary"] == {
        "records": 5,
        "unique_endpoint_groups": 3,
        "duplicate_groups": 2,
        "records_in_duplicate_groups": 4,
        "redundant_records": 2,
        "same_direction_duplicate_groups": 1,
        "reverse_direction_duplicate_groups": 1,
    }
    assert {group["classification"] for group in report["groups"]} == {
        "same_direction_duplicates",
        "reverse_direction_duplicates",
    }


def test_endpoint_group_audit_keeps_chains_from_one_file_distinct():
    records = [
        _record("chain-a", "AAA", "AYA", "a", "b"),
        _record(
            "chain-b",
            "AAA",
            "AYA",
            "a",
            "b",
            source_chain="B",
            target_chain="B",
        ),
    ]

    report = audit_manifest_endpoint_groups(records)

    assert report["summary"]["unique_endpoint_groups"] == 2
    assert report["summary"]["duplicate_groups"] == 0
