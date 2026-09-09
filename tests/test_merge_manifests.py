import numpy as np
import pytest

from ospedit.data import PairRecord, StructurePair, write_manifest
from scripts.merge_manifests import merge_manifest_files


def _record(pair_id: str, split: str) -> PairRecord:
    coords = np.zeros((2, 4, 3), dtype=float)
    pair = StructurePair(pair_id, "AA", "AY", coords, coords.copy(), (1,))
    return PairRecord(pair, pair_id + "-parent", pair_id + "-family", split)


def test_merge_manifests_preserves_splits_and_reports_fingerprint(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_manifest([_record("train-pair", "train")], first)
    write_manifest([_record("dev-pair", "dev")], second)

    records, report = merge_manifest_files([first, second])

    assert [record.pair.pair_id for record in records] == ["train-pair", "dev-pair"]
    assert report["split_counts"] == {"train": 1, "dev": 1, "test": 0}
    assert len(report["manifest_fingerprint"]) == 64


def test_merge_manifests_rejects_duplicate_pair_ids(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_manifest([_record("duplicate", "train")], first)
    write_manifest([_record("duplicate", "train")], second)

    with pytest.raises(ValueError, match="duplicate pair_ids"):
        merge_manifest_files([first, second])
