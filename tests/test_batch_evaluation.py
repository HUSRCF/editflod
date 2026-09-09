import numpy as np

from ospedit.data import PairRecord, StructurePair
from ospedit.experiment import evaluate_manifest
from ospedit.models import CopyParentEditor


def record(pair_id: str, family_id: str, split: str) -> PairRecord:
    coords = np.zeros((3, 1, 3))
    pair = StructurePair(pair_id, "AAA", "AYA", coords, coords, (1,))
    return PairRecord(pair, "parent", family_id, split)


def test_manifest_evaluation_filters_split_and_reports_families():
    records = [record("a", "fam-a", "dev"), record("b", "fam-a", "dev"), record("c", "fam-b", "test")]
    report = evaluate_manifest(records, CopyParentEditor(), split="dev")
    assert len(report.records) == 2
    assert set(report.family_summary) == {"fam-a"}
    assert report.runtime_summary["records"] == 2
