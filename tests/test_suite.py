import numpy as np

from ospedit.data import PairRecord, StructurePair
from ospedit.experiment import evaluate_editor_suite
from ospedit.models import CopyParentEditor, TargetUpdateEditor


class ZeroField:
    def field(self, coords, sequence, noise_level):
        return np.zeros_like(coords)


def test_editor_suite_uses_same_records_for_each_method():
    coords = np.zeros((3, 1, 3))
    records = [
        PairRecord(StructurePair("a", "AAA", "AYA", coords, coords, (1,)), "p", "f", "dev"),
        PairRecord(StructurePair("b", "AAA", "AFA", coords, coords, (1,)), "p", "g", "dev"),
    ]
    suite = evaluate_editor_suite(
        records,
        {"copy": CopyParentEditor(), "target": TargetUpdateEditor(ZeroField())},
        split="dev",
    )
    assert set(suite.methods) == {"copy", "target"}
    assert [row["pair_id"] for row in suite.methods["copy"].records] == ["a", "b"]
    assert [row["pair_id"] for row in suite.methods["target"].records] == ["a", "b"]


def test_editor_suite_uses_batch_path_when_editor_supports_it():
    class BatchedCopy(CopyParentEditor):
        def predict_batch(self, pairs):
            return [pair.parent_coords.copy() for pair in pairs]

    coords = np.zeros((3, 1, 3))
    records = [
        PairRecord(StructurePair("a", "AAA", "AYA", coords, coords, (1,)), "p", "f", "dev"),
        PairRecord(StructurePair("b", "AAA", "AFA", coords, coords, (1,)), "p", "g", "dev"),
    ]
    suite = evaluate_editor_suite(records, {"copy": BatchedCopy()}, split="dev", batch_size=2)
    assert suite.methods["copy"].runtime_summary["batches"] == 1.0
