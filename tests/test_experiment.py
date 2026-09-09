import numpy as np

from ospedit.data import PairRecord, StructurePair
from ospedit.experiment import evaluate_editor, evaluate_manifest
from ospedit.models import ConditionalDifferenceEditor, CopyParentEditor, MultiNoiseLocalFrameDifferenceEditor, StudentEditor, TargetUpdateEditor
from ospedit.student import ParentEditStudent
from ospedit.student_inference import ParentContextCache


class ZeroModel:
    def field(self, coords, sequence, noise_level):
        return np.zeros_like(coords)


def record(pair_id, family_id, split):
    coords = np.zeros((3, 1, 3))
    return PairRecord(StructurePair(pair_id, "AAA", "AYA", coords, coords, (1,)), "parent", family_id, split)


def test_cost_accounting_does_not_hide_condition_branches():
    coords = np.zeros((4, 1, 3))
    pair = StructurePair("cost", "AAAA", "AYAA", coords, coords, (1,))
    copy = evaluate_editor(pair, CopyParentEditor())
    diff = evaluate_editor(pair, ConditionalDifferenceEditor(ZeroModel()))
    assert copy.runtime.network_calls == 0
    assert copy.runtime.structure_updates == 0
    assert diff.runtime.network_calls == 2
    assert diff.runtime.condition_branches == 2


def test_manifest_summary_reports_condition_branches():
    report = evaluate_manifest([record("branches", "family", "dev")], ConditionalDifferenceEditor(ZeroModel()), split="dev")
    assert report.runtime_summary["condition_branches"] == 2.0


def test_multi_noise_cost_counts_each_condition_query():
    residue = np.array([[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    coords = np.repeat(residue[None, :, :], 4, axis=0)
    pair = StructurePair("cost-multi", "AAAA", "AYAA", coords, coords, (1,))

    class Endpoint:
        def endpoint(self, coords, sequence, noise_level):
            return np.tile(np.eye(3), (len(coords), 1, 1)), coords[:, 1].copy()

    result = evaluate_editor(pair, MultiNoiseLocalFrameDifferenceEditor(Endpoint(), (0.2, 0.8)))
    assert result.runtime.network_calls == 4
    assert result.runtime.condition_branches == 4


def test_manifest_batch_matches_individual_metrics():
    records = [record("a", "fam-a", "dev"), record("b", "fam-b", "dev")]
    report = evaluate_manifest(records, CopyParentEditor(), split="dev")
    individual = [evaluate_editor(row.pair, CopyParentEditor()).metrics for row in records]
    for row, expected in zip(report.records, individual, strict=True):
        for key, value in expected.items():
            actual = row["metrics"][key]
            assert actual == value or (np.isnan(actual) and np.isnan(value))


def test_no_edit_runtime_counts_identity_shortcut():
    coords = np.zeros((3, 1, 3))
    pair = StructurePair("identity-cost", "AAA", "AAA", coords, coords.copy(), ())
    result = evaluate_editor(pair, TargetUpdateEditor(ZeroModel()))
    assert result.runtime.network_calls == 0
    assert result.runtime.structure_updates == 0


def test_manifest_summary_accumulates_parent_cache_runtime_fields():
    residue = np.array([[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    coords = np.repeat(residue[None, :, :], 2, axis=0)
    records = [
        PairRecord(StructurePair("cache-a", "AA", "AY", coords, coords, (1,), ("N", "CA", "C", "O")), "p", "f", "dev"),
        PairRecord(StructurePair("cache-b", "AA", "AF", coords, coords, (1,), ("N", "CA", "C", "O")), "p", "g", "dev"),
    ]
    model = ParentEditStudent(parent_dim=16, hidden_dim=8, blocks=1, heads=2)
    editor = StudentEditor(model, parent_cache=ParentContextCache())
    report = evaluate_manifest(records, editor, split="dev")
    assert report.runtime_summary["parent_cache_misses"] == 1.0
    assert report.runtime_summary["parent_cache_hits"] == 1.0


def test_manifest_runtime_summary_uses_max_for_peak_and_validation_metrics():
    class RuntimeEditor:
        def __init__(self):
            self.calls = 0
            self.last_runtime = {}

        def predict(self, pair):
            self.calls += 1
            self.last_runtime = {
                "peak_vram_bytes": float(100 + self.calls),
                "input_reconstruction_rmsd": 0.1 * self.calls,
                "inference_seconds": float(self.calls),
            }
            return pair.parent_coords.copy()

    report = evaluate_manifest(
        [record("runtime-a", "family-a", "dev"), record("runtime-b", "family-b", "dev")],
        RuntimeEditor(),
        split="dev",
    )

    assert report.runtime_summary["peak_vram_bytes"] == 102.0
    assert report.runtime_summary["input_reconstruction_rmsd"] == 0.2
    assert report.runtime_summary["inference_seconds"] == 3.0
