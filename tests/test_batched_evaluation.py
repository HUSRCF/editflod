import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ospedit.data import PairRecord, StructurePair
from ospedit.experiment import evaluate_manifest, evaluate_manifest_batched
from ospedit.models import StudentEditor
from ospedit.student_inference import ParentContextCache


class ZeroStudent(torch.nn.Module):
    def forward(self, parent, edit, residue_mask=None):
        return torch.zeros((parent.shape[0], parent.shape[1], 6), dtype=parent.dtype, device=parent.device)


def make_record(pair_id, length):
    residue = np.array([[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    coords = np.repeat(residue[None], length, axis=0)
    return PairRecord(StructurePair(pair_id, "A" * length, "A" + "Y" + "A" * (length - 2), coords, coords.copy(), (1,)), "p", pair_id, "dev")


def test_batched_student_evaluation_preserves_metrics_and_counts_batches():
    records = [make_record("a", 2), make_record("b", 3)]
    editor = StudentEditor(ZeroStudent())
    regular = evaluate_manifest(records, editor, method="student", split="dev")
    batched = evaluate_manifest_batched(records, editor, batch_size=2, method="student", split="dev")
    for regular_row, batch_row in zip(regular.records, batched.records, strict=True):
        for key, value in regular_row["metrics"].items():
            actual = batch_row["metrics"][key]
            assert actual == value or (isinstance(value, float) and np.isnan(value) and np.isnan(actual))
    assert batched.runtime_summary["batches"] == 1
    assert batched.runtime_summary["conditional_batches"] == 1
    assert batched.runtime_summary["network_calls"] == 1
    assert batched.runtime_summary["sequence_encoder_calls"] == 0


def test_batched_student_evaluation_reports_parent_cache_reuse():
    records = [make_record("a", 2), make_record("b", 2)]
    cache = ParentContextCache()
    editor = StudentEditor(ZeroStudent(), parent_cache=cache)
    report = evaluate_manifest_batched(records, editor, batch_size=2, method="student", split="dev")
    assert report.runtime_summary["parent_cache_misses"] == 1.0
    assert report.runtime_summary["parent_cache_hits"] == 1.0
    assert report.runtime_summary["parent_cache_entries"] == 1.0
    second = evaluate_manifest_batched(records, editor, batch_size=2, method="student", split="dev")
    assert second.runtime_summary["parent_cache_misses"] == 0.0
    assert second.runtime_summary["parent_cache_hits"] == 2.0


def test_batched_identity_only_shortcut_reports_zero_model_calls():
    coords = np.zeros((2, 4, 3), dtype=float)
    pair = StructurePair("identity-batch", "AA", "AA", coords, coords.copy(), ())
    record = PairRecord(pair, "identity-parent", "identity-family", "dev")
    report = evaluate_manifest_batched(
        [record], StudentEditor(ZeroStudent()), batch_size=1, method="student", split="dev"
    )
    assert report.runtime_summary["batches"] == 1.0
    assert report.runtime_summary["conditional_batches"] == 0.0
    assert report.runtime_summary["network_calls"] == 0.0
    assert report.runtime_summary["condition_branches"] == 0.0


def test_batched_mixed_identity_and_edit_counts_one_forward():
    coords = make_record("identity-template", 2).pair.parent_coords
    identity = PairRecord(
        StructurePair("mixed-identity", "AA", "AA", coords, coords.copy(), ()),
        "parent-identity",
        "family-identity",
        "dev",
    )
    edited = make_record("mixed-edited", 2)
    report = evaluate_manifest_batched(
        [identity, edited], StudentEditor(ZeroStudent()), batch_size=2, method="student", split="dev"
    )
    assert report.runtime_summary["batches"] == 1.0
    assert report.runtime_summary["conditional_batches"] == 1.0
    assert report.runtime_summary["network_calls"] == 1.0
    assert report.runtime_summary["structure_updates"] == 1.0
