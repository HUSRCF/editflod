import pytest

torch = pytest.importorskip("torch")

from ospedit.student import ParentEditStudent
from ospedit.student_training import save_student_checkpoint
from scripts.export_student_checkpoint_report import checkpoint_report


def test_checkpoint_report_counts_optimizer_steps_and_macro_metrics(tmp_path):
    model = ParentEditStudent(parent_dim=8, hidden_dim=8, blocks=1, heads=2)
    path = tmp_path / "student.pt"
    evaluation = {
        "family_summary": {"a": {"local_backbone_error": 1.0}, "b": {"local_backbone_error": 3.0}},
        "copy_parent_baseline": {
            "family_summary": {"a": {"local_backbone_error": 2.0}, "b": {"local_backbone_error": 4.0}}
        },
    }
    save_student_checkpoint(
        model,
        path,
        epoch=3,
        history=[3.0, 2.0, 1.0],
        config={"record_count": 7, "batch_size": 2, "grad_accumulation_steps": 3, "evaluation": evaluation},
    )

    report = checkpoint_report(path)

    assert report["optimizer_steps"] == 6
    assert report["comparison"]["student_parent_family_macro"]["local_backbone_error"] == 2.0
    assert report["comparison"]["student_minus_copy"]["local_backbone_error"] == -1.0
