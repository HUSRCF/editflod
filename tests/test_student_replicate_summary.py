import pytest

from scripts.summarize_student_replicates import REPORT_FORMAT, summarize_reports


def _report(seed: int, local_error: float, *, geometry: bool = False):
    return {
        "checkpoint": f"seed-{seed}.pt",
        "checkpoint_sha256": str(seed) * 64,
        "epoch": 20,
        "optimizer_steps": 100,
        "loss_history": [0.2 + seed, 0.1 + seed],
        "configuration": {
            "manifest_fingerprint": "a" * 64,
            "student_architecture": "spatial_graph",
            "geometry_features": geometry,
            "seed": seed,
        },
        "comparison": {
            "student_parent_family_macro": {
                "local_backbone_error": local_error,
                "distance_change_cosine": None,
            }
        },
    }


def test_replicate_summary_aggregates_numeric_metrics():
    report = summarize_reports([_report(0, 0.2), _report(1, 0.4)])
    assert report["format"] == REPORT_FORMAT
    assert report["aggregate"]["local_backbone_error"]["mean"] == pytest.approx(0.3)
    assert report["aggregate"]["local_backbone_error"]["std"] == pytest.approx(0.1)
    assert [run["seed"] for run in report["runs"]] == [0, 1]
    assert report["runs"][0]["final_loss"] == pytest.approx(0.1)


def test_replicate_summary_rejects_incomparable_configuration():
    with pytest.raises(ValueError, match="geometry_features"):
        summarize_reports([_report(0, 0.2), _report(1, 0.2, geometry=True)])
