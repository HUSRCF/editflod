import pytest

from scripts.analyze_student_generalization import analyze_evaluations


def _evaluation(local: float, site: float, *, family: str = "family"):
    return {
        "split": "dev",
        "metric_schema": "ospedit.structure_metrics.v2",
        "records": [{
            "pair_id": "pair",
            "family_id": family,
            "metrics": {
                "local_backbone_error": local,
                "mutation_site_backbone_error": site,
                "distance_change_error": 2.0,
                "distance_change_cosine": None,
                "remote_target_error": 3.0,
                "remote_scaffold_drift": 0.0,
                "predicted_distance_change_norm": 0.0,
                "true_distance_change_norm": 4.0,
            },
        }],
    }


def test_generalization_diagnostic_counts_paired_regressions():
    report = analyze_evaluations(
        _evaluation(1.0, 2.0),
        [
            ("graph", 0, _evaluation(1.5, 1.0)),
            ("graph", 1, _evaluation(1.25, 3.0)),
        ],
    )

    graph = report["architectures"]["graph"]
    local = graph["overall_student_minus_copy"]["local_backbone_error"]
    site = graph["overall_student_minus_copy"]["mutation_site_backbone_error"]
    assert local["mean"] == pytest.approx(0.375)
    assert local["improved"] == 0
    assert site["improved"] == 1
    assert graph["all_record_seed_local_errors_worse_than_copy"] is True


def test_generalization_diagnostic_rejects_mismatched_pairs():
    student = _evaluation(1.0, 1.0)
    student["records"][0]["pair_id"] = "other"

    with pytest.raises(ValueError, match="identical pair_id"):
        analyze_evaluations(_evaluation(1.0, 1.0), [("graph", 0, student)])


def test_generalization_diagnostic_rejects_mismatched_metric_schema():
    student = _evaluation(1.0, 1.0)
    student["metric_schema"] = "old"

    with pytest.raises(ValueError, match="same metric schema"):
        analyze_evaluations(_evaluation(1.0, 1.0), [("graph", 0, student)])
