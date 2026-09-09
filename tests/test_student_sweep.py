import json

import pytest

from scripts.run_student_bound_sweep import _last_json, _result_summary

def test_last_json_extracts_training_result():
    assert _last_json("warning\n{\"output\": \"student.pt\", \"epochs\": 2}\n") == {
        "output": "student.pt",
        "epochs": 2,
    }


def test_last_json_requires_result_object():
    with pytest.raises(ValueError, match="JSON result"):
        _last_json(json.dumps({"status": "failed"}))


def test_result_summary_extracts_family_macro_metrics():
    result = {
        "evaluation": {
            "family_summary": {
                "a": {"local_backbone_error": 1.0, "mutation_site_backbone_error": 0.5, "parent_to_prediction_frame_rmsd": 2.0, "distance_change_error": 3.0},
                "b": {"local_backbone_error": 3.0, "mutation_site_backbone_error": 1.5, "parent_to_prediction_frame_rmsd": 4.0, "distance_change_error": 5.0},
            },
            "runtime_summary": {"mean_seconds": 0.5},
            "copy_parent_baseline": {
                "family_summary": {
                    "a": {"local_backbone_error": 0.5, "mutation_site_backbone_error": 0.25},
                    "b": {"local_backbone_error": 1.5, "mutation_site_backbone_error": 0.75},
                }
            },
        }
    }
    assert _result_summary(result) == {
        "local_backbone_error": 2.0,
        "mutation_site_backbone_error": 1.0,
        "mutation_site_global_rmsd": None,
        "copy_parent_local_backbone_error": 1.0,
        "copy_parent_mutation_site_backbone_error": 0.5,
        "copy_parent_mutation_site_global_rmsd": None,
        "local_error_minus_copy": 1.0,
        "mutation_site_error_minus_copy": 0.5,
        "mutation_site_global_error_minus_copy": None,
        "parent_to_prediction_frame_rmsd": 3.0,
        "distance_change_error": 4.0,
        "mean_seconds": 0.5,
    }
