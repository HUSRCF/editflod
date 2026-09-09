import json

import pytest

from scripts.analyze_background_signal import background_signal_report


def _reports(tmp_path):
    background = {
        "format": "ospedit.background_control_coverage.v1",
        "metric_schema": "ospedit.structure_metrics.v2",
        "manifest_fingerprint": "same",
        "records": [{
            "pair_id": "pair-1",
            "parent_id": "parent-1",
            "family_id": "family-1",
            "split": "dev",
            "repeat_structures": 2,
            "background_max": {
                "neighborhood_rmsd_angstrom": 0.5,
                "mutation_site_rmsd_angstrom": 0.25,
                "distance_change_rms_angstrom": 0.2,
            },
            "background_median": {
                "neighborhood_rmsd_angstrom": 0.4,
                "mutation_site_rmsd_angstrom": 0.2,
                "distance_change_rms_angstrom": 0.1,
            },
        }],
    }
    response = {
        "format": "ospedit.response_learnability_audit.v1",
        "manifest_fingerprint": "same",
        "records": [{
            "pair_id": "pair-1",
            "copy_error": {
                "local_backbone_error": 1.0,
                "mutation_site_backbone_error": 0.5,
                "distance_change_error": 0.4,
            },
        }],
    }
    background_path = tmp_path / "background.json"
    response_path = tmp_path / "response.json"
    background_path.write_text(json.dumps(background))
    response_path.write_text(json.dumps(response))
    return background_path, response_path


def test_background_signal_reports_max_and_median_ratios(tmp_path):
    background, response = _reports(tmp_path)

    report = background_signal_report(background, response)

    assert report["records"][0]["local"]["signal_to_background_max"] == 2.0
    assert report["records"][0]["local"]["signal_to_background_median"] == 2.5
    assert report["summary"]["at_least_two_controls"]["splits"]["dev"]["records"] == 1
    assert report["usage"] == "diagnostic_only_not_a_training_weight"


def test_background_signal_rejects_manifest_mismatch(tmp_path):
    background, response = _reports(tmp_path)
    payload = json.loads(response.read_text())
    payload["manifest_fingerprint"] = "different"
    response.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="different manifests"):
        background_signal_report(background, response)


def test_background_signal_adds_context_prescreened_cohort(tmp_path):
    background, response = _reports(tmp_path)
    context = tmp_path / "context.json"
    context.write_text(json.dumps({
        "format": "ospedit.repeat_control_context_audit.v1",
        "manifest_fingerprint": "same",
        "records": [{
            "pair_id": "pair-1",
            "selected": True,
            "background": {
                "neighborhood_rmsd_angstrom": 0.25,
                "mutation_site_rmsd_angstrom": 0.1,
                "distance_change_rms_angstrom": 0.2,
            },
        }],
    }))

    report = background_signal_report(background, response, context)

    assert report["summary"]["context_prescreened_controls"]["overall"]["records"] == 1
    assert report["context_prescreened_records"][0]["local"]["signal_to_background_max"] == 4.0
