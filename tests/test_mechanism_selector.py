from __future__ import annotations

import pytest

from scripts.select_mechanism_config import select_configuration


def _payload():
    def run(run_id, local, change, remote, seconds):
        return {
            "run_id": run_id,
            "config": {"noise_level": 0.5, "difference_step_size": run_id / 10},
            "manifest_fingerprint": "abc",
            "rows": [{
                "method": "C0_copy_parent",
                "mean_local_backbone_error": 0.15,
                "mean_remote_scaffold_frame_drift": 0.0,
            }, {
                "method": "C3_two_noise_shared_difference",
                "mean_local_backbone_error": local,
                "mean_distance_change_error": change,
                "mean_remote_scaffold_frame_drift": remote,
                "network_calls": seconds,
                "mean_seconds": seconds,
            }],
        }
    return {"format": "ospedit.mechanism_sweep.v1", "status": "completed", "expected_configurations": 3, "completed_configurations": 3, "manifest": "pairs.jsonl", "split": "dev", "runs": [
        run(0, 0.30, 0.20, 0.01, 3.0),
        run(1, 0.20, 0.30, 0.03, 1.0),
        run(2, 0.10, 0.10, 0.20, 1.0),
    ]}


def test_selector_applies_remote_constraint_and_deterministic_sort():
    selected = select_configuration(_payload(), method="C3_two_noise_shared_difference", max_remote_drift=0.05)
    assert selected["run_id"] == 1
    assert selected["selection_policy"]["max_remote_scaffold_frame_drift"] == 0.05


def test_selector_rejects_empty_candidate_set():
    with pytest.raises(ValueError, match="no .* satisfies"):
        select_configuration(_payload(), method="C3_two_noise_shared_difference", max_remote_drift=0.001)


def test_selector_can_bound_regression_against_copy_parent():
    selected = select_configuration(
        _payload(), method="C3_two_noise_shared_difference", max_remote_drift=0.05, max_local_regression=0.06
    )
    assert selected["run_id"] == 1
    assert selected["selection_policy"]["max_local_regression_vs_copy_parent"] == 0.06


def test_selector_skips_run_without_copy_parent_when_regression_bound_is_set():
    payload = _payload()
    for run in payload["runs"]:
        run["rows"] = [row for row in run["rows"] if row["method"] != "C0_copy_parent"]
    with pytest.raises(ValueError, match="no .* satisfies"):
        select_configuration(payload, method="C3_two_noise_shared_difference", max_remote_drift=0.05, max_local_regression=0.1)


def test_selector_rejects_test_split_without_explicit_diagnostic_override():
    payload = _payload()
    payload["split"] = "test"
    with pytest.raises(ValueError, match="requires a dev split"):
        select_configuration(payload, method="C3_two_noise_shared_difference", max_remote_drift=0.05)
    selected = select_configuration(
        payload, method="C3_two_noise_shared_difference", max_remote_drift=0.05, allow_non_dev=True
    )
    assert selected["split"] == "test"


def test_selector_rejects_mixed_manifest_fingerprints():
    payload = _payload()
    payload["runs"][1]["manifest_fingerprint"] = "different"
    with pytest.raises(ValueError, match="multiple manifest fingerprints"):
        select_configuration(payload, method="C3_two_noise_shared_difference", max_remote_drift=0.05)


def test_selector_rejects_incomplete_sweep():
    payload = _payload()
    payload["status"] = "running"
    with pytest.raises(ValueError, match="incomplete sweep"):
        select_configuration(payload, method="C3_two_noise_shared_difference", max_remote_drift=0.05)


def test_selector_can_bound_network_calls():
    selected = select_configuration(
        _payload(), method="C3_two_noise_shared_difference", max_remote_drift=0.05, max_network_calls=1.0
    )
    assert selected["run_id"] == 1
    assert selected["selection_policy"]["max_network_calls"] == 1.0


def test_selector_rejects_over_budget_network_calls():
    with pytest.raises(ValueError, match="no .* satisfies"):
        select_configuration(
            _payload(), method="C3_two_noise_shared_difference", max_remote_drift=0.05, max_network_calls=0.5
        )
