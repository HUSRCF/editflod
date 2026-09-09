import json

import pytest

from scripts.assemble_runtime_table import assemble


def _suite(path, *, fingerprint="abc", pair_id="pair"):
    path.write_text(json.dumps({
        "manifest_fingerprint": fingerprint,
        "run_metadata": {"setup": 2.0},
        "methods": {
            "source": {
                "records": [{"pair_id": pair_id}, {"pair_id": pair_id + "-2"}],
                "runtime_summary": {
                    "total_seconds": 8.0,
                    "model_load_seconds_amortized": 2.0,
                    "network_calls": 4.0,
                    "sequence_encoder_calls": 2.0,
                },
            }
        },
    }))


def test_runtime_table_separates_setup_and_warm_candidate_cost(tmp_path):
    suite = tmp_path / "suite.json"
    _suite(suite)

    payload = assemble([f"method={suite}::source::metadata.setup::cpu"], (1, 3))

    row = payload["rows"][0]
    assert row["setup_seconds"] == 2.0
    assert row["observed_candidate_seconds"] == 6.0
    assert row["observed_candidate_count"] == 2
    assert row["mean_warm_candidate_seconds"] == 3.0
    assert row["resident_total_seconds_n3"] == 11.0
    assert row["network_calls_per_candidate"] == 2.0
    assert payload["cross_device_ranking_allowed"] is False
    assert payload["only_observed_resident_total_is_measured"] is True
    assert payload["projection_model"] == "linear_from_observed_candidate_mean"


def test_runtime_table_reads_external_setup_report(tmp_path):
    suite = tmp_path / "suite.json"
    setup = tmp_path / "setup.json"
    _suite(suite)
    setup.write_text(json.dumps({"runtime": {"load": 4.0}}))

    payload = assemble([f"method={suite}::source::{setup}#runtime.load::gpu"], (1,))

    assert payload["rows"][0]["setup_seconds"] == 4.0


def test_runtime_table_rejects_identity_mismatch(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _suite(first, fingerprint="abc")
    _suite(second, fingerprint="def")

    with pytest.raises(ValueError, match="identity mismatch"):
        assemble([
            f"one={first}::source::metadata.setup::cpu",
            f"two={second}::source::metadata.setup::cpu",
        ], (1,))
