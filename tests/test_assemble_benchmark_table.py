import json

import pytest

from scripts.assemble_benchmark_table import assemble


def _report(path, *, split="test", fingerprint="abc", pair_id="pair"):
    path.write_text(json.dumps({
        "split": split,
        "manifest_fingerprint": fingerprint,
        "methods": {
            "source": {
                "records": [{
                    "pair_id": pair_id,
                    "parent_id": "parent",
                    "family_id": "family",
                    "metrics": {"pair_id": pair_id, "local_backbone_error": 1.0},
                    "runtime": {"total_seconds": 2.0, "extras": {"inference_seconds": 1.5}},
                }]
            }
        },
    }))


def test_assemble_flattens_metrics_runtime_and_provenance(tmp_path):
    report = tmp_path / "report.json"
    _report(report)

    payload = assemble([f"method={report}::source"])

    row = payload["rows"][0]
    assert row["method"] == "method"
    assert row["metric_local_backbone_error"] == 1.0
    assert row["runtime_total_seconds"] == 2.0
    assert row["runtime_inference_seconds"] == 1.5


def test_assemble_rejects_mismatched_split_identity(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _report(first, fingerprint="abc")
    _report(second, fingerprint="def")

    with pytest.raises(ValueError, match="split identity mismatch"):
        assemble([f"one={first}::source", f"two={second}::source"])


def test_assemble_rejects_duplicate_output_method(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _report(first)
    _report(second)

    with pytest.raises(ValueError, match="duplicate output method"):
        assemble([f"same={first}::source", f"same={second}::source"])
