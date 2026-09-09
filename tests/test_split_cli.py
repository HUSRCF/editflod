import json
import sys

import numpy as np

from ospedit.data import PairRecord, StructurePair, write_manifest
from ospedit.split_cli import main


def _record(index: int) -> PairRecord:
    coords = np.zeros((2, 4, 3), dtype=float)
    pair = StructurePair(f"split-{index}", "AA", "AB", coords, coords.copy(), (1,), ("N", "CA", "C", "O"))
    return PairRecord(pair, f"parent-{index}", f"family-{index}", "dev")


def test_split_cli_writes_provenance_report(tmp_path, monkeypatch):
    manifest = tmp_path / "input.jsonl"
    output = tmp_path / "nested" / "split.jsonl"
    report = tmp_path / "nested" / "split.report.json"
    write_manifest([_record(index) for index in range(4)], manifest)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ospedit-split-manifest",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--seed",
            "17",
            "--report",
            str(report),
            "--require-nonempty",
        ],
    )
    main()
    payload = json.loads(report.read_text())
    assert payload["seed"] == 17
    assert payload["group_by"] == "family"
    assert payload["split_counts"] == {"train": 2, "dev": 1, "test": 1}
    assert isinstance(payload["manifest_fingerprint"], str)
    assert output.exists()


def test_split_cli_reports_duplicate_pair_ids_cleanly(tmp_path, monkeypatch):
    manifest = tmp_path / "duplicate.jsonl"
    write_manifest([_record(0), _record(0)], manifest)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ospedit-split-manifest",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "split.jsonl"),
        ],
    )
    try:
        main()
    except SystemExit as error:
        assert "split assignment failed" in str(error)
        assert "duplicate pair_id" in str(error)
    else:
        raise AssertionError("expected duplicate pair_id failure")
