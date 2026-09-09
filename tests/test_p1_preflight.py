import json
import sys

import numpy as np
import pytest

from ospedit.data import PairRecord, StructurePair, write_manifest
from ospedit.p1_preflight import p1_preflight_report


def test_p1_preflight_combines_manifest_and_environment_report(tmp_path):
    coords = np.zeros((2, 1, 3))
    record = PairRecord(StructurePair("p1", "AA", "AY", coords, coords, (1,)), "parent", "family", "dev")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([record], manifest)
    report = p1_preflight_report(str(manifest))
    assert report["manifest"]["records"] == 1
    assert report["manifest"]["errors"] == []
    assert report["manifest"]["max_mutations"] is None
    assert report["manifest"]["dataset_structure"]["split_capacity_ok"] is False
    assert report["manifest"]["dataset_structure"]["split_record_counts"] == {"train": 0, "dev": 1, "test": 0}
    assert report["manifest"]["dataset_structure"]["assigned_splits_nonempty"] is False
    assert report["manifest"]["require_split_capacity"] is False
    assert report["ready"] is report["environment"]["ready_for_foldflow_import"]
    json.dumps(report, allow_nan=False)


def test_p1_preflight_cli_writes_report(tmp_path, monkeypatch, capsys):
    from ospedit.p1_preflight import main

    coords = np.zeros((2, 1, 3))
    record = PairRecord(StructurePair("p1-cli", "AA", "AY", coords, coords, (1,)), "parent", "family", "dev")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([record], manifest)
    output = tmp_path / "preflight.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ospedit-p1-preflight",
            "--manifest",
            str(manifest),
            "--max-mutations",
            "1",
            "--require-split-capacity",
            "--foldflow-root",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit):
        main()
    saved = json.loads(output.read_text())
    assert json.loads(capsys.readouterr().out) == saved
    assert saved["manifest"]["max_mutations"] == 1
    assert saved["manifest"]["require_split_capacity"] is True
    assert saved["environment"]["foldflow_root"] == str(tmp_path)


def test_p1_preflight_reports_group_capacity(tmp_path):
    coords = np.zeros((2, 1, 3))
    records = [
        PairRecord(StructurePair("p1-a", "AA", "AY", coords, coords, (1,)), "parent", "family-a", "dev"),
        PairRecord(StructurePair("p1-b", "AA", "AY", coords, coords, (1,)), "parent", "family-b", "dev"),
    ]
    manifest = tmp_path / "capacity.jsonl"
    write_manifest(records, manifest)
    report = p1_preflight_report(str(manifest))
    structure = report["manifest"]["dataset_structure"]
    assert structure["unique_parents"] == 1
    assert structure["connected_group_count"] == 1
    assert structure["split_capacity_ok"] is False
    assert structure["warnings"]


def test_p1_preflight_can_make_split_capacity_a_hard_gate(tmp_path):
    coords = np.zeros((2, 1, 3))
    record = PairRecord(
        StructurePair("capacity-gate", "AA", "AY", coords, coords, (1,)),
        "parent",
        "family",
        "dev",
    )
    manifest = tmp_path / "capacity-gate.jsonl"
    write_manifest([record], manifest)
    report = p1_preflight_report(str(manifest), require_split_capacity=True)
    assert report["manifest"]["require_split_capacity"] is True
    assert any("split is impossible" in error for error in report["manifest"]["errors"])
    assert report["ready"] is False


def test_p1_preflight_rejects_empty_assigned_split_in_strict_mode(tmp_path):
    coords = np.zeros((2, 1, 3))
    records = [
        PairRecord(
            StructurePair(f"assigned-{index}", "AA", "AY", coords, coords, (1,)),
            f"parent-{index}",
            f"family-{index}",
            "train",
        )
        for index in range(3)
    ]
    manifest = tmp_path / "assigned-split.jsonl"
    write_manifest(records, manifest)
    report = p1_preflight_report(str(manifest), require_split_capacity=True)
    assert report["manifest"]["dataset_structure"]["split_capacity_ok"] is True
    assert report["manifest"]["dataset_structure"]["assigned_splits_nonempty"] is False
    assert any("split assignment has an empty" in error for error in report["manifest"]["errors"])
