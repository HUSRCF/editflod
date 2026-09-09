import json
import sys

import numpy as np
import pytest

from ospedit.data import PairRecord, StructurePair, write_manifest
from ospedit.student_data import target_local_delta
from scripts.evaluate_teacher_cache import main


def _fixture(tmp_path):
    coords = np.asarray(
        [
            [[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
            [[-1.0, 4.3, 0.0], [0.0, 3.8, 0.0], [1.0, 3.8, 0.0], [1.5, 3.8, 0.0]],
        ]
    )
    mutant = coords.copy()
    mutant[1, :, 0] += 0.25
    pair = StructurePair("cli-eval", "AA", "AY", coords, mutant, (1,))
    manifest = tmp_path / "pairs.jsonl"
    write_manifest([PairRecord(pair, "parent", "family", "dev")], manifest)
    delta, valid = target_local_delta(pair)
    identity = np.tile(np.eye(3), (2, 1, 1))
    np.savez_compressed(
        tmp_path / "pair.npz",
        **{
            "level_0.25_source_rotations": identity,
            "level_0.25_source_origins": coords[:, 1],
            "level_0.25_target_rotations": identity,
            "level_0.25_target_origins": mutant[:, 1],
            "level_0.25_local_delta": delta,
            "level_0.25_valid": valid.astype(np.uint8),
        },
    )
    index = tmp_path / "index.json"
    index.write_text(json.dumps({
        "format": "ospedit.teacher_cache.v1",
        "split": "dev",
        "noise_levels": [0.25],
        "entries": [{"pair_id": "cli-eval", "file": "pair.npz", "length": 2}],
    }))
    return manifest, index


def test_teacher_evaluation_cli_fails_gate_after_writing_report(tmp_path, monkeypatch):
    manifest, index = _fixture(tmp_path)
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "evaluate_teacher_cache", "--manifest", str(manifest), "--teacher-cache", str(index),
        "--noise-level", "0.25", "--split", "dev", "--min-mutation-cosine", "1.1",
        "--output", str(output),
    ])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    payload = json.loads(output.read_text())
    assert payload["direction_gate"]["accepted"] is False
    assert payload["direction_gate"]["reasons"]


def test_teacher_evaluation_cli_rejects_negative_error_threshold(tmp_path, monkeypatch):
    manifest, index = _fixture(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "evaluate_teacher_cache", "--manifest", str(manifest), "--teacher-cache", str(index),
        "--noise-level", "0.25", "--split", "dev", "--max-mutation-rmse", "-0.1",
        "--output", str(tmp_path / "report.json"),
    ])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_teacher_evaluation_cli_verifies_manifest_checksums(tmp_path, monkeypatch):
    manifest, index = _fixture(tmp_path)
    row = json.loads(manifest.read_text())
    row["source_file"] = str(tmp_path / "missing-parent.pdb")
    row["source_checksum"] = "a" * 64
    manifest.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(sys, "argv", [
        "evaluate_teacher_cache", "--manifest", str(manifest), "--teacher-cache", str(index),
        "--noise-level", "0.25", "--split", "dev", "--verify-checksums",
        "--output", str(tmp_path / "report.json"),
    ])
    with pytest.raises(SystemExit) as error:
        main()
    assert "manifest audit failed" in str(error.value)
