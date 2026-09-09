import json

import numpy as np
import pytest

from ospedit.data import load_manifest, validate_manifest


def row(split, family="fam-a"):
    return {
        "pair_id": f"pair-{split}",
        "parent_id": "parent-1",
        "family_id": family,
        "split": split,
        "source_sequence": "AAA",
        "target_sequence": "AYA",
        "parent_coords": np.zeros((3, 1, 3)).tolist(),
        "mutant_coords": np.zeros((3, 1, 3)).tolist(),
        "mutation_indices": [1],
    }


def test_manifest_rejects_family_leakage(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps([row("train"), row("test")]))
    records = load_manifest(path)
    errors = validate_manifest(records)
    assert any("crosses" in error for error in errors)


def test_manifest_can_explicitly_allow_diagnostic_split_overlap(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps([row("train"), row("dev")]))

    assert validate_manifest(load_manifest(path), allow_split_overlap=True) == []


def test_manifest_accepts_clean_jsonl(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(row("train")) + "\n")
    assert validate_manifest(load_manifest(path)) == []


def test_manifest_rejects_duplicate_pairs_and_parent_leakage(tmp_path):
    path = tmp_path / "leak.json"
    first = row("train", family="fam-a")
    second = row("test", family="fam-b")
    second["pair_id"] = first["pair_id"]
    second["parent_id"] = first["parent_id"]
    path.write_text(json.dumps([first, second]))
    errors = validate_manifest(load_manifest(path))
    assert any("duplicate pair_id" in error for error in errors)
    assert any("parent" in error and "crosses" in error for error in errors)


def test_manifest_can_enforce_single_point_scope(tmp_path):
    path = tmp_path / "multi.json"
    payload = row("dev")
    payload["target_sequence"] = "AYG"
    payload["mutation_indices"] = [1, 2]
    path.write_text(json.dumps([payload]))
    records = load_manifest(path)
    errors = validate_manifest(records, max_mutations=1)
    assert any("max_mutations=1" in error for error in errors)


def test_manifest_rejects_missing_sequence_fields_with_actionable_error(tmp_path):
    path = tmp_path / "missing-sequence.json"
    payload = row("dev")
    del payload["source_sequence"]
    path.write_text(json.dumps([payload]))
    with pytest.raises(ValueError, match="requires string source_sequence and target_sequence"):
        load_manifest(path)
