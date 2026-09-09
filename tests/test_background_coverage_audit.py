import json
from pathlib import Path

import numpy as np
import pytest

from ospedit.data import PairRecord, StructurePair, write_manifest
from scripts.audit_background_coverage import RMSD_DEFINITION, background_coverage_report


def _pair(pair_id: str) -> StructurePair:
    parent = np.zeros((3, 1, 3))
    mutant = parent.copy()
    mutant[1, 0, 0] = 0.5
    return StructurePair(pair_id, "AAA", "AYA", parent, mutant, (1,))


def _record(pair: StructurePair, tmp_path: Path, split: str) -> PairRecord:
    return PairRecord(
        pair,
        "parent-1",
        "family-1",
        split,
        source_file=str(tmp_path / "parent.pdb"),
        source_chain="A",
    )


def _audit(path: Path, pair_id: str, repeat: str, site: float = 0.2, chain: str = "A") -> Path:
    row = {
        "pair_id": pair_id,
        "parent_structure": str(path.parent / "parent.pdb"),
        "parent_chain": "A",
        "repeat_structure": str(path.parent / repeat),
        "repeat_chain": chain,
        "mutation_index": 1,
        "backbone_rmsd_angstrom": 0.1,
        "mutation_site_rmsd_angstrom": site,
        "neighborhood_rmsd_angstrom": 0.3,
        "distance_change_rms_angstrom": 0.4,
        "max_translation_angstrom": 0.5,
        "max_rotation_radian": 0.6,
        "max_normalized_delta_norm": 0.7,
    }
    path.write_text(json.dumps({
        "format": "ospedit.repeat_structure_audit.v3",
        "coordinate_rmsd_definition": RMSD_DEFINITION,
        "records": [row],
    }))
    return path


def test_background_coverage_deduplicates_controls_and_excludes_mutant_labels(tmp_path):
    pair = _pair("pair-1")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([_record(pair, tmp_path, "train")], manifest)
    first = _audit(tmp_path / "first.json", pair.pair_id, "repeat.pdb")
    second = _audit(tmp_path / "second.json", pair.pair_id, "repeat.pdb")

    report = background_coverage_report(manifest, [first, second])

    assert report["summary"]["unique_controls"] == 1
    assert report["summary"]["overall"]["covered_pairs"] == 1
    assert report["summary"]["splits"]["train"]["pair_fraction"] == 1.0
    assert report["records"][0]["repeat_structures"] == 1
    assert report["records"][0]["background_max"]["mutation_site_rmsd_angstrom"] == 0.2
    assert report["records"][0]["background_median"]["mutation_site_rmsd_angstrom"] == 0.2
    assert len(report["records"][0]["controls"][0]["sources"]) == 2
    assert "mutant" not in json.dumps(report)


def test_background_coverage_rejects_old_metric_schema(tmp_path):
    pair = _pair("pair-1")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([_record(pair, tmp_path, "dev")], manifest)
    audit = _audit(tmp_path / "audit.json", pair.pair_id, "repeat.pdb")
    payload = json.loads(audit.read_text())
    payload["format"] = "ospedit.repeat_structure_audit.v2"
    audit.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="expected ospedit.repeat_structure_audit.v3"):
        background_coverage_report(manifest, [audit])


def test_background_coverage_rejects_inconsistent_duplicate(tmp_path):
    pair = _pair("pair-1")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([_record(pair, tmp_path, "test")], manifest)
    first = _audit(tmp_path / "first.json", pair.pair_id, "repeat.pdb", site=0.2)
    second = _audit(tmp_path / "second.json", pair.pair_id, "repeat.pdb", site=0.25)

    with pytest.raises(ValueError, match="inconsistent duplicate"):
        background_coverage_report(manifest, [first, second])


def test_background_coverage_keeps_distinct_chains_from_same_file(tmp_path):
    pair = _pair("pair-1")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([_record(pair, tmp_path, "train")], manifest)
    first = _audit(tmp_path / "first.json", pair.pair_id, "repeat.pdb", chain="A")
    second = _audit(tmp_path / "second.json", pair.pair_id, "repeat.pdb", chain="B")

    report = background_coverage_report(manifest, [first, second])

    assert report["summary"]["unique_controls"] == 2
    assert report["records"][0]["repeat_structures"] == 2
