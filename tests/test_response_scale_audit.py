import numpy as np
import pytest

from ospedit.data import PairRecord, StructurePair, write_manifest
from scripts.audit_response_scale import audit_response_scale


def _record(pair_id: str, shift: float) -> PairRecord:
    residue = np.array([[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    parent = np.repeat(residue[None, :, :], 2, axis=0)
    mutant = parent.copy()
    mutant[1, :, 0] += shift
    pair = StructurePair(pair_id, "AA", "AY", parent, mutant, (1,), ("N", "CA", "C", "O"))
    return PairRecord(pair, "parent", pair_id, "dev")


def test_response_audit_reports_and_filters_by_normalized_norm(tmp_path):
    manifest = tmp_path / "pairs.jsonl"
    write_manifest([_record("small", 0.2), _record("large", 2.0)], manifest)
    report, selected = audit_response_scale(manifest, max_normalized_norm=1.0)
    assert report["format"] == "ospedit.response_scale_audit.v1"
    assert len(report["manifest_fingerprint"]) == 64
    assert report["summary"]["dev"]["count"] == 2
    assert report["records"][0]["max_translation_angstrom"] == pytest.approx(0.2)
    assert report["records"][0]["max_rotation_radian"] == pytest.approx(0.0)
    assert [record.pair.pair_id for record in selected] == ["small"]


def test_response_audit_rejects_invalid_scales(tmp_path):
    manifest = tmp_path / "pairs.jsonl"
    write_manifest([_record("pair", 0.2)], manifest)
    with pytest.raises(ValueError, match="must be positive"):
        audit_response_scale(manifest, translation_scale=0.0)
