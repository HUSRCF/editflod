import numpy as np
import pytest

from ospedit.data import PairRecord, StructurePair, write_manifest
from scripts.audit_response_learnability import REPORT_FORMAT, response_learnability_report


def _record(pair_id: str, family_id: str, split: str, shift: float) -> PairRecord:
    residue = np.array([[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    parent = np.repeat(residue[None, :, :], 4, axis=0)
    parent[:, :, 1] += np.arange(4)[:, None] * 4.0
    mutant = parent.copy()
    mutant[1, :, 0] += shift
    pair = StructurePair(pair_id, "AAAA", "AYAA", parent, mutant, (1,), ("N", "CA", "C", "O"))
    return PairRecord(pair, pair_id + "-parent", family_id, split)


def test_response_learnability_reports_family_macro_and_oracle_gain(tmp_path):
    manifest = tmp_path / "pairs.jsonl"
    write_manifest(
        [
            _record("a", "family-a", "train", 0.2),
            _record("b", "family-a", "train", 0.4),
            _record("c", "family-b", "dev", 0.3),
        ],
        manifest,
    )
    report = response_learnability_report(manifest)
    assert report["format"] == REPORT_FORMAT
    assert len(report["manifest_fingerprint"]) == 64
    assert report["summary"]["train"]["records"] == 2
    assert report["summary"]["train"]["families"] == 1
    assert report["families"]["family-a"]["records"] == 2
    row = report["records"][0]
    assert row["copy_error"]["local_backbone_error"] > 0
    assert row["oracle_error"]["local_backbone_error"] == pytest.approx(0.0, abs=1e-6)
    assert row["recoverable_fraction"]["local_backbone_error"] == pytest.approx(1.0)
    assert row["response_scope"]["translation_rms_angstrom"] > 0
    assert report["summary"]["test"]["records"] == 0


def test_response_learnability_rejects_invalid_scales(tmp_path):
    manifest = tmp_path / "pairs.jsonl"
    write_manifest([_record("a", "family-a", "train", 0.2)], manifest)
    with pytest.raises(ValueError, match="must be positive"):
        response_learnability_report(manifest, rotation_scale=0.0)
