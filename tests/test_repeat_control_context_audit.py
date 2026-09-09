import json
from pathlib import Path

from scripts.audit_repeat_control_context import audit_repeat_control_context


def _pdb(path: Path, ligand: str) -> None:
    path.write_text(
        "HEADER    TEST STRUCTURE\n"
        "EXPDTA    X-RAY DIFFRACTION\n"
        "REMARK   2 RESOLUTION.    1.50 ANGSTROMS.\n"
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
        "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C\n"
        f"HETATM    3  C1  {ligand:>3s} B 101       2.000   0.000   0.000  1.00 20.00           C\n"
        "END\n"
    )


def _background(tmp_path: Path, repeat_ligand: str) -> Path:
    parent = tmp_path / "parent.pdb"
    repeat = tmp_path / "repeat.pdb"
    _pdb(parent, "LIG")
    _pdb(repeat, repeat_ligand)
    payload = {
        "format": "ospedit.background_control_coverage.v1",
        "manifest_fingerprint": "fingerprint",
        "records": [{
            "pair_id": "pair-1",
            "parent_id": "parent-1",
            "family_id": "family-1",
            "split": "train",
            "parent_structure": str(parent),
            "parent_chain": "A",
            "controls": [{
                "repeat_structure": str(repeat),
                "repeat_chain": "A",
                "background": {
                    "neighborhood_rmsd_angstrom": 0.3,
                    "mutation_site_rmsd_angstrom": 0.2,
                    "distance_change_rms_angstrom": 0.1,
                },
            }],
        }],
    }
    path = tmp_path / "background.json"
    path.write_text(json.dumps(payload))
    return path


def test_repeat_context_accepts_matching_proximal_ligand(tmp_path):
    report = audit_repeat_control_context(_background(tmp_path, "LIG"))

    assert report["summary"]["selected_controls"] == 1
    assert report["records"][0]["repeat"]["proximal_hetero"] == ["LIG"]
    assert report["biological_assembly_verified"] is False


def test_repeat_context_rejects_proximal_ligand_mismatch(tmp_path):
    report = audit_repeat_control_context(_background(tmp_path, "ATP"))

    assert report["summary"]["selected_controls"] == 0
    assert report["records"][0]["rejection_reasons"] == ["proximal_hetero_mismatch"]
