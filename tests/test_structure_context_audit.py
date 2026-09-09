from pathlib import Path

import numpy as np

from ospedit.data import PairRecord, StructurePair
from ospedit.structure_context import structure_context
from scripts.audit_structure_context import audit_structure_context


def _pdb(
    path: Path,
    *,
    residue: str,
    ligand: str = "LIG",
    ligand_chain: str = "A",
    extra_chain: bool = False,
) -> None:
    lines = [
        "HEADER    TEST STRUCTURE",
        "EXPDTA    X-RAY DIFFRACTION",
        "REMARK   2 RESOLUTION.    1.50 ANGSTROMS.",
        f"ATOM      1  N   {residue:>3s} A   1       0.000   0.000   0.000  1.00 20.00           N",
        f"ATOM      2  CA  {residue:>3s} A   1       1.000   0.000   0.000  1.00 20.00           C",
        f"HETATM    3  C1  {ligand:>3s} {ligand_chain} 101       2.000   0.000   0.000  1.00 20.00           C",
    ]
    if extra_chain:
        lines.append(
            "ATOM      4  CA  GLY B   1       3.000   0.000   0.000  1.00 20.00           C"
        )
    path.write_text("\n".join(lines) + "\nEND\n")


def _record(tmp_path: Path, *, mutant_ligand: str = "LIG", extra_chain: bool = False) -> PairRecord:
    parent_path = tmp_path / "parent.pdb"
    mutant_path = tmp_path / "mutant.pdb"
    _pdb(parent_path, residue="ALA", ligand="LIG")
    _pdb(mutant_path, residue="GLY", ligand=mutant_ligand, extra_chain=extra_chain)
    coords = np.zeros((1, 1, 3), dtype=float)
    pair = StructurePair("pair", "A", "G", coords, coords, (0,), ("CA",))
    return PairRecord(
        pair,
        parent_id="parent_A",
        family_id="family",
        split="test",
        source_file=str(parent_path),
        target_file=str(mutant_path),
        source_chain="A",
        target_chain="A",
    )


def test_structure_context_accepts_matching_single_chain_pair(tmp_path):
    record = _record(tmp_path)

    report, selected = audit_structure_context([record])

    assert selected == [record]
    assert report["selected_records"] == 1
    assert report["records"][0]["parent"]["target_hetero"] == ["LIG"]


def test_structure_context_rejects_ligand_mismatch(tmp_path):
    record = _record(tmp_path, mutant_ligand="ATP")

    report, selected = audit_structure_context([record])

    assert selected == []
    assert report["records"][0]["rejection_reasons"] == ["proximal_hetero_mismatch"]


def test_structure_context_rejects_multichain_complex(tmp_path):
    record = _record(tmp_path, extra_chain=True)

    report, selected = audit_structure_context([record])

    assert selected == []
    assert report["records"][0]["rejection_reasons"] == ["multiple_protein_chains"]


def test_structure_context_finds_proximal_ligand_on_another_chain(tmp_path):
    path = tmp_path / "structure.pdb"
    _pdb(path, residue="ALA", ligand_chain="B")

    context = structure_context(path, "A", ignored_hetero=frozenset({"HOH"}))

    assert context["target_hetero"] == []
    assert context["proximal_hetero"] == ["LIG"]


def test_structure_context_reports_actual_protein_contacts(tmp_path):
    path = tmp_path / "structure.pdb"
    _pdb(path, residue="ALA", extra_chain=True)

    context = structure_context(path, "A", ignored_hetero=frozenset({"HOH"}))

    assert context["protein_chain_count"] == 2
    assert context["protein_contact_chains"] == ["B"]
