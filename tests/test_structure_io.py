from dataclasses import replace
from pathlib import Path

import numpy as np

from Bio.PDB import MMCIFIO, PDBParser

from ospedit.data import BACKBONE_ATOMS, pair_parsed_structures, parse_structure


PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00 10.00           C
ATOM      3  C   ALA A   1       2.000   1.400   0.000  1.00 10.00           C
ATOM      4  O   ALA A   1       1.500   2.450   0.000  1.00 10.00           O
ATOM      5  N   TYR A   2       3.200   1.450   0.000  1.00 10.00           N
ATOM      6  CA  TYR A   2       3.900   2.750   0.000  1.00 10.00           C
ATOM      7  C   TYR A   2       5.350   2.600   0.000  1.00 10.00           C
ATOM      8  O   TYR A   2       6.000   3.600   0.000  1.00 10.00           O
TER
END
"""


def test_parse_and_pair_pdb(tmp_path: Path):
    parent_path = tmp_path / "parent.pdb"
    mutant_path = tmp_path / "mutant.pdb"
    parent_path.write_text(PDB.replace("TYR", "ALA"))
    mutant_path.write_text(PDB)
    parent = parse_structure(parent_path, "A")
    mutant = parse_structure(mutant_path, "A")
    assert parent.sequence == "AA"
    assert mutant.sequence == "AY"
    assert parent.coords.shape == (2, 4, 3)
    assert parent.atom_names == BACKBONE_ATOMS
    pair = pair_parsed_structures(parent, mutant, "pdb-toy")
    assert pair.mutation_indices == (1,)
    assert np.isfinite(pair.parent_coords).all()


def test_pair_rigidly_aligns_mutant_into_parent_frame(tmp_path: Path):
    parent_path = tmp_path / "parent.pdb"
    mutant_path = tmp_path / "mutant.pdb"
    parent_path.write_text(PDB.replace("TYR", "ALA"))
    mutant_path.write_text(PDB)
    parent = parse_structure(parent_path, "A")
    mutant = parse_structure(mutant_path, "A")
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = mutant.coords @ rotation.T + np.array([12.0, -3.0, 8.0])

    pair = pair_parsed_structures(parent, replace(mutant, coords=transformed), "rigid-pose")

    assert np.allclose(pair.mutant_coords, parent.coords, atol=1e-6)


def test_missing_backbone_atom_is_nan(tmp_path: Path):
    path = tmp_path / "missing.pdb"
    path.write_text(PDB.replace("  O   TYR", "  O   TYR", 1).replace("ATOM      4  O   ALA A   1       1.500   2.450   0.000  1.00 10.00           O\n", ""))
    parsed = parse_structure(path, "A")
    assert np.isnan(parsed.coords[0, 3]).all()


def test_parse_prefers_altloc_a(tmp_path: Path):
    path = tmp_path / "altloc.pdb"
    path.write_text(PDB.replace(
        "ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00 10.00           C\n",
        "ATOM      2  CA AALA A   1       1.450   0.000   0.000  1.00 10.00           C\n"
        "ATOM      9  CA BALA A   1       9.450   0.000   0.000  1.00 10.00           C\n"
    ).replace("TYR", "ALA"))
    parsed = parse_structure(path, "A")
    assert np.allclose(parsed.coords[0, 1], [1.45, 0.0, 0.0])


def test_parse_mmcif_roundtrip(tmp_path: Path):
    pdb_path = tmp_path / "source.pdb"
    cif_path = tmp_path / "source.cif"
    pdb_path.write_text(PDB)
    structure = PDBParser(QUIET=True).get_structure("source", str(pdb_path))
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(str(cif_path))
    parsed = parse_structure(cif_path, "A")
    assert parsed.sequence == "AY"
    assert parsed.coords.shape == (2, 4, 3)
    assert parsed.residue_ids == (("A", 1, ""), ("A", 2, ""))
