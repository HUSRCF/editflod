import csv
from pathlib import Path

from scripts.build_platinum_manifest import records_from_platinum_csv


def _pdb(path: Path, residues: tuple[tuple[str, int], ...], ligand: str = "ATP") -> None:
    lines = ["HEADER    TEST", "EXPDTA    X-RAY DIFFRACTION"]
    serial = 1
    for residue, number in residues:
        for atom, x in (("N", 0.0), ("CA", 1.0), ("C", 2.0), ("O", 3.0)):
            lines.append(
                f"ATOM  {serial:5d} {atom:>4s} {residue:>3s} A{number:4d}"
                f"    {x + number:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C"
            )
            serial += 1
    lines.append(
        f"HETATM{serial:5d}  P   {ligand:>3s} A 900       9.000   0.000   0.000  1.00 20.00           P"
    )
    path.write_text("\n".join(lines) + "\nEND\n")


def _csv(path: Path) -> None:
    fields = [
        "mutation",
        "affin.chain",
        "affin.lig_id",
        "mut.mt_pdb",
        "mut.wt_pdb",
        "mut.is_single_point",
        "mut.uniprot",
        "prot.stoichiometry",
        "affin.exptal_method",
        "mut.pmid",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "mutation": "A10G",
            "affin.chain": "A",
            "affin.lig_id": "ATP",
            "mut.mt_pdb": "2MTT",
            "mut.wt_pdb": "1WTT",
            "mut.is_single_point": "YES",
            "mut.uniprot": "P12345",
            "prot.stoichiometry": "MONOMERIC (AUTHOR)",
            "affin.exptal_method": "kinetic",
            "mut.pmid": "123",
        })


def test_platinum_import_requires_exact_author_mapped_mutation_and_shared_ligand(tmp_path):
    _pdb(tmp_path / "1WTT.pdb", (("ALA", 10), ("GLY", 11)))
    _pdb(tmp_path / "2MTT.pdb", (("GLY", 10), ("GLY", 11)))
    source = tmp_path / "platinum.csv"
    _csv(source)

    records, counters = records_from_platinum_csv(source, tmp_path, min_length=1)

    assert counters["accepted"] == 1
    assert records[0].pair.mutation_indices == (0,)
    assert records[0].environment_metadata["shared_declared_ligand_ids"] == ["ATP"]


def test_platinum_import_rejects_missing_declared_ligand(tmp_path):
    _pdb(tmp_path / "1WTT.pdb", (("ALA", 10),), ligand="ATP")
    _pdb(tmp_path / "2MTT.pdb", (("GLY", 10),), ligand="HEM")
    source = tmp_path / "platinum.csv"
    _csv(source)
    rejections = []

    records, counters = records_from_platinum_csv(
        source, tmp_path, min_length=1, rejections=rejections
    )

    assert records == []
    assert counters["ligand_mismatch"] == 1
    assert "not present in both" in rejections[0]["reason"]


def test_platinum_import_allows_high_coverage_terminal_crop(tmp_path):
    _pdb(tmp_path / "1WTT.pdb", (("ALA", 9), ("ALA", 10), ("GLY", 11)))
    _pdb(tmp_path / "2MTT.pdb", (("GLY", 10), ("GLY", 11)))
    source = tmp_path / "platinum.csv"
    _csv(source)

    records, counters = records_from_platinum_csv(
        source,
        tmp_path,
        min_length=1,
        min_mapping_coverage=0.65,
    )

    assert counters["terminal_overlap_mapping"] == 1
    assert records[0].residue_map[0] == ("A", 10, "")
    mapping = records[0].environment_metadata["residue_mapping"]
    assert mapping["mode"] == "terminal_overlap_crop"
    assert mapping["parent_terminal_trim"] == [1, 0]


def test_platinum_import_rejects_internal_coordinate_gap(tmp_path):
    _pdb(
        tmp_path / "1WTT.pdb",
        (("ALA", 9), ("ALA", 10), ("GLY", 11), ("GLY", 12)),
    )
    _pdb(tmp_path / "2MTT.pdb", (("ALA", 9), ("GLY", 10), ("GLY", 12)))
    source = tmp_path / "platinum.csv"
    _csv(source)
    rejections = []

    records, counters = records_from_platinum_csv(
        source,
        tmp_path,
        min_length=1,
        min_mapping_coverage=0.5,
        rejections=rejections,
    )

    assert records == []
    assert counters["invalid_mapping"] == 1
    assert "internal gap or reordering" in rejections[0]["reason"]
