import csv
from pathlib import Path

from scripts.build_microminer_manifest import records_from_microminer_candidates
from scripts.select_microminer_candidates import COLUMNS


def _pdb(path: Path, residues: tuple[tuple[str, int], ...]) -> None:
    lines = ["HEADER    TEST", "EXPDTA    X-RAY DIFFRACTION"]
    serial = 1
    for residue, number in residues:
        for atom, x in (("N", 0.0), ("CA", 1.0), ("C", 2.0), ("O", 3.0)):
            lines.append(
                f"ATOM  {serial:5d} {atom:>4s} {residue:>3s} A{number:4d}"
                f"    {x + number:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C"
            )
            serial += 1
    path.write_text("\n".join(lines) + "\nEND\n")


def _candidates(path: Path, **changes: str) -> None:
    row = {
        "queryName": "1AAA", "queryChain": "A", "queryAA": "ALA", "queryPos": "10",
        "hitName": "2AAA", "hitChain": "A", "hitAA": "VAL", "hitPos": "20",
        "siteIdentity": "0.95", "siteBackBoneRMSD": "0.4", "siteAllAtomRMSD": "0.7",
        "nofSiteResidues": "12", "alignmentLDDT": "0.95", "fullSeqId": "0.99",
    }
    row.update(changes)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow(row)


def test_microminer_import_accepts_equal_length_sequence_index_mapping(tmp_path):
    _pdb(tmp_path / "1AAA.pdb", (("ALA", 10), ("GLY", 11)))
    _pdb(tmp_path / "2AAA.pdb", (("VAL", 20), ("GLY", 21)))
    candidates = tmp_path / "candidates.csv"
    _candidates(candidates)

    records, counters = records_from_microminer_candidates(
        candidates, tmp_path, min_length=1
    )

    assert counters["accepted"] == 1
    assert counters["sequence_index_equal_length_mapping"] == 1
    assert records[0].pair.mutation_indices == (0,)
    assert records[0].residue_map[0] == ("A", 10, "")
    assert records[0].environment_metadata["residue_mapping"]["mode"] == (
        "sequence_index_equal_length"
    )
    assert records[0].split == "train"


def test_microminer_import_rejects_more_than_one_observed_difference(tmp_path):
    _pdb(tmp_path / "1AAA.pdb", (("ALA", 10), ("GLY", 11)))
    _pdb(tmp_path / "2AAA.pdb", (("VAL", 20), ("SER", 21)))
    candidates = tmp_path / "candidates.csv"
    _candidates(candidates)
    rejections = []

    records, counters = records_from_microminer_candidates(
        candidates, tmp_path, min_length=1, rejections=rejections
    )

    assert records == []
    assert counters["not_single_substitution"] == 1
    assert "found 2" in rejections[0]["reason"]


def test_microminer_import_groups_shared_uniprot(tmp_path):
    _pdb(tmp_path / "1AAA.pdb", (("ALA", 10), ("GLY", 11)))
    _pdb(tmp_path / "2AAA.pdb", (("VAL", 20), ("GLY", 21)))
    candidates = tmp_path / "candidates.csv"
    _candidates(candidates)

    records, counters = records_from_microminer_candidates(
        candidates,
        tmp_path,
        min_length=1,
        chain_uniprot={"1AAA_A": ["P12345"], "2AAA_A": ["P12345", "Q99999"]},
    )

    assert counters["accepted"] == 1
    assert records[0].family_id == "microminer_uniprot_P12345"
    assert records[0].experiment_metadata["uniprot"]["shared"] == ["P12345"]
    assert records[0].experiment_metadata["family_grouping"] == "shared_uniprot"


def test_microminer_import_rejects_missing_shared_uniprot(tmp_path):
    _pdb(tmp_path / "1AAA.pdb", (("ALA", 10), ("GLY", 11)))
    _pdb(tmp_path / "2AAA.pdb", (("VAL", 20), ("GLY", 21)))
    candidates = tmp_path / "candidates.csv"
    _candidates(candidates)
    rejections = []

    records, counters = records_from_microminer_candidates(
        candidates,
        tmp_path,
        min_length=1,
        chain_uniprot={"1AAA_A": ["P12345"], "2AAA_A": ["Q99999"]},
        rejections=rejections,
    )

    assert records == []
    assert counters["accepted"] == 0
    assert "no shared UniProt" in rejections[0]["reason"]


def test_microminer_import_optionally_allows_terminal_overlap(tmp_path):
    _pdb(tmp_path / "1AAA.pdb", (("ALA", 9), ("ALA", 10), ("GLY", 11)))
    _pdb(tmp_path / "2AAA.pdb", (("VAL", 10), ("GLY", 11)))
    candidates = tmp_path / "candidates.csv"
    _candidates(candidates, hitPos="10")

    records, counters = records_from_microminer_candidates(
        candidates,
        tmp_path,
        min_length=1,
        allow_terminal_overlap=True,
        min_mapping_coverage=0.65,
    )

    assert counters["terminal_overlap_mapping"] == 1
    assert records[0].pair.mutation_indices == (0,)
    mapping = records[0].environment_metadata["residue_mapping"]
    assert mapping["mode"] == "terminal_overlap_crop"
    assert mapping["parent_terminal_trim"] == [1, 0]


def test_microminer_import_terminal_overlap_rejects_internal_gap(tmp_path):
    _pdb(tmp_path / "1AAA.pdb", (("ALA", 9), ("ALA", 10), ("GLY", 11), ("GLY", 12)))
    _pdb(tmp_path / "2AAA.pdb", (("ALA", 9), ("VAL", 10), ("GLY", 12)))
    candidates = tmp_path / "candidates.csv"
    _candidates(candidates, hitPos="10")
    rejections = []

    records, counters = records_from_microminer_candidates(
        candidates,
        tmp_path,
        min_length=1,
        allow_terminal_overlap=True,
        min_mapping_coverage=0.5,
        rejections=rejections,
    )

    assert records == []
    assert counters["invalid_terminal_overlap"] == 1
    assert "internal gap or reordering" in rejections[0]["reason"]
