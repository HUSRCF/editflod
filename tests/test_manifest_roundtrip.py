import numpy as np
import pytest
import sys

from ospedit.data import PairRecord, StructurePair, append_manifest, file_sha256, load_manifest, verify_record_checksums, write_manifest


def test_pair_record_roundtrip(tmp_path):
    coords = np.zeros((3, 1, 3))
    pair = StructurePair("roundtrip", "AAA", "AYA", coords, coords, (1,))
    record = PairRecord(pair, "parent", "family", "dev", residue_map=(("A", 1, ""), ("A", 2, ""), ("A", 3, "")))
    path = tmp_path / "nested" / "pairs.jsonl"
    write_manifest([record], path)
    loaded = load_manifest(path)
    assert loaded[0].pair.parent_sequence == "AAA"
    assert loaded[0].pair.mutation_indices == (1,)
    assert loaded[0].residue_map == (("A", 1, ""), ("A", 2, ""), ("A", 3, ""))


def test_append_manifest_preserves_existing_records(tmp_path):
    coords = np.zeros((2, 1, 3))
    path = tmp_path / "pairs.jsonl"
    first = PairRecord(StructurePair("first", "AA", "AY", coords, coords, (1,)), "p1", "f1", "dev")
    second = PairRecord(StructurePair("second", "AA", "AG", coords, coords, (1,)), "p2", "f2", "dev")
    write_manifest([first], path)
    append_manifest([second], path)
    assert [record.pair.pair_id for record in load_manifest(path)] == ["first", "second"]


def test_manifest_fingerprint_ignores_machine_local_paths(tmp_path):
    from ospedit.data import manifest_fingerprint

    coords = np.zeros((2, 1, 3))
    pair = StructurePair("portable", "AA", "AY", coords, coords, (1,))
    first = PairRecord(pair, "parent", "family", "dev", source_file="/machine-a/parent.pdb", target_file="/machine-a/mutant.pdb")
    second = PairRecord(pair, "parent", "family", "dev", source_file="/machine-b/parent.pdb", target_file="/machine-b/mutant.pdb")
    assert manifest_fingerprint([first]) == manifest_fingerprint([second])


def test_records_from_csv_resolves_structure_paths_and_metadata(tmp_path):
    from ospedit.manifest_cli import records_from_csv

    parent = tmp_path / "parent.pdb"
    mutant = tmp_path / "mutant.pdb"
    pdb = """ATOM      1  N   ALA A   1       0.000   1.000   0.000  1.00 20.00           N  \nATOM      2  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C  \nATOM      3  C   ALA A   1       1.460   0.000   0.000  1.00 20.00           C  \nATOM      4  O   ALA A   1       1.460  -1.240   0.000  1.00 20.00           O  \nTER\nEND\n"""
    parent.write_text(pdb)
    mutant.write_text(pdb.replace("ALA", "TYR"))
    table = tmp_path / "pairs.csv"
    table.write_text("pair_id,parent_structure,mutant_structure,parent_id,family_id,split\np1,parent.pdb,mutant.pdb,parent,family,dev\n")
    records = records_from_csv(table)
    assert len(records) == 1
    assert records[0].source_file == str(parent.resolve())
    assert records[0].pair.mutant_sequence == "Y"


def test_manifest_cli_validates_before_replacing_output(tmp_path, monkeypatch):
    from ospedit.manifest_cli import main

    csv_path = tmp_path / "pairs.csv"
    parent = tmp_path / "parent.pdb"
    mutant = tmp_path / "mutant.pdb"
    pdb = """ATOM      1  N   ALA A   1       0.000   1.000   0.000  1.00 20.00           N  \nATOM      2  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C  \nATOM      3  C   ALA A   1       1.460   0.000   0.000  1.00 20.00           C  \nATOM      4  O   ALA A   1       1.460  -1.240   0.000  1.00 20.00           O  \nTER\nEND\n"""
    parent.write_text(pdb)
    mutant.write_text(pdb.replace("ALA", "TYR"))
    csv_path.write_text(
        "pair_id,parent_structure,mutant_structure,parent_id,family_id,split\n"
        "duplicate,parent.pdb,mutant.pdb,parent,family,dev\n"
        "duplicate,parent.pdb,mutant.pdb,parent,family,dev\n"
    )
    output = tmp_path / "manifest.jsonl"
    sentinel = "sentinel\n"
    output.write_text(sentinel)
    monkeypatch.setattr(sys, "argv", ["ospedit-build-manifest", "--pairs-csv", str(csv_path), "--output", str(output)])
    with pytest.raises(SystemExit, match="duplicate pair_id"):
        main()
    assert output.read_text() == sentinel


def test_missing_coordinates_use_strict_json_null(tmp_path):
    coords = np.zeros((2, 1, 3))
    coords[1, 0, 0] = np.nan
    pair = StructurePair("missing", "AA", "AY", coords, coords.copy(), (1,))
    path = tmp_path / "missing.jsonl"
    write_manifest([PairRecord(pair, "parent", "family", "dev")], path)
    text = path.read_text()
    assert "NaN" not in text
    assert "null" in text
    loaded = load_manifest(path)[0]
    assert np.isnan(loaded.pair.parent_coords[1, 0, 0])


def test_record_checksum_detects_source_tampering(tmp_path):
    source = tmp_path / "source.pdb"
    target = tmp_path / "target.pdb"
    source.write_text("source-v1")
    target.write_text("target-v1")
    pair = StructurePair("checksummed", "AA", "AY", np.zeros((2, 1, 3)), np.zeros((2, 1, 3)), (1,))
    record = PairRecord(
        pair,
        "parent",
        "family",
        "dev",
        source_file=str(source),
        target_file=str(target),
        source_checksum=file_sha256(source),
        target_checksum=file_sha256(target),
    )
    assert verify_record_checksums(record) == []
    source.write_text("source-v2")
    assert any("source checksum mismatch" in error for error in verify_record_checksums(record))


def test_pair_record_rejects_inconsistent_metadata():
    coords = np.zeros((2, 1, 3))
    pair = StructurePair("invalid", "AA", "AY", coords, coords.copy(), (1,))
    for kwargs, message in (
        ({"family_id": "", "split": "dev"}, "family_id"),
        ({"family_id": "family", "split": "holdout"}, "split"),
        ({"family_id": "family", "split": "dev", "residue_map": (("A", 1, ""),)}, "residue_map"),
        ({"family_id": "family", "split": "dev", "source_checksum": "bad"}, "checksum"),
    ):
        try:
            PairRecord(pair, "parent", **kwargs)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"expected validation error for {message}")
