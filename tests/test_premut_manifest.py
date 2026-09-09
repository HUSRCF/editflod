import pickle

import pytest

from scripts.build_premut_manifest import _load_cluster_map, records_from_premut_csv


def test_premut_import_requires_expected_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("wrong\nvalue\n")
    with pytest.raises(ValueError, match="Mutated_PDB"):
        records_from_premut_csv(csv_path, tmp_path, dataset_name="test")


def test_premut_import_counts_missing_structures(tmp_path):
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text("Mutated_PDB,Mutation INFO,Possible Wilds\n1abc_A,A_0_V,2def_A\n")
    records, counters = records_from_premut_csv(csv_path, tmp_path, dataset_name="test")
    assert records == []
    assert counters == {"rows": 1, "accepted": 0, "missing_parent": 0, "missing_mutant": 1, "invalid": 0}


def test_premut_import_records_rejection_details(tmp_path):
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "Mutated_PDB,Mutation INFO,Possible Wilds\n"
        "bad-token,V_1_A,1abc_A\n"
    )
    rejections: list[dict[str, object]] = []

    records, counters = records_from_premut_csv(
        csv_path,
        tmp_path,
        dataset_name="test",
        rejections=rejections,
    )

    assert records == []
    assert counters["invalid"] == 1
    assert rejections == [
        {
            "row": 2,
            "mutant": "bad-token",
            "mutation": "V_1_A",
            "parent": "1abc_A",
            "error_type": "ValueError",
            "reason": "expected PDB_CHAIN token, got 'bad-token'",
        }
    ]


def test_premut_import_strict_missing_raises(tmp_path):
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text("Mutated_PDB,Mutation INFO,Possible Wilds\n1abc_A,A_0_V,2def_A\n")
    with pytest.raises(FileNotFoundError, match="mutant PDB is missing"):
        records_from_premut_csv(csv_path, tmp_path, dataset_name="test", strict_missing=True)


def test_premut_import_max_rows_counts_processed_rows(tmp_path):
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "Mutated_PDB,Mutation INFO,Possible Wilds\n"
        "1abc_A,A_0_V,2def_A\n"
        "1abc_A,A_0_V,2def_A\n"
    )
    _, counters = records_from_premut_csv(csv_path, tmp_path, dataset_name="test", max_rows=1)
    assert counters["rows"] == 1


def test_premut_import_accepts_zero_based_mutation_index(tmp_path):
    def write_structure(path, middle_residue):
        atoms = []
        serial = 1
        for residue, name in enumerate(("ALA", middle_residue, "ALA"), start=1):
            x = float(residue * 3)
            for atom, dx, dy in (("N", -1.0, 0.5), ("CA", 0.0, 0.0), ("C", 1.0, 0.0), ("O", 1.5, 0.8)):
                atoms.append(f"ATOM  {serial:5d} {atom:>4s} {name:>3s} A{residue:4d}    {x + dx:8.3f}{dy:8.3f}{0.0:8.3f}  1.00 20.00           C\n")
                serial += 1
        path.write_text("".join(atoms) + "END\n")

    write_structure(tmp_path / "wild.pdb", "ALA")
    write_structure(tmp_path / "mutant.pdb", "TYR")
    cluster_path = tmp_path / "cluster_dict"
    with cluster_path.open("wb") as handle:
        pickle.dump({"cluster-1": ["WILD_A"]}, handle)
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text("Mutated_PDB,Mutation INFO,Possible Wilds\nmutant_A,A_1_Y,wild_A\n")
    records, counters = records_from_premut_csv(
        csv_path, tmp_path, dataset_name="test", cluster_dict=cluster_path
    )
    assert counters["accepted"] == 1
    assert records[0].pair.mutation_indices == (1,)
    assert records[0].family_id == "test_cluster_cluster-1"


def test_premut_import_falls_back_to_mutant_structure_family(tmp_path):
    def write_structure(path, middle_residue):
        atoms = []
        serial = 1
        for residue, name in enumerate(("ALA", middle_residue, "ALA"), start=1):
            x = float(residue * 3)
            for atom, dx, dy in (("N", -1.0, 0.5), ("CA", 0.0, 0.0), ("C", 1.0, 0.0), ("O", 1.5, 0.8)):
                atoms.append(f"ATOM  {serial:5d} {atom:>4s} {name:>3s} A{residue:4d}    {x + dx:8.3f}{dy:8.3f}{0.0:8.3f}  1.00 20.00           C\n")
                serial += 1
        path.write_text("".join(atoms) + "END\n")

    write_structure(tmp_path / "wild.pdb", "ALA")
    write_structure(tmp_path / "mutant.pdb", "TYR")
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text("Mutated_PDB,Mutation INFO,Possible Wilds\nmutant_A,A_1_Y,wild_A\n")

    records, _ = records_from_premut_csv(csv_path, tmp_path, dataset_name="test")

    assert records[0].family_id == "test_mutant_MUTANT_A"


def test_premut_import_uses_optional_cluster_dictionary(tmp_path):
    cluster_path = tmp_path / "cluster_dict"
    with cluster_path.open("wb") as handle:
        pickle.dump({"cluster-7": ["WILD_A"]}, handle)
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text("Mutated_PDB,Mutation INFO,Possible Wilds\nMUT_A,A_0_V,WILD_A\n")
    # Missing structures are sufficient to exercise cluster-map parsing before
    # the importer reaches the file-resolution branch.
    records, counters = records_from_premut_csv(
        csv_path, tmp_path, dataset_name="test", cluster_dict=cluster_path
    )
    assert records == []
    assert counters["missing_mutant"] == 1
    assert _load_cluster_map(cluster_path)["WILD_A"] == "cluster-7"
