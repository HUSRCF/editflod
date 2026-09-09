from scripts.select_premut_repeat_candidates import select_repeat_candidates


def test_select_repeat_candidates_requires_parent_plus_repeats(tmp_path):
    source = tmp_path / "premut.csv"
    source.write_text(
        "Mutated_PDB,Mutation INFO,Possible Wilds\n"
        "1abc_A,A_1_V,2def_A\n"
        "1abc_A,A_1_V,3ghi_B\n"
        "1abc_A,A_1_V,4jkl_A\n"
        "5mno_A,G_2_D,6pqr_A\n"
        "5mno_A,G_2_D,7stu_A\n"
    )

    report = select_repeat_candidates(
        source,
        dataset_name="test",
        pdb_root=tmp_path / "pdbs",
        max_candidates=10,
        repeat_structures=2,
    )

    assert len(report["candidates"]) == 1
    assert report["candidates"][0]["pair_id"] == "test_2DEF_1ABC_A_A_1_V"
    assert [row["token"] for row in report["candidates"][0]["repeats"]] == ["3GHI_B", "4JKL_A"]
    assert report["structure_ids"] == ["1ABC", "2DEF", "3GHI", "4JKL"]


def test_select_repeat_candidates_excludes_used_mutants(tmp_path):
    source = tmp_path / "premut.csv"
    source.write_text(
        "Mutated_PDB,Mutation INFO,Possible Wilds\n"
        "1abc_A,A_1_V,2def_A\n"
        "1abc_A,A_1_V,3ghi_A\n"
        "1abc_A,A_1_V,4jkl_A\n"
    )

    report = select_repeat_candidates(
        source,
        dataset_name="test",
        pdb_root=tmp_path,
        max_candidates=10,
        excluded_mutants=["1ABC_A"],
    )

    assert report["candidates"] == []


def test_select_repeat_candidates_deduplicates_primary_parent(tmp_path):
    source = tmp_path / "premut.csv"
    source.write_text(
        "Mutated_PDB,Mutation INFO,Possible Wilds\n"
        "1abc_A,A_1_V,2def_A\n1abc_A,A_1_V,3ghi_A\n1abc_A,A_1_V,4jkl_A\n"
        "5mno_A,G_2_D,2def_A\n5mno_A,G_2_D,3ghi_A\n5mno_A,G_2_D,4jkl_A\n"
    )

    report = select_repeat_candidates(
        source,
        dataset_name="test",
        pdb_root=tmp_path,
        max_candidates=10,
    )

    assert [row["mutant_token"] for row in report["candidates"]] == ["1ABC_A"]


def test_select_repeat_candidates_can_keep_multiple_mutations_per_parent(tmp_path):
    source = tmp_path / "premut.csv"
    source.write_text(
        "Mutated_PDB,Mutation INFO,Possible Wilds\n"
        "1abc_A,A_1_V,2def_A\n1abc_A,A_1_V,3ghi_A\n1abc_A,A_1_V,4jkl_A\n"
        "5mno_A,G_2_D,2def_A\n5mno_A,G_2_D,3ghi_A\n5mno_A,G_2_D,4jkl_A\n"
    )

    report = select_repeat_candidates(
        source,
        dataset_name="test",
        pdb_root=tmp_path,
        max_candidates=10,
        unique_parents=False,
    )

    assert [row["mutant_token"] for row in report["candidates"]] == ["1ABC_A", "5MNO_A"]


def test_select_repeat_candidates_filters_large_mutation_index(tmp_path):
    source = tmp_path / "premut.csv"
    source.write_text(
        "Mutated_PDB,Mutation INFO,Possible Wilds\n"
        "1abc_A,A_300_V,2def_A\n1abc_A,A_300_V,3ghi_A\n1abc_A,A_300_V,4jkl_A\n"
    )

    report = select_repeat_candidates(
        source,
        dataset_name="test",
        pdb_root=tmp_path,
        max_candidates=10,
        max_mutation_index=255,
    )

    assert report["candidates"] == []
