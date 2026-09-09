from scripts.audit_microminer_metadata import filter_candidate_rows


def _row() -> dict[str, str]:
    return {
        "queryName": "1AAA", "queryChain": "A", "queryAA": "ALA", "queryPos": "10",
        "hitName": "2AAA", "hitChain": "B", "hitAA": "VAL", "hitPos": "10",
        "siteIdentity": "0.95", "siteBackBoneRMSD": "0.4", "siteAllAtomRMSD": "0.7",
        "nofSiteResidues": "12", "alignmentLDDT": "0.95", "fullSeqId": "0.99",
    }


def _metadata(*, hit_hetero: list[str] | None = None, hit_chains: list[str] | None = None):
    return {
        "1AAA": {
            "protein_chains": ["A"], "protein_chain_count": 1, "hetero": ["ATP", "CL"],
            "experimental_method": "X-ray", "resolution_angstrom": 1.5, "chain_uniprot": {"A": ["P1"]},
        },
        "2AAA": {
            "protein_chains": hit_chains or ["B"], "protein_chain_count": len(hit_chains or ["B"]),
            "hetero": hit_hetero or ["ATP"], "experimental_method": "X-ray",
            "resolution_angstrom": 1.8, "chain_uniprot": {"B": ["P1"]},
        },
    }


def test_metadata_filter_accepts_matching_single_chain_context_with_ignored_additive():
    selected, counters = filter_candidate_rows(
        [_row()], _metadata(), ignored_hetero=("HOH", "CL")
    )

    assert selected == [_row()]
    assert counters["selected_rows"] == 1


def test_metadata_filter_rejects_hetero_and_multiple_protein_chains():
    selected, counters = filter_candidate_rows([_row()], _metadata(hit_hetero=["ADP"]))
    assert selected == []
    assert counters["hetero_mismatch"] == 1

    selected, counters = filter_candidate_rows(
        [_row()], _metadata(hit_chains=["B", "C"])
    )
    assert selected == []
    assert counters["multiple_protein_chains"] == 1


def test_metadata_filter_rejects_different_uniprot_accessions():
    metadata = _metadata()
    metadata["2AAA"]["chain_uniprot"] = {"B": ["P2"]}

    selected, counters = filter_candidate_rows([_row()], metadata)

    assert selected == []
    assert counters["no_shared_uniprot"] == 1
