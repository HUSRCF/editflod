from scripts.audit_microminer_group_targets import filter_target_groups


def _row(hit: str = "1AAA", chain: str = "A") -> dict[str, str]:
    return {
        "queryName": "2AAA", "queryChain": "A", "queryAA": "ALA", "queryPos": "10",
        "hitName": hit, "hitChain": chain, "hitAA": "VAL", "hitPos": "10",
    }


def _metadata() -> dict:
    return {
        "1AAA": {
            "protein_chains": ["A"],
            "protein_chain_count": 1,
            "chain_uniprot": {"A": ["P1"]},
        },
        "3AAA": {
            "protein_chains": ["B", "C"],
            "protein_chain_count": 2,
            "chain_uniprot": {"B": ["P2"], "C": ["P2"]},
        },
    }


def test_target_audit_deduplicates_and_accepts_single_chain_uniprot_target():
    selected, counters = filter_target_groups([_row(), _row()], _metadata())

    assert selected == [{
        "hitName": "1AAA", "hitChain": "A", "hitAA": "VAL",
        "hitPos": "10", "queryAA": "ALA",
    }]
    assert counters["candidate_groups"] == 1
    assert counters["selected_groups"] == 1


def test_target_audit_rejects_multichain_and_missing_metadata():
    selected, counters = filter_target_groups(
        [_row("3AAA", "B"), _row("4AAA", "A")], _metadata()
    )

    assert selected == []
    assert counters["multiple_protein_chains"] == 1
    assert counters["missing_metadata"] == 1
