from pathlib import Path

import pytest

from scripts.select_microminer_candidates import COLUMNS, select_microminer_candidates


def _write_tsv(path: Path, rows: list[list[str]]) -> None:
    path.write_text("\t".join(COLUMNS) + "\n" + "\n".join("\t".join(row) for row in rows) + "\n")


def _row(
    query: str,
    hit: str,
    *,
    full_identity: str = "0.99",
    backbone_rmsd: str = "0.4",
) -> list[str]:
    values = [
        query, "A", "ALA", "10", hit, "A", "GLY", "10", "0.95", backbone_rmsd,
        "0.6", "12", "0.95", full_identity,
    ]
    return [*values, full_identity]


def test_selector_validates_duplicate_column_and_is_deterministic(tmp_path):
    source = tmp_path / "micro.tsv"
    _write_tsv(source, [_row("1AAA", "2AAA"), _row("3AAA", "4AAA"), _row("5AAA", "6AAA")])

    first, report = select_microminer_candidates(source, max_candidates=2, seed=9)
    second, _ = select_microminer_candidates(source, max_candidates=2, seed=9)

    assert first == second
    assert len(first) == 2
    assert report["counters"]["duplicated_terminal_column_rows"] == 3
    assert report["configuration"]["uses_structural_rmsd_for_ranking"] is False


def test_selector_applies_quality_and_unique_query_filters(tmp_path):
    source = tmp_path / "micro.tsv"
    _write_tsv(
        source,
        [
            _row("1AAA", "2AAA"),
            _row("1AAA", "3AAA"),
            _row("4AAA", "5AAA", full_identity="0.5"),
        ],
    )

    rows, report = select_microminer_candidates(source, max_candidates=3, seed=0)

    assert len(rows) == 1
    assert report["counters"]["below_quality_threshold"] == 1


def test_selector_rejects_unequal_unlabelled_terminal_column(tmp_path):
    source = tmp_path / "micro.tsv"
    row = _row("1AAA", "2AAA")
    row[-1] = "0.98"
    _write_tsv(source, [row])

    with pytest.raises(ValueError, match="unequal terminal values"):
        select_microminer_candidates(source, max_candidates=1)


def test_selector_balances_explicit_discovery_rmsd_strata(tmp_path):
    source = tmp_path / "micro.tsv"
    rows = [
        _row(f"{index}AAA", f"{index}BBB", backbone_rmsd=rmsd)
        for index, rmsd in enumerate(("0.05", "0.10", "0.20", "0.25", "0.50", "0.80"), 1)
    ]
    _write_tsv(source, rows)

    selected, report = select_microminer_candidates(
        source,
        max_candidates=6,
        candidate_pool_multiplier=1,
        discovery_site_backbone_rmsd_cutoffs=(0.15, 0.4),
    )

    assert len(selected) == 6
    assert report["strata"]["selected_rows"] == [2, 2, 2]
    assert report["configuration"]["uses_structural_rmsd_for_ranking"] is True
    assert report["configuration"]["intended_use"] == "discovery_only"
    assert report["configuration"]["eligible_for_unbiased_test"] is False


@pytest.mark.parametrize("cutoffs", [(0.2, 0.2), (-0.1,), (float("inf"),)])
def test_selector_rejects_invalid_discovery_cutoffs(tmp_path, cutoffs):
    source = tmp_path / "micro.tsv"
    _write_tsv(source, [_row("1AAA", "2AAA")])

    with pytest.raises(ValueError, match="discovery RMSD cutoffs"):
        select_microminer_candidates(
            source,
            max_candidates=1,
            discovery_site_backbone_rmsd_cutoffs=cutoffs,
        )
