from pathlib import Path

import pytest

from scripts.select_microminer_candidates import COLUMNS
from scripts.select_microminer_repeat_groups import select_repeat_groups


def _row(query: str, hit: str, *, position: str, rmsd: str) -> list[str]:
    values = [
        query, "A", "ALA", position, hit, "B", "GLY", position,
        "0.95", rmsd, "0.6", "12", "0.95", "0.99",
    ]
    return [*values, values[-1]]


def _write(path: Path, rows: list[list[str]]) -> None:
    path.write_text("\t".join(COLUMNS) + "\n" + "\n".join("\t".join(row) for row in rows) + "\n")


def test_repeat_group_selector_keeps_multiple_unique_parents_per_target(tmp_path):
    source = tmp_path / "micro.tsv"
    _write(source, [
        _row("1AAA", "9AAA", position="10", rmsd="0.1"),
        _row("2AAA", "9AAA", position="10", rmsd="0.2"),
        _row("3AAA", "9AAA", position="10", rmsd="0.3"),
        _row("4AAA", "8AAA", position="20", rmsd="0.7"),
        _row("5AAA", "8AAA", position="20", rmsd="0.8"),
        _row("6AAA", "8AAA", position="20", rmsd="0.9"),
    ])

    rows, report = select_repeat_groups(
        source,
        max_groups=2,
        parent_candidates_per_group=3,
        min_group_rows=3,
        seed=4,
    )

    assert len(rows) == 6
    assert report["selected_groups"] == 2
    assert report["selected_groups_by_stratum"] == [0, 0, 1, 1]
    assert report["configuration"]["eligible_for_unbiased_test"] is False
    assert {row["hitName"] for row in rows} == {"8AAA", "9AAA"}


def test_repeat_group_selector_drops_groups_below_minimum(tmp_path):
    source = tmp_path / "micro.tsv"
    _write(source, [
        _row("1AAA", "9AAA", position="10", rmsd="0.2"),
        _row("2AAA", "9AAA", position="10", rmsd="0.2"),
    ])

    rows, report = select_repeat_groups(
        source,
        max_groups=1,
        parent_candidates_per_group=3,
        min_group_rows=3,
    )

    assert rows == []
    assert report["eligible_groups"] == 0


def test_repeat_group_selector_can_prioritize_largest_groups(tmp_path):
    source = tmp_path / "micro.tsv"
    rows = [
        _row(f"{index}AAA", "9AAA", position="10", rmsd="0.2")
        for index in range(1, 6)
    ] + [
        _row(f"{index}BBB", "8AAA", position="20", rmsd="0.2")
        for index in range(1, 4)
    ]
    _write(source, rows)

    selected, report = select_repeat_groups(
        source,
        max_groups=1,
        parent_candidates_per_group=5,
        min_group_rows=3,
        group_selection="largest",
    )

    assert len(selected) == 5
    assert {row["hitName"] for row in selected} == {"9AAA"}
    assert report["configuration"]["group_selection"] == "largest"


@pytest.mark.parametrize("cutoffs", [(0.2, 0.2), (-0.1,), (float("inf"),)])
def test_repeat_group_selector_rejects_invalid_cutoffs(tmp_path, cutoffs):
    source = tmp_path / "micro.tsv"
    _write(source, [_row("1AAA", "9AAA", position="10", rmsd="0.2")])

    with pytest.raises(ValueError, match="RMSD cutoffs"):
        select_repeat_groups(
            source,
            max_groups=1,
            parent_candidates_per_group=3,
            group_rmsd_cutoffs=cutoffs,
        )
