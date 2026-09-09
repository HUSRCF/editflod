"""Select MicroMiner mutation groups with multiple candidate parent structures."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Any, Sequence

from scripts.select_microminer_candidates import (
    COLUMNS,
    STANDARD_AA3,
    _finite_float,
    _parse_row,
    _stable_rank,
    write_candidates,
)


GroupKey = tuple[str, str, str, str, str]


def _group_key(row: dict[str, str]) -> GroupKey:
    return (
        row["hitName"].upper(),
        row["hitChain"],
        row["hitAA"].upper(),
        row["hitPos"],
        row["queryAA"].upper(),
    )


def _group_rank(key: GroupKey, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{'|'.join(key)}".encode()).digest(), "big")


def _validate_cutoffs(cutoffs: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in cutoffs)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("RMSD cutoffs must be finite and non-negative")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError("RMSD cutoffs must be strictly increasing")
    return values


def _stratum(value: float, cutoffs: tuple[float, ...]) -> int:
    return sum(value >= cutoff for cutoff in cutoffs)


def _eligible(
    row: dict[str, str],
    line_number: int,
    *,
    min_full_sequence_identity: float,
    min_alignment_lddt: float,
    min_site_residues: int,
) -> bool:
    query_aa = row["queryAA"].upper()
    hit_aa = row["hitAA"].upper()
    return (
        query_aa in STANDARD_AA3
        and hit_aa in STANDARD_AA3
        and query_aa != hit_aa
        and row["queryName"].upper() != row["hitName"].upper()
        and _finite_float(row, "fullSeqId", line_number) >= min_full_sequence_identity
        and _finite_float(row, "alignmentLDDT", line_number) >= min_alignment_lddt
        and _finite_float(row, "nofSiteResidues", line_number) >= min_site_residues
    )


def _rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = tuple(value.strip() for value in next(reader))
        except StopIteration as error:
            raise ValueError("MicroMiner TSV is empty") from error
        if header != COLUMNS:
            raise ValueError("MicroMiner TSV header does not match the expected release schema")
        for line_number, values in enumerate(reader, start=2):
            row, duplicated = _parse_row(values, line_number)
            yield line_number, row, duplicated


def select_repeat_groups(
    path: str | Path,
    *,
    max_groups: int,
    parent_candidates_per_group: int,
    min_group_rows: int = 3,
    seed: int = 0,
    min_full_sequence_identity: float = 0.98,
    min_alignment_lddt: float = 0.9,
    min_site_residues: int = 8,
    group_rmsd_cutoffs: Sequence[float] = (0.15, 0.3, 0.6),
    row_pool_multiplier: int = 4,
    group_selection: str = "hash",
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Two-pass selection of repeated-target groups and bounded parent candidates."""
    if (
        max_groups <= 0
        or parent_candidates_per_group <= 0
        or min_group_rows < 2
        or min_site_residues <= 0
        or row_pool_multiplier <= 0
    ):
        raise ValueError("group, row, site-residue, and pool limits must be positive")
    if not 0 <= min_full_sequence_identity <= 1 or not 0 <= min_alignment_lddt <= 1:
        raise ValueError("identity and lDDT thresholds must be in [0, 1]")
    if group_selection not in {"hash", "largest"}:
        raise ValueError("group_selection must be 'hash' or 'largest'")
    cutoffs = _validate_cutoffs(group_rmsd_cutoffs)
    source = Path(path)
    counts: Counter[GroupKey] = Counter()
    max_rmsd: dict[GroupKey, float] = {}
    counters: Counter[str] = Counter()
    for line_number, row, duplicated in _rows(source):
        counters["data_rows"] += 1
        counters["duplicated_terminal_column_rows"] += int(duplicated)
        if not _eligible(
            row,
            line_number,
            min_full_sequence_identity=min_full_sequence_identity,
            min_alignment_lddt=min_alignment_lddt,
            min_site_residues=min_site_residues,
        ):
            counters["ineligible_rows"] += 1
            continue
        counters["eligible_rows"] += 1
        key = _group_key(row)
        counts[key] += 1
        max_rmsd[key] = max(
            max_rmsd.get(key, 0.0),
            _finite_float(row, "siteBackBoneRMSD", line_number),
        )

    eligible_groups = [key for key, count in counts.items() if count >= min_group_rows]
    by_stratum: list[list[GroupKey]] = [[] for _ in range(len(cutoffs) + 1)]
    for key in eligible_groups:
        by_stratum[_stratum(max_rmsd[key], cutoffs)].append(key)
    for keys in by_stratum:
        if group_selection == "largest":
            keys.sort(key=lambda key: (-counts[key], _group_rank(key, seed), key))
        else:
            keys.sort(key=lambda key: (_group_rank(key, seed), key))

    selected_groups: list[GroupKey] = []
    positions = [0] * len(by_stratum)
    while len(selected_groups) < max_groups:
        progress = False
        for stratum, keys in enumerate(by_stratum):
            if positions[stratum] >= len(keys):
                continue
            selected_groups.append(keys[positions[stratum]])
            positions[stratum] += 1
            progress = True
            if len(selected_groups) == max_groups:
                break
        if not progress:
            break

    selected_set = set(selected_groups)
    pool_size = parent_candidates_per_group * row_pool_multiplier
    heaps: dict[GroupKey, list[tuple[int, int, dict[str, str]]]] = {
        key: [] for key in selected_groups
    }
    for line_number, row, _ in _rows(source):
        key = _group_key(row)
        if key not in selected_set or not _eligible(
            row,
            line_number,
            min_full_sequence_identity=min_full_sequence_identity,
            min_alignment_lddt=min_alignment_lddt,
            min_site_residues=min_site_residues,
        ):
            continue
        item = (-_stable_rank(row, seed), -line_number, row)
        heap = heaps[key]
        if len(heap) < pool_size:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    output = []
    summaries = []
    output_groups_by_stratum = [0] * len(by_stratum)
    for key in selected_groups:
        ranked = [item[2] for item in sorted(heaps[key], key=lambda item: (-item[0], -item[1]))]
        group_rows = []
        used_queries: set[tuple[str, str]] = set()
        for row in ranked:
            query = (row["queryName"].upper(), row["queryChain"])
            if query in used_queries:
                continue
            used_queries.add(query)
            group_rows.append(row)
            if len(group_rows) == parent_candidates_per_group:
                break
        if len(group_rows) < min_group_rows:
            continue
        output.extend(group_rows)
        stratum = _stratum(max_rmsd[key], cutoffs)
        output_groups_by_stratum[stratum] += 1
        summaries.append({
            "target": {
                "pdb_id": key[0],
                "chain": key[1],
                "target_aa": key[2],
                "target_author_position": key[3],
                "source_aa": key[4],
            },
            "eligible_rows": counts[key],
            "selected_parent_candidates": len(group_rows),
            "max_site_backbone_rmsd": max_rmsd[key],
            "rmsd_stratum": stratum,
        })

    report = {
        "format": "ospedit.microminer_repeat_group_selection.v1",
        "source": str(source.expanduser().resolve()),
        "configuration": {
            "max_groups": max_groups,
            "parent_candidates_per_group": parent_candidates_per_group,
            "min_group_rows": min_group_rows,
            "seed": seed,
            "min_full_sequence_identity": min_full_sequence_identity,
            "min_alignment_lddt": min_alignment_lddt,
            "min_site_residues": min_site_residues,
            "group_rmsd_cutoffs": list(cutoffs),
            "row_pool_multiplier": row_pool_multiplier,
            "group_selection": group_selection,
            "intended_use": "repeat_first_discovery_only",
            "eligible_for_unbiased_test": False,
        },
        "counters": dict(sorted(counters.items())),
        "eligible_groups": len(eligible_groups),
        "eligible_groups_by_stratum": [len(keys) for keys in by_stratum],
        "selected_groups": len(summaries),
        "selected_groups_by_stratum": output_groups_by_stratum,
        "selected_rows": len(output),
        "groups": summaries,
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Select repeat-first MicroMiner groups")
    parser.add_argument("--tsv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-groups", type=int, default=128)
    parser.add_argument("--parent-candidates-per-group", type=int, default=12)
    parser.add_argument("--min-group-rows", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-full-sequence-identity", type=float, default=0.98)
    parser.add_argument("--min-alignment-lddt", type=float, default=0.9)
    parser.add_argument("--min-site-residues", type=int, default=8)
    parser.add_argument("--group-rmsd-cutoffs", default="0.15,0.30,0.60")
    parser.add_argument("--row-pool-multiplier", type=int, default=4)
    parser.add_argument("--group-selection", choices=("hash", "largest"), default="hash")
    args = parser.parse_args()
    try:
        rows, report = select_repeat_groups(
            args.tsv,
            max_groups=args.max_groups,
            parent_candidates_per_group=args.parent_candidates_per_group,
            min_group_rows=args.min_group_rows,
            seed=args.seed,
            min_full_sequence_identity=args.min_full_sequence_identity,
            min_alignment_lddt=args.min_alignment_lddt,
            min_site_residues=args.min_site_residues,
            group_rmsd_cutoffs=tuple(float(value) for value in args.group_rmsd_cutoffs.split(",")),
            row_pool_multiplier=args.row_pool_multiplier,
            group_selection=args.group_selection,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    write_candidates(rows, args.output)
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"MicroMiner repeat-first candidates written: {args.output} "
        f"({report['selected_groups']} groups, {len(rows)} rows)"
    )


if __name__ == "__main__":
    main()
