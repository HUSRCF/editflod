"""Stream and deterministically sample candidate mutation pairs from MicroMiner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


COLUMNS = (
    "queryName",
    "queryChain",
    "queryAA",
    "queryPos",
    "hitName",
    "hitChain",
    "hitAA",
    "hitPos",
    "siteIdentity",
    "siteBackBoneRMSD",
    "siteAllAtomRMSD",
    "nofSiteResidues",
    "alignmentLDDT",
    "fullSeqId",
)
STANDARD_AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def _stable_rank(row: dict[str, str], seed: int) -> int:
    key = "|".join(row[column] for column in COLUMNS[:8])
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest(), "big")


def _parse_row(values: list[str], line_number: int) -> tuple[dict[str, str], bool]:
    values = [value.strip() for value in values]
    duplicated_terminal = False
    if len(values) == len(COLUMNS) + 1:
        if values[-1] != values[-2]:
            raise ValueError(
                f"line {line_number}: unexpected 15-column row with unequal terminal values"
            )
        values = values[:-1]
        duplicated_terminal = True
    if len(values) != len(COLUMNS):
        raise ValueError(f"line {line_number}: expected 14 or validated 15 columns, found {len(values)}")
    return dict(zip(COLUMNS, values)), duplicated_terminal


def _finite_float(row: dict[str, str], field: str, line_number: int) -> float:
    try:
        value = float(row[field])
    except ValueError as error:
        raise ValueError(f"line {line_number}: invalid {field} value {row[field]!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"line {line_number}: non-finite {field} value")
    return value


def _validate_discovery_cutoffs(cutoffs: Sequence[float] | None) -> tuple[float, ...] | None:
    if cutoffs is None:
        return None
    values = tuple(float(value) for value in cutoffs)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("discovery RMSD cutoffs must be finite and non-negative")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError("discovery RMSD cutoffs must be strictly increasing")
    return values


def _rmsd_stratum(value: float, cutoffs: tuple[float, ...] | None) -> int:
    if cutoffs is None:
        return 0
    return sum(value >= cutoff for cutoff in cutoffs)


def select_microminer_candidates(
    path: str | Path,
    *,
    max_candidates: int,
    seed: int = 0,
    min_full_sequence_identity: float = 0.98,
    min_alignment_lddt: float = 0.9,
    min_site_residues: int = 8,
    candidate_pool_multiplier: int = 50,
    require_distinct_entries: bool = True,
    unique_query_chains: bool = True,
    unique_hit_chains: bool = True,
    discovery_site_backbone_rmsd_cutoffs: Sequence[float] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Select hash-ranked rows without loading the full TSV into memory."""
    if max_candidates <= 0 or min_site_residues <= 0 or candidate_pool_multiplier <= 0:
        raise ValueError("candidate counts, site residues, and pool multiplier must be positive")
    if not 0 <= min_full_sequence_identity <= 1 or not 0 <= min_alignment_lddt <= 1:
        raise ValueError("identity and lDDT thresholds must be in [0, 1]")
    cutoffs = _validate_discovery_cutoffs(discovery_site_backbone_rmsd_cutoffs)
    strata = 1 if cutoffs is None else len(cutoffs) + 1
    quota = math.ceil(max_candidates / strata)
    pool_size_per_stratum = quota * candidate_pool_multiplier
    heaps: list[list[tuple[int, int, dict[str, str]]]] = [[] for _ in range(strata)]
    eligible_by_stratum = [0] * strata
    counts = {
        "data_rows": 0,
        "duplicated_terminal_column_rows": 0,
        "eligible_rows": 0,
        "nonstandard_or_identity_mutation": 0,
        "same_entry_rows": 0,
        "below_quality_threshold": 0,
    }
    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = tuple(value.strip() for value in next(reader))
        except StopIteration as error:
            raise ValueError("MicroMiner TSV is empty") from error
        if header != COLUMNS:
            raise ValueError("MicroMiner TSV header does not match the expected release schema")
        for line_number, values in enumerate(reader, start=2):
            counts["data_rows"] += 1
            row, duplicated = _parse_row(values, line_number)
            counts["duplicated_terminal_column_rows"] += int(duplicated)
            query_aa = row["queryAA"].upper()
            hit_aa = row["hitAA"].upper()
            if query_aa not in STANDARD_AA3 or hit_aa not in STANDARD_AA3 or query_aa == hit_aa:
                counts["nonstandard_or_identity_mutation"] += 1
                continue
            if require_distinct_entries and row["queryName"].upper() == row["hitName"].upper():
                counts["same_entry_rows"] += 1
                continue
            if (
                _finite_float(row, "fullSeqId", line_number) < min_full_sequence_identity
                or _finite_float(row, "alignmentLDDT", line_number) < min_alignment_lddt
                or _finite_float(row, "nofSiteResidues", line_number) < min_site_residues
            ):
                counts["below_quality_threshold"] += 1
                continue
            counts["eligible_rows"] += 1
            stratum = _rmsd_stratum(
                _finite_float(row, "siteBackBoneRMSD", line_number), cutoffs
            )
            eligible_by_stratum[stratum] += 1
            rank = _stable_rank(row, seed)
            item = (-rank, -line_number, row)
            heap = heaps[stratum]
            if len(heap) < pool_size_per_stratum:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)

    ranked = [
        [item[2] for item in sorted(heap, key=lambda item: (-item[0], -item[1]))]
        for heap in heaps
    ]
    selected: list[dict[str, str]] = []
    selected_by_stratum = [0] * strata
    used_queries: set[tuple[str, str]] = set()
    used_hits: set[tuple[str, str]] = set()
    positions = [0] * strata
    while len(selected) < max_candidates:
        progress = False
        for stratum, rows in enumerate(ranked):
            while positions[stratum] < len(rows):
                row = rows[positions[stratum]]
                positions[stratum] += 1
                query = (row["queryName"].upper(), row["queryChain"])
                hit = (row["hitName"].upper(), row["hitChain"])
                if unique_query_chains and query in used_queries:
                    continue
                if unique_hit_chains and hit in used_hits:
                    continue
                selected.append(row)
                selected_by_stratum[stratum] += 1
                used_queries.add(query)
                used_hits.add(hit)
                progress = True
                break
            if len(selected) == max_candidates:
                break
        if not progress:
            break
    report = {
        "format": "ospedit.microminer_candidate_selection.v1",
        "source": str(source.expanduser().resolve()),
        "configuration": {
            "max_candidates": max_candidates,
            "seed": seed,
            "min_full_sequence_identity": min_full_sequence_identity,
            "min_alignment_lddt": min_alignment_lddt,
            "min_site_residues": min_site_residues,
            "candidate_pool_multiplier": candidate_pool_multiplier,
            "require_distinct_entries": require_distinct_entries,
            "unique_query_chains": unique_query_chains,
            "unique_hit_chains": unique_hit_chains,
            "selection_key": "sha256(seed,query/hit mutation tuple)",
            "uses_structural_rmsd_for_ranking": cutoffs is not None,
            "discovery_site_backbone_rmsd_cutoffs": list(cutoffs) if cutoffs else None,
            "intended_use": "discovery_only" if cutoffs is not None else "general",
            "eligible_for_unbiased_test": cutoffs is None,
        },
        "counters": counts,
        "strata": {
            "eligible_rows": eligible_by_stratum,
            "candidate_pool_rows": [len(rows) for rows in ranked],
            "selected_rows": selected_by_stratum,
        },
        "candidate_pool_rows": sum(len(rows) for rows in ranked),
        "selected_rows": len(selected),
    }
    return selected, report


def write_candidates(rows: Iterable[dict[str, str]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select MicroMiner mutation-pair candidates")
    parser.add_argument("--tsv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-full-sequence-identity", type=float, default=0.98)
    parser.add_argument("--min-alignment-lddt", type=float, default=0.9)
    parser.add_argument("--min-site-residues", type=int, default=8)
    parser.add_argument("--candidate-pool-multiplier", type=int, default=50)
    parser.add_argument("--allow-same-entry", action="store_true")
    parser.add_argument("--allow-repeated-query", action="store_true")
    parser.add_argument("--allow-repeated-hit", action="store_true")
    parser.add_argument(
        "--discovery-site-backbone-rmsd-cutoffs",
        help=(
            "comma-separated RMSD cutoffs for equal-allocation discovery strata; "
            "outputs are ineligible for an unbiased test set"
        ),
    )
    args = parser.parse_args()
    try:
        discovery_cutoffs = (
            tuple(float(value) for value in args.discovery_site_backbone_rmsd_cutoffs.split(","))
            if args.discovery_site_backbone_rmsd_cutoffs
            else None
        )
        rows, report = select_microminer_candidates(
            args.tsv,
            max_candidates=args.max_candidates,
            seed=args.seed,
            min_full_sequence_identity=args.min_full_sequence_identity,
            min_alignment_lddt=args.min_alignment_lddt,
            min_site_residues=args.min_site_residues,
            candidate_pool_multiplier=args.candidate_pool_multiplier,
            require_distinct_entries=not args.allow_same_entry,
            unique_query_chains=not args.allow_repeated_query,
            unique_hit_chains=not args.allow_repeated_hit,
            discovery_site_backbone_rmsd_cutoffs=discovery_cutoffs,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    write_candidates(rows, args.output)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"MicroMiner candidates written: {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
