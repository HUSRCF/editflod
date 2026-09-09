"""Build a target-context allowlist for MicroMiner repeat-group discovery."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError

from scripts.audit_microminer_metadata import fetch_rcsb_metadata
from scripts.select_microminer_candidates import COLUMNS


GROUP_COLUMNS = ("hitName", "hitChain", "hitAA", "hitPos", "queryAA")


def _group_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row["hitName"].strip().upper(),
        row["hitChain"].strip(),
        row["hitAA"].strip().upper(),
        row["hitPos"].strip(),
        row["queryAA"].strip().upper(),
    )


def filter_target_groups(
    rows: Iterable[dict[str, str]],
    metadata: dict[str, dict[str, Any]],
    *,
    require_single_protein_chain: bool = True,
    require_uniprot: bool = True,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Deduplicate mutation groups and retain targets with usable context metadata."""
    groups = {_group_key(row) for row in rows}
    selected = []
    counters: Counter[str] = Counter(candidate_groups=len(groups))
    for key in sorted(groups):
        pdb_id, chain, target_aa, position, source_aa = key
        entry = metadata.get(pdb_id)
        if entry is None:
            counters["missing_metadata"] += 1
            continue
        if chain not in entry["protein_chains"]:
            counters["target_chain_not_protein"] += 1
            continue
        if require_single_protein_chain and entry["protein_chain_count"] != 1:
            counters["multiple_protein_chains"] += 1
            continue
        accessions = entry["chain_uniprot"].get(chain, [])
        if require_uniprot and not accessions:
            counters["missing_target_uniprot"] += 1
            continue
        selected.append({
            "hitName": pdb_id,
            "hitChain": chain,
            "hitAA": target_aa,
            "hitPos": position,
            "queryAA": source_aa,
        })
    counters["selected_groups"] = len(selected)
    return selected, dict(sorted(counters.items()))


def _read_candidates(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("candidate CSV header does not match the MicroMiner selection schema")
        return list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MicroMiner repeat-group targets")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--metadata-cache")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--allow-multiple-protein-chains", action="store_true")
    parser.add_argument("--allow-missing-uniprot", action="store_true")
    args = parser.parse_args()
    try:
        rows = _read_candidates(args.candidates)
        target_ids = {row["hitName"].strip().upper() for row in rows}
        existing = (
            json.loads(Path(args.metadata_cache).read_text())
            if args.metadata_cache and Path(args.metadata_cache).is_file()
            else {}
        )
        metadata = fetch_rcsb_metadata(
            target_ids,
            batch_size=args.batch_size,
            timeout=args.timeout,
            retries=args.retries,
            existing=existing,
            cache_path=args.metadata_cache,
        )
        selected, counters = filter_target_groups(
            rows,
            metadata,
            require_single_protein_chain=not args.allow_multiple_protein_chains,
            require_uniprot=not args.allow_missing_uniprot,
        )
    except (HTTPError, URLError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GROUP_COLUMNS)
        writer.writeheader()
        writer.writerows(selected)
    report = {
        "format": "ospedit.microminer_group_target_audit.v1",
        "candidates": str(Path(args.candidates).expanduser().resolve()),
        "configuration": {
            "require_single_protein_chain": not args.allow_multiple_protein_chains,
            "require_uniprot": not args.allow_missing_uniprot,
            "network_retries": args.retries,
        },
        "target_entry_ids": len(target_ids),
        "metadata_entries": len(metadata),
        "counters": counters,
        "selected_groups": len(selected),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"MicroMiner target allowlist written: {destination} ({len(selected)} groups)")


if __name__ == "__main__":
    main()
