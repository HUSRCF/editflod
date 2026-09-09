"""Prefilter MicroMiner candidate pairs with batched RCSB entry metadata."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from scripts.select_microminer_candidates import COLUMNS, write_candidates


GRAPHQL_ENDPOINT = "https://data.rcsb.org/graphql"
ENTRY_QUERY = """
query($ids:[String!]!) {
  entries(entry_ids:$ids) {
    rcsb_id
    rcsb_entry_info { experimental_method resolution_combined }
    polymer_entities {
      entity_poly { rcsb_entity_polymer_type }
      rcsb_polymer_entity_container_identifiers {
        auth_asym_ids
        reference_sequence_identifiers { database_name database_accession }
      }
    }
    nonpolymer_entities { nonpolymer_comp { chem_comp { id } } }
  }
}
"""


def _parse_entry(raw: dict[str, Any]) -> dict[str, Any]:
    protein_chains: list[str] = []
    chain_uniprot: dict[str, list[str]] = {}
    for entity in raw.get("polymer_entities") or []:
        if (entity.get("entity_poly") or {}).get("rcsb_entity_polymer_type") != "Protein":
            continue
        identifiers = entity.get("rcsb_polymer_entity_container_identifiers") or {}
        chains = [str(value) for value in identifiers.get("auth_asym_ids") or []]
        protein_chains.extend(chains)
        accessions = sorted({
            str(item["database_accession"])
            for item in identifiers.get("reference_sequence_identifiers") or []
            if item.get("database_name") == "UniProt" and item.get("database_accession")
        })
        for chain in chains:
            chain_uniprot[chain] = accessions
    info = raw.get("rcsb_entry_info") or {}
    resolutions = [
        float(value)
        for value in info.get("resolution_combined") or []
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    hetero = sorted({
        str(((entity.get("nonpolymer_comp") or {}).get("chem_comp") or {}).get("id", "")).upper()
        for entity in raw.get("nonpolymer_entities") or []
        if ((entity.get("nonpolymer_comp") or {}).get("chem_comp") or {}).get("id")
    })
    return {
        "pdb_id": str(raw["rcsb_id"]).upper(),
        "experimental_method": info.get("experimental_method"),
        "resolution_angstrom": min(resolutions) if resolutions else None,
        "protein_chains": sorted(protein_chains),
        "protein_chain_count": len(protein_chains),
        "hetero": hetero,
        "chain_uniprot": chain_uniprot,
    }


def fetch_rcsb_metadata(
    entry_ids: Iterable[str],
    *,
    batch_size: int = 100,
    timeout: float = 30.0,
    retries: int = 2,
    existing: dict[str, dict[str, Any]] | None = None,
    cache_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch normalized RCSB entry metadata in bounded GraphQL batches."""
    if batch_size <= 0 or timeout <= 0 or retries < 0:
        raise ValueError("batch_size and timeout must be positive; retries must be non-negative")
    output = dict(existing or {})
    identifiers = sorted({
        value.strip().upper()
        for value in entry_ids
        if value.strip() and value.strip().upper() not in output
    })
    destination = Path(cache_path) if cache_path is not None else None
    for start in range(0, len(identifiers), batch_size):
        batch = identifiers[start : start + batch_size]
        request = Request(
            GRAPHQL_ENDPOINT,
            data=json.dumps({"query": ENTRY_QUERY, "variables": {"ids": batch}}).encode(),
            headers={"Content-Type": "application/json", "User-Agent": "ospedit/0.1"},
        )
        payload = None
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                with urlopen(request, timeout=timeout) as response:  # noqa: S310
                    payload = json.load(response)
                last_error = None
                break
            except (HTTPError, URLError, OSError) as error:
                last_error = error
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))
        if payload is None:
            assert last_error is not None
            raise last_error
        if payload.get("errors"):
            raise ValueError(f"RCSB GraphQL error: {payload['errors']}")
        for raw in (payload.get("data") or {}).get("entries") or []:
            parsed = _parse_entry(raw)
            output[str(parsed["pdb_id"])] = parsed
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
            temporary.replace(destination)
    return output


def filter_candidate_rows(
    rows: Iterable[dict[str, str]],
    metadata: dict[str, dict[str, Any]],
    *,
    ignored_hetero: Iterable[str] = ("HOH",),
    require_single_protein_chain: bool = True,
    require_matching_hetero: bool = True,
    require_matching_method: bool = True,
    require_shared_uniprot: bool = True,
    max_resolution_difference: float | None = None,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Apply entry-level context filters before coordinate download."""
    if max_resolution_difference is not None and max_resolution_difference < 0:
        raise ValueError("max_resolution_difference must be non-negative")
    ignored = {value.strip().upper() for value in ignored_hetero if value.strip()}
    selected = []
    counters: Counter[str] = Counter()
    for row in rows:
        counters["candidate_rows"] += 1
        query_id = row["queryName"].strip().upper()
        hit_id = row["hitName"].strip().upper()
        query = metadata.get(query_id)
        hit = metadata.get(hit_id)
        if query is None or hit is None:
            counters["missing_metadata"] += 1
            continue
        if row["queryChain"] not in query["protein_chains"] or row["hitChain"] not in hit["protein_chains"]:
            counters["target_chain_not_protein"] += 1
            continue
        query_uniprot = set(query["chain_uniprot"].get(row["queryChain"], []))
        hit_uniprot = set(hit["chain_uniprot"].get(row["hitChain"], []))
        if require_shared_uniprot and not query_uniprot.intersection(hit_uniprot):
            counters["no_shared_uniprot"] += 1
            continue
        if require_single_protein_chain and (
            query["protein_chain_count"] != 1 or hit["protein_chain_count"] != 1
        ):
            counters["multiple_protein_chains"] += 1
            continue
        query_hetero = sorted(set(query["hetero"]) - ignored)
        hit_hetero = sorted(set(hit["hetero"]) - ignored)
        if require_matching_hetero and query_hetero != hit_hetero:
            counters["hetero_mismatch"] += 1
            continue
        if require_matching_method and query["experimental_method"] != hit["experimental_method"]:
            counters["experimental_method_mismatch"] += 1
            continue
        if max_resolution_difference is not None:
            query_resolution = query["resolution_angstrom"]
            hit_resolution = hit["resolution_angstrom"]
            if (
                query_resolution is None
                or hit_resolution is None
                or abs(float(query_resolution) - float(hit_resolution)) > max_resolution_difference
            ):
                counters["resolution_gate_failed"] += 1
                continue
        selected.append(row)
    counters["selected_rows"] = len(selected)
    return selected, dict(sorted(counters.items()))


def _read_candidates(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("candidate CSV header does not match the MicroMiner selection schema")
        return list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefilter MicroMiner rows with RCSB metadata")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--metadata-cache")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--ignore-hetero", nargs="*", default=("HOH",))
    parser.add_argument("--allow-multiple-protein-chains", action="store_true")
    parser.add_argument("--allow-hetero-mismatch", action="store_true")
    parser.add_argument("--allow-method-mismatch", action="store_true")
    parser.add_argument("--allow-uniprot-mismatch", action="store_true")
    parser.add_argument("--max-resolution-difference", type=float)
    args = parser.parse_args()
    try:
        rows = _read_candidates(args.candidates)
        entry_ids = {row[field] for row in rows for field in ("queryName", "hitName")}
        existing = (
            json.loads(Path(args.metadata_cache).read_text())
            if args.metadata_cache and Path(args.metadata_cache).is_file()
            else {}
        )
        metadata = fetch_rcsb_metadata(
            entry_ids,
            batch_size=args.batch_size,
            timeout=args.timeout,
            retries=args.retries,
            existing=existing,
            cache_path=args.metadata_cache,
        )
        selected, counters = filter_candidate_rows(
            rows,
            metadata,
            ignored_hetero=args.ignore_hetero,
            require_single_protein_chain=not args.allow_multiple_protein_chains,
            require_matching_hetero=not args.allow_hetero_mismatch,
            require_matching_method=not args.allow_method_mismatch,
            require_shared_uniprot=not args.allow_uniprot_mismatch,
            max_resolution_difference=args.max_resolution_difference,
        )
    except (HTTPError, URLError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    write_candidates(selected, args.output)
    report = {
        "format": "ospedit.microminer_metadata_audit.v1",
        "candidates": str(Path(args.candidates).expanduser().resolve()),
        "configuration": {
            "ignored_hetero": sorted({value.strip().upper() for value in args.ignore_hetero}),
            "require_single_protein_chain": not args.allow_multiple_protein_chains,
            "require_matching_hetero": not args.allow_hetero_mismatch,
            "require_matching_method": not args.allow_method_mismatch,
            "require_shared_uniprot": not args.allow_uniprot_mismatch,
            "max_resolution_difference": args.max_resolution_difference,
            "network_retries": args.retries,
        },
        "metadata_entries": len(metadata),
        "counters": counters,
        "selected_rows": len(selected),
        "selected_chain_uniprot": {
            f"{row[field]}_{row[chain_field]}": metadata[row[field].upper()]["chain_uniprot"].get(
                row[chain_field], []
            )
            for row in selected
            for field, chain_field in (("queryName", "queryChain"), ("hitName", "hitChain"))
        },
    }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"MicroMiner metadata audit written: {args.output} ({len(selected)}/{len(rows)} selected)")


if __name__ == "__main__":
    main()
