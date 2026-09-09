"""Audit parent-mutant endpoint compatibility with RCSB assembly metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError

from ospedit.data import PairRecord, load_manifest, manifest_fingerprint, validate_manifest
from ospedit.rcsb_metadata import (
    cell_compatible,
    fetch_rcsb_environment_metadata,
    load_metadata_cache,
    pdb_id_from_path,
    target_profile,
)


REPORT_FORMAT = "ospedit.rcsb_endpoint_environment_audit.v1"


def _citation_ids(entry: dict[str, Any] | None) -> list[str]:
    citation = (entry or {}).get("primary_citation") or {}
    values = []
    if citation.get("pdbx_database_id_DOI"):
        values.append("DOI:" + str(citation["pdbx_database_id_DOI"]).lower())
    if citation.get("pdbx_database_id_PubMed"):
        values.append("PMID:" + str(citation["pdbx_database_id_PubMed"]))
    return values


def _growth_signatures(entry: dict[str, Any] | None) -> list[list[Any]]:
    signatures = {
        (growth.get("method"), growth.get("pH"), growth.get("temp"))
        for growth in (entry or {}).get("crystal_growth") or []
        if any(growth.get(name) is not None for name in ("method", "pH", "temp"))
    }
    return [list(value) for value in sorted(signatures, key=str)]


def audit_rcsb_endpoints(
    records: Iterable[PairRecord],
    metadata: dict[str, dict[str, Any]],
    *,
    relative_cell_length_tolerance: float = 0.1,
    cell_angle_tolerance: float = 5.0,
) -> dict[str, Any]:
    rows = list(records)
    errors = validate_manifest(rows)
    if errors:
        raise ValueError("manifest audit failed: " + "; ".join(errors))
    if relative_cell_length_tolerance < 0 or cell_angle_tolerance < 0:
        raise ValueError("cell tolerances must be non-negative")
    results = []
    used_entries: set[str] = set()
    for record in rows:
        if not all((record.source_file, record.source_chain, record.target_file, record.target_chain)):
            raise ValueError(f"missing structure provenance for {record.pair.pair_id}")
        parent_id = pdb_id_from_path(record.source_file)
        mutant_id = pdb_id_from_path(record.target_file)
        used_entries.update((parent_id, mutant_id))
        parent_entry, mutant_entry = metadata.get(parent_id), metadata.get(mutant_id)
        reasons = []
        if parent_entry is None or mutant_entry is None:
            reasons.append("missing_rcsb_entry_metadata")
        parent_profile = (
            target_profile(parent_entry, record.source_chain) if parent_entry is not None else None
        )
        mutant_profile = (
            target_profile(mutant_entry, record.target_chain) if mutant_entry is not None else None
        )
        if parent_profile is None or mutant_profile is None:
            reasons.append("target_author_chain_not_mapped")
        parent_signatures = {
            tuple(value) for value in (parent_profile or {}).get("assembly_signatures", [])
        }
        mutant_signatures = {
            tuple(value) for value in (mutant_profile or {}).get("assembly_signatures", [])
        }
        shared_signatures = sorted(parent_signatures & mutant_signatures, key=str)
        if not parent_signatures or not mutant_signatures:
            reasons.append("target_biological_assembly_missing")
        elif not shared_signatures:
            reasons.append("biological_assembly_signature_mismatch")
        if parent_entry is not None and mutant_entry is not None:
            if parent_entry.get("experimental_method") != mutant_entry.get("experimental_method"):
                reasons.append("experimental_method_mismatch")
        parent_lengths = (parent_profile or {}).get("sample_sequence_lengths", [])
        mutant_lengths = (mutant_profile or {}).get("sample_sequence_lengths", [])
        if parent_lengths and mutant_lengths and parent_lengths != mutant_lengths:
            reasons.append("deposited_construct_length_mismatch")
        assembly_compatible = not reasons

        parent_citations = set(_citation_ids(parent_entry))
        mutant_citations = set(_citation_ids(mutant_entry))
        shared_citations = sorted(parent_citations & mutant_citations)
        parent_growth = _growth_signatures(parent_entry)
        mutant_growth = _growth_signatures(mutant_entry)
        shared_growth = sorted({tuple(value) for value in parent_growth} & {
            tuple(value) for value in mutant_growth
        }, key=str)

        crystal_reasons = []
        cell_comparison = None
        if assembly_compatible:
            assert parent_entry is not None and mutant_entry is not None
            parent_space = parent_entry.get("space_group")
            mutant_space = mutant_entry.get("space_group")
            if not parent_space or not mutant_space:
                crystal_reasons.append("missing_space_group")
            elif parent_space != mutant_space:
                crystal_reasons.append("space_group_mismatch")
            cells_match, cell_comparison = cell_compatible(
                parent_entry,
                mutant_entry,
                relative_length_tolerance=relative_cell_length_tolerance,
                angle_tolerance=cell_angle_tolerance,
            )
            if cell_comparison is None:
                crystal_reasons.append("missing_cell_metadata")
            elif not cells_match:
                crystal_reasons.append("unit_cell_mismatch")
        else:
            crystal_reasons.append("assembly_incompatible")
        results.append({
            "pair_id": record.pair.pair_id,
            "parent_id": record.parent_id,
            "family_id": record.family_id,
            "split": record.split,
            "parent_pdb_id": parent_id,
            "mutant_pdb_id": mutant_id,
            "parent_profile": parent_profile,
            "mutant_profile": mutant_profile,
            "shared_assembly_signatures": [list(value) for value in shared_signatures],
            "shared_uniprot": sorted(
                set((parent_profile or {}).get("uniprot_accessions", []))
                & set((mutant_profile or {}).get("uniprot_accessions", []))
            ),
            "assembly_compatible": assembly_compatible,
            "assembly_rejection_reasons": reasons,
            "crystal_form_compatible": assembly_compatible and not crystal_reasons,
            "crystal_rejection_reasons": crystal_reasons,
            "cell_comparison": cell_comparison,
            "shared_primary_citation_ids": shared_citations,
            "primary_citation_matched": bool(shared_citations),
            "parent_crystal_growth_signatures": parent_growth,
            "mutant_crystal_growth_signatures": mutant_growth,
            "shared_crystal_growth_signatures": [list(value) for value in shared_growth],
            "declared_crystal_growth_matched": bool(shared_growth),
            "citation_crystal_compatible": (
                assembly_compatible and not crystal_reasons and bool(shared_citations)
            ),
            "citation_growth_compatible": (
                assembly_compatible
                and not crystal_reasons
                and bool(shared_citations)
                and bool(shared_growth)
            ),
        })

    def summary(field: str) -> dict[str, Any]:
        selected = [row for row in results if row[field]]
        return {
            "pairs": len(selected),
            "families": len({row["family_id"] for row in selected}),
            "splits": {
                split: {
                    "pairs": sum(row[field] and row["split"] == split for row in results),
                    "families": len({
                        row["family_id"] for row in results if row[field] and row["split"] == split
                    }),
                }
                for split in ("train", "dev", "test")
            },
        }
    return {
        "format": REPORT_FORMAT,
        "manifest_fingerprint": manifest_fingerprint(rows),
        "metadata_format": "ospedit.rcsb_environment_metadata.v2",
        "configuration": {
            "relative_cell_length_tolerance": relative_cell_length_tolerance,
            "cell_angle_tolerance_degree": cell_angle_tolerance,
        },
        "selection_semantics": {
            "assembly": "mapped_target_chain_shared_assembly_signature_method_and_construct_length",
            "crystal_form": "assembly_plus_space_group_and_unit_cell_tolerance",
            "citation_crystal": "crystal_form_plus_shared_primary_doi_or_pubmed_id",
            "citation_growth": "citation_crystal_plus_exact_declared_growth_signature",
        },
        "limitations": [
            "assembly_signature_matches_composition_not_interface_geometry",
            "crystal_form_match_does_not_establish_identical_crystallization_conditions",
            "manual_review_is_required_before_mutation_response_attribution",
        ],
        "summary": {
            "pairs": len(results),
            "entries": len(used_entries),
            "entries_with_metadata": len(used_entries & set(metadata)),
            "assembly_compatible": summary("assembly_compatible"),
            "crystal_form_compatible": summary("crystal_form_compatible"),
            "citation_crystal_compatible": summary("citation_crystal_compatible"),
            "citation_growth_compatible": summary("citation_growth_compatible"),
        },
        "metadata": {entry_id: metadata[entry_id] for entry_id in sorted(used_entries) if entry_id in metadata},
        "records": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--metadata-cache", required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--relative-cell-length-tolerance", type=float, default=0.1)
    parser.add_argument("--cell-angle-tolerance", type=float, default=5.0)
    args = parser.parse_args()
    try:
        records = load_manifest(args.manifest)
        entry_ids = {
            pdb_id_from_path(path)
            for record in records
            for path in (record.source_file, record.target_file)
            if path
        }
        metadata = load_metadata_cache(args.metadata_cache)
        if not args.offline:
            metadata = fetch_rcsb_environment_metadata(
                entry_ids,
                batch_size=args.batch_size,
                timeout=args.timeout,
                retries=args.retries,
                existing=metadata,
                cache_path=args.metadata_cache,
            )
        report = audit_rcsb_endpoints(
            records,
            metadata,
            relative_cell_length_tolerance=args.relative_cell_length_tolerance,
            cell_angle_tolerance=args.cell_angle_tolerance,
        )
    except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    summary = report["summary"]
    print(
        f"RCSB endpoint audit written: {destination}; "
        f"assembly={summary['assembly_compatible']['pairs']}/{summary['pairs']}, "
        f"crystal={summary['crystal_form_compatible']['pairs']}/{summary['pairs']}"
    )


if __name__ == "__main__":
    main()
