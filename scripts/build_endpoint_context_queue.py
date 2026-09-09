"""Build a response-independent review queue from matched endpoint metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ospedit.data import load_manifest, manifest_fingerprint, validate_manifest


REPORT_FORMAT = "ospedit.endpoint_context_review_queue.v1"
ENDPOINT_FORMAT = "ospedit.rcsb_endpoint_environment_audit.v1"
ANNOTATION_FORMAT = "ospedit.endpoint_mutation_annotation_audit.v1"
VALID_TIERS = ("citation_crystal_compatible", "citation_growth_compatible")


def build_endpoint_context_queue(
    manifest: str | Path,
    endpoint_report: str | Path,
    *,
    tier: str = "citation_crystal_compatible",
    annotation_report: str | Path | None = None,
) -> dict[str, Any]:
    if tier not in VALID_TIERS:
        raise ValueError(f"tier must be one of {VALID_TIERS}")
    records = load_manifest(manifest)
    errors = validate_manifest(records)
    if errors:
        raise ValueError("manifest audit failed: " + "; ".join(errors))
    fingerprint = manifest_fingerprint(records)
    endpoint_path = Path(endpoint_report).resolve()
    endpoint = json.loads(endpoint_path.read_text())
    if endpoint.get("format") != ENDPOINT_FORMAT:
        raise ValueError(f"endpoint report must use {ENDPOINT_FORMAT}")
    if endpoint.get("manifest_fingerprint") != fingerprint:
        raise ValueError("endpoint report uses a different manifest")
    annotation_path = Path(annotation_report).resolve() if annotation_report else None
    annotations: dict[str, dict[str, Any]] = {}
    if annotation_path is not None:
        annotation_payload = json.loads(annotation_path.read_text())
        if annotation_payload.get("format") != ANNOTATION_FORMAT:
            raise ValueError(f"annotation report must use {ANNOTATION_FORMAT}")
        if annotation_payload.get("manifest_fingerprint") != fingerprint:
            raise ValueError("annotation report uses a different manifest")
        annotations = {
            row["pair_id"]: row for row in annotation_payload.get("records", [])
        }
    by_pair = {record.pair.pair_id: record for record in records}
    metadata = endpoint.get("metadata", {})

    def entry_review(pdb_id: str) -> dict[str, Any] | None:
        entry = metadata.get(pdb_id)
        if entry is None:
            return None
        return {
            "title": entry.get("title"),
            "primary_citation": entry.get("primary_citation"),
            "experimental_method": entry.get("experimental_method"),
            "resolution_angstrom": entry.get("resolution_angstrom"),
            "space_group": entry.get("space_group"),
            "cell": entry.get("cell"),
            "crystal_growth": entry.get("crystal_growth"),
        }

    queue = []
    for row in endpoint["records"]:
        if not row.get(tier):
            continue
        record = by_pair[row["pair_id"]]
        annotation = annotations.get(row["pair_id"])
        queue.append({
            "pair_id": row["pair_id"],
            "parent_id": row["parent_id"],
            "family_id": row["family_id"],
            "split": row["split"],
            "frozen_test_record": row["split"] == "test",
            "mutation": [
                {
                    "index": index,
                    "source": record.pair.parent_sequence[index],
                    "target": record.pair.mutant_sequence[index],
                }
                for index in record.pair.mutation_indices
            ],
            "parent_pdb_id": row["parent_pdb_id"],
            "mutant_pdb_id": row["mutant_pdb_id"],
            "parent_entry": entry_review(row["parent_pdb_id"]),
            "mutant_entry": entry_review(row["mutant_pdb_id"]),
            "shared_primary_citation_ids": row["shared_primary_citation_ids"],
            "shared_assembly_signatures": row["shared_assembly_signatures"],
            "shared_uniprot": row["shared_uniprot"],
            "cell_comparison": row["cell_comparison"],
            "shared_crystal_growth_signatures": row[
                "shared_crystal_growth_signatures"
            ],
            "declared_crystal_growth_matched": row[
                "declared_crystal_growth_matched"
            ],
            "mutation_annotation_classification": (
                annotation.get("classification") if annotation else None
            ),
            "parent_engineered_mutations": (
                annotation.get("parent_engineered_mutations", []) if annotation else []
            ),
            "mutant_engineered_mutations": (
                annotation.get("mutant_engineered_mutations", []) if annotation else []
            ),
        })
    queue.sort(key=lambda row: (
        row["frozen_test_record"],
        not row["declared_crystal_growth_matched"],
        row["split"],
        row["family_id"],
        row["pair_id"],
    ))
    summary: dict[str, Any] = {
        "candidates": len(queue),
        "families": len({row["family_id"] for row in queue}),
        "declared_growth_matched": sum(
            row["declared_crystal_growth_matched"] for row in queue
        ),
        "splits": {
            split: sum(row["split"] == split for row in queue)
            for split in ("train", "dev", "test")
        },
    }
    if annotations:
        summary["mutation_annotation_classifications"] = {
            classification: sum(
                row["mutation_annotation_classification"] == classification
                for row in queue
            )
            for classification in sorted({
                str(row["mutation_annotation_classification"])
                for row in queue
                if row["mutation_annotation_classification"] is not None
            })
        }
    return {
        "format": REPORT_FORMAT,
        "manifest": str(Path(manifest).resolve()),
        "manifest_fingerprint": fingerprint,
        "endpoint_report": str(endpoint_path),
        "annotation_report": str(annotation_path) if annotation_path else None,
        "usage": "manual_context_review_without_response_magnitude_selection",
        "configuration": {
            "selection_tier": tier,
            "mutation_indexing": "zero_based_internal_sequence_index",
            "observed_structure_response_used_for_selection": False,
            "mutation_annotation_used_for_selection": False,
            "test_records_frozen_for_final_evaluation": True,
        },
        "limitations": [
            "shared_primary_citation_does_not_establish_identical_sample_state",
            "declared_growth_metadata_may_be_incomplete_or_coarsely_normalized",
            "test_records_must_not_be_used_for_model_or_threshold_selection",
            "manual_primary_source_review_is_required_before_response_attribution",
            "seqadv_annotations_are_diagnostic_and_not_a_hard_filter",
        ],
        "summary": summary,
        "records": queue,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("endpoint_report")
    parser.add_argument("output")
    parser.add_argument("--tier", choices=VALID_TIERS, default=VALID_TIERS[0])
    parser.add_argument("--annotation-report")
    args = parser.parse_args()
    try:
        report = build_endpoint_context_queue(
            args.manifest,
            args.endpoint_report,
            tier=args.tier,
            annotation_report=args.annotation_report,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        f"endpoint context queue written: {destination} "
        f"({report['summary']['candidates']} candidates)"
    )


if __name__ == "__main__":
    main()
