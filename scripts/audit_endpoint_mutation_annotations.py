"""Compare manifest edits with engineered-mutation annotations in PDB headers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from ospedit.data import PairRecord, load_manifest, manifest_fingerprint, validate_manifest
from ospedit.pdb_annotations import parse_engineered_mutations


REPORT_FORMAT = "ospedit.endpoint_mutation_annotation_audit.v1"


def audit_endpoint_mutation_annotations(
    records: Iterable[PairRecord],
) -> dict[str, Any]:
    rows = list(records)
    errors = validate_manifest(rows)
    if errors:
        raise ValueError("manifest audit failed: " + "; ".join(errors))
    results = []
    for record in rows:
        if not record.source_file or not record.target_file or not record.source_chain:
            raise ValueError(f"missing structure provenance for {record.pair.pair_id}")
        parent_annotations = parse_engineered_mutations(record.source_file)
        mutant_annotations = parse_engineered_mutations(record.target_file)
        expected = []
        direct_support = False
        reverse_support = False
        paired_endpoint_support = False
        for index in record.pair.mutation_indices:
            if index >= len(record.residue_map):
                raise ValueError(f"missing residue map at edit index for {record.pair.pair_id}")
            chain, residue_number, insertion_code = record.residue_map[index]
            source = record.pair.parent_sequence[index]
            target = record.pair.mutant_sequence[index]
            expected.append({
                "index": index,
                "chain": chain,
                "residue_number": residue_number,
                "insertion_code": insertion_code,
                "source": source,
                "target": target,
            })
            direct_support = direct_support or any(
                annotation["chain"] == (record.target_chain or chain)
                and annotation["residue_number"] == residue_number
                and annotation["insertion_code"] == insertion_code
                and annotation["deposited_residue"] == target
                and annotation["reference_residue"] == source
                for annotation in mutant_annotations
            )
            reverse_support = reverse_support or any(
                annotation["chain"] == record.source_chain
                and annotation["residue_number"] == residue_number
                and annotation["insertion_code"] == insertion_code
                and annotation["deposited_residue"] == source
                and annotation["reference_residue"] == target
                for annotation in parent_annotations
            )
            parent_site_residues = {
                annotation["deposited_residue"]
                for annotation in parent_annotations
                if annotation["chain"] == record.source_chain
                and annotation["residue_number"] == residue_number
                and annotation["insertion_code"] == insertion_code
            }
            mutant_site_residues = {
                annotation["deposited_residue"]
                for annotation in mutant_annotations
                if annotation["chain"] == (record.target_chain or chain)
                and annotation["residue_number"] == residue_number
                and annotation["insertion_code"] == insertion_code
            }
            paired_endpoint_support = paired_endpoint_support or (
                source in parent_site_residues and target in mutant_site_residues
            )
        relevant_annotations = [
            annotation
            for annotation in parent_annotations + mutant_annotations
            if annotation["chain"] in {record.source_chain, record.target_chain}
        ]
        if direct_support or reverse_support or paired_endpoint_support:
            classification = "edit_supported_by_seqadv"
        elif relevant_annotations:
            classification = "other_engineered_mutation_only"
        else:
            classification = "no_engineered_seqadv_annotation"
        results.append({
            "pair_id": record.pair.pair_id,
            "parent_id": record.parent_id,
            "family_id": record.family_id,
            "split": record.split,
            "expected_edit": expected,
            "parent_engineered_mutations": parent_annotations,
            "mutant_engineered_mutations": mutant_annotations,
            "direct_support": direct_support,
            "reverse_support": reverse_support,
            "paired_endpoint_support": paired_endpoint_support,
            "classification": classification,
        })
    classes: list[str] = sorted({str(row["classification"]) for row in results})
    return {
        "format": REPORT_FORMAT,
        "manifest_fingerprint": manifest_fingerprint(rows),
        "selection_semantics": "diagnostic_only_not_a_hard_filter",
        "limitations": [
            "legacy_seqadv_annotations_are_optional_and_incomplete",
            "absence_of_seqadv_does_not_mean_the_edit_is_invalid",
            "residue_map_is_assumed_to_use_shared_parent_mutant_author_numbering",
            "other_engineered_mutation_only_requires_manual_construct_review",
        ],
        "summary": {
            "pairs": len(results),
            "classifications": {
                classification: sum(
                    row["classification"] == classification for row in results
                )
                for classification in classes
            },
            "splits": {
                split: {
                    classification: sum(
                        row["split"] == split
                        and row["classification"] == classification
                        for row in results
                    )
                    for classification in classes
                }
                for split in ("train", "dev", "test")
            },
        },
        "records": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    args = parser.parse_args()
    try:
        report = audit_endpoint_mutation_annotations(load_manifest(args.manifest))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        f"endpoint annotation audit written: {destination} "
        f"({report['summary']['pairs']} pairs)"
    )


if __name__ == "__main__":
    main()
