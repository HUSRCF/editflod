"""Build the response-independent train/dev structure-prediction layer."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ospedit.data import (
    PairRecord,
    load_manifest,
    manifest_fingerprint,
    validate_manifest,
    write_manifest,
)


REPORT_FORMAT = "ospedit.structure_prediction_layer.v1"
ENDPOINT_FORMAT = "ospedit.rcsb_endpoint_environment_audit.v1"
CONTEXT_FORMAT = "ospedit.structure_context_audit.v1"
ANNOTATION_FORMAT = "ospedit.endpoint_mutation_annotation_audit.v1"
DECISION_FORMAT = "ospedit.endpoint_context_decisions.v1"
DEVELOPMENT_SPLITS = frozenset({"train", "dev"})


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _indexed_records(
    payload: dict[str, Any], expected_format: str, fingerprint: str
) -> dict[str, dict[str, Any]]:
    if payload.get("format") != expected_format:
        raise ValueError(f"report must use {expected_format}")
    if payload.get("manifest_fingerprint") != fingerprint:
        raise ValueError(f"{expected_format} report uses a different manifest")
    indexed: dict[str, dict[str, Any]] = {}
    for row in payload.get("records", []):
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"{expected_format} report contains a record without pair_id")
        if pair_id in indexed:
            raise ValueError(f"{expected_format} report duplicates {pair_id}")
        indexed[pair_id] = row
    return indexed


def select_structure_prediction_layer(
    records: Iterable[PairRecord],
    endpoint_report: dict[str, Any],
    context_report: dict[str, Any],
    annotation_report: dict[str, Any],
    context_decisions: dict[str, Any],
) -> tuple[list[PairRecord], dict[str, Any]]:
    rows = list(records)
    errors = validate_manifest(rows)
    if errors:
        raise ValueError("manifest audit failed: " + "; ".join(errors))
    fingerprint = manifest_fingerprint(rows)
    endpoint = _indexed_records(endpoint_report, ENDPOINT_FORMAT, fingerprint)
    context = _indexed_records(context_report, CONTEXT_FORMAT, fingerprint)
    annotations = _indexed_records(annotation_report, ANNOTATION_FORMAT, fingerprint)
    decisions = _indexed_records(context_decisions, DECISION_FORMAT, fingerprint)

    expected = {record.pair.pair_id for record in rows}
    for name, indexed in (
        ("endpoint", endpoint),
        ("context", context),
        ("annotation", annotations),
    ):
        missing = sorted(expected - set(indexed))
        if missing:
            raise ValueError(f"{name} report misses manifest records: {', '.join(missing)}")
    if context_decisions.get("selection_used_observed_response") is not False:
        raise ValueError("context decisions must declare response-independent selection")

    selected = []
    audit_rows = []
    for record in rows:
        if record.split not in DEVELOPMENT_SPLITS:
            continue
        pair_id = record.pair.pair_id
        reasons = []
        if endpoint[pair_id].get("crystal_form_compatible") is not True:
            reasons.append("crystal_form_incompatible")
        if context[pair_id].get("selected") is not True:
            reasons.extend(
                f"target_context:{reason}"
                for reason in context[pair_id].get("rejection_reasons", [])
            )
        annotation = annotations[pair_id]
        annotation_class = annotation.get("classification")
        if annotation_class == "other_engineered_mutation_only":
            reasons.append("conflicting_mutation_provenance")
        manual = decisions.get(pair_id)
        if manual is not None and manual.get("disposition") != "matched_context_candidate":
            reasons.append("manual_context:" + str(manual.get("disposition")))
        is_selected = not reasons
        if is_selected:
            selected.append(record)
        audit_rows.append({
            "pair_id": pair_id,
            "parent_id": record.parent_id,
            "family_id": record.family_id,
            "split": record.split,
            "selected": is_selected,
            "rejection_reasons": reasons,
            "mutation_annotation_classification": annotation_class,
            "manual_disposition": manual.get("disposition") if manual else None,
        })
    if not selected:
        raise ValueError("structure-prediction layer selects no records")
    selected.sort(key=lambda record: (record.split, record.family_id, record.pair.pair_id))
    audit_rows.sort(key=lambda row: (row["split"], row["family_id"], row["pair_id"]))
    rejection_counts = Counter(
        reason for row in audit_rows for reason in row["rejection_reasons"]
    )
    report = {
        "format": REPORT_FORMAT,
        "source_manifest_fingerprint": fingerprint,
        "selected_manifest_fingerprint": manifest_fingerprint(selected),
        "source_report_sha256": {
            "endpoint": _digest(endpoint_report),
            "target_context": _digest(context_report),
            "mutation_annotations": _digest(annotation_report),
            "manual_context_decisions": _digest(context_decisions),
        },
        "selection_used_observed_response": False,
        "frozen_test_records_processed": 0,
        "configuration": {
            "splits": sorted(DEVELOPMENT_SPLITS),
            "require_crystal_form_compatible": True,
            "require_target_neighborhood_context_selected": True,
            "reject_other_site_only_seqadv": True,
            "manual_nonmatched_dispositions_veto": True,
            "allow_absent_seqadv_annotation": True,
        },
        "limitations": [
            "matching_ligand_names_do_not_establish_matching_occupancy_or_chemical_state",
            "matching_contact_counts_do_not_establish_identical_interface_geometry",
            "records_without_manual_review_are_structure_prediction_supervision_not_strict_mutation_attribution_truth",
            "development_layer_cannot_be_used_for_final_test_claims",
        ],
        "summary": {
            "records_considered": len(audit_rows),
            "records_selected": len(selected),
            "families_selected": len({record.family_id for record in selected}),
            "parents_selected": len({record.parent_id for record in selected}),
            "selected_splits": {
                split: {
                    "records": sum(record.split == split for record in selected),
                    "families": len({
                        record.family_id for record in selected if record.split == split
                    }),
                }
                for split in sorted(DEVELOPMENT_SPLITS)
            },
            "rejection_reasons": dict(sorted(rejection_counts.items())),
        },
        "records": audit_rows,
    }
    return selected, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("endpoint_report")
    parser.add_argument("context_report")
    parser.add_argument("annotation_report")
    parser.add_argument("context_decisions")
    parser.add_argument("output_manifest")
    parser.add_argument("output_report")
    args = parser.parse_args()
    try:
        records = load_manifest(args.manifest)
        payloads = [
            json.loads(Path(path).read_text())
            for path in (
                args.endpoint_report,
                args.context_report,
                args.annotation_report,
                args.context_decisions,
            )
        ]
        selected, report = select_structure_prediction_layer(records, *payloads)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    write_manifest(selected, args.output_manifest)
    destination = Path(args.output_report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        f"structure-prediction layer written: {args.output_manifest}; "
        f"{report['summary']['records_selected']} records"
    )


if __name__ == "__main__":
    main()
