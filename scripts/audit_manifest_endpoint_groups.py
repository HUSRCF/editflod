"""Audit duplicate and reverse-direction physical endpoint pairs in a manifest."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

from ospedit.data import PairRecord, load_manifest, manifest_fingerprint, validate_manifest
from ospedit.endpoint_groups import (
    endpoint_group_id,
    endpoint_group_key,
    endpoint_identity,
)


REPORT_FORMAT = "ospedit.manifest_endpoint_group_audit.v1"


def audit_manifest_endpoint_groups(records: Iterable[PairRecord]) -> dict[str, Any]:
    rows = list(records)
    errors = validate_manifest(rows)
    if errors:
        raise ValueError("manifest audit failed: " + "; ".join(errors))
    grouped: dict[tuple[str, str], list[PairRecord]] = defaultdict(list)
    for record in rows:
        grouped[endpoint_group_key(record)].append(record)
    duplicate_groups = []
    for key, members in sorted(grouped.items()):
        if len(members) < 2:
            continue
        canonical = min(members, key=lambda record: record.pair.pair_id)
        # The unordered key alone cannot recover direction. Compare endpoint identity order.
        canonical_order = (
            endpoint_identity(canonical, source=True),
            endpoint_identity(canonical, source=False),
        )
        member_rows = []
        observed_directions = set()
        for record in sorted(members, key=lambda item: item.pair.pair_id):
            order = (
                endpoint_identity(record, source=True),
                endpoint_identity(record, source=False),
            )
            direction = "canonical" if order == canonical_order else "reverse"
            observed_directions.add(direction)
            member_rows.append({
                "pair_id": record.pair.pair_id,
                "split": record.split,
                "family_id": record.family_id,
                "direction": direction,
            })
        duplicate_groups.append({
            "endpoint_group_id": endpoint_group_id(canonical),
            "canonical_pair_id": canonical.pair.pair_id,
            "classification": (
                "reverse_direction_duplicates"
                if "reverse" in observed_directions
                else "same_direction_duplicates"
            ),
            "records": member_rows,
        })
    return {
        "format": REPORT_FORMAT,
        "manifest_fingerprint": manifest_fingerprint(rows),
        "grouping": "unordered_physical_endpoint_identity_pair",
        "limitations": [
            "checksums_identify_files_not_biologically_equivalent_re_refinements",
            "reverse_directions_may_be_valid_training_augmentation_but_not_independent_evidence",
        ],
        "summary": {
            "records": len(rows),
            "unique_endpoint_groups": len(grouped),
            "duplicate_groups": len(duplicate_groups),
            "records_in_duplicate_groups": sum(
                len(group["records"]) for group in duplicate_groups
            ),
            "redundant_records": sum(
                len(group["records"]) - 1 for group in duplicate_groups
            ),
            "same_direction_duplicate_groups": sum(
                group["classification"] == "same_direction_duplicates"
                for group in duplicate_groups
            ),
            "reverse_direction_duplicate_groups": sum(
                group["classification"] == "reverse_direction_duplicates"
                for group in duplicate_groups
            ),
        },
        "groups": duplicate_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    args = parser.parse_args()
    try:
        report = audit_manifest_endpoint_groups(load_manifest(args.manifest))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        f"endpoint-group audit written: {destination}; "
        f"{report['summary']['duplicate_groups']} duplicate groups"
    )


if __name__ == "__main__":
    main()
