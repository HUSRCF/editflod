"""Validate static endpoint-context decisions against their source queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


QUEUE_FORMAT = "ospedit.endpoint_context_review_queue.v1"
DECISION_FORMAT = "ospedit.endpoint_context_decisions.v1"
DISPOSITIONS = {
    "matched_context_candidate",
    "state_mismatch_challenge",
    "assembly_dependent_challenge",
    "provenance_reject",
    "reverse_duplicate_alias",
}


def _payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_endpoint_context_decisions(
    queue: dict[str, Any], decisions: dict[str, Any]
) -> list[str]:
    errors = []
    if queue.get("format") != QUEUE_FORMAT:
        errors.append(f"queue must use {QUEUE_FORMAT}")
    if decisions.get("format") != DECISION_FORMAT:
        errors.append(f"decisions must use {DECISION_FORMAT}")
    if decisions.get("selection_used_observed_response") is not False:
        errors.append("decisions must declare response-independent selection")
    if decisions.get("manifest_fingerprint") != queue.get("manifest_fingerprint"):
        errors.append("decision manifest fingerprint does not match queue")
    if decisions.get("source_queue_sha256") != _payload_sha256(queue):
        errors.append("decision source queue digest does not match queue")
    expected = {
        row["pair_id"]: row for row in queue.get("records", []) if row["split"] != "test"
    }
    rows = decisions.get("records", [])
    actual: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            errors.append("decision record has no pair_id")
            continue
        if pair_id in actual:
            errors.append(f"duplicate decision for {pair_id}")
        actual[pair_id] = row
        if row.get("split") == "test":
            errors.append(f"frozen test record was reviewed: {pair_id}")
        if row.get("disposition") not in DISPOSITIONS:
            errors.append(f"unknown disposition for {pair_id}: {row.get('disposition')}")
        if row.get("direct_supervision_candidate") is True and row.get(
            "disposition"
        ) != "matched_context_candidate":
            errors.append(f"ineligible direct-supervision disposition for {pair_id}")
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing:
        errors.append("missing non-test decisions: " + ", ".join(missing))
    if extra:
        errors.append("unexpected decisions: " + ", ".join(extra))
    for pair_id in sorted(set(expected) & set(actual)):
        if expected[pair_id]["split"] != actual[pair_id].get("split"):
            errors.append(f"split mismatch for {pair_id}")
    for pair_id, row in actual.items():
        canonical = row.get("canonical_pair_id")
        if row.get("disposition") == "reverse_duplicate_alias":
            if canonical not in actual:
                errors.append(f"missing canonical decision for {pair_id}: {canonical}")
            elif actual[canonical].get("canonical_pair_id"):
                errors.append(f"canonical decision is itself an alias for {pair_id}")

    computed_summary: dict[str, Any] = {
        "reviewed": len(rows),
        **{
            disposition: sum(row.get("disposition") == disposition for row in rows)
            for disposition in sorted(DISPOSITIONS)
        },
        "direct_supervision_candidates": {
            split: sum(
                row.get("direct_supervision_candidate") is True
                and row.get("split") == split
                for row in rows
            )
            for split in ("train", "dev", "test")
        },
        "strict_mutation_attribution_admitted": 0,
    }
    if decisions.get("summary") != computed_summary:
        errors.append("summary does not match decision records")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue")
    parser.add_argument("decisions")
    args = parser.parse_args()
    try:
        queue = json.loads(Path(args.queue).read_text())
        decisions = json.loads(Path(args.decisions).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    errors = validate_endpoint_context_decisions(queue, decisions)
    if errors:
        raise SystemExit("decision validation failed: " + "; ".join(errors))
    print(f"endpoint context decisions valid: {len(decisions['records'])} records")


if __name__ == "__main__":
    main()
