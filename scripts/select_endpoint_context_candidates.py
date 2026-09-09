"""Select response-independent direct-supervision candidates from a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from ospedit.data import (
    PairRecord,
    load_manifest,
    manifest_fingerprint,
    validate_manifest,
    write_manifest,
)


DECISION_FORMAT = "ospedit.endpoint_context_decisions.v1"
REPORT_FORMAT = "ospedit.endpoint_context_candidate_selection.v1"


def select_endpoint_context_candidates(
    records: Iterable[PairRecord], decisions: dict[str, Any]
) -> tuple[list[PairRecord], dict[str, Any]]:
    rows = list(records)
    errors = validate_manifest(rows)
    if errors:
        raise ValueError("manifest audit failed: " + "; ".join(errors))
    fingerprint = manifest_fingerprint(rows)
    if decisions.get("format") != DECISION_FORMAT:
        raise ValueError(f"decisions must use {DECISION_FORMAT}")
    if decisions.get("manifest_fingerprint") != fingerprint:
        raise ValueError("decisions use a different manifest")
    if decisions.get("selection_used_observed_response") is not False:
        raise ValueError("decisions must declare response-independent selection")

    by_pair = {record.pair.pair_id: record for record in rows}
    selected: list[PairRecord] = []
    decision_rows = []
    seen: set[str] = set()
    for decision in decisions.get("records", []):
        if decision.get("direct_supervision_candidate") is not True:
            continue
        pair_id = decision.get("pair_id")
        if pair_id in seen:
            raise ValueError(f"duplicate selected decision: {pair_id}")
        seen.add(pair_id)
        record = by_pair.get(pair_id)
        if record is None:
            raise ValueError(f"selected pair is absent from manifest: {pair_id}")
        if record.split == "test" or decision.get("split") == "test":
            raise ValueError(f"frozen test record cannot be selected: {pair_id}")
        if decision.get("split") != record.split:
            raise ValueError(f"split mismatch for selected pair: {pair_id}")
        if decision.get("disposition") != "matched_context_candidate":
            raise ValueError(f"selected pair lacks matched-context disposition: {pair_id}")
        selected.append(record)
        decision_rows.append({
            "pair_id": pair_id,
            "parent_id": record.parent_id,
            "family_id": record.family_id,
            "split": record.split,
            "disposition": decision["disposition"],
        })
    if not selected:
        raise ValueError("decisions select no direct-supervision candidates")
    selected.sort(key=lambda record: (record.split, record.family_id, record.pair.pair_id))
    decision_rows.sort(key=lambda row: (row["split"], row["family_id"], row["pair_id"]))
    report = {
        "format": REPORT_FORMAT,
        "source_manifest_fingerprint": fingerprint,
        "decision_sha256": hashlib.sha256(
            json.dumps(
                decisions,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest(),
        "selected_manifest_fingerprint": manifest_fingerprint(selected),
        "selection_used_observed_response": False,
        "usage": "mechanism_development_probe_not_generalization_benchmark",
        "summary": {
            "records": len(selected),
            "families": len({record.family_id for record in selected}),
            "parents": len({record.parent_id for record in selected}),
            "splits": {
                split: sum(record.split == split for record in selected)
                for split in ("train", "dev", "test")
            },
        },
        "records": decision_rows,
    }
    return selected, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("decisions")
    parser.add_argument("output_manifest")
    parser.add_argument("output_report")
    args = parser.parse_args()
    try:
        records = load_manifest(args.manifest)
        decisions = json.loads(Path(args.decisions).read_text())
        selected, report = select_endpoint_context_candidates(records, decisions)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    write_manifest(selected, args.output_manifest)
    destination = Path(args.output_report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        f"context candidates written: {args.output_manifest}; "
        f"{report['summary']['records']} records"
    )


if __name__ == "__main__":
    main()
