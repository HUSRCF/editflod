"""Build a response-independent endpoint holdout within represented families."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from ospedit.data import (
    PairRecord,
    json_safe,
    load_manifest,
    manifest_fingerprint,
    write_manifest,
)
from ospedit.endpoint_groups import endpoint_group_id, endpoint_group_key


REPORT_FORMAT = "ospedit.within_family_probe_split.v1"


def _rank(seed: int, family_id: str, key: tuple[str, str]) -> str:
    payload = "\n".join((str(seed), family_id, *key)).encode()
    return hashlib.sha256(payload).hexdigest()


def build_within_family_probe(
    records: Iterable[PairRecord],
    *,
    dev_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[PairRecord], dict[str, Any]]:
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be in (0, 1)")
    source = list(records)
    if any(record.split != "train" for record in source):
        raise ValueError("within-family probe input must contain only train records")

    families: dict[str, dict[tuple[str, str], list[PairRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in source:
        families[record.family_id][endpoint_group_key(record)].append(record)

    output: list[PairRecord] = []
    family_rows: list[dict[str, Any]] = []
    excluded_records = 0
    for family_id, groups in sorted(families.items()):
        if len(groups) < 2:
            record_count = sum(len(group) for group in groups.values())
            excluded_records += record_count
            family_rows.append({
                "family_id": family_id,
                "eligible": False,
                "reason": "fewer_than_two_endpoint_groups",
                "endpoint_groups": len(groups),
                "records": record_count,
            })
            continue
        ordered = sorted(groups, key=lambda key: (_rank(seed, family_id, key), key))
        dev_groups = max(1, min(len(ordered) - 1, math.ceil(len(ordered) * dev_fraction)))
        dev_keys = set(ordered[:dev_groups])
        assignments: list[dict[str, Any]] = []
        for key in ordered:
            split = "dev" if key in dev_keys else "train"
            members = sorted(groups[key], key=lambda record: record.pair.pair_id)
            output.extend(replace(record, split=split) for record in members)
            assignments.append({
                "endpoint_group_id": endpoint_group_id(members[0]),
                "split": split,
                "pair_ids": [record.pair.pair_id for record in members],
            })
        family_rows.append({
            "family_id": family_id,
            "eligible": True,
            "endpoint_groups": len(groups),
            "records": sum(len(group) for group in groups.values()),
            "train_endpoint_groups": len(groups) - dev_groups,
            "dev_endpoint_groups": dev_groups,
            "assignments": assignments,
        })

    output.sort(key=lambda record: (record.split, record.family_id, record.pair.pair_id))
    report = {
        "format": REPORT_FORMAT,
        "source_manifest_fingerprint": manifest_fingerprint(source),
        "probe_manifest_fingerprint": manifest_fingerprint(output),
        "selection_used_observed_response": False,
        "frozen_dev_or_test_records_used": 0,
        "configuration": {"dev_fraction": dev_fraction, "seed": seed},
        "summary": {
            "source_records": len(source),
            "eligible_families": sum(row["eligible"] for row in family_rows),
            "excluded_families": sum(not row["eligible"] for row in family_rows),
            "excluded_records": excluded_records,
            "probe_records": len(output),
            "train_records": sum(record.split == "train" for record in output),
            "dev_records": sum(record.split == "dev" for record in output),
            "train_families": len({r.family_id for r in output if r.split == "train"}),
            "dev_families": len({r.family_id for r in output if r.split == "dev"}),
        },
        "families": family_rows,
        "interpretation": [
            "diagnostic_only_not_a_replacement_for_frozen_dev",
            "physical_endpoint_groups_never_cross_splits",
            "all_probe_dev_families_are_represented_in_probe_train",
            "excluded_single_endpoint_families_are_not_model_failures",
        ],
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output_manifest")
    parser.add_argument("output_report")
    parser.add_argument("--source-split", default="train", choices=("train",))
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source = [
        record
        for record in load_manifest(args.manifest)
        if record.split == args.source_split
    ]
    probe, report = build_within_family_probe(
        source, dev_fraction=args.dev_fraction, seed=args.seed
    )
    write_manifest(probe, args.output_manifest)
    destination = Path(args.output_report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        f"within-family probe written: {args.output_manifest} "
        f"({report['summary']['train_records']} train, "
        f"{report['summary']['dev_records']} dev)"
    )


if __name__ == "__main__":
    main()
