"""Combined preflight gate for the first real mechanism experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .data import load_manifest, validate_manifest, verify_record_checksums
from .foldflow_env import inspect_foldflow_environment


def _dataset_structure(records: list[Any]) -> dict[str, Any]:
    """Summarize grouping capacity before attempting held-out splits."""
    parents = {record.parent_id for record in records}
    families = {record.family_id for record in records}
    roots: dict[str, str] = {}

    def find(node: str) -> str:
        roots.setdefault(node, node)
        if roots[node] != node:
            roots[node] = find(roots[node])
        return roots[node]

    for record in records:
        left, right = find(f"family:{record.family_id}"), find(f"parent:{record.parent_id}")
        if left != right:
            roots[right] = left
    components = {find(f"family:{family}") for family in families}
    parent_sizes: dict[str, int] = {}
    for record in records:
        parent_sizes[record.parent_id] = parent_sizes.get(record.parent_id, 0) + 1
    split_counts = {
        split: sum(record.split == split for record in records)
        for split in ("train", "dev", "test")
    }
    return {
        "unique_parents": len(parents),
        "unique_families": len(families),
        "connected_group_count": len(components),
        "split_capacity_ok": len(components) >= 3,
        "split_record_counts": split_counts,
        "assigned_splits_nonempty": all(count > 0 for count in split_counts.values()),
        "largest_parent_record_count": max(parent_sizes.values(), default=0),
        "warnings": [
            "fewer than three connected parent-family groups; nonempty train/dev/test split is impossible"
        ] if len(components) < 3 else [],
    }


def p1_preflight_report(
    manifest: str,
    *,
    verify_checksums: bool = False,
    max_mutations: int | None = None,
    require_split_capacity: bool = False,
) -> dict[str, Any]:
    records = load_manifest(manifest)
    manifest_errors = validate_manifest(records, max_mutations=max_mutations)
    if verify_checksums:
        manifest_errors.extend(error for record in records for error in verify_record_checksums(record))
    dataset_structure = _dataset_structure(records)
    if require_split_capacity:
        if not dataset_structure["split_capacity_ok"]:
            manifest_errors.append(
                "dataset has fewer than three connected parent-family groups; nonempty train/dev/test split is impossible"
            )
        if not dataset_structure["assigned_splits_nonempty"]:
            manifest_errors.append(
                "manifest split assignment has an empty train/dev/test split"
            )
    environment = inspect_foldflow_environment()
    return {
        "ready": not manifest_errors and bool(environment["ready_for_foldflow_import"]),
        "manifest": {
            "path": manifest,
            "records": len(records),
            "errors": manifest_errors,
            "max_mutations": max_mutations,
            "dataset_structure": dataset_structure,
            "require_split_capacity": require_split_capacity,
        },
        "environment": environment,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check all prerequisites for a real P1 mechanism run")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--output", help="Write the JSON report to this path as well as stdout")
    parser.add_argument("--max-mutations", type=int, default=None)
    parser.add_argument("--foldflow-root", help="upstream FoldFlow source checkout to include in the environment audit")
    parser.add_argument(
        "--require-split-capacity",
        action="store_true",
        help="Fail unless at least three connected parent-family groups are available",
    )
    args = parser.parse_args()
    if args.foldflow_root:
        os.environ["OSPEDIT_FOLDFLOW_ROOT"] = args.foldflow_root
    report = p1_preflight_report(
        args.manifest,
        verify_checksums=args.verify_checksums,
        max_mutations=args.max_mutations,
        require_split_capacity=args.require_split_capacity,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload + "\n")
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
