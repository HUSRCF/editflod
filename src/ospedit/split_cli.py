"""Deterministically split a pair manifest by family or parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import assign_group_splits, load_manifest, manifest_fingerprint, validate_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign deterministic group-wise train/dev/test splits")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--group-by", choices=("family", "parent"), default="family")
    parser.add_argument("--require-nonempty", action="store_true", help="fail unless train/dev/test all contain records")
    parser.add_argument("--report", help="Optional JSON provenance report path")
    args = parser.parse_args()
    records = load_manifest(args.manifest)
    try:
        split_records = assign_group_splits(
            records,
            seed=args.seed,
            train_fraction=args.train_fraction,
            dev_fraction=args.dev_fraction,
            group_by=args.group_by,
        )
    except ValueError as error:
        raise SystemExit(f"split assignment failed: {error}") from error
    errors = validate_manifest(split_records)
    if errors:
        raise SystemExit("split manifest validation failed: " + "; ".join(errors))
    counts = {split: sum(row.split == split for row in split_records) for split in ("train", "dev", "test")}
    if args.require_nonempty and any(counts[split] == 0 for split in counts):
        raise SystemExit(f"split assignment has an empty split: {counts}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(split_records, output_path)
    if args.report:
        report = {
            "input": str(Path(args.manifest).resolve()),
            "output": str(Path(args.output).resolve()),
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "dev_fraction": args.dev_fraction,
            "test_fraction": 1.0 - args.train_fraction - args.dev_fraction,
            "group_by": args.group_by,
            "require_nonempty": args.require_nonempty,
            "records": len(split_records),
            "split_counts": counts,
            "manifest_fingerprint": manifest_fingerprint(split_records),
        }
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(split_records)} records to {args.output}: {counts}")


if __name__ == "__main__":
    main()
