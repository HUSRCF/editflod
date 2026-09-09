"""Merge audited manifests while preserving record provenance and splits."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

from ospedit.data import (
    PairRecord,
    load_manifest,
    manifest_fingerprint,
    validate_manifest,
    verify_record_checksums,
    write_manifest,
)


def merge_manifest_files(
    paths: Iterable[str | Path],
    *,
    verify_checksums: bool = False,
) -> tuple[list[PairRecord], dict[str, Any]]:
    sources = [Path(path) for path in paths]
    if len(sources) < 2:
        raise ValueError("at least two manifests are required")
    records: list[PairRecord] = []
    inputs = []
    for source in sources:
        rows = load_manifest(source)
        errors = validate_manifest(rows)
        if verify_checksums:
            errors.extend(error for record in rows for error in verify_record_checksums(record))
        if errors:
            raise ValueError(f"input manifest failed validation ({source}): " + "; ".join(errors))
        inputs.append({
            "path": str(source.resolve()),
            "records": len(rows),
            "manifest_fingerprint": manifest_fingerprint(rows),
        })
        records.extend(rows)
    pair_ids = [record.pair.pair_id for record in records]
    duplicates = sorted(pair_id for pair_id, count in Counter(pair_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"input manifests contain duplicate pair_ids: {duplicates}")
    errors = validate_manifest(records)
    if errors:
        raise ValueError("merged manifest failed validation: " + "; ".join(errors))
    split_counts = {
        split: sum(record.split == split for record in records)
        for split in ("train", "dev", "test")
    }
    report = {
        "format": "ospedit.manifest_merge.v1",
        "inputs": inputs,
        "records": len(records),
        "split_counts": split_counts,
        "manifest_fingerprint": manifest_fingerprint(records),
        "checksums_verified": verify_checksums,
    }
    return records, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge validated ospedit manifests")
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--verify-checksums", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output in {Path(path).resolve() for path in args.manifest}:
        parser.error("--output must differ from every input manifest")
    try:
        records, report = merge_manifest_files(args.manifest, verify_checksums=args.verify_checksums)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    write_manifest(records, output)
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"merged manifest written: {output} ({len(records)} records; {report['split_counts']})")


if __name__ == "__main__":
    main()
