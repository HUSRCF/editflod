"""Assemble identity-compatible suite methods into one benchmark table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from ospedit.data import json_safe


def _parse_entry(value: str) -> tuple[str, Path, str]:
    if "=" not in value or "::" not in value:
        raise ValueError("entry must use OUTPUT_METHOD=REPORT_PATH::SOURCE_METHOD")
    output_method, remainder = value.split("=", 1)
    report_path, source_method = remainder.rsplit("::", 1)
    if not output_method or not report_path or not source_method:
        raise ValueError("entry must use OUTPUT_METHOD=REPORT_PATH::SOURCE_METHOD")
    return output_method, Path(report_path).expanduser().resolve(), source_method


def assemble(entries: Iterable[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    sources = []
    split_identity: dict[str, tuple[str, tuple[str, ...]]] = {}
    seen_methods: set[tuple[str, str]] = set()
    for specification in entries:
        output_method, path, source_method = _parse_entry(specification)
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"unable to read suite report {path}: {error}") from error
        split = payload.get("split")
        fingerprint = payload.get("manifest_fingerprint")
        methods = payload.get("methods")
        if split not in {"train", "dev", "test"} or not isinstance(fingerprint, str):
            raise ValueError(f"suite report has invalid split/fingerprint: {path}")
        if not isinstance(methods, dict) or source_method not in methods:
            raise ValueError(f"suite report {path} has no method {source_method!r}")
        method = methods[source_method]
        records = method.get("records") if isinstance(method, dict) else None
        if not isinstance(records, list) or not records:
            raise ValueError(f"suite method has no records: {path}::{source_method}")
        pair_ids = tuple(str(record.get("pair_id")) for record in records)
        identity = (fingerprint, pair_ids)
        if split in split_identity and split_identity[split] != identity:
            raise ValueError(
                f"split identity mismatch for {split}: {split_identity[split]} != {identity}"
            )
        split_identity[split] = identity
        method_key = (split, output_method)
        if method_key in seen_methods:
            raise ValueError(f"duplicate output method for split {split}: {output_method}")
        seen_methods.add(method_key)
        for record in records:
            metrics = record.get("metrics")
            runtime = record.get("runtime")
            if not isinstance(metrics, dict) or not isinstance(runtime, dict):
                raise ValueError(f"suite record lacks metrics/runtime: {path}::{source_method}")
            row: dict[str, Any] = {
                "split": split,
                "pair_id": record.get("pair_id"),
                "parent_id": record.get("parent_id"),
                "family_id": record.get("family_id"),
                "method": output_method,
                "source_method": source_method,
                "source_report": str(path),
                "manifest_fingerprint": fingerprint,
            }
            row.update({f"metric_{key}": value for key, value in metrics.items() if key != "pair_id"})
            row.update({f"runtime_{key}": value for key, value in runtime.items() if key != "extras"})
            extras = runtime.get("extras", {})
            if isinstance(extras, dict):
                row.update({f"runtime_{key}": value for key, value in extras.items()})
            rows.append(row)
        sources.append({
            "output_method": output_method,
            "source_method": source_method,
            "report": str(path),
            "split": split,
            "manifest_fingerprint": fingerprint,
            "pair_ids": list(pair_ids),
        })
    if not rows:
        raise ValueError("at least one benchmark entry is required")
    split_order = {"train": 0, "dev": 1, "test": 2}
    rows.sort(key=lambda row: (split_order[str(row["split"])], str(row["method"]), str(row["pair_id"])))
    return {
        "format": "ospedit.benchmark_table.v1",
        "split_identity": {
            split: {"manifest_fingerprint": identity[0], "pair_ids": list(identity[1])}
            for split, identity in sorted(split_identity.items(), key=lambda item: split_order[item[0]])
        },
        "sources": sources,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble suite reports into a strict benchmark table")
    parser.add_argument("--entry", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", required=True)
    args = parser.parse_args()
    try:
        payload = assemble(args.entry)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    rows = payload["rows"]
    fieldnames = sorted({key for row in rows for key in row})
    csv_output = Path(args.csv_output)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(json_safe(rows))
    print(f"benchmark table written: {output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
