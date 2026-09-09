"""Build an auditable pair manifest from a small CSV pairing table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .data import append_manifest, load_manifest, pair_record_from_structures, parse_structure, validate_manifest, write_manifest


_REQUIRED = ("pair_id", "parent_structure", "mutant_structure", "parent_id", "family_id", "split")


def _path(value: str, base: Path) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (base / path).resolve())


def records_from_csv(path: str | Path):
    source = Path(path)
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = [name for name in _REQUIRED if name not in fields]
        if missing:
            raise ValueError("pair CSV is missing columns: " + ", ".join(missing))
        records = []
        for line_number, row in enumerate(reader, start=2):
            try:
                parent = parse_structure(_path(row["parent_structure"], source.parent), row.get("parent_chain") or "A")
                mutant = parse_structure(_path(row["mutant_structure"], source.parent), row.get("mutant_chain") or "A")
                records.append(
                    pair_record_from_structures(
                        parent,
                        mutant,
                        pair_id=row["pair_id"],
                        parent_id=row["parent_id"],
                        family_id=row["family_id"],
                        split=row["split"],
                        label_source=row.get("label_source") or "experimental",
                    )
                )
            except (KeyError, OSError, ValueError) as error:
                raise ValueError(f"pair CSV row {line_number}: {error}") from error
    if not records:
        raise ValueError("pair CSV contains no records")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an auditable ospedit manifest from a CSV pairing table")
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--append", action="store_true", help="Append to an existing manifest")
    parser.add_argument("--max-mutations", type=int, default=None)
    args = parser.parse_args()
    records = records_from_csv(args.pairs_csv)
    existing = load_manifest(args.output) if args.append and Path(args.output).exists() else []
    all_records = [*existing, *records]
    errors = validate_manifest(all_records, max_mutations=args.max_mutations)
    if errors:
        raise SystemExit("manifest validation failed: " + "; ".join(errors))
    if args.append:
        append_manifest(records, args.output)
    else:
        write_manifest(records, args.output)
    print(f"wrote {len(all_records)} records to {args.output}")


if __name__ == "__main__":
    main()
