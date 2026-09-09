"""Convert a local PreMut CSV/PDB release into an audited ospedit manifest."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

from ospedit.data import (
    PairRecord,
    assign_group_splits,
    manifest_fingerprint,
    pair_record_from_structures,
    parse_structure,
    validate_manifest,
    write_manifest,
)


def _pdb_path(root: Path, token: str) -> Path | None:
    """Resolve ``1abc_A`` to a case-insensitive ``1abc.pdb`` below root."""
    pdb_id = token.split("_", 1)[0]
    candidates = [root / f"{pdb_id}.pdb", root / f"{pdb_id.lower()}.pdb", root / f"{pdb_id.upper()}.pdb"]
    return next((path for path in candidates if path.is_file()), None)


def _token_parts(token: str) -> tuple[str, str]:
    parts = token.strip().split("_")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"expected PDB_CHAIN token, got {token!r}")
    return parts[0].upper(), parts[1]


def _mutation_parts(value: str) -> tuple[str, int, str]:
    parts = value.strip().split("_")
    if len(parts) != 3 or len(parts[0]) != 1 or len(parts[2]) != 1:
        raise ValueError(f"expected WT_POSITION_MUT token, got {value!r}")
    return parts[0].upper(), int(parts[1]), parts[2].upper()


def _load_cluster_map(path: str | Path | None) -> dict[str, str]:
    """Load PreMut's pickle cluster dictionary as structure-token -> cluster."""
    if path is None:
        return {}
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("PreMut cluster dictionary must contain a mapping")
    output: dict[str, str] = {}
    for cluster, members in payload.items():
        if not isinstance(members, (list, tuple, set)):
            raise ValueError("PreMut cluster dictionary members must be sequences")
        for member in members:
            output[str(member).upper()] = str(cluster)
    return output


def records_from_premut_csv(
    csv_path: str | Path,
    pdb_root: str | Path,
    *,
    dataset_name: str,
    max_rows: int | None = None,
    strict_missing: bool = False,
    cluster_dict: str | Path | None = None,
    rejections: list[dict[str, object]] | None = None,
) -> tuple[list[PairRecord], dict[str, int]]:
    """Build records and return ``(records, counters)`` without writing files."""
    source = Path(csv_path)
    root = Path(pdb_root)
    if not root.is_dir():
        raise NotADirectoryError(f"PDB root does not exist: {root}")
    cluster_map = _load_cluster_map(cluster_dict)
    required = {"Mutated_PDB", "Mutation INFO", "Possible Wilds"}
    records: list[PairRecord] = []
    counters = {"rows": 0, "accepted": 0, "missing_parent": 0, "missing_mutant": 0, "invalid": 0}
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(set(reader.fieldnames or ())):
            raise ValueError(f"PreMut CSV must contain columns: {', '.join(sorted(required))}")
        for row_number, row in enumerate(reader, start=2):
            if max_rows is not None and counters["rows"] >= max_rows:
                break
            counters["rows"] += 1
            try:
                mutant_token = row["Mutated_PDB"].strip()
                parent_token = row["Possible Wilds"].strip()
                mutation = row["Mutation INFO"].strip()
                mutant_id, mutant_chain = _token_parts(mutant_token)
                parent_id, parent_chain = _token_parts(parent_token)
                wild, position, target = _mutation_parts(mutation)
                mutant_path = _pdb_path(root, mutant_token)
                parent_path = _pdb_path(root, parent_token)
                if mutant_path is None:
                    counters["missing_mutant"] += 1
                    if strict_missing:
                        raise FileNotFoundError(f"mutant PDB is missing: {mutant_token}")
                    continue
                if parent_path is None:
                    counters["missing_parent"] += 1
                    if strict_missing:
                        raise FileNotFoundError(f"parent PDB is missing: {parent_token}")
                    continue
                parent = parse_structure(parent_path, parent_chain)
                mutant = parse_structure(mutant_path, mutant_chain)
                # PreMut's mutation index is explicitly zero-based.
                if position < 0:
                    raise ValueError(f"mutation position must be non-negative: {position}")
                if position >= len(parent.sequence) or parent.sequence[position] != wild:
                    raise ValueError(f"parent residue mismatch at {parent_id}_{parent_chain}:{position}")
                if position >= len(mutant.sequence) or mutant.sequence[position] != target:
                    raise ValueError(f"mutant residue mismatch at {mutant_id}_{mutant_chain}:{position}")
                pair_id = f"{dataset_name}_{parent_id}_{mutant_id}_{parent_chain}_{mutation}"
                cluster = cluster_map.get(parent_token.upper())
                family_id = (
                    f"{dataset_name}_cluster_{cluster}"
                    if cluster is not None
                    else f"{dataset_name}_mutant_{mutant_id}_{mutant_chain}"
                )
                records.append(
                    pair_record_from_structures(
                        parent,
                        mutant,
                        pair_id=pair_id,
                        parent_id=f"{parent_id}_{parent_chain}",
                        family_id=family_id,
                        split="train",
                    )
                )
                counters["accepted"] += 1
            except (KeyError, OSError, ValueError) as exc:
                counters["invalid"] += 1
                if rejections is not None:
                    rejections.append(
                        {
                            "row": row_number,
                            "mutant": row.get("Mutated_PDB", ""),
                            "mutation": row.get("Mutation INFO", ""),
                            "parent": row.get("Possible Wilds", ""),
                            "error_type": type(exc).__name__,
                            "reason": str(exc),
                        }
                    )
                if strict_missing:
                    raise
    return records, counters


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an audited manifest from a local PreMut release")
    parser.add_argument("--csv", required=True, help="MutData2022.csv or MutData2023.csv")
    parser.add_argument("--pdb-root", required=True, help="Directory containing one PDB file per structure")
    parser.add_argument("--dataset-name", default="premut")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--strict-missing", action="store_true")
    parser.add_argument("--cluster-dict", help="Optional PreMut *_cluster_dict pickle")
    parser.add_argument("--report", help="Optional JSON audit report path")
    args = parser.parse_args()
    rejections: list[dict[str, object]] = []
    records, counters = records_from_premut_csv(
        args.csv,
        args.pdb_root,
        dataset_name=args.dataset_name,
        max_rows=args.max_rows,
        strict_missing=args.strict_missing,
        cluster_dict=args.cluster_dict,
        rejections=rejections,
    )
    if not records:
        raise SystemExit(f"no usable rows found: {counters}")
    records = assign_group_splits(records, seed=args.seed, group_by="family")
    errors = validate_manifest(records, max_mutations=1)
    if errors:
        raise SystemExit("generated manifest failed validation: " + "; ".join(errors))
    write_manifest(records, args.output)
    counts = {split: sum(record.split == split for record in records) for split in ("train", "dev", "test")}
    if args.report:
        report = {
            "csv": str(Path(args.csv).resolve()),
            "pdb_root": str(Path(args.pdb_root).resolve()),
            "cluster_dict": str(Path(args.cluster_dict).resolve()) if args.cluster_dict else None,
            "dataset_name": args.dataset_name,
            "coordinate_alignment": "backbone_kabsch_mutant_to_parent",
            "counters": counters,
            "splits": counts,
            "records": len(records),
            "rejections": rejections,
            "manifest_fingerprint": manifest_fingerprint(records),
        }
        Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(records)} records to {args.output}; splits={counts}; counters={counters}")


if __name__ == "__main__":
    main()
