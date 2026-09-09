"""Select PreMut mutation groups with multiple wild-type structure repeats."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from ospedit.data import load_manifest


def _token_parts(token: str) -> tuple[str, str]:
    parts = token.strip().split("_")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"expected PDB_CHAIN token, got {token!r}")
    return parts[0].upper(), parts[1]


def _excluded_mutants(manifests: Iterable[str | Path]) -> set[str]:
    excluded = set()
    for manifest in manifests:
        for record in load_manifest(manifest):
            if record.target_file and record.target_chain:
                excluded.add(f"{Path(record.target_file).stem.upper()}_{record.target_chain}")
    return excluded


def _excluded_parents(manifests: Iterable[str | Path]) -> set[str]:
    return {
        record.parent_id.upper()
        for manifest in manifests
        for record in load_manifest(manifest)
    }


def _excluded_reports(reports: Iterable[str | Path]) -> tuple[set[str], set[str]]:
    mutants: set[str] = set()
    parents: set[str] = set()
    for report in reports:
        payload = json.loads(Path(report).read_text())
        if payload.get("format") != "ospedit.premut_repeat_candidates.v1":
            raise ValueError(f"candidate exclusion report has unsupported format: {report}")
        for candidate in payload.get("candidates", []):
            mutants.add(str(candidate["mutant_token"]).upper())
            parents.add(str(candidate["parent_token"]).upper())
    return mutants, parents


def select_repeat_candidates(
    csv_path: str | Path,
    *,
    dataset_name: str,
    pdb_root: str | Path,
    max_candidates: int,
    repeat_structures: int = 2,
    excluded_mutants: Iterable[str] = (),
    excluded_parents: Iterable[str] = (),
    unique_parents: bool = True,
    max_mutation_index: int | None = None,
) -> dict[str, Any]:
    if max_candidates <= 0 or repeat_structures <= 0:
        raise ValueError("max_candidates and repeat_structures must be positive")
    if max_mutation_index is not None and max_mutation_index < 0:
        raise ValueError("max_mutation_index must be non-negative")
    groups: OrderedDict[tuple[str, str], list[str]] = OrderedDict()
    with Path(csv_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Mutated_PDB", "Mutation INFO", "Possible Wilds"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("PreMut CSV is missing required columns")
        for row in reader:
            mutant = row["Mutated_PDB"].strip().upper()
            mutation = row["Mutation INFO"].strip().upper()
            wild = row["Possible Wilds"].strip().upper()
            _token_parts(mutant)
            _token_parts(wild)
            wilds = groups.setdefault((mutant, mutation), [])
            if wild not in wilds:
                wilds.append(wild)
    excluded = {token.upper() for token in excluded_mutants}
    excluded_parent_tokens = {token.upper() for token in excluded_parents}
    root = Path(pdb_root).resolve()
    candidates = []
    structure_ids: set[str] = set()
    selected_parents: set[str] = set()
    for (mutant, mutation), wilds in groups.items():
        try:
            mutation_index = int(mutation.split("_")[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"invalid mutation token: {mutation!r}") from error
        if (
            mutant in excluded
            or len(wilds) < repeat_structures + 1
            or wilds[0] in excluded_parent_tokens
            or (unique_parents and wilds[0] in selected_parents)
            or (max_mutation_index is not None and mutation_index > max_mutation_index)
        ):
            continue
        mutant_id, mutant_chain = _token_parts(mutant)
        parent_id, parent_chain = _token_parts(wilds[0])
        pair_id = f"{dataset_name}_{parent_id}_{mutant_id}_{parent_chain}_{mutation}"
        repeats = []
        for token in wilds[1 : repeat_structures + 1]:
            repeat_id, repeat_chain = _token_parts(token)
            repeats.append({
                "token": token,
                "structure": str(root / f"{repeat_id}.pdb"),
                "chain": repeat_chain,
            })
            structure_ids.add(repeat_id)
        candidates.append({
            "pair_id": pair_id,
            "mutant_token": mutant,
            "mutation": mutation,
            "parent_token": wilds[0],
            "parent_structure": str(root / f"{parent_id}.pdb"),
            "parent_chain": parent_chain,
            "mutant_structure": str(root / f"{mutant_id}.pdb"),
            "mutant_chain": mutant_chain,
            "repeats": repeats,
        })
        selected_parents.add(wilds[0])
        structure_ids.update((mutant_id, parent_id))
        if len(candidates) >= max_candidates:
            break
    return {
        "format": "ospedit.premut_repeat_candidates.v1",
        "source_csv": str(Path(csv_path).resolve()),
        "dataset_name": dataset_name,
        "pdb_root": str(root),
        "repeat_structures": repeat_structures,
        "excluded_mutants": sorted(excluded),
        "excluded_parents": sorted(excluded_parent_tokens),
        "unique_parents": unique_parents,
        "max_mutation_index": max_mutation_index,
        "candidates": candidates,
        "structure_ids": sorted(structure_ids),
    }


def _write_csvs(payload: dict[str, Any], mutations_path: Path, repeats_path: Path) -> None:
    mutations_path.parent.mkdir(parents=True, exist_ok=True)
    with mutations_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("Mutated_PDB", "Mutation INFO", "Possible Wilds"))
        writer.writeheader()
        for candidate in payload["candidates"]:
            writer.writerow({
                "Mutated_PDB": candidate["mutant_token"],
                "Mutation INFO": candidate["mutation"],
                "Possible Wilds": candidate["parent_token"],
            })
    repeats_path.parent.mkdir(parents=True, exist_ok=True)
    with repeats_path.open("w", newline="") as handle:
        fields = ("pair_id", "parent_structure", "parent_chain", "repeat_structure", "repeat_chain", "mutation_index")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in payload["candidates"]:
            mutation_index = int(str(candidate["mutation"]).split("_")[1])
            for repeat in candidate["repeats"]:
                writer.writerow({
                    "pair_id": candidate["pair_id"],
                    "parent_structure": candidate["parent_structure"],
                    "parent_chain": candidate["parent_chain"],
                    "repeat_structure": repeat["structure"],
                    "repeat_chain": repeat["chain"],
                    "mutation_index": mutation_index,
                })


def main() -> None:
    parser = argparse.ArgumentParser(description="Select PreMut groups with repeated wild structures")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--dataset-name", default="premut")
    parser.add_argument("--pdb-root", required=True)
    parser.add_argument("--max-candidates", type=int, required=True)
    parser.add_argument("--repeat-structures", type=int, default=2)
    parser.add_argument("--exclude-manifest", nargs="*", default=())
    parser.add_argument("--exclude-report", nargs="*", default=())
    parser.add_argument("--exclude-mutant", nargs="*", default=())
    parser.add_argument("--allow-repeated-parent", action="store_true")
    parser.add_argument(
        "--allow-excluded-parents",
        action="store_true",
        help="Exclude audited mutants while permitting other mutations of their parents",
    )
    parser.add_argument("--max-mutation-index", type=int)
    parser.add_argument("--mutations-output", required=True)
    parser.add_argument("--repeats-output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    exclude_manifests = tuple(args.exclude_manifest)
    report_mutants, report_parents = _excluded_reports(args.exclude_report)
    try:
        payload = select_repeat_candidates(
            args.csv,
            dataset_name=args.dataset_name,
            pdb_root=args.pdb_root,
            max_candidates=args.max_candidates,
            repeat_structures=args.repeat_structures,
            excluded_mutants=(
                _excluded_mutants(exclude_manifests)
                | report_mutants
                | {value.upper() for value in args.exclude_mutant}
            ),
            excluded_parents=(
                ()
                if args.allow_excluded_parents
                else _excluded_parents(exclude_manifests) | report_parents
            ),
            unique_parents=not args.allow_repeated_parent,
            max_mutation_index=args.max_mutation_index,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if not payload["candidates"]:
        raise SystemExit("no eligible repeat candidates found")
    _write_csvs(payload, Path(args.mutations_output), Path(args.repeats_output))
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"selected {len(payload['candidates'])} repeat candidates; report: {report}")


if __name__ == "__main__":
    main()
