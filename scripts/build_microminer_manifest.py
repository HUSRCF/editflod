"""Build strictly mapped structure-pair manifests from selected MicroMiner rows."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from ospedit.data import (
    PairRecord,
    THREE_TO_ONE,
    map_terminal_overlap,
    manifest_fingerprint,
    pair_record_from_structures,
    parse_structure,
    validate_manifest,
    write_manifest,
)
from scripts.select_microminer_candidates import COLUMNS


def _structure_path(root: Path, pdb_id: str) -> Path | None:
    names = (pdb_id.upper(), pdb_id.lower(), pdb_id)
    for name in names:
        for suffix in (".pdb", ".cif", ".mmcif"):
            candidate = root / f"{name}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def records_from_microminer_candidates(
    csv_path: str | Path,
    structure_root: str | Path,
    *,
    dataset_name: str = "microminer",
    min_length: int | None = 64,
    max_length: int | None = 256,
    chain_uniprot: dict[str, list[str]] | None = None,
    allow_terminal_overlap: bool = False,
    min_mapping_coverage: float = 0.95,
    rejections: list[dict[str, Any]] | None = None,
) -> tuple[list[PairRecord], dict[str, int]]:
    """Validate selected rows against observed chains and return all-train records."""
    if not 0.0 < min_mapping_coverage <= 1.0:
        raise ValueError("min_mapping_coverage must be in (0, 1]")
    root = Path(structure_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"structure root does not exist: {root}")
    counters = {
        "candidate_rows": 0,
        "accepted": 0,
        "missing_structure": 0,
        "parse_error": 0,
        "unequal_observed_length": 0,
        "not_single_substitution": 0,
        "mutation_identity_mismatch": 0,
        "mutation_author_number_mismatch": 0,
        "out_of_scope_length": 0,
        "exact_residue_id_mapping": 0,
        "sequence_index_equal_length_mapping": 0,
        "terminal_overlap_mapping": 0,
        "invalid_terminal_overlap": 0,
    }
    records = []
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("candidate CSV header does not match the MicroMiner selection schema")
        for line_number, row in enumerate(reader, start=2):
            counters["candidate_rows"] += 1
            query_id = row["queryName"].strip().upper()
            hit_id = row["hitName"].strip().upper()
            query_chain = row["queryChain"].strip()
            hit_chain = row["hitChain"].strip()
            rejection = {
                "line_number": line_number,
                "query_pdb": query_id,
                "query_chain": query_chain,
                "hit_pdb": hit_id,
                "hit_chain": hit_chain,
            }
            try:
                query_path = _structure_path(root, query_id)
                hit_path = _structure_path(root, hit_id)
                if query_path is None or hit_path is None:
                    counters["missing_structure"] += 1
                    raise FileNotFoundError(f"missing structure: query={query_id} hit={hit_id}")
                try:
                    parent = parse_structure(query_path, query_chain)
                    mutant = parse_structure(hit_path, hit_chain)
                except ValueError:
                    counters["parse_error"] += 1
                    raise
                mapping_metadata: dict[str, Any]
                if parent.residue_ids == mutant.residue_ids:
                    mapping_metadata = {"mode": "exact_residue_ids", "coverage": 1.0}
                elif allow_terminal_overlap:
                    try:
                        parent, mutant, mapping_metadata = map_terminal_overlap(
                            parent,
                            mutant,
                            min_coverage=min_mapping_coverage,
                        )
                    except ValueError:
                        counters["invalid_terminal_overlap"] += 1
                        raise
                elif len(parent.sequence) != len(mutant.sequence):
                    counters["unequal_observed_length"] += 1
                    raise ValueError(
                        f"observed chain lengths differ: {len(parent.sequence)} != {len(mutant.sequence)}"
                    )
                else:
                    mapping_metadata = {
                        "mode": "sequence_index_equal_length",
                        "coverage": 1.0,
                        "requires_equal_observed_length": True,
                    }
                differences = [
                    index
                    for index, (source, target) in enumerate(zip(parent.sequence, mutant.sequence))
                    if source != target
                ]
                if len(differences) != 1:
                    counters["not_single_substitution"] += 1
                    raise ValueError(f"expected one observed sequence difference, found {len(differences)}")
                mutation_index = differences[0]
                source_aa = THREE_TO_ONE.get(row["queryAA"].strip().upper())
                target_aa = THREE_TO_ONE.get(row["hitAA"].strip().upper())
                if (
                    source_aa is None
                    or target_aa is None
                    or parent.sequence[mutation_index] != source_aa
                    or mutant.sequence[mutation_index] != target_aa
                ):
                    counters["mutation_identity_mismatch"] += 1
                    raise ValueError("MicroMiner mutation identities do not match observed structures")
                try:
                    query_position = int(row["queryPos"])
                    hit_position = int(row["hitPos"])
                except ValueError as error:
                    counters["mutation_author_number_mismatch"] += 1
                    raise ValueError("MicroMiner mutation positions are not integers") from error
                parent_residue_id = parent.residue_ids[mutation_index]
                mutant_residue_id = mutant.residue_ids[mutation_index]
                if parent_residue_id[1] != query_position or mutant_residue_id[1] != hit_position:
                    counters["mutation_author_number_mismatch"] += 1
                    raise ValueError("MicroMiner positions do not match author residue numbers")
                length = len(parent.sequence)
                if (min_length is not None and length < min_length) or (
                    max_length is not None and length > max_length
                ):
                    counters["out_of_scope_length"] += 1
                    raise ValueError(f"length {length} is outside configured scope")
                exact_ids = mapping_metadata["mode"] == "exact_residue_ids"
                original_mutant_residue_ids = mutant.residue_ids
                if mapping_metadata["mode"] == "sequence_index_equal_length":
                    mutant = replace(mutant, residue_ids=parent.residue_ids)
                pair_id = (
                    f"{dataset_name}_{query_id}_{query_chain}_{hit_id}_{hit_chain}_"
                    f"{source_aa}{query_position}{target_aa}"
                )
                query_uniprot = set((chain_uniprot or {}).get(f"{query_id}_{query_chain}", []))
                hit_uniprot = set((chain_uniprot or {}).get(f"{hit_id}_{hit_chain}", []))
                shared_uniprot = sorted(query_uniprot.intersection(hit_uniprot))
                if chain_uniprot is not None and not shared_uniprot:
                    raise ValueError("metadata report has no shared UniProt accession")
                family_id = (
                    f"{dataset_name}_uniprot_{'+'.join(shared_uniprot)}"
                    if shared_uniprot
                    else f"{dataset_name}_provisional_{query_id}_{query_chain}"
                )
                record = pair_record_from_structures(
                    parent,
                    mutant,
                    pair_id=pair_id,
                    parent_id=f"{query_id}_{query_chain}",
                    family_id=family_id,
                    split="train",
                    environment_metadata={
                        "source_database": "MicroMiner/PDB",
                        "residue_mapping": {
                            **mapping_metadata,
                            "original_mutant_mutation_residue_id": list(
                                original_mutant_residue_ids[mutation_index]
                            ),
                        },
                    },
                    experiment_metadata={
                        "microminer": {
                            key: (
                                float(row[key])
                                if key
                                in {
                                    "siteIdentity",
                                    "siteBackBoneRMSD",
                                    "siteAllAtomRMSD",
                                    "nofSiteResidues",
                                    "alignmentLDDT",
                                    "fullSeqId",
                                }
                                else row[key]
                            )
                            for key in COLUMNS
                        },
                        "uniprot": {
                            "query": sorted(query_uniprot),
                            "hit": sorted(hit_uniprot),
                            "shared": shared_uniprot,
                        },
                        "family_grouping": (
                            "shared_uniprot"
                            if shared_uniprot
                            else "provisional_query_chain_only"
                        ),
                    },
                )
                records.append(record)
                counters["accepted"] += 1
                counters[
                    "exact_residue_id_mapping"
                    if exact_ids
                    else (
                        "terminal_overlap_mapping"
                        if mapping_metadata["mode"] == "terminal_overlap_crop"
                        else "sequence_index_equal_length_mapping"
                    )
                ] += 1
            except (KeyError, OSError, ValueError) as error:
                if rejections is not None:
                    rejections.append({
                        **rejection,
                        "reason": str(error),
                        "error_type": type(error).__name__,
                    })
    return records, counters


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a manifest from MicroMiner candidates")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--structure-root", required=True)
    parser.add_argument("--dataset-name", default="microminer")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--min-length", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--allow-terminal-overlap", action="store_true")
    parser.add_argument("--min-mapping-coverage", type=float, default=0.95)
    parser.add_argument(
        "--metadata-report",
        help="metadata-audit JSON used to attach shared UniProt family groups",
    )
    args = parser.parse_args()
    rejections: list[dict[str, Any]] = []
    try:
        metadata_report = (
            json.loads(Path(args.metadata_report).read_text())
            if args.metadata_report
            else None
        )
        chain_uniprot = (
            metadata_report.get("selected_chain_uniprot", {})
            if metadata_report is not None
            else None
        )
        if chain_uniprot is not None and not isinstance(chain_uniprot, dict):
            raise ValueError("metadata report selected_chain_uniprot must be an object")
        records, counters = records_from_microminer_candidates(
            args.candidates,
            args.structure_root,
            dataset_name=args.dataset_name,
            min_length=args.min_length,
            max_length=args.max_length,
            chain_uniprot=chain_uniprot,
            allow_terminal_overlap=args.allow_terminal_overlap,
            min_mapping_coverage=args.min_mapping_coverage,
            rejections=rejections,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    report = {
        "format": "ospedit.microminer_manifest.v1",
        "candidates": str(Path(args.candidates).expanduser().resolve()),
        "structure_root": str(Path(args.structure_root).expanduser().resolve()),
        "metadata_report": (
            str(Path(args.metadata_report).expanduser().resolve())
            if args.metadata_report
            else None
        ),
        "length_scope": [args.min_length, args.max_length],
        "mapping_policy": {
            "allow_terminal_overlap": args.allow_terminal_overlap,
            "minimum_coverage": args.min_mapping_coverage,
            "internal_gaps_allowed": False,
        },
        "split_policy": "all_train_pending_sequence_family_clustering",
        "counters": counters,
        "records": len(records),
        "manifest_fingerprint": manifest_fingerprint(records),
        "rejections": rejections,
    }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not records:
        raise SystemExit(f"no usable MicroMiner pairs found: {counters}")
    errors = validate_manifest(records, max_mutations=1)
    if errors:
        raise SystemExit("generated manifest failed validation: " + "; ".join(errors))
    write_manifest(records, args.output)
    print(f"MicroMiner manifest written: {args.output} ({len(records)} records, all train)")


if __name__ == "__main__":
    main()
