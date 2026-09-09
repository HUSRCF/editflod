"""Audit parent/mutant experimental context before mutation attribution."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import is_aa

from ospedit.data import (
    PairRecord,
    json_safe,
    load_manifest,
    manifest_fingerprint,
    validate_manifest,
    verify_record_checksums,
    write_manifest,
)


def _structure_context(
    path: str | Path,
    chain_id: str,
    *,
    ignored_hetero: frozenset[str],
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    parser = MMCIFParser(QUIET=True) if source.suffix.lower() in {".cif", ".mmcif"} else PDBParser(QUIET=True, PERMISSIVE=True)
    structure = parser.get_structure(source.stem, str(source))
    try:
        model = next(iter(structure))
        target_chain = model[chain_id]
    except (StopIteration, KeyError) as error:
        raise ValueError(f"chain/model not found: {source} chain={chain_id}") from error
    protein_chains = [
        chain.id for chain in model if any(is_aa(residue, standard=True) for residue in chain)
    ]
    target_hetero = sorted({
        residue.resname.strip().upper()
        for residue in target_chain
        if residue.id[0].strip() and residue.resname.strip().upper() not in ignored_hetero
    })
    model_hetero = sorted({
        residue.resname.strip().upper()
        for chain in model
        for residue in chain
        if residue.id[0].strip() and residue.resname.strip().upper() not in ignored_hetero
    })
    header = getattr(structure, "header", {}) or {}
    resolution = header.get("resolution")
    return {
        "path": str(source),
        "target_chain": chain_id,
        "protein_chains": protein_chains,
        "protein_chain_count": len(protein_chains),
        "target_hetero": target_hetero,
        "model_hetero": model_hetero,
        "structure_method": header.get("structure_method"),
        "resolution_angstrom": (
            float(resolution)
            if isinstance(resolution, (int, float)) and math.isfinite(float(resolution))
            else None
        ),
        "name": header.get("name"),
    }


def audit_structure_context(
    records: Iterable[PairRecord],
    *,
    require_single_protein_chain: bool = True,
    require_matching_target_hetero: bool = True,
    require_matching_method: bool = True,
    max_resolution_difference: float | None = None,
    ignored_hetero: Iterable[str] = ("HOH",),
) -> tuple[dict[str, Any], list[PairRecord]]:
    rows = list(records)
    errors = validate_manifest(rows)
    if errors:
        raise ValueError("manifest audit failed: " + "; ".join(errors))
    if max_resolution_difference is not None and max_resolution_difference < 0:
        raise ValueError("max_resolution_difference must be non-negative")
    ignored = frozenset(value.strip().upper() for value in ignored_hetero if value.strip())
    results = []
    selected = []
    for record in rows:
        parent = _structure_context(record.source_file, record.source_chain, ignored_hetero=ignored)
        mutant = _structure_context(record.target_file, record.target_chain, ignored_hetero=ignored)
        reasons = []
        if require_single_protein_chain and (
            parent["protein_chain_count"] != 1 or mutant["protein_chain_count"] != 1
        ):
            reasons.append("multiple_protein_chains")
        if require_matching_target_hetero and parent["target_hetero"] != mutant["target_hetero"]:
            reasons.append("target_hetero_mismatch")
        if require_matching_method and parent["structure_method"] != mutant["structure_method"]:
            reasons.append("structure_method_mismatch")
        resolution_difference = None
        if parent["resolution_angstrom"] is not None and mutant["resolution_angstrom"] is not None:
            resolution_difference = abs(
                float(parent["resolution_angstrom"]) - float(mutant["resolution_angstrom"])
            )
            if (
                max_resolution_difference is not None
                and resolution_difference > max_resolution_difference
            ):
                reasons.append("resolution_difference_exceeds_limit")
        row = {
            "pair_id": record.pair.pair_id,
            "parent_id": record.parent_id,
            "parent": parent,
            "mutant": mutant,
            "resolution_difference_angstrom": resolution_difference,
            "selected": not reasons,
            "rejection_reasons": reasons,
        }
        results.append(row)
        if not reasons:
            selected.append(record)
    payload = {
        "format": "ospedit.structure_context_audit.v1",
        "manifest_fingerprint": manifest_fingerprint(rows),
        "configuration": {
            "require_single_protein_chain": require_single_protein_chain,
            "require_matching_target_hetero": require_matching_target_hetero,
            "require_matching_method": require_matching_method,
            "max_resolution_difference": max_resolution_difference,
            "ignored_hetero": sorted(ignored),
        },
        "records": results,
        "selected_records": len(selected),
        "rejected_records": len(rows) - len(selected),
    }
    return payload, selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit experimental structure context")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--filtered-output")
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--allow-multiple-protein-chains", action="store_true")
    parser.add_argument("--allow-target-hetero-mismatch", action="store_true")
    parser.add_argument("--allow-method-mismatch", action="store_true")
    parser.add_argument("--max-resolution-difference", type=float)
    parser.add_argument("--ignore-hetero", nargs="*", default=("HOH",))
    args = parser.parse_args()
    records = load_manifest(args.manifest)
    if args.verify_checksums:
        checksum_errors = [error for record in records for error in verify_record_checksums(record)]
        if checksum_errors:
            raise SystemExit("checksum audit failed: " + "; ".join(checksum_errors))
    try:
        payload, selected = audit_structure_context(
            records,
            require_single_protein_chain=not args.allow_multiple_protein_chains,
            require_matching_target_hetero=not args.allow_target_hetero_mismatch,
            require_matching_method=not args.allow_method_mismatch,
            max_resolution_difference=args.max_resolution_difference,
            ignored_hetero=args.ignore_hetero,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    if args.filtered_output:
        write_manifest(selected, args.filtered_output)
    print(
        f"structure-context audit written: {destination} "
        f"({len(selected)}/{len(records)} selected)"
    )


if __name__ == "__main__":
    main()
