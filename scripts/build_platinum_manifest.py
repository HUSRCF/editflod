"""Build strict experimental WT/mutant structure pairs from Platinum."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
import json
from pathlib import Path
import re
from typing import Any

from Bio.PDB import PDBParser
import numpy as np

from ospedit.data import (
    PairRecord,
    ParsedStructure,
    assign_group_splits,
    manifest_fingerprint,
    pair_record_from_structures,
    parse_structure,
    validate_manifest,
    write_manifest,
)


MISSING = {"", "NO", "NR", "NONE", "N/A", "NULL", "-"}
MUTATION_PATTERN = re.compile(r"([A-Z])(\d+)([A-Z])")


def _present(value: str) -> bool:
    return value.strip().upper() not in MISSING


def _pdb_path(root: Path, pdb_id: str) -> Path | None:
    candidates = (
        root / f"{pdb_id.upper()}.pdb",
        root / f"{pdb_id.lower()}.pdb",
        root / f"{pdb_id}.pdb",
    )
    return next((path for path in candidates if path.is_file()), None)


def _model_hetero(path: Path) -> set[str]:
    structure = PDBParser(QUIET=True, PERMISSIVE=True).get_structure(path.stem, str(path))
    model = next(iter(structure))
    return {
        residue.resname.strip().upper()
        for chain in model
        for residue in chain
        if residue.id[0].strip() and residue.resname.strip().upper() != "HOH"
    }


def _map_terminal_overlap(
    parent: ParsedStructure,
    mutant: ParsedStructure,
    *,
    min_coverage: float,
) -> tuple[ParsedStructure, ParsedStructure, dict[str, Any]]:
    """Crop terminal coordinate differences while rejecting internal gaps."""
    if parent.residue_ids == mutant.residue_ids:
        return parent, mutant, {
            "mode": "exact_residue_ids",
            "coverage": 1.0,
            "parent_terminal_trim": [0, 0],
            "mutant_terminal_trim": [0, 0],
        }
    mutant_indices = {residue_id: index for index, residue_id in enumerate(mutant.residue_ids)}
    shared = [
        (parent_index, mutant_indices[residue_id])
        for parent_index, residue_id in enumerate(parent.residue_ids)
        if residue_id in mutant_indices
    ]
    if not shared:
        raise ValueError("parent and mutant have no shared residue identifiers")
    parent_positions = [item[0] for item in shared]
    mutant_positions = [item[1] for item in shared]
    if parent_positions != list(range(parent_positions[0], parent_positions[-1] + 1)) or (
        mutant_positions != list(range(mutant_positions[0], mutant_positions[-1] + 1))
    ):
        raise ValueError("residue identifier mismatch contains an internal gap or reordering")
    retained = len(shared)
    coverage = retained / max(len(parent.sequence), len(mutant.sequence))
    if coverage < min_coverage:
        raise ValueError(
            f"terminal-overlap coverage {coverage:.4f} is below minimum {min_coverage:.4f}"
        )
    parent_start, parent_stop = parent_positions[0], parent_positions[-1] + 1
    mutant_start, mutant_stop = mutant_positions[0], mutant_positions[-1] + 1

    def crop(structure: ParsedStructure, start: int, stop: int) -> ParsedStructure:
        return ParsedStructure(
            path=structure.path,
            chain_id=structure.chain_id,
            sequence=structure.sequence[start:stop],
            coords=np.array(structure.coords[start:stop], copy=True),
            residue_ids=structure.residue_ids[start:stop],
            atom_names=structure.atom_names,
        )

    return crop(parent, parent_start, parent_stop), crop(mutant, mutant_start, mutant_stop), {
        "mode": "terminal_overlap_crop",
        "coverage": coverage,
        "parent_original_length": len(parent.sequence),
        "mutant_original_length": len(mutant.sequence),
        "parent_terminal_trim": [parent_start, len(parent.sequence) - parent_stop],
        "mutant_terminal_trim": [mutant_start, len(mutant.sequence) - mutant_stop],
    }


def records_from_platinum_csv(
    csv_path: str | Path,
    pdb_root: str | Path,
    *,
    dataset_name: str = "platinum",
    monomer_only: bool = True,
    min_length: int | None = 64,
    max_length: int | None = 256,
    allow_terminal_crop: bool = True,
    min_mapping_coverage: float = 0.95,
    rejections: list[dict[str, Any]] | None = None,
) -> tuple[list[PairRecord], dict[str, int]]:
    """Return strict, deduplicated experimental structure pairs and counters."""
    if not 0.0 < min_mapping_coverage <= 1.0:
        raise ValueError("min_mapping_coverage must be in (0, 1]")
    root = Path(pdb_root)
    if not root.is_dir():
        raise NotADirectoryError(f"PDB root does not exist: {root}")
    groups: OrderedDict[tuple[str, str, str, str], list[dict[str, str]]] = OrderedDict()
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "mutation",
            "affin.chain",
            "affin.lig_id",
            "mut.mt_pdb",
            "mut.wt_pdb",
            "mut.is_single_point",
            "mut.uniprot",
            "prot.stoichiometry",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Platinum CSV is missing required columns")
        for row in reader:
            mutation = row["mutation"].strip().upper()
            wt = row["mut.wt_pdb"].strip().upper()
            mt = row["mut.mt_pdb"].strip().upper()
            chain = row["affin.chain"].strip()
            if (
                row["mut.is_single_point"].strip().upper() != "YES"
                or MUTATION_PATTERN.fullmatch(mutation) is None
                or not _present(wt)
                or not _present(mt)
                or not chain
                or (monomer_only and row["prot.stoichiometry"].strip() != "MONOMERIC (AUTHOR)")
            ):
                continue
            groups.setdefault((wt, mt, chain, mutation), []).append(row)

    counters = {
        "candidate_structure_pairs": len(groups),
        "accepted": 0,
        "missing_structure": 0,
        "invalid_mapping": 0,
        "ligand_mismatch": 0,
        "out_of_scope_length": 0,
        "exact_mapping": 0,
        "terminal_overlap_mapping": 0,
    }
    records = []
    for (wt, mt, chain, mutation), rows in groups.items():
        try:
            wt_path = _pdb_path(root, wt)
            mt_path = _pdb_path(root, mt)
            if wt_path is None or mt_path is None:
                counters["missing_structure"] += 1
                raise FileNotFoundError(f"missing structure: WT={wt} MT={mt}")
            parent = parse_structure(wt_path, chain)
            mutant = parse_structure(mt_path, chain)
            if parent.residue_ids != mutant.residue_ids and not allow_terminal_crop:
                counters["invalid_mapping"] += 1
                raise ValueError("parent and mutant residue identifiers do not match")
            try:
                parent, mutant, mapping_metadata = _map_terminal_overlap(
                    parent,
                    mutant,
                    min_coverage=min_mapping_coverage,
                )
            except ValueError:
                counters["invalid_mapping"] += 1
                raise
            differences = [
                index
                for index, (source, target) in enumerate(zip(parent.sequence, mutant.sequence))
                if source != target
            ]
            match = MUTATION_PATTERN.fullmatch(mutation)
            assert match is not None
            source_residue, author_number, target_residue = match.group(1), int(match.group(2)), match.group(3)
            if len(differences) != 1:
                counters["invalid_mapping"] += 1
                raise ValueError(f"expected one sequence difference, found {len(differences)}")
            index = differences[0]
            if (
                parent.sequence[index] != source_residue
                or mutant.sequence[index] != target_residue
                or parent.residue_ids[index][1] != author_number
                or mutant.residue_ids[index][1] != author_number
            ):
                counters["invalid_mapping"] += 1
                raise ValueError("mutation identity or author residue number does not match structures")
            length = len(parent.sequence)
            if (min_length is not None and length < min_length) or (
                max_length is not None and length > max_length
            ):
                counters["out_of_scope_length"] += 1
                raise ValueError(f"length {length} is outside configured scope")
            ligand_ids = sorted({row["affin.lig_id"].strip().upper() for row in rows if _present(row["affin.lig_id"])})
            parent_hetero = _model_hetero(wt_path)
            mutant_hetero = _model_hetero(mt_path)
            shared_ligands = [
                ligand for ligand in ligand_ids if ligand in parent_hetero and ligand in mutant_hetero
            ]
            if not shared_ligands:
                counters["ligand_mismatch"] += 1
                raise ValueError(
                    f"declared ligands are not present in both structures: {ligand_ids}"
                )
            uniprot_ids = sorted({row["mut.uniprot"].strip() for row in rows if _present(row["mut.uniprot"])})
            family = uniprot_ids[0] if len(uniprot_ids) == 1 else f"{wt}_{chain}"
            pair_id = f"{dataset_name}_{wt}_{mt}_{chain}_{mutation}"
            records.append(
                pair_record_from_structures(
                    parent,
                    mutant,
                    pair_id=pair_id,
                    parent_id=f"{wt}_{chain}",
                    family_id=f"{dataset_name}_uniprot_{family}",
                    split="train",
                    environment_metadata={
                        "source_database": "Platinum",
                        "declared_ligand_ids": ligand_ids,
                        "shared_declared_ligand_ids": shared_ligands,
                        "parent_model_hetero": sorted(parent_hetero),
                        "mutant_model_hetero": sorted(mutant_hetero),
                        "author_stoichiometry": sorted({row["prot.stoichiometry"] for row in rows}),
                        "residue_mapping": mapping_metadata,
                    },
                    experiment_metadata={
                        "platinum_rows": len(rows),
                        "uniprot_ids": uniprot_ids,
                        "affinity_methods": sorted({row.get("affin.exptal_method", "") for row in rows}),
                        "pmids": sorted({row.get("mut.pmid", "") for row in rows if _present(row.get("mut.pmid", ""))}),
                    },
                )
            )
            counters["accepted"] += 1
            counters[
                "exact_mapping"
                if mapping_metadata["mode"] == "exact_residue_ids"
                else "terminal_overlap_mapping"
            ] += 1
        except (KeyError, OSError, ValueError) as error:
            if rejections is not None:
                rejections.append({
                    "wt_pdb": wt,
                    "mt_pdb": mt,
                    "chain": chain,
                    "mutation": mutation,
                    "reason": str(error),
                    "error_type": type(error).__name__,
                })
    return records, counters


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an ospedit manifest from Platinum")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--pdb-root", required=True)
    parser.add_argument("--dataset-name", default="platinum")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-nonmonomers", action="store_true")
    parser.add_argument("--min-length", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--require-exact-residue-ids", action="store_true")
    parser.add_argument("--min-mapping-coverage", type=float, default=0.95)
    args = parser.parse_args()
    rejections: list[dict[str, Any]] = []
    try:
        records, counters = records_from_platinum_csv(
            args.csv,
            args.pdb_root,
            dataset_name=args.dataset_name,
            monomer_only=not args.include_nonmonomers,
            min_length=args.min_length,
            max_length=args.max_length,
            allow_terminal_crop=not args.require_exact_residue_ids,
            min_mapping_coverage=args.min_mapping_coverage,
            rejections=rejections,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if not records:
        raise SystemExit(f"no usable Platinum pairs found: {counters}")
    records = assign_group_splits(records, seed=args.seed, group_by="family")
    errors = validate_manifest(records, max_mutations=1)
    if errors:
        raise SystemExit("generated manifest failed validation: " + "; ".join(errors))
    write_manifest(records, args.output)
    report = {
        "format": "ospedit.platinum_manifest.v1",
        "csv": str(Path(args.csv).resolve()),
        "pdb_root": str(Path(args.pdb_root).resolve()),
        "monomer_only": not args.include_nonmonomers,
        "length_scope": [args.min_length, args.max_length],
        "allow_terminal_crop": not args.require_exact_residue_ids,
        "min_mapping_coverage": args.min_mapping_coverage,
        "counters": counters,
        "records": len(records),
        "split_counts": {
            split: sum(record.split == split for record in records)
            for split in ("train", "dev", "test")
        },
        "manifest_fingerprint": manifest_fingerprint(records),
        "rejections": rejections,
    }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Platinum manifest written: {args.output} ({len(records)} records)")


if __name__ == "__main__":
    main()
