"""Small parsers for provenance annotations in legacy PDB coordinate files."""

from __future__ import annotations

from pathlib import Path
from typing import Any


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def parse_engineered_mutations(path: str | Path) -> list[dict[str, Any]]:
    """Return structured engineered-mutation SEQADV records from a PDB file."""
    records = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if not line.startswith("SEQADV") or "ENGINEERED MUTATION" not in line[49:]:
            continue
        deposited_name = line[12:15].strip().upper()
        reference_name = line[39:42].strip().upper()
        try:
            residue_number = int(line[18:22])
        except ValueError:
            continue
        records.append({
            "chain": line[16:17].strip(),
            "residue_number": residue_number,
            "insertion_code": line[22:23].strip(),
            "deposited_residue_name": deposited_name,
            "deposited_residue": THREE_TO_ONE.get(deposited_name),
            "reference_residue_name": reference_name,
            "reference_residue": THREE_TO_ONE.get(reference_name),
        })
    return records


def engineered_edit_supported(
    parent_annotations: list[dict[str, Any]],
    mutant_annotations: list[dict[str, Any]],
    *,
    parent_chain: str,
    mutant_chain: str,
    parent_residue_number: int,
    mutant_residue_number: int,
    parent_insertion_code: str,
    mutant_insertion_code: str,
    source_residue: str,
    target_residue: str,
) -> bool:
    """Check whether SEQADV records support an endpoint residue substitution."""
    parent_site = [
        row
        for row in parent_annotations
        if row["chain"] == parent_chain
        and row["residue_number"] == parent_residue_number
        and row["insertion_code"] == parent_insertion_code
    ]
    mutant_site = [
        row
        for row in mutant_annotations
        if row["chain"] == mutant_chain
        and row["residue_number"] == mutant_residue_number
        and row["insertion_code"] == mutant_insertion_code
    ]
    direct = any(
        row["deposited_residue"] == target_residue
        and row["reference_residue"] == source_residue
        for row in mutant_site
    )
    reverse = any(
        row["deposited_residue"] == source_residue
        and row["reference_residue"] == target_residue
        for row in parent_site
    )
    paired = (
        any(row["deposited_residue"] == source_residue for row in parent_site)
        and any(row["deposited_residue"] == target_residue for row in mutant_site)
    )
    return direct or reverse or paired
