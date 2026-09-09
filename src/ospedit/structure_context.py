"""Coordinate-file context features for experimental structure audits."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from Bio.PDB import MMCIFParser, NeighborSearch, PDBParser
from Bio.PDB.Polypeptide import is_aa


def structure_context(
    path: str | Path,
    chain_id: str,
    *,
    ignored_hetero: frozenset[str],
    protein_contact_radius: float = 5.0,
    hetero_contact_radius: float = 6.0,
) -> dict[str, Any]:
    """Describe ligands and protein contacts around one target chain."""
    if protein_contact_radius <= 0 or hetero_contact_radius <= 0:
        raise ValueError("contact radii must be positive")
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
    target_atoms = [
        atom
        for residue in target_chain
        if is_aa(residue, standard=True)
        for atom in residue.get_atoms()
    ]
    neighbors = NeighborSearch(list(model.get_atoms()))
    proximal_hetero: set[str] = set()
    protein_contact_chains: set[str] = set()
    protein_contact_residues: set[tuple[str, str]] = set()
    for atom in target_atoms:
        for residue in neighbors.search(atom.coord, hetero_contact_radius, level="R"):
            residue_name = residue.resname.strip().upper()
            if residue.id[0].strip() and residue_name not in ignored_hetero:
                proximal_hetero.add(residue_name)
        for residue in neighbors.search(atom.coord, protein_contact_radius, level="R"):
            chain = residue.get_parent()
            if chain.id != chain_id and is_aa(residue, standard=True):
                protein_contact_chains.add(chain.id)
                protein_contact_residues.add((chain.id, str(residue.id)))
    header = getattr(structure, "header", {}) or {}
    resolution = header.get("resolution")
    return {
        "path": str(source),
        "target_chain": chain_id,
        "protein_chains": protein_chains,
        "protein_chain_count": len(protein_chains),
        "target_hetero": target_hetero,
        "model_hetero": model_hetero,
        "proximal_hetero": sorted(proximal_hetero),
        "protein_contact_chains": sorted(protein_contact_chains),
        "protein_contact_chain_count": len(protein_contact_chains),
        "protein_contact_residue_count": len(protein_contact_residues),
        "protein_contact_radius_angstrom": protein_contact_radius,
        "hetero_contact_radius_angstrom": hetero_contact_radius,
        "structure_method": header.get("structure_method"),
        "resolution_angstrom": (
            float(resolution)
            if isinstance(resolution, (int, float)) and math.isfinite(float(resolution))
            else None
        ),
        "name": header.get("name"),
    }
