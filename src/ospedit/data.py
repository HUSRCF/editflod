from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
import hashlib
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from Bio.PDB import MMCIFParser, PDBParser
from Bio.Data.IUPACData import protein_letters_3to1


BACKBONE_ATOMS = ("N", "CA", "C", "O")
THREE_TO_ONE = {key.upper(): value for key, value in protein_letters_3to1.items()}


def json_safe(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, (set, frozenset)):
        return [json_safe(item) for item in sorted(value, key=repr)]
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        output: dict[Any, Any] = {}
        for key, item in value.items():
            safe_key = json_safe(key)
            if not isinstance(safe_key, (str, int, float, bool)) and safe_key is not None:
                safe_key = str(safe_key)
            output[safe_key] = json_safe(item)
        return output
    return value


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_fingerprint(records: Iterable[PairRecord]) -> str:
    """Return a deterministic SHA-256 fingerprint for manifest records."""
    rows = [record.to_dict() for record in records]
    # Absolute source paths are machine-local; checksums and embedded arrays
    # carry the data identity needed for resume/evaluation provenance.
    for row in rows:
        row["source_file"] = ""
        row["target_file"] = ""
    canonical = json.dumps(json_safe(rows), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class ParsedStructure:
    """Single-chain structure reduced to a residue-aligned backbone tensor."""

    path: str
    chain_id: str
    sequence: str
    coords: np.ndarray
    residue_ids: tuple[tuple[str, int, str], ...]
    atom_names: tuple[str, ...] = BACKBONE_ATOMS


def parse_structure(path: str | Path, chain_id: str, model_index: int = 0) -> ParsedStructure:
    """Parse one chain from PDB/mmCIF while preserving residue identifiers.

    Only standard amino-acid residues are included. Alternate locations prefer
    blank/A, and absent backbone atoms remain NaN rather than changing indices.
    """
    source = Path(path)
    structure_id = source.stem
    if source.suffix.lower() in {".cif", ".mmcif"}:
        structure = MMCIFParser(QUIET=True).get_structure(structure_id, str(source))
    else:
        structure = PDBParser(QUIET=True, PERMISSIVE=True).get_structure(structure_id, str(source))
    try:
        model = list(structure)[model_index]
        chain = model[chain_id]
    except (IndexError, KeyError) as exc:
        raise ValueError(f"chain/model not found: {path} model={model_index} chain={chain_id}") from exc

    residues: list[Any] = []
    for residue in chain:
        hetflag, resseq, icode = residue.id
        if hetflag.strip() or residue.resname.strip().upper() not in THREE_TO_ONE:
            continue
        residues.append(residue)
    coords = np.full((len(residues), len(BACKBONE_ATOMS), 3), np.nan, dtype=float)
    sequence: list[str] = []
    residue_ids: list[tuple[str, int, str]] = []
    for index, residue in enumerate(residues):
        resname = residue.resname.strip().upper()
        sequence.append(THREE_TO_ONE[resname])
        _, resseq, icode = residue.id
        residue_ids.append((chain_id, int(resseq), str(icode).strip()))
        for atom_index, atom_name in enumerate(BACKBONE_ATOMS):
            if atom_name not in residue:
                continue
            atom = residue[atom_name]
            if atom.is_disordered():
                # Prefer the canonical blank/A conformer; otherwise choose the
                # highest-occupancy conformer deterministically.
                children = list(atom.child_dict.values())
                preferred = next((child for child in children if child.altloc in {" ", "A"}), None)
                atom = preferred or max(children, key=lambda child: (child.occupancy or 0.0, child.altloc))
            coords[index, atom_index] = np.asarray(atom.coord, dtype=float)
    return ParsedStructure(str(source), chain_id, "".join(sequence), coords, tuple(residue_ids))


def map_terminal_overlap(
    parent: ParsedStructure,
    mutant: ParsedStructure,
    *,
    min_coverage: float,
) -> tuple[ParsedStructure, ParsedStructure, dict[str, Any]]:
    """Crop continuous terminal coordinate differences using shared residue IDs."""
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage must be in (0, 1]")
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


def align_coordinates_to_reference(reference: np.ndarray, mobile: np.ndarray) -> np.ndarray:
    """Rigidly align ``mobile`` onto ``reference`` using shared finite atoms."""
    valid = np.isfinite(reference).all(axis=-1) & np.isfinite(mobile).all(axis=-1)
    ref = reference[valid]
    mob = mobile[valid]
    if len(ref) < 3:
        raise ValueError("at least three shared finite atoms are required for rigid alignment")
    ref_center = ref.mean(axis=0)
    mob_center = mob.mean(axis=0)
    covariance = (mob - mob_center).T @ (ref - ref_center)
    u, _, vh = np.linalg.svd(covariance)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return (mobile - mob_center) @ rotation + ref_center


def pair_parsed_structures(parent: ParsedStructure, mutant: ParsedStructure, pair_id: str) -> StructurePair:
    if parent.residue_ids != mutant.residue_ids:
        raise ValueError("parent and mutant residue identifiers do not match; explicit alignment is required")
    mutation_indices = tuple(i for i, (a, b) in enumerate(zip(parent.sequence, mutant.sequence)) if a != b)
    if not mutation_indices:
        raise ValueError("parent and mutant structures have no sequence substitution")
    # Experimental coordinate files use arbitrary global poses. Store the
    # mutant in the parent's frame so local-frame supervision represents
    # internal structural response rather than a file-level rigid transform.
    aligned_mutant_coords = align_coordinates_to_reference(parent.coords, mutant.coords)
    return StructurePair(
        pair_id=pair_id,
        parent_sequence=parent.sequence,
        mutant_sequence=mutant.sequence,
        parent_coords=parent.coords,
        mutant_coords=aligned_mutant_coords,
        mutation_indices=mutation_indices,
        atom_names=BACKBONE_ATOMS,
    )


def structure_pair_payload(pair: StructurePair) -> dict[str, Any]:
    return {
        "pair_id": pair.pair_id,
        "parent_sequence": pair.parent_sequence,
        "mutant_sequence": pair.mutant_sequence,
        "parent_coords": json_safe(pair.parent_coords.tolist()),
        "mutant_coords": json_safe(pair.mutant_coords.tolist()),
        "mutation_indices": list(pair.mutation_indices),
        "atom_names": list(pair.atom_names),
    }


@dataclass(frozen=True)
class StructurePair:
    """A residue-mapped parent/mutant pair in a shared atom representation.

    Coordinates are shaped ``(L, A, 3)``.  Missing atoms should be represented
    by NaN in both structures; the first atom is conventionally C-alpha.
    """

    pair_id: str
    parent_sequence: str
    mutant_sequence: str
    parent_coords: np.ndarray
    mutant_coords: np.ndarray
    mutation_indices: tuple[int, ...]
    atom_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.parent_sequence) != len(self.mutant_sequence):
            raise ValueError("parent and mutant sequences must have equal length")
        if self.parent_coords.shape != self.mutant_coords.shape:
            raise ValueError("parent and mutant coordinates must have equal shape")
        if self.parent_coords.ndim != 3 or self.parent_coords.shape[-1] != 3:
            raise ValueError("coordinates must have shape (length, atoms, 3)")
        if not self.atom_names:
            inferred = ("N", "CA", "C", "O") if self.parent_coords.shape[1] == 4 else ("CA",) if self.parent_coords.shape[1] == 1 else ()
            if not inferred:
                raise ValueError("atom_names are required unless coordinates contain 1 or 4 atoms")
            object.__setattr__(self, "atom_names", inferred)
        if len(self.atom_names) != self.parent_coords.shape[1]:
            raise ValueError("atom_names must match the coordinate atom axis")
        if "CA" not in self.atom_names:
            raise ValueError("atom_names must contain CA")
        if any(not isinstance(index, (int, np.integer)) or isinstance(index, bool) for index in self.mutation_indices):
            raise ValueError("mutation_indices must contain integers")
        if len(set(self.mutation_indices)) != len(self.mutation_indices):
            raise ValueError("mutation_indices must be unique")
        if any(index < 0 or index >= len(self.parent_sequence) for index in self.mutation_indices):
            raise ValueError("mutation_indices are out of bounds")
        expected = {i for i, (a, b) in enumerate(zip(self.parent_sequence, self.mutant_sequence)) if a != b}
        if set(self.mutation_indices) != expected:
            raise ValueError(f"mutation_indices {self.mutation_indices} do not match sequences {sorted(expected)}")

    @property
    def length(self) -> int:
        return len(self.parent_sequence)

    @property
    def mutation_mask(self) -> np.ndarray:
        mask = np.zeros(self.length, dtype=bool)
        mask[list(self.mutation_indices)] = True
        return mask

    @property
    def ca_atom_index(self) -> int:
        return self.atom_names.index("CA")

    @property
    def atom_mask(self) -> np.ndarray:
        return np.isfinite(self.parent_coords).all(axis=-1) & np.isfinite(self.mutant_coords).all(axis=-1)

    @classmethod
    def from_json(cls, path: str | Path) -> "StructurePair":
        payload = json.loads(Path(path).read_text())
        return cls(
            pair_id=payload["pair_id"],
            parent_sequence=payload["parent_sequence"],
            mutant_sequence=payload["mutant_sequence"],
            parent_coords=np.asarray(payload["parent_coords"], dtype=float),
            mutant_coords=np.asarray(payload["mutant_coords"], dtype=float),
            mutation_indices=tuple(payload["mutation_indices"]),
            atom_names=tuple(payload.get("atom_names", ())),
        )


@dataclass(frozen=True)
class PairRecord:
    """Auditable manifest row pointing to one experimental pair."""

    pair: StructurePair
    parent_id: str
    family_id: str
    split: str
    source_file: str = ""
    target_file: str = ""
    source_chain: str = ""
    target_chain: str = ""
    residue_map: tuple[Any, ...] = ()
    environment_metadata: dict[str, Any] = field(default_factory=dict)
    experiment_metadata: dict[str, Any] = field(default_factory=dict)
    label_source: str = "experimental"
    teacher_config_hash: str | None = None
    source_checksum: str | None = None
    target_checksum: str | None = None

    def __post_init__(self) -> None:
        if not self.parent_id:
            raise ValueError("parent_id must not be empty")
        if not self.family_id:
            raise ValueError("family_id must not be empty")
        if self.split not in {"train", "dev", "test"}:
            raise ValueError(f"invalid split: {self.split!r}")
        if self.residue_map and len(self.residue_map) != self.pair.length:
            raise ValueError("residue_map length must match pair sequence length")
        for label, checksum in (("source", self.source_checksum), ("target", self.target_checksum)):
            if checksum is not None and (len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum.lower())):
                raise ValueError(f"{label}_checksum must be a SHA-256 hexadecimal digest")

    def to_dict(self) -> dict[str, Any]:
        payload = structure_pair_payload(self.pair)
        payload.update(
            {
                "source_sequence": payload.pop("parent_sequence"),
                "target_sequence": payload.pop("mutant_sequence"),
                "parent_id": self.parent_id,
                "family_id": self.family_id,
                "split": self.split,
                "source_file": self.source_file,
                "target_file": self.target_file,
                "source_chain": self.source_chain,
                "target_chain": self.target_chain,
                "residue_map": [list(item) if isinstance(item, tuple) else item for item in self.residue_map],
                "environment_metadata": self.environment_metadata,
                "experiment_metadata": self.experiment_metadata,
                "label_source": self.label_source,
                "teacher_config_hash": self.teacher_config_hash,
                "source_checksum": self.source_checksum,
                "target_checksum": self.target_checksum,
            }
        )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PairRecord":
        parent_sequence = payload.get("source_sequence", payload.get("parent_sequence"))
        mutant_sequence = payload.get("target_sequence", payload.get("mutant_sequence"))
        if not isinstance(parent_sequence, str) or not isinstance(mutant_sequence, str):
            raise ValueError("manifest record requires string source_sequence and target_sequence")
        pair = StructurePair(
            pair_id=payload["pair_id"],
            parent_sequence=parent_sequence,
            mutant_sequence=mutant_sequence,
            parent_coords=np.asarray(payload["parent_coords"], dtype=float),
            mutant_coords=np.asarray(payload["mutant_coords"], dtype=float),
            mutation_indices=tuple(payload["mutation_indices"]),
            atom_names=tuple(payload.get("atom_names", ())),
        )
        return cls(
            pair=pair,
            parent_id=payload["parent_id"],
            family_id=payload["family_id"],
            split=payload["split"],
            source_file=payload.get("source_file", ""),
            target_file=payload.get("target_file", ""),
            source_chain=payload.get("source_chain", ""),
            target_chain=payload.get("target_chain", ""),
            residue_map=tuple(tuple(item) if isinstance(item, list) else item for item in payload.get("residue_map", ())),
            environment_metadata=dict(payload.get("environment_metadata", {})),
            experiment_metadata=dict(payload.get("experiment_metadata", {})),
            label_source=payload.get("label_source", "experimental"),
            teacher_config_hash=payload.get("teacher_config_hash"),
            source_checksum=payload.get("source_checksum"),
            target_checksum=payload.get("target_checksum"),
        )


def pair_record_from_structures(
    parent: ParsedStructure,
    mutant: ParsedStructure,
    *,
    pair_id: str,
    parent_id: str,
    family_id: str,
    split: str,
    label_source: str = "experimental",
    environment_metadata: dict[str, Any] | None = None,
    experiment_metadata: dict[str, Any] | None = None,
    compute_checksums: bool = True,
) -> PairRecord:
    pair = pair_parsed_structures(parent, mutant, pair_id)
    return PairRecord(
        pair=pair,
        parent_id=parent_id,
        family_id=family_id,
        split=split,
        source_file=parent.path,
        target_file=mutant.path,
        source_chain=parent.chain_id,
        target_chain=mutant.chain_id,
        residue_map=parent.residue_ids,
        environment_metadata=environment_metadata or {},
        experiment_metadata=experiment_metadata or {},
        label_source=label_source,
        source_checksum=file_sha256(parent.path) if compute_checksums else None,
        target_checksum=file_sha256(mutant.path) if compute_checksums else None,
    )
def load_manifest(path: str | Path) -> list[PairRecord]:
    """Load a development manifest from JSON, JSONL, or a JSON list.

    Parquet conversion is intentionally kept outside the core package so this
    prototype does not require a dataframe dependency.
    """
    source = Path(path)
    text = source.read_text()
    if source.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        rows = payload["records"] if isinstance(payload, dict) and "records" in payload else payload
    if not isinstance(rows, list):
        raise ValueError("manifest must contain a list of records")
    return [PairRecord.from_dict(row) for row in rows]


def write_manifest(records: Iterable[PairRecord], path: str | Path) -> None:
    rows = [record.to_dict() for record in records]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".jsonl":
        text = "\n".join(json.dumps(json_safe(row), allow_nan=False) for row in rows) + ("\n" if rows else "")
    else:
        text = json.dumps(json_safe({"records": rows}), indent=2, allow_nan=False) + "\n"
    destination.write_text(text)


def append_manifest(records: Iterable[PairRecord], path: str | Path) -> None:
    """Append records to an existing manifest, preserving its format."""
    destination = Path(path)
    existing = load_manifest(destination) if destination.exists() else []
    write_manifest([*existing, *records], destination)


def assign_group_splits(
    records: Iterable[PairRecord],
    *,
    seed: int = 0,
    train_fraction: float = 0.7,
    dev_fraction: float = 0.15,
    group_by: str = "family",
) -> list[PairRecord]:
    """Assign deterministic splits without separating related records.

    Groups are hashed rather than sampled from individual rows, so all pairs
    connected through a family or parent receive one split. The input order is
    kept.
    """
    if not 0.0 < train_fraction < 1.0 or not 0.0 <= dev_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1) and dev_fraction in [0, 1)")
    if train_fraction + dev_fraction >= 1.0:
        raise ValueError("train_fraction + dev_fraction must be below 1")
    if group_by not in {"family", "parent"}:
        raise ValueError("group_by must be 'family' or 'parent'")
    rows = list(records)
    pair_ids = [row.pair.pair_id for row in rows]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("records must not contain duplicate pair_id values")
    # Build connected components because a family-based split can otherwise
    # separate records that share a parent, and vice versa.
    parent: dict[str, str] = {}
    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
    for row in rows:
        union(f"family:{row.family_id}", f"parent:{row.parent_id}")
    row_groups = {
        row.pair.pair_id: find(f"family:{row.family_id}")
        for row in rows
    }
    groups = set(row_groups.values())
    ranked_groups = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest(),
    )
    group_count = len(ranked_groups)
    train_count = max(1, int(round(train_fraction * group_count))) if group_count else 0
    dev_count = max(1, int(round(dev_fraction * group_count))) if group_count >= 3 else 0
    if group_count >= 3:
        # Preserve one group for dev and test even when rounding the requested
        # fractions would otherwise consume the entire assignment budget.
        train_count = min(train_count, group_count - 2)
        dev_count = min(dev_count, group_count - train_count - 1)
    elif group_count and train_count >= group_count:
        train_count = group_count - 1 if group_count > 1 else 1
    assignments: dict[str, str] = {}
    for index, group in enumerate(ranked_groups):
        split = "train" if index < train_count else "dev" if index < train_count + dev_count else "test"
        assignments[group] = split
    return [
        replace(row, split=assignments[row_groups[row.pair.pair_id]])
        for row in rows
    ]


def verify_record_checksums(record: PairRecord) -> list[str]:
    """Return checksum errors without requiring source files for old records."""
    errors: list[str] = []
    for label, path, expected in (
        ("source", record.source_file, record.source_checksum),
        ("target", record.target_file, record.target_checksum),
    ):
        if not expected:
            continue
        if not path or not Path(path).exists():
            errors.append(f"{record.pair.pair_id}: {label} file missing for checksum verification")
            continue
        actual = file_sha256(path)
        if actual != expected:
            errors.append(f"{record.pair.pair_id}: {label} checksum mismatch")
    return errors


def validate_manifest(records: Iterable[PairRecord], *, max_mutations: int | None = None) -> list[str]:
    """Return protocol violations without mutating or dropping records."""
    if max_mutations is not None and max_mutations < 0:
        raise ValueError("max_mutations must be non-negative or None")
    rows = list(records)
    errors: list[str] = []
    seen_groups: dict[str, str] = {}
    seen_parents: dict[str, str] = {}
    seen_pairs: set[str] = set()
    for row in rows:
        if max_mutations is not None and len(row.pair.mutation_indices) > max_mutations:
            errors.append(
                f"{row.pair.pair_id}: {len(row.pair.mutation_indices)} mutations exceed max_mutations={max_mutations}"
            )
        if row.pair.pair_id in seen_pairs:
            errors.append(f"duplicate pair_id {row.pair.pair_id!r}")
        seen_pairs.add(row.pair.pair_id)
        if row.split not in {"train", "dev", "test"}:
            errors.append(f"{row.pair.pair_id}: invalid split {row.split!r}")
        previous = seen_groups.setdefault(row.family_id, row.split)
        if previous != row.split:
            errors.append(f"family {row.family_id!r} crosses {previous}/{row.split}")
        previous_parent = seen_parents.setdefault(row.parent_id, row.split)
        if previous_parent != row.split:
            errors.append(f"parent {row.parent_id!r} crosses {previous_parent}/{row.split}")
        if row.label_source not in {"experimental", "synthetic", "teacher"}:
            errors.append(f"{row.pair.pair_id}: invalid label_source {row.label_source!r}")
        if not row.pair.mutation_indices:
            errors.append(f"{row.pair.pair_id}: no sequence substitution found")
        if row.pair.parent_coords.shape[1] < 1:
            errors.append(f"{row.pair.pair_id}: missing coordinate atom axis")
    return errors
