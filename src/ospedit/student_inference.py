from __future__ import annotations

from typing import Any
import inspect
import hashlib
from dataclasses import dataclass, field
from collections import OrderedDict

import numpy as np

from .data import StructurePair
from .geometry import apply_local_frame_update
from .student import encode_edit_features
from .student_data import parent_local_features, parent_residue_mask
from .student_data import PairDataset, collate_pair_records


@dataclass
class ParentContextCache:
    """Cache parent-local features for many mutations of the same parent."""

    max_entries: int | None = None
    include_geometry: bool = False
    _values: OrderedDict[str, np.ndarray] = field(default_factory=OrderedDict)
    hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        if self.max_entries is not None and self.max_entries <= 0:
            raise ValueError("max_entries must be positive or None")

    def _key(self, pair: StructurePair) -> str:
        digest = hashlib.sha256()
        digest.update(pair.parent_sequence.encode("ascii"))
        digest.update("\0".join(pair.atom_names).encode("ascii"))
        digest.update(np.ascontiguousarray(pair.parent_coords).tobytes())
        return digest.hexdigest()

    def get(self, pair: StructurePair) -> np.ndarray:
        key = self._key(pair)
        if key not in self._values:
            self.misses += 1
            features = parent_local_features(pair, include_geometry=self.include_geometry)
            features.setflags(write=False)
            self._values[key] = features
        else:
            self.hits += 1
            self._values.move_to_end(key)
        if self.max_entries is not None:
            self._values.move_to_end(key)
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)
        return self._values[key]

    def clear(self) -> None:
        self._values.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._values)


def apply_student_delta(
    pair: StructurePair,
    delta: np.ndarray,
    translation_scale: float = 1.0,
    rotation_scale: float = 1.0,
) -> np.ndarray:
    """Apply one student's ``(L, 6)`` local-frame output to parent atoms."""
    delta = np.asarray(delta, dtype=float)
    if delta.shape != (pair.length, 6):
        raise ValueError(f"student delta must have shape {(pair.length, 6)}, got {delta.shape}")
    if not np.isfinite(delta).all():
        raise ValueError("student delta contains non-finite values")
    return apply_local_frame_update(
        pair.parent_coords,
        pair.atom_names,
        delta[:, :3],
        delta[:, 3:],
        translation_scale,
        rotation_scale,
    )


def predict_student(
    model: Any,
    pair: StructurePair,
    *,
    device: str = "cpu",
    translation_scale: float = 1.0,
    rotation_scale: float = 1.0,
    parent_cache: ParentContextCache | None = None,
    include_geometry: bool = False,
) -> np.ndarray:
    """Run one student forward pass and return edited backbone coordinates."""
    if pair.parent_sequence == pair.mutant_sequence:
        return pair.parent_coords.copy()
    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise ImportError("student inference requires torch; install ospedit[torch]") from error
    parent_features = (
        np.array(parent_cache.get(pair), copy=True)
        if parent_cache is not None
        else parent_local_features(pair, include_geometry=include_geometry)
    )
    parent = torch.as_tensor(parent_features[None], dtype=torch.float32, device=device)
    edit = encode_edit_features([pair.parent_sequence], [pair.mutant_sequence]).to(device=device)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            residue_mask = torch.as_tensor(parent_residue_mask(pair)[None], dtype=torch.bool, device=device)
            if "residue_mask" in inspect.signature(model.forward).parameters:
                prediction = model(parent, edit, residue_mask=residue_mask)
            else:
                prediction = model(parent, edit)
    finally:
        if was_training:
            model.train()
    if prediction.shape != (1, pair.length, 6):
        raise ValueError(f"student output must have shape {(1, pair.length, 6)}, got {tuple(prediction.shape)}")
    return apply_student_delta(
        pair,
        prediction[0].detach().cpu().numpy(),
        translation_scale,
        rotation_scale,
    )


def predict_student_batch(
    model: Any,
    pairs: list[StructurePair],
    *,
    device: str = "cpu",
    translation_scale: float = 1.0,
    rotation_scale: float = 1.0,
    parent_cache: ParentContextCache | None = None,
    include_geometry: bool = False,
) -> list[np.ndarray]:
    """Run one padded student forward for multiple residue-mapped pairs."""
    if not pairs:
        raise ValueError("pairs must not be empty")
    if all(pair.parent_sequence == pair.mutant_sequence for pair in pairs):
        return [pair.parent_coords.copy() for pair in pairs]
    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise ImportError("student inference requires torch; install ospedit[torch]") from error
    if parent_cache is None:
        dataset = PairDataset([_record_for_pair(pair) for pair in pairs], include_geometry=include_geometry)
        batch = collate_pair_records([dataset[index] for index in range(len(dataset))])
    else:
        # Keep edit/target handling identical while substituting cached parent
        # features for each candidate in the padded batch.
        dataset = PairDataset([_record_for_pair(pair) for pair in pairs], include_geometry=parent_cache.include_geometry)
        items = [dataset[index] for index in range(len(dataset))]
        for item, pair in zip(items, pairs, strict=True):
            item["parent_features"] = parent_cache.get(pair)
            item["residue_mask"] = parent_residue_mask(pair).astype(np.float32)
        batch = collate_pair_records(items)
    parent = torch.as_tensor(batch["parent_features"], dtype=torch.float32, device=device)
    edit = torch.as_tensor(batch["edit_features"], dtype=torch.float32, device=device)
    residue_mask = torch.as_tensor(batch["residue_mask"], dtype=torch.bool, device=device)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            if "residue_mask" in inspect.signature(model.forward).parameters:
                prediction = model(parent, edit, residue_mask=residue_mask)
            else:
                prediction = model(parent, edit)
    finally:
        if was_training:
            model.train()
    if prediction.ndim != 3 or prediction.shape[0] != len(pairs) or prediction.shape[-1] != 6:
        raise ValueError("student batch output must have shape (batch, length, 6)")
    return [
        apply_student_delta(pair, prediction[index, : pair.length].detach().cpu().numpy(), translation_scale, rotation_scale)
        for index, pair in enumerate(pairs)
    ]


def _record_for_pair(pair: StructurePair):
    """Create an in-memory PairRecord without inventing source metadata."""
    from .data import PairRecord

    return PairRecord(pair, parent_id=pair.pair_id, family_id=pair.pair_id, split="dev")
