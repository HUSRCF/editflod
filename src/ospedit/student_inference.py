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
from .student_data import edit_geometry_features, parent_context_features, parent_local_features, parent_residue_mask, parent_spatial_graph


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
            features = parent_context_features(pair, include_geometry=self.include_geometry)
            features.setflags(write=False)
            self._values[key] = features
        else:
            self.hits += 1
            self._values.move_to_end(key)
        if self.max_entries is not None:
            self._values.move_to_end(key)
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)
        cached = self._values[key]
        if not self.include_geometry:
            return cached
        features = np.array(cached, copy=True)
        features[:, -2:] = edit_geometry_features(pair)
        features.setflags(write=False)
        return features

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
    update_scale: float = 1.0,
) -> np.ndarray:
    """Apply one student's ``(L, 6)`` local-frame output to parent atoms."""
    delta = np.asarray(delta, dtype=float)
    if delta.shape != (pair.length, 6):
        raise ValueError(f"student delta must have shape {(pair.length, 6)}, got {delta.shape}")
    if not np.isfinite(delta).all():
        raise ValueError("student delta contains non-finite values")
    if not np.isfinite(update_scale) or update_scale < 0:
        raise ValueError("update_scale must be finite and non-negative")
    delta = delta * update_scale
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
    include_spatial_graph: bool = False,
    spatial_neighbors: int = 24,
    update_scale: float = 1.0,
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
            parameters = inspect.signature(model.forward).parameters
            kwargs: dict[str, Any] = {}
            if "residue_mask" in parameters:
                kwargs["residue_mask"] = residue_mask
            if "edge_features" in parameters:
                if not include_spatial_graph:
                    raise ValueError("model requires include_spatial_graph=True")
                edge_features, edge_mask = parent_spatial_graph(
                    pair, spatial_neighbors=spatial_neighbors
                )
                kwargs["edge_features"] = torch.as_tensor(
                    edge_features[None], dtype=torch.float32, device=device
                )
                kwargs["edge_mask"] = torch.as_tensor(
                    edge_mask[None], dtype=torch.bool, device=device
                )
            prediction = model(parent, edit, **kwargs)
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
        update_scale,
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
    include_spatial_graph: bool = False,
    spatial_neighbors: int = 24,
    update_scale: float = 1.0,
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
    use_geometry = parent_cache.include_geometry if parent_cache is not None else include_geometry
    feature_rows = [
        parent_cache.get(pair) if parent_cache is not None else parent_local_features(pair, include_geometry=use_geometry)
        for pair in pairs
    ]
    max_length = max(pair.length for pair in pairs)
    parent_values = np.zeros((len(pairs), max_length, feature_rows[0].shape[-1]), dtype=np.float32)
    edit_values = np.zeros((len(pairs), max_length, 41), dtype=np.float32)
    mask_values = np.zeros((len(pairs), max_length), dtype=bool)
    edge_values = np.zeros((len(pairs), max_length, max_length, 16), dtype=np.float32) if include_spatial_graph else None
    edge_masks = np.zeros((len(pairs), max_length, max_length), dtype=bool) if include_spatial_graph else None
    for index, (pair, features) in enumerate(zip(pairs, feature_rows, strict=True)):
        parent_values[index, :pair.length] = features
        edit_values[index, :pair.length] = encode_edit_features(
            [pair.parent_sequence], [pair.mutant_sequence]
        )[0].numpy()
        mask_values[index, :pair.length] = parent_residue_mask(pair)
        if edge_values is not None and edge_masks is not None:
            pair_edges, pair_edge_mask = parent_spatial_graph(
                pair, spatial_neighbors=spatial_neighbors
            )
            edge_values[index, :pair.length, :pair.length] = pair_edges
            edge_masks[index, :pair.length, :pair.length] = pair_edge_mask
    parent = torch.as_tensor(parent_values, dtype=torch.float32, device=device)
    edit = torch.as_tensor(edit_values, dtype=torch.float32, device=device)
    residue_mask = torch.as_tensor(mask_values, dtype=torch.bool, device=device)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            parameters = inspect.signature(model.forward).parameters
            kwargs: dict[str, Any] = {}
            if "residue_mask" in parameters:
                kwargs["residue_mask"] = residue_mask
            if "edge_features" in parameters:
                if edge_values is None or edge_masks is None:
                    raise ValueError("model requires include_spatial_graph=True")
                kwargs["edge_features"] = torch.as_tensor(
                    edge_values, dtype=torch.float32, device=device
                )
                kwargs["edge_mask"] = torch.as_tensor(
                    edge_masks, dtype=torch.bool, device=device
                )
            prediction = model(parent, edit, **kwargs)
    finally:
        if was_training:
            model.train()
    if prediction.ndim != 3 or prediction.shape[0] != len(pairs) or prediction.shape[-1] != 6:
        raise ValueError("student batch output must have shape (batch, length, 6)")
    return [
        apply_student_delta(pair, prediction[index, : pair.length].detach().cpu().numpy(), translation_scale, rotation_scale, update_scale)
        for index, pair in enumerate(pairs)
    ]
