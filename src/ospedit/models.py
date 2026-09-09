from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Protocol

import numpy as np

from .data import StructurePair
from .geometry import apply_local_frame_update, local_frame_difference, residue_frames
from .noise import SharedNoise, make_shared_noise
from .student_inference import ParentContextCache, predict_student, predict_student_batch


class FieldModel(Protocol):
    """底座模型的最小适配协议。

    ``field`` 返回与坐标同形状的几何更新场。真实 flow/score 模型可在
    适配器中完成序列编码、刚体切空间转换和模型输出参数化转换。
    """

    def field(self, coords: np.ndarray, sequence: str, noise_level: float, **kwargs: object) -> np.ndarray:
        ...


class EndpointModel(Protocol):
    """Adapter boundary for a sequence-conditioned frame predictor."""

    def endpoint(self, coords: np.ndarray, sequence: str, noise_level: float, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        """Return per-residue rotation matrices and origins in global coordinates."""
        ...


def _endpoint(
    model: EndpointModel,
    coords: np.ndarray,
    sequence: str,
    noise_level: float,
    noise_state: SharedNoise,
) -> tuple[np.ndarray, np.ndarray]:
    """Call legacy adapters or explicit shared-noise adapters."""
    parameters = inspect.signature(model.endpoint).parameters
    if "noise_state" in parameters:
        return model.endpoint(coords, sequence, noise_level, noise_state=noise_state)
    return model.endpoint(coords, sequence, noise_level)


def _field(
    model: FieldModel,
    coords: np.ndarray,
    sequence: str,
    noise_level: float,
    noise_state: SharedNoise,
) -> np.ndarray:
    """Call field adapters with shared noise when they explicitly support it."""
    parameters = inspect.signature(model.field).parameters
    if "noise_state" in parameters:
        return model.field(coords, sequence, noise_level, noise_state=noise_state)
    return model.field(coords, sequence, noise_level)


class Editor(Protocol):
    def predict(self, pair: StructurePair) -> np.ndarray:
        ...


@dataclass
class StudentEditor:
    """Adapter exposing a trained ParentEditStudent as the common Editor API."""

    model: object
    device: str = "cpu"
    translation_scale: float = 1.0
    rotation_scale: float = 1.0
    parent_cache: ParentContextCache | None = None
    include_geometry: bool = False

    def __post_init__(self) -> None:
        if self.parent_cache is not None and self.parent_cache.include_geometry != self.include_geometry:
            raise ValueError("StudentEditor geometry setting must match parent_cache.include_geometry")

    def predict(self, pair: StructurePair) -> np.ndarray:
        return predict_student(
            self.model,
            pair,
            device=self.device,
            translation_scale=self.translation_scale,
            rotation_scale=self.rotation_scale,
            parent_cache=self.parent_cache,
            include_geometry=self.include_geometry,
        )

    def predict_batch(self, pairs: list[StructurePair]) -> list[np.ndarray]:
        return predict_student_batch(
            self.model,
            pairs,
            device=self.device,
            translation_scale=self.translation_scale,
            rotation_scale=self.rotation_scale,
            parent_cache=self.parent_cache,
            include_geometry=self.include_geometry,
        )


@dataclass
class CopyParentEditor:
    def predict(self, pair: StructurePair) -> np.ndarray:
        return pair.parent_coords.copy()


@dataclass
class TargetUpdateEditor:
    model: FieldModel
    step_size: float = 1.0
    noise_level: float = 0.0

    def predict(self, pair: StructurePair) -> np.ndarray:
        if pair.parent_sequence == pair.mutant_sequence:
            return pair.parent_coords.copy()
        noise_state = make_shared_noise(pair.parent_coords.shape, self.noise_level, seed=0)
        update = _field(self.model, pair.parent_coords, pair.mutant_sequence, self.noise_level, noise_state)
        return pair.parent_coords + self.step_size * update


@dataclass
class ConditionalDifferenceEditor:
    model: FieldModel
    step_size: float = 1.0
    noise_level: float = 0.0
    source_weight: float = 1.0

    def predict(self, pair: StructurePair) -> np.ndarray:
        if pair.parent_sequence == pair.mutant_sequence:
            return pair.parent_coords.copy()
        noise_state = make_shared_noise(pair.parent_coords.shape, self.noise_level, seed=0)
        target = _field(self.model, pair.parent_coords, pair.mutant_sequence, self.noise_level, noise_state)
        source = _field(self.model, pair.parent_coords, pair.parent_sequence, self.noise_level, noise_state)
        update = target - self.source_weight * source
        return pair.parent_coords + self.step_size * update


def _local_frame_difference_update(
    model: EndpointModel,
    pair: StructurePair,
    noise_levels: tuple[float, ...],
    weights: tuple[float, ...],
    translation_scale: float,
    rotation_scale: float,
    shared_noise: bool = True,
    seed_offset: int = 0,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    parent_rotations, parent_origins, valid = residue_frames(pair.parent_coords, pair.atom_names)
    translations = np.zeros((pair.length, 3), dtype=float)
    rotations = np.zeros((pair.length, 3), dtype=float)
    for noise_level, weight in zip(noise_levels, weights, strict=True):
        target_noise = make_shared_noise(pair.parent_coords.shape, noise_level, seed=seed_offset)
        source_noise = target_noise if shared_noise else make_shared_noise(pair.parent_coords.shape, noise_level, seed=seed_offset + 1)
        target_rotations, target_origins = _endpoint(model, pair.parent_coords, pair.mutant_sequence, noise_level, target_noise)
        source_rotations, source_origins = _endpoint(model, pair.parent_coords, pair.parent_sequence, noise_level, source_noise)
        level_translation, level_rotation = local_frame_difference(
            parent_rotations,
            parent_origins,
            target_rotations,
            target_origins,
            source_rotations,
            source_origins,
        )
        translations += weight * level_translation
        rotations += weight * level_rotation
    translations[~valid] = 0.0
    rotations[~valid] = 0.0
    if active_mask is not None:
        mask = np.asarray(active_mask, dtype=bool)
        if mask.shape != (pair.length,):
            raise ValueError(f"active_mask must have shape {(pair.length,)}, got {mask.shape}")
        translations[~mask] = 0.0
        rotations[~mask] = 0.0
    return apply_local_frame_update(
        pair.parent_coords,
        pair.atom_names,
        translations,
        rotations,
        translation_scale,
        rotation_scale,
    )


@dataclass
class LocalFrameDifferenceEditor:
    """Source-target endpoint difference expressed in parent residue charts."""

    model: EndpointModel
    translation_scale: float = 1.0
    rotation_scale: float = 1.0
    noise_level: float = 0.0

    def predict(self, pair: StructurePair) -> np.ndarray:
        if pair.parent_sequence == pair.mutant_sequence:
            return pair.parent_coords.copy()
        return _local_frame_difference_update(
            self.model, pair, (self.noise_level,), (1.0,), self.translation_scale, self.rotation_scale
        )


@dataclass
class MutationNeighborhoodDifferenceEditor(LocalFrameDifferenceEditor):
    """Local-frame difference restricted to a parent-structure neighborhood."""

    radius: float = 10.0

    def predict(self, pair: StructurePair) -> np.ndarray:
        if self.radius < 0.0:
            raise ValueError("radius must be non-negative")
        if pair.parent_sequence == pair.mutant_sequence:
            return pair.parent_coords.copy()
        ca_index = pair.atom_names.index("CA")
        ca = pair.parent_coords[:, ca_index]
        mutation_ca = ca[list(pair.mutation_indices)]
        active = np.isfinite(ca).all(axis=-1) & np.isfinite(mutation_ca).all(axis=-1).all()
        if len(mutation_ca):
            distances = np.linalg.norm(ca[:, None, :] - mutation_ca[None, :, :], axis=-1)
            active = np.isfinite(distances).any(axis=1) & (np.nanmin(distances, axis=1) <= self.radius)
        return _local_frame_difference_update(
            self.model, pair, (self.noise_level,), (1.0,), self.translation_scale,
            self.rotation_scale, active_mask=active,
        )


@dataclass
class MultiNoiseLocalFrameDifferenceEditor:
    """Shared-input multi-noise conditional-difference editor."""

    model: EndpointModel
    noise_levels: tuple[float, ...]
    weights: tuple[float, ...] | None = None
    translation_scale: float = 1.0
    rotation_scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.noise_levels:
            raise ValueError("noise_levels must not be empty")
        if self.weights is None:
            object.__setattr__(self, "weights", tuple(1.0 for _ in self.noise_levels))
        weights = self.weights
        if weights is None:
            raise ValueError("weights must be initialized")
        if len(weights) != len(self.noise_levels):
            raise ValueError("weights and noise_levels must have equal length")

    def predict(self, pair: StructurePair) -> np.ndarray:
        if pair.parent_sequence == pair.mutant_sequence:
            return pair.parent_coords.copy()
        return _local_frame_difference_update(
            self.model,
            pair,
            tuple(self.noise_levels),
            tuple(self.weights or ()),
            self.translation_scale,
            self.rotation_scale,
        )


@dataclass
class IndependentNoiseLocalFrameDifferenceEditor:
    """Control editor using independent source/target noise realizations."""

    model: EndpointModel
    noise_level: float = 0.0
    translation_scale: float = 1.0
    rotation_scale: float = 1.0

    def predict(self, pair: StructurePair) -> np.ndarray:
        if pair.parent_sequence == pair.mutant_sequence:
            return pair.parent_coords.copy()
        return _local_frame_difference_update(
            self.model,
            pair,
            (self.noise_level,),
            (1.0,),
            self.translation_scale,
            self.rotation_scale,
            shared_noise=False,
        )


@dataclass
class RepeatedSingleNoiseLocalFrameDifferenceEditor:
    """Matched-budget control repeating one noise level with shared branches."""

    model: EndpointModel
    noise_level: float = 0.0
    repeats: int = 2
    translation_scale: float = 1.0
    rotation_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.repeats <= 0:
            raise ValueError("repeats must be positive")

    def predict(self, pair: StructurePair) -> np.ndarray:
        if pair.parent_sequence == pair.mutant_sequence:
            return pair.parent_coords.copy()
        predictions = [
            _local_frame_difference_update(
                self.model,
                pair,
                (self.noise_level,),
                (1.0,),
                self.translation_scale,
                self.rotation_scale,
                seed_offset=2 * repeat,
            )
            for repeat in range(self.repeats)
        ]
        return np.mean(predictions, axis=0)


class ZeroFieldModel:
    """Useful smoke-test model and explicit copy-parent equivalent."""

    def field(self, coords: np.ndarray, sequence: str, noise_level: float) -> np.ndarray:
        return np.zeros_like(coords)


def build_editor(name: str, model: Any | None = None) -> Editor:
    """Build only model-independent baselines; external models use adapters."""
    if name == "copy_source_backbone":
        return CopyParentEditor()
    if name == "target_only_partial_denoising":
        if model is None:
            raise ValueError("target_only_partial_denoising requires a FieldModel")
        return TargetUpdateEditor(model)
    if name == "single_noise_conditional_difference":
        if model is None:
            raise ValueError("single_noise_conditional_difference requires a FieldModel")
        return ConditionalDifferenceEditor(model)
    if name == "single_noise_local_frame_difference":
        if model is None:
            raise ValueError("single_noise_local_frame_difference requires an EndpointModel")
        return LocalFrameDifferenceEditor(model)
    if name == "two_noise_local_frame_difference":
        if model is None:
            raise ValueError("two_noise_local_frame_difference requires an EndpointModel")
        return MultiNoiseLocalFrameDifferenceEditor(model, (0.25, 0.75), (0.5, 0.5))
    if name == "independent_noise_local_frame_difference":
        if model is None:
            raise ValueError("independent_noise_local_frame_difference requires an EndpointModel")
        return IndependentNoiseLocalFrameDifferenceEditor(model)
    if name == "repeated_single_noise_local_frame_difference":
        if model is None:
            raise ValueError("repeated_single_noise_local_frame_difference requires an EndpointModel")
        return RepeatedSingleNoiseLocalFrameDifferenceEditor(model, noise_level=0.5, repeats=2)
    raise ValueError(f"unknown or external baseline: {name}")
