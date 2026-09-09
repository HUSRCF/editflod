from __future__ import annotations

import numpy as np

from .data import StructurePair
from .student_data import mutation_localization_weights, target_local_delta
from .student_inference import apply_student_delta


def oracle_local_delta(
    pair: StructurePair,
    *,
    translation_scale: float = 1.0,
    rotation_scale: float = 1.0,
    max_normalized_delta: float | None = None,
    localization_radius: float | None = None,
    localization_transition: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the experimental local-frame target, optionally component-clipped.

    Clipping matches ``ParentEditStudent``'s component-wise tanh output bound.
    """
    if max_normalized_delta is not None and max_normalized_delta <= 0:
        raise ValueError("max_normalized_delta must be positive or None")
    delta, valid = target_local_delta(
        pair,
        translation_scale=translation_scale,
        rotation_scale=rotation_scale,
    )
    if max_normalized_delta is not None:
        delta = np.clip(delta, -max_normalized_delta, max_normalized_delta)
    if localization_radius is not None:
        delta *= mutation_localization_weights(
            pair,
            radius=localization_radius,
            transition=localization_transition,
        )[:, None]
    delta[~valid] = 0.0
    return delta, valid


def oracle_prediction(
    pair: StructurePair,
    *,
    translation_scale: float = 1.0,
    rotation_scale: float = 1.0,
    max_normalized_delta: float | None = None,
    localization_radius: float | None = None,
    localization_transition: float = 5.0,
) -> np.ndarray:
    """Apply the exact representable target delta to the parent structure."""
    delta, _ = oracle_local_delta(
        pair,
        translation_scale=translation_scale,
        rotation_scale=rotation_scale,
        max_normalized_delta=max_normalized_delta,
        localization_radius=localization_radius,
        localization_transition=localization_transition,
    )
    return apply_student_delta(
        pair,
        delta,
        translation_scale=translation_scale,
        rotation_scale=rotation_scale,
    )
