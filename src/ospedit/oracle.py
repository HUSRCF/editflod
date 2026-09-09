from __future__ import annotations

import numpy as np

from .data import StructurePair
from .student_data import target_local_delta
from .student_inference import apply_student_delta


def oracle_local_delta(
    pair: StructurePair,
    *,
    translation_scale: float = 1.0,
    rotation_scale: float = 1.0,
    max_normalized_delta: float | None = None,
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
    delta[~valid] = 0.0
    return delta, valid


def oracle_prediction(
    pair: StructurePair,
    *,
    translation_scale: float = 1.0,
    rotation_scale: float = 1.0,
    max_normalized_delta: float | None = None,
) -> np.ndarray:
    """Apply the exact representable target delta to the parent structure."""
    delta, _ = oracle_local_delta(
        pair,
        translation_scale=translation_scale,
        rotation_scale=rotation_scale,
        max_normalized_delta=max_normalized_delta,
    )
    return apply_student_delta(
        pair,
        delta,
        translation_scale=translation_scale,
        rotation_scale=rotation_scale,
    )
