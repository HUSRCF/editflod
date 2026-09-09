from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .data import StructurePair
from .geometry import so3_log
from .models import EndpointModel
from .models import _endpoint
from .noise import make_shared_noise


def assess_teacher_admission(
    report: dict[str, Any],
    *,
    min_mutation_response: float = 1e-5,
    repeat_error: dict[str, float] | None = None,
    max_repeat_error: float | None = None,
) -> dict[str, Any]:
    """Apply explicit engineering gates before using endpoint outputs as labels."""
    reasons: list[str] = []
    usable_levels = []
    for level in report.get("levels", []):
        mutation_signal = float(level["mutation_translation_response"]) + float(level["mutation_rotation_response"])
        if np.isfinite(mutation_signal):
            usable_levels.append(mutation_signal)
    if not usable_levels:
        reasons.append("no finite mutation response")
    elif max(usable_levels) < min_mutation_response:
        reasons.append("mutation response is below admission threshold")
    if repeat_error is not None and max_repeat_error is not None:
        translation_error = float(repeat_error.get("max_translation_error", float("nan")))
        rotation_error = float(repeat_error.get("max_rotation_error", float("nan")))
        if not np.isfinite(translation_error) or not np.isfinite(rotation_error):
            reasons.append("condition response is not reproducible")
        elif max(translation_error, rotation_error) > max_repeat_error:
            reasons.append("condition response repeat error exceeds admission threshold")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "max_mutation_response": max(usable_levels) if usable_levels else float("nan"),
        "levels_checked": len(report.get("levels", [])),
        "max_repeat_error": (
            max(
                float(repeat_error.get("max_translation_error", float("nan"))),
                float(repeat_error.get("max_rotation_error", float("nan"))),
            )
            if repeat_error is not None
            else None
        ),
    }


def condition_response_repeat_error(
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, float]:
    """Compare two condition-response diagnostics level by level.

    The helper intentionally compares the reported response summaries rather
    than raw endpoint coordinates, so it remains usable across endpoint
    implementations with different geometric parameterizations.
    """
    first_levels = first.get("levels", [])
    second_levels = second.get("levels", [])
    if len(first_levels) != len(second_levels):
        return {"max_translation_error": float("inf"), "max_rotation_error": float("inf")}
    translation_keys = (
        "mean_translation_response",
        "mutation_translation_response",
        "remote_translation_response",
    )
    rotation_keys = (
        "mean_rotation_response",
        "mutation_rotation_response",
        "remote_rotation_response",
    )

    def max_error(keys: tuple[str, ...]) -> float:
        errors: list[float] = []
        for first_level, second_level in zip(first_levels, second_levels):
            for key in keys:
                left = float(first_level.get(key, float("nan")))
                right = float(second_level.get(key, float("nan")))
                if not np.isfinite(left) or not np.isfinite(right):
                    return float("inf")
                errors.append(abs(left - right))
        return max(errors, default=0.0)

    return {
        "max_translation_error": max_error(translation_keys),
        "max_rotation_error": max_error(rotation_keys),
    }


def _validate_endpoint(rotations: np.ndarray, origins: np.ndarray, length: int) -> None:
    if rotations.shape != (length, 3, 3):
        raise ValueError(f"endpoint rotations must have shape {(length, 3, 3)}, got {rotations.shape}")
    if origins.shape != (length, 3):
        raise ValueError(f"endpoint origins must have shape {(length, 3)}, got {origins.shape}")
    if not np.isfinite(rotations).all() or not np.isfinite(origins).all():
        raise ValueError("endpoint output contains non-finite values")


def condition_response_diagnostic(
    model: EndpointModel,
    pair: StructurePair,
    noise_levels: Iterable[float],
) -> dict[str, Any]:
    """Measure source/target endpoint separation under shared input conditions.

    This is a mechanism diagnostic, not a structural accuracy metric. A useful
    teacher should expose a reproducible nonzero response for real edits while
    an identical source/target sequence should produce zero separation.
    """
    mutation_mask = pair.mutation_mask
    remote_mask = ~mutation_mask
    levels: list[dict[str, float | int]] = []
    for noise_level in noise_levels:
        noise_state = make_shared_noise(pair.parent_coords.shape, noise_level, seed=0)
        target_rotations, target_origins = _endpoint(model, pair.parent_coords, pair.mutant_sequence, noise_level, noise_state)
        source_rotations, source_origins = _endpoint(model, pair.parent_coords, pair.parent_sequence, noise_level, noise_state)
        _validate_endpoint(target_rotations, target_origins, pair.length)
        _validate_endpoint(source_rotations, source_origins, pair.length)
        translation_norms = np.linalg.norm(target_origins - source_origins, axis=-1)
        rotation_norms = np.asarray(
            [np.linalg.norm(so3_log(source_rotations[i].T @ target_rotations[i])) for i in range(pair.length)]
        )
        finite = np.isfinite(translation_norms) & np.isfinite(rotation_norms)
        if not finite.any():
            raise ValueError("endpoint condition response has no finite residues")

        def mean_for(mask: np.ndarray) -> float:
            selected = finite & mask
            return float(np.mean(translation_norms[selected])) if selected.any() else float("nan")

        def rotation_mean_for(mask: np.ndarray) -> float:
            selected = finite & mask
            return float(np.mean(rotation_norms[selected])) if selected.any() else float("nan")

        levels.append(
            {
                "noise_level": float(noise_level),
                "finite_residues": int(finite.sum()),
                "mean_translation_response": mean_for(np.ones(pair.length, dtype=bool)),
                "mean_rotation_response": rotation_mean_for(np.ones(pair.length, dtype=bool)),
                "mutation_translation_response": mean_for(mutation_mask),
                "remote_translation_response": mean_for(remote_mask),
                "mutation_rotation_response": rotation_mean_for(mutation_mask),
                "remote_rotation_response": rotation_mean_for(remote_mask),
            }
        )
    return {"pair_id": pair.pair_id, "mutation_count": len(pair.mutation_indices), "levels": levels}
