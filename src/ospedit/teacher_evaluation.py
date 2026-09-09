"""Evaluate admitted teacher deltas against experimental structure changes."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np

from .data import PairRecord, json_safe
from .metrics import region_masks
from .student_data import target_local_delta
from .teacher_cache import TeacherCache


METRIC_NAMES = (
    "all_rmse", "mutation_rmse", "local_rmse", "transition_rmse", "remote_rmse",
    "all_translation_rmse", "mutation_translation_rmse", "local_translation_rmse",
    "transition_translation_rmse", "remote_translation_rmse",
    "all_rotation_rmse", "mutation_rotation_rmse", "local_rotation_rmse",
    "transition_rotation_rmse", "remote_rotation_rmse",
    "all_cosine", "mutation_cosine", "local_cosine", "transition_cosine", "remote_cosine",
)


def _rmse(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    values = predicted[mask] - target[mask]
    return float(np.sqrt(np.mean(values * values))) if len(values) else float("nan")


def _channel_rmse(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray, channels: slice) -> float:
    values = predicted[mask, channels] - target[mask, channels]
    return float(np.sqrt(np.mean(values * values))) if len(values) else float("nan")


def _cosine(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    left = predicted[mask].reshape(-1)
    right = target[mask].reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else float("nan")


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for name in METRIC_NAMES:
        values = np.asarray([row[name] for row in rows], dtype=float)
        summary[f"mean_{name}"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    return summary


def _macro_summary(rows: list[dict[str, Any]], group_key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[group_key]), []).append(row)
    group_rows = [
        {name: _mean_metrics(group)[f"mean_{name}"] for name in METRIC_NAMES}
        for group in groups.values()
    ]
    return {"groups": len(groups), **_mean_metrics(group_rows)}


def evaluate_teacher_cache(
    cache: TeacherCache,
    records: Iterable[PairRecord],
    noise_level: float,
    *,
    local_radius: float = 10.0,
    remote_min_distance: float = 15.0,
) -> dict[str, Any]:
    """Compare cached condition responses with experimental local frame deltas."""
    cached_deltas = cache.local_deltas(noise_level)
    report = evaluate_teacher_deltas(
        records,
        cached_deltas,
        local_radius=local_radius,
        remote_min_distance=remote_min_distance,
    )
    report["noise_level"] = float(noise_level)
    return report


def evaluate_teacher_combination(
    cache: TeacherCache,
    records: Iterable[PairRecord],
    noise_weights: Mapping[float, float],
    *,
    local_radius: float = 10.0,
    remote_min_distance: float = 15.0,
) -> dict[str, Any]:
    """Evaluate a weighted combination of cached condition responses."""
    if not noise_weights:
        raise ValueError("noise_weights must not be empty")
    if any(not math.isfinite(level) or not math.isfinite(weight) for level, weight in noise_weights.items()):
        raise ValueError("noise levels and weights must be finite")
    level_deltas = {float(level): cache.local_deltas(float(level)) for level in noise_weights}
    pair_ids = set.intersection(*(set(values) for values in level_deltas.values()))
    combined: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pair_id in pair_ids:
        arrays = [level_deltas[level][pair_id] for level in level_deltas]
        delta = sum(
            float(noise_weights[level]) * arrays[index][0]
            for index, level in enumerate(level_deltas)
        )
        valid = np.logical_and.reduce([array[1] for array in arrays])
        combined[pair_id] = np.asarray(delta), np.asarray(valid)
    report = evaluate_teacher_deltas(
        records,
        combined,
        local_radius=local_radius,
        remote_min_distance=remote_min_distance,
    )
    report["noise_weights"] = {str(level): float(weight) for level, weight in noise_weights.items()}
    return report


def evaluate_teacher_deltas(
    records: Iterable[PairRecord],
    cached_deltas: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    local_radius: float = 10.0,
    remote_min_distance: float = 15.0,
) -> dict[str, Any]:
    """Compare supplied teacher deltas with experimental local-frame deltas."""
    rows: list[dict[str, Any]] = []
    for record in records:
        pair = record.pair
        if pair.pair_id not in cached_deltas:
            raise KeyError(f"pair id is not present in teacher cache: {pair.pair_id}")
        teacher, teacher_valid = cached_deltas[pair.pair_id]
        target, target_valid = target_local_delta(pair)
        valid = teacher_valid & target_valid
        regions = region_masks(pair, local_radius, remote_min_distance)
        mutation = np.zeros(pair.length, dtype=bool)
        mutation[list(pair.mutation_indices)] = True
        row: dict[str, Any] = {
            "pair_id": pair.pair_id,
            "parent_id": record.parent_id,
            "family_id": record.family_id,
            "length": pair.length,
        }
        for name, mask in (
            ("all", valid),
            ("mutation", valid & mutation),
            ("local", valid & regions["local"]),
            ("transition", valid & regions["transition"]),
            ("remote", valid & regions["remote"]),
        ):
            row[f"{name}_rmse"] = _rmse(teacher, target, mask)
            row[f"{name}_translation_rmse"] = _channel_rmse(teacher, target, mask, slice(0, 3))
            row[f"{name}_rotation_rmse"] = _channel_rmse(teacher, target, mask, slice(3, 6))
            row[f"{name}_cosine"] = _cosine(teacher, target, mask)
            row[f"{name}_residues"] = int(mask.sum())
        rows.append(row)

    return json_safe({
        "records": rows,
        "summary": _mean_metrics(rows),
        "family_macro_summary": _macro_summary(rows, "family_id"),
        "parent_macro_summary": _macro_summary(rows, "parent_id"),
    })


__all__ = ["evaluate_teacher_cache", "evaluate_teacher_combination", "evaluate_teacher_deltas"]
