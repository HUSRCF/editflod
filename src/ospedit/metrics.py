from __future__ import annotations

import numpy as np

from .data import StructurePair


def _flat_valid(coords_a: np.ndarray, coords_b: np.ndarray, residue_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(coords_a).all(axis=-1) & np.isfinite(coords_b).all(axis=-1)
    if residue_mask is not None:
        valid &= residue_mask[:, None]
    return coords_a[valid], coords_b[valid]


def kabsch_align(reference: np.ndarray, mobile: np.ndarray) -> np.ndarray:
    """Align mobile onto reference without changing either input."""
    ref, mob = _flat_valid(reference, mobile)
    if len(ref) < 3:
        return mobile.copy()
    ref_center = ref.mean(axis=0)
    mob_center = mob.mean(axis=0)
    covariance = (mob - mob_center).T @ (ref - ref_center)
    u, _, vh = np.linalg.svd(covariance)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return (mobile - mob_center) @ rotation + ref_center


def _rmsd(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    aa = kabsch_align(a, b)
    x, y = _flat_valid(a, aa, mask)
    return float(np.sqrt(np.mean((x - y) ** 2))) if len(x) else float("nan")


def _direct_rmsd(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """RMSD in the supplied coordinate frame, without a new rigid fit."""
    x, y = _flat_valid(a, b, mask)
    return float(np.sqrt(np.mean((x - y) ** 2))) if len(x) else float("nan")


def _distance_matrix(coords: np.ndarray, ca_atom_index: int) -> np.ndarray:
    ca = coords[:, ca_atom_index, :]
    delta = ca[:, None, :] - ca[None, :, :]
    out = np.sqrt(np.sum(delta * delta, axis=-1))
    return out


def _distance_change_vector(
    predicted: np.ndarray, target: np.ndarray, parent: np.ndarray, ca_atom_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite off-diagonal predicted/true pairwise distance changes."""
    pred_delta = _distance_matrix(predicted, ca_atom_index) - _distance_matrix(parent, ca_atom_index)
    true_delta = _distance_matrix(target, ca_atom_index) - _distance_matrix(parent, ca_atom_index)
    mask = ~np.eye(len(pred_delta), dtype=bool)
    mask &= np.isfinite(pred_delta) & np.isfinite(true_delta)
    return pred_delta[mask], true_delta[mask]


def _masked_distance_change_error(
    predicted: np.ndarray,
    target: np.ndarray,
    parent: np.ndarray,
    ca_atom_index: int,
    residue_mask: np.ndarray,
) -> float:
    """Compare pairwise CA changes for residue pairs inside one region."""
    pred_values, true_values = _masked_distance_change_values(
        predicted, target, parent, ca_atom_index, residue_mask
    )
    return float(np.sqrt(np.mean((pred_values - true_values) ** 2))) if len(pred_values) else float("nan")


def _masked_distance_change_values(
    predicted: np.ndarray,
    target: np.ndarray,
    parent: np.ndarray,
    ca_atom_index: int,
    residue_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite off-diagonal distance changes for one residue region."""
    pred_delta = _distance_matrix(predicted, ca_atom_index) - _distance_matrix(parent, ca_atom_index)
    true_delta = _distance_matrix(target, ca_atom_index) - _distance_matrix(parent, ca_atom_index)
    pair_mask = residue_mask[:, None] & residue_mask[None, :]
    pair_mask &= ~np.eye(len(residue_mask), dtype=bool)
    finite = pair_mask & np.isfinite(pred_delta) & np.isfinite(true_delta)
    return pred_delta[finite], true_delta[finite]


def _cosine_similarity(predicted: np.ndarray, target: np.ndarray) -> float:
    if not len(predicted):
        return float("nan")
    denominator = float(np.linalg.norm(predicted) * np.linalg.norm(target))
    return float(np.dot(predicted, target) / denominator) if denominator > 1e-12 else float("nan")


def region_masks(pair: StructurePair, local_radius: float = 10.0, remote_min_distance: float = 15.0) -> dict[str, np.ndarray]:
    ca = pair.parent_coords[:, pair.ca_atom_index, :]
    valid = np.isfinite(ca).all(axis=-1)
    if not pair.mutation_indices:
        return {"local": valid.copy(), "transition": np.zeros(pair.length, dtype=bool), "remote": np.zeros(pair.length, dtype=bool)}
    mutation_ca = ca[list(pair.mutation_indices)]
    distances = np.sqrt(((ca[:, None, :] - mutation_ca[None, :, :]) ** 2).sum(axis=-1))
    nearest = np.nanmin(distances, axis=1)
    local = valid & (nearest <= local_radius)
    remote = valid & (nearest > remote_min_distance)
    transition = valid & ~local & ~remote
    return {"local": local, "transition": transition, "remote": remote}


def backbone_geometry_violations(coords: np.ndarray, atom_names: tuple[str, ...], tolerance: float = 0.20) -> tuple[int, int]:
    """Count implausible backbone bond lengths, returning violations/checked."""
    indices = {name: atom_names.index(name) for name in ("N", "CA", "C", "O") if name in atom_names}
    expected = (("N", "CA", 1.46), ("CA", "C", 1.53), ("C", "O", 1.24))
    violations = 0
    checked = 0

    def check(first: np.ndarray, second: np.ndarray, target: float) -> None:
        nonlocal violations, checked
        if np.isfinite(first).all() and np.isfinite(second).all():
            checked += 1
            violations += int(abs(float(np.linalg.norm(first - second))) - target > tolerance or target - float(np.linalg.norm(first - second)) > tolerance)

    for residue in coords:
        for first_name, second_name, target in expected:
            if first_name in indices and second_name in indices:
                check(residue[indices[first_name]], residue[indices[second_name]], target)
    if "C" in indices and "N" in indices:
        for index in range(len(coords) - 1):
            check(coords[index, indices["C"]], coords[index + 1, indices["N"]], 1.33)
    return violations, checked


def backbone_angle_violations(
    coords: np.ndarray, atom_names: tuple[str, ...], tolerance_degrees: float = 20.0
) -> tuple[int, int]:
    """Count implausible intra-residue backbone angles."""
    indices = {name: atom_names.index(name) for name in ("N", "CA", "C", "O") if name in atom_names}
    expected = (("N", "CA", "C", 111.0), ("CA", "C", "O", 120.0))
    violations = 0
    checked = 0
    for residue in coords:
        for first_name, vertex_name, last_name, target in expected:
            if not all(name in indices for name in (first_name, vertex_name, last_name)):
                continue
            first, vertex, last = (residue[indices[name]] for name in (first_name, vertex_name, last_name))
            if not (np.isfinite(first).all() and np.isfinite(vertex).all() and np.isfinite(last).all()):
                continue
            left = first - vertex
            right = last - vertex
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denominator <= 1e-8:
                continue
            angle = float(np.degrees(np.arccos(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))))
            checked += 1
            violations += int(abs(angle - target) > tolerance_degrees)
    return violations, checked


def backbone_clash_violations(
    coords: np.ndarray, atom_names: tuple[str, ...], threshold: float = 2.0
) -> tuple[int, int]:
    """Count close backbone atom pairs outside the same/adjacent residue."""
    coordinates = np.asarray(coords, dtype=float)
    violations = 0
    checked = 0
    for residue_index in range(len(coordinates)):
        for other_index in range(residue_index + 1, len(coordinates)):
            if other_index - residue_index <= 1:
                continue
            for first in coordinates[residue_index]:
                for second in coordinates[other_index]:
                    if not (np.isfinite(first).all() and np.isfinite(second).all()):
                        continue
                    checked += 1
                    violations += int(float(np.linalg.norm(first - second)) < threshold)
    return violations, checked


def evaluate_pair(
    pair: StructurePair,
    predicted_coords: np.ndarray,
    neighborhood: int = 4,
    local_radius: float = 10.0,
    remote_min_distance: float = 15.0,
) -> dict[str, float | str]:
    if predicted_coords.shape != pair.parent_coords.shape:
        raise ValueError("predicted coordinates must match pair coordinates")
    regions = region_masks(pair, local_radius, remote_min_distance)
    local = regions["local"]
    remote = regions["remote"]
    mutation = np.zeros(pair.length, dtype=bool)
    mutation[list(pair.mutation_indices)] = True
    globally_aligned_prediction = kabsch_align(pair.mutant_coords, predicted_coords)
    parent_to_pred = _rmsd(pair.parent_coords, predicted_coords)
    local_rmsd = _rmsd(pair.mutant_coords, predicted_coords, local)
    remote_drift = _rmsd(pair.parent_coords, predicted_coords, remote)
    remote_frame_drift = _direct_rmsd(pair.parent_coords, predicted_coords, remote)
    pred_delta_values, true_delta_values = _distance_change_vector(
        predicted_coords, pair.mutant_coords, pair.parent_coords, pair.ca_atom_index
    )
    delta_error = (
        float(np.sqrt(np.mean((pred_delta_values - true_delta_values) ** 2)))
        if len(pred_delta_values)
        else float("nan")
    )
    pred_norm = float(np.linalg.norm(pred_delta_values)) if len(pred_delta_values) else float("nan")
    true_norm = float(np.linalg.norm(true_delta_values)) if len(true_delta_values) else float("nan")
    denominator = pred_norm * true_norm
    delta_cosine = (
        float(np.dot(pred_delta_values, true_delta_values) / denominator)
        if np.isfinite(denominator) and denominator > 1e-12
        else float("nan")
    )
    local_delta_error = _masked_distance_change_error(
        predicted_coords, pair.mutant_coords, pair.parent_coords, pair.ca_atom_index, local
    )
    remote_delta_error = _masked_distance_change_error(
        predicted_coords, pair.mutant_coords, pair.parent_coords, pair.ca_atom_index, remote
    )
    local_pred_delta, local_true_delta = _masked_distance_change_values(
        predicted_coords, pair.mutant_coords, pair.parent_coords, pair.ca_atom_index, local
    )
    remote_pred_delta, remote_true_delta = _masked_distance_change_values(
        predicted_coords, pair.mutant_coords, pair.parent_coords, pair.ca_atom_index, remote
    )
    geometry_violations, geometry_checked = backbone_geometry_violations(predicted_coords, pair.atom_names)
    angle_violations, angle_checked = backbone_angle_violations(predicted_coords, pair.atom_names)
    clash_violations, clash_checked = backbone_clash_violations(predicted_coords, pair.atom_names)
    return {
        "pair_id": pair.pair_id,
        "parent_to_prediction_rmsd": parent_to_pred,
        "mutation_neighborhood_rmsd": local_rmsd,
        "local_backbone_error": local_rmsd,
        "mutation_site_backbone_error": _rmsd(pair.mutant_coords, predicted_coords, mutation),
        "mutation_site_global_rmsd": _direct_rmsd(pair.mutant_coords, globally_aligned_prediction, mutation),
        "remote_scaffold_drift": remote_drift,
        "remote_scaffold_frame_drift": remote_frame_drift,
        "parent_to_prediction_frame_rmsd": _direct_rmsd(pair.parent_coords, predicted_coords),
        "remote_target_error": _rmsd(pair.mutant_coords, predicted_coords, remote),
        "distance_change_error": delta_error,
        "distance_change_cosine": delta_cosine,
        "predicted_distance_change_norm": pred_norm,
        "true_distance_change_norm": true_norm,
        "local_distance_change_error": local_delta_error,
        "remote_distance_change_error": remote_delta_error,
        "local_distance_change_cosine": _cosine_similarity(local_pred_delta, local_true_delta),
        "remote_distance_change_cosine": _cosine_similarity(remote_pred_delta, remote_true_delta),
        "backbone_geometry_violations": float(geometry_violations / geometry_checked) if geometry_checked else float("nan"),
        "backbone_geometry_checked_bonds": float(geometry_checked),
        "backbone_angle_violations": float(angle_violations / angle_checked) if angle_checked else float("nan"),
        "backbone_angle_checked": float(angle_checked),
        "backbone_clash_violations": float(clash_violations / clash_checked) if clash_checked else float("nan"),
        "backbone_clash_checked": float(clash_checked),
    }
