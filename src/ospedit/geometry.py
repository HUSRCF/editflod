from __future__ import annotations

import numpy as np


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(vector: np.ndarray) -> np.ndarray:
    """Exponential map from a rotation vector to a 3x3 matrix."""
    vector = np.asarray(vector, dtype=float)
    theta = float(np.linalg.norm(vector))
    omega = _skew(vector)
    if theta < 1e-8:
        return np.eye(3) + omega + 0.5 * (omega @ omega)
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * omega + b * (omega @ omega)


def so3_log(rotation: np.ndarray) -> np.ndarray:
    """Principal rotation vector for a proper 3D rotation matrix."""
    matrix = np.asarray(rotation, dtype=float)
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cosine))
    skew_vector = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]])
    if theta < 1e-7:
        return 0.5 * skew_vector
    if np.pi - theta < 1e-5:
        # The antisymmetric formula is ill-conditioned at pi. Recover the
        # axis from the symmetric part and choose the largest diagonal entry.
        diagonal = np.diag(matrix)
        axis_index = int(np.argmax(diagonal))
        axis = np.zeros(3, dtype=float)
        axis[axis_index] = np.sqrt(max((diagonal[axis_index] + 1.0) / 2.0, 0.0))
        if axis[axis_index] < 1e-8:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            others = [index for index in range(3) if index != axis_index]
            for other in others:
                axis[other] = (matrix[other, axis_index] + matrix[axis_index, other]) / (4.0 * axis[axis_index])
            axis /= np.linalg.norm(axis)
        if np.dot(axis, skew_vector) < 0.0:
            axis = -axis
        return theta * axis
    scale = theta / (2.0 * np.sin(theta))
    return scale * skew_vector


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        raise ValueError("cannot construct a frame from a zero-length vector")
    return vector / norm


def residue_frames(coords: np.ndarray, atom_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build N-CA-C residue frames; returns rotations, origins, and validity."""
    coords = np.asarray(coords, dtype=float)
    indices = {name: atom_names.index(name) for name in ("N", "CA", "C")}
    rotations = np.tile(np.eye(3), (len(coords), 1, 1))
    origins = np.full((len(coords), 3), np.nan, dtype=float)
    valid = np.zeros(len(coords), dtype=bool)
    for index, residue in enumerate(coords):
        n, ca, carbon = (residue[indices[name]] for name in ("N", "CA", "C"))
        if not (np.isfinite(n).all() and np.isfinite(ca).all() and np.isfinite(carbon).all()):
            continue
        x = _normalize(carbon - ca)
        y_hint = n - ca
        z = _normalize(np.cross(x, y_hint))
        y = _normalize(np.cross(z, x))
        rotations[index] = np.column_stack((x, y, z))
        origins[index] = ca
        valid[index] = True
    return rotations, origins, valid


def residue_frames_masked(
    coords: np.ndarray, atom_names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct frames per residue, masking missing or degenerate residues."""
    coordinates = np.asarray(coords, dtype=float)
    length = len(coordinates)
    rotations = np.tile(np.eye(3), (length, 1, 1))
    origins = np.zeros((length, 3), dtype=float)
    valid = np.zeros(length, dtype=bool)
    ca_index = atom_names.index("CA") if "CA" in atom_names else None
    if ca_index is not None:
        ca = coordinates[:, ca_index]
        finite_ca = np.isfinite(ca).all(axis=-1)
        origins[finite_ca] = ca[finite_ca]
    for index in range(length):
        try:
            row_rotations, row_origins, row_valid = residue_frames(coordinates[index : index + 1], atom_names)
        except (ValueError, IndexError):
            continue
        rotations[index] = row_rotations[0]
        origins[index] = row_origins[0]
        valid[index] = bool(row_valid[0])
    return rotations, origins, valid


def local_frame_difference(
    parent_rotations: np.ndarray,
    parent_origins: np.ndarray,
    target_rotations: np.ndarray,
    target_origins: np.ndarray,
    source_rotations: np.ndarray,
    source_origins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return target-minus-source translation and rotation in parent charts."""
    translations = np.einsum("nij,nj->ni", parent_rotations.transpose(0, 2, 1), target_origins - parent_origins)
    source_translations = np.einsum("nij,nj->ni", parent_rotations.transpose(0, 2, 1), source_origins - parent_origins)
    rotations = np.empty((len(parent_rotations), 3), dtype=float)
    source_rotation_vectors = np.empty_like(rotations)
    for index, parent_rotation in enumerate(parent_rotations):
        target_relative = parent_rotation.T @ target_rotations[index]
        source_relative = parent_rotation.T @ source_rotations[index]
        rotations[index] = so3_log(target_relative)
        source_rotation_vectors[index] = so3_log(source_relative)
    return translations - source_translations, rotations - source_rotation_vectors


def apply_local_frame_update(
    coords: np.ndarray,
    atom_names: tuple[str, ...],
    translations: np.ndarray,
    rotations: np.ndarray,
    translation_scale: float = 1.0,
    rotation_scale: float = 1.0,
) -> np.ndarray:
    """Apply per-residue local rigid updates to actual input atom coordinates."""
    parent_rotations, origins, valid = residue_frames(coords, atom_names)
    updated = np.array(coords, dtype=float, copy=True)
    for index in np.flatnonzero(valid):
        local_rotation = so3_exp(rotation_scale * rotations[index])
        # Local-frame rotations must be conjugated into the global chart.
        global_rotation = parent_rotations[index] @ local_rotation @ parent_rotations[index].T
        origin = origins[index]
        shifted = coords[index] - origin
        finite = np.isfinite(shifted).all(axis=-1)
        moved = shifted[finite] @ global_rotation.T + origin + parent_rotations[index] @ (translation_scale * translations[index])
        updated[index, finite] = moved
    return updated
