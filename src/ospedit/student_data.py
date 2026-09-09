from __future__ import annotations

from typing import Any, Mapping, Sequence
from collections import Counter

import numpy as np

from .data import PairRecord, StructurePair
from .endpoint_groups import endpoint_group_key
from .geometry import local_frame_difference, residue_frames_masked
from .student import encode_edit_features


def parent_context_features(pair: StructurePair, *, include_geometry: bool = False) -> np.ndarray:
    """Encode parent backbone atoms in each residue's local frame.

    The feature layout is flattened local xyz for every atom followed by one
    finite-coordinate bit per atom. It is intentionally independent of the
    global coordinate frame.
    """
    parent_rotations, origins, valid_frames = residue_frames_masked(pair.parent_coords, pair.atom_names)
    base_dim = len(pair.atom_names) * 4
    geometry_dim = 8 if include_geometry else 0
    features = np.zeros((pair.length, base_dim + geometry_dim), dtype=np.float32)
    for residue_index in range(pair.length):
        if not valid_frames[residue_index]:
            continue
        finite = np.isfinite(pair.parent_coords[residue_index]).all(axis=-1)
        local = (pair.parent_coords[residue_index] - origins[residue_index]) @ parent_rotations[residue_index]
        features[residue_index, : 3 * len(pair.atom_names)] = np.where(finite[:, None], local, 0.0).reshape(-1)
        features[residue_index, 3 * len(pair.atom_names) : base_dim] = finite.astype(np.float32)
    if include_geometry:
        ca_index = pair.ca_atom_index
        ca = pair.parent_coords[:, ca_index]
        finite_ca = np.isfinite(ca).all(axis=-1)
        for residue_index in range(pair.length):
            if not valid_frames[residue_index] or not finite_ca[residue_index]:
                continue
            origin = origins[residue_index]
            rotation = parent_rotations[residue_index]
            cursor = base_dim
            for neighbor_index in (residue_index - 1, residue_index + 1):
                if 0 <= neighbor_index < pair.length and finite_ca[neighbor_index]:
                    relative = (ca[neighbor_index] - origin) @ rotation
                    features[residue_index, cursor : cursor + 3] = relative.astype(np.float32)
                cursor += 3
    return features


def edit_geometry_features(pair: StructurePair) -> np.ndarray:
    """Return candidate-specific distance and mutation-marker channels."""
    features = np.zeros((pair.length, 2), dtype=np.float32)
    ca = pair.parent_coords[:, pair.ca_atom_index]
    finite_ca = np.isfinite(ca).all(axis=-1)
    mutation_indices = [index for index in pair.mutation_indices if 0 <= index < pair.length and finite_ca[index]]
    if not mutation_indices:
        return features
    distances = np.linalg.norm(ca[:, None, :] - ca[mutation_indices][None, :, :], axis=-1)
    features[finite_ca, 0] = np.min(distances[finite_ca], axis=1).astype(np.float32)
    features[mutation_indices, 1] = 1.0
    return features


def parent_local_features(pair: StructurePair, *, include_geometry: bool = False) -> np.ndarray:
    """Combine cacheable parent context with candidate-specific edit geometry."""
    features = parent_context_features(pair, include_geometry=include_geometry)
    if include_geometry:
        features[:, -2:] = edit_geometry_features(pair)
    return features


def parent_residue_mask(pair: StructurePair) -> np.ndarray:
    """Return residues with a usable parent N/CA/C frame."""
    return residue_frames_masked(pair.parent_coords, pair.atom_names)[2]


def mutation_neighborhood_mask(pair: StructurePair, radius: float = 10.0) -> np.ndarray:
    """Return finite-CA residues within ``radius`` Angstrom of a mutation."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    ca = pair.parent_coords[:, pair.ca_atom_index, :]
    valid = np.isfinite(ca).all(axis=-1)
    if not pair.mutation_indices:
        return np.zeros(pair.length, dtype=np.float32)
    mutation_ca = ca[list(pair.mutation_indices)]
    mutation_valid = np.isfinite(mutation_ca).all(axis=-1)
    if not mutation_valid.any():
        return np.zeros(pair.length, dtype=np.float32)
    distances = np.linalg.norm(ca[:, None, :] - mutation_ca[None, :, :], axis=-1)
    nearest = np.min(np.where(mutation_valid[None, :], distances, np.inf), axis=1)
    return (valid & (nearest <= radius)).astype(np.float32)


def mutation_localization_weights(
    pair: StructurePair,
    radius: float = 10.0,
    transition: float = 5.0,
) -> np.ndarray:
    """Return a cosine-tapered spatial window around mutated residues."""
    if radius <= 0 or transition <= 0:
        raise ValueError("localization radius and transition must be positive")
    ca = pair.parent_coords[:, pair.ca_atom_index, :]
    valid = np.isfinite(ca).all(axis=-1)
    mutation_indices = [index for index in pair.mutation_indices if valid[index]]
    weights = np.zeros(pair.length, dtype=np.float32)
    if not mutation_indices:
        return weights
    distances = np.linalg.norm(ca[:, None, :] - ca[mutation_indices][None, :, :], axis=-1)
    nearest = np.min(distances, axis=1)
    weights[valid & (nearest <= radius)] = 1.0
    taper = valid & (nearest > radius) & (nearest < radius + transition)
    phase = (nearest[taper] - radius) / transition
    weights[taper] = (0.5 * (1.0 + np.cos(np.pi * phase))).astype(np.float32)
    return weights


def parent_spatial_graph(
    pair: StructurePair,
    *,
    spatial_neighbors: int = 24,
    sequence_window: int = 1,
    distance_scale: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build invariant directed edges from parent residue frames.

    Features are distance, local relative translation, relative rotation,
    clipped sequence offset, and mutation markers for both edge endpoints.
    """
    if spatial_neighbors <= 0 or sequence_window < 0 or distance_scale <= 0:
        raise ValueError("graph neighborhood and distance scale must be positive")
    rotations, origins, valid = residue_frames_masked(pair.parent_coords, pair.atom_names)
    length = pair.length
    edge_features = np.zeros((length, length, 16), dtype=np.float32)
    edge_mask = np.zeros((length, length), dtype=bool)
    distances = np.linalg.norm(origins[:, None, :] - origins[None, :, :], axis=-1)
    mutation = np.zeros(length, dtype=np.float32)
    mutation[list(pair.mutation_indices)] = 1.0
    for source in np.flatnonzero(valid):
        candidates = np.flatnonzero(valid & (np.arange(length) != source))
        nearest = candidates[np.argsort(distances[source, candidates])[:spatial_neighbors]]
        sequential = candidates[np.abs(candidates - source) <= sequence_window]
        for target in np.union1d(nearest, sequential):
            edge_mask[source, target] = True
            relative_translation = rotations[source].T @ (origins[target] - origins[source])
            relative_rotation = rotations[source].T @ rotations[target]
            edge_features[source, target] = np.concatenate(
                (
                    np.asarray([distances[source, target] / distance_scale]),
                    relative_translation / distance_scale,
                    relative_rotation.reshape(-1),
                    np.asarray(
                        [
                            np.clip((target - source) / 32.0, -1.0, 1.0),
                            mutation[source],
                            mutation[target],
                        ]
                    ),
                )
            ).astype(np.float32)
    return edge_features, edge_mask


def _validate_delta_scales(translation_scale: float, rotation_scale: float) -> tuple[float, float]:
    if translation_scale <= 0 or rotation_scale <= 0:
        raise ValueError("translation_scale and rotation_scale must be positive")
    return float(translation_scale), float(rotation_scale)


def target_local_delta(
    pair: StructurePair,
    *,
    translation_scale: float = 1.0,
    rotation_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return scaled experimental mutant frame delta and valid-residue mask.

    The first three channels are in Angstroms and the last three are in
    radians.  Scaling keeps their contribution to the student loss explicit;
    inference multiplies the predicted channels by the same factors.
    """
    translation_scale, rotation_scale = _validate_delta_scales(translation_scale, rotation_scale)
    parent_rotations, parent_origins, parent_valid = residue_frames_masked(pair.parent_coords, pair.atom_names)
    mutant_rotations, mutant_origins, mutant_valid = residue_frames_masked(pair.mutant_coords, pair.atom_names)
    valid = parent_valid & mutant_valid
    translations, rotations = local_frame_difference(
        parent_rotations,
        parent_origins,
        mutant_rotations,
        mutant_origins,
        parent_rotations,
        parent_origins,
    )
    target = np.concatenate((translations, rotations), axis=-1).astype(np.float32)
    target[..., :3] /= translation_scale
    target[..., 3:] /= rotation_scale
    target[~valid] = 0.0
    return target, valid


class PairDataset:
    """Lightweight dataset over already-audited PairRecords."""

    def __init__(
        self,
        records: Sequence[PairRecord],
        *,
        translation_scale: float = 1.0,
        rotation_scale: float = 1.0,
        include_geometry: bool = False,
        neighborhood_radius: float = 10.0,
        teacher_deltas: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
        include_spatial_graph: bool = False,
        spatial_neighbors: int = 24,
        family_balanced_loss: bool = False,
        endpoint_group_balanced_loss: bool = False,
        include_biochemical: bool = False,
        include_target_residue: bool = True,
        target_localization_radius: float | None = None,
        target_localization_transition: float = 5.0,
    ):
        if not records:
            raise ValueError("PairDataset requires at least one record")
        self.records = list(records)
        atom_names = self.records[0].pair.atom_names
        if any(record.pair.atom_names != atom_names for record in self.records[1:]):
            raise ValueError("all PairRecords must use the same atom_names layout")
        if any(name not in atom_names for name in ("N", "CA", "C", "O")):
            raise ValueError("student training requires backbone atom layout containing N, CA, C, and O")
        self.translation_scale, self.rotation_scale = _validate_delta_scales(translation_scale, rotation_scale)
        self.include_geometry = bool(include_geometry)
        if neighborhood_radius <= 0:
            raise ValueError("neighborhood_radius must be positive")
        self.neighborhood_radius = float(neighborhood_radius)
        self.teacher_deltas = teacher_deltas
        self.include_spatial_graph = bool(include_spatial_graph)
        if spatial_neighbors <= 0:
            raise ValueError("spatial_neighbors must be positive")
        self.spatial_neighbors = int(spatial_neighbors)
        self.family_balanced_loss = bool(family_balanced_loss)
        self.endpoint_group_balanced_loss = bool(endpoint_group_balanced_loss)
        self.include_biochemical = bool(include_biochemical)
        self.include_target_residue = bool(include_target_residue)
        if target_localization_radius is not None and target_localization_radius <= 0:
            raise ValueError("target_localization_radius must be positive or None")
        if target_localization_transition <= 0:
            raise ValueError("target_localization_transition must be positive")
        self.target_localization_radius = target_localization_radius
        self.target_localization_transition = float(target_localization_transition)
        family_counts = Counter(record.family_id for record in self.records)
        self.family_weights = {
            family: len(self.records) / (len(family_counts) * count)
            for family, count in family_counts.items()
        }
        self.sample_weights = {
            record.pair.pair_id: self.family_weights[record.family_id]
            if self.family_balanced_loss
            else 1.0
            for record in self.records
        }
        if self.endpoint_group_balanced_loss:
            endpoint_members: dict[tuple[str, str], list[PairRecord]] = {}
            for record in self.records:
                endpoint_members.setdefault(endpoint_group_key(record), []).append(record)
            for members in endpoint_members.values():
                if len({record.family_id for record in members}) != 1:
                    raise ValueError("one endpoint group spans multiple families")
            if self.family_balanced_loss:
                family_group_counts = Counter(
                    members[0].family_id for members in endpoint_members.values()
                )
                group_mass = {
                    key: len(self.records)
                    / (len(family_counts) * family_group_counts[members[0].family_id])
                    for key, members in endpoint_members.items()
                }
            else:
                group_mass = {
                    key: len(self.records) / len(endpoint_members)
                    for key in endpoint_members
                }
            self.sample_weights = {
                record.pair.pair_id: group_mass[key] / len(members)
                for key, members in endpoint_members.items()
                for record in members
            }
        if teacher_deltas is not None:
            missing = [record.pair.pair_id for record in self.records if record.pair.pair_id not in teacher_deltas]
            if missing:
                raise ValueError(f"teacher deltas are missing pair ids: {missing}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        target, valid = target_local_delta(
            record.pair,
            translation_scale=self.translation_scale,
            rotation_scale=self.rotation_scale,
        )
        localization = None
        if self.target_localization_radius is not None:
            localization = mutation_localization_weights(
                record.pair,
                radius=self.target_localization_radius,
                transition=self.target_localization_transition,
            )
            target *= localization[:, None]
        input_valid = parent_residue_mask(record.pair)
        parent_rotations, parent_origins, _ = residue_frames_masked(
            record.pair.parent_coords, record.pair.atom_names
        )
        edit = encode_edit_features(
            [record.pair.parent_sequence],
            [record.pair.mutant_sequence],
            include_biochemical=self.include_biochemical,
            include_target_residue=self.include_target_residue,
        )[0].numpy()
        item = {
            "pair_id": record.pair.pair_id,
            "parent_features": parent_local_features(record.pair, include_geometry=self.include_geometry),
            "edit_features": edit,
            "target_delta": target,
            "parent_frame_origins": parent_origins.astype(np.float32),
            "parent_frame_rotations": parent_rotations.astype(np.float32),
            "input_mask": input_valid.astype(np.float32),
            "loss_mask": valid.astype(np.float32),
            "neighborhood_mask": mutation_neighborhood_mask(record.pair, self.neighborhood_radius),
            "sample_weight": self.sample_weights[record.pair.pair_id],
        }
        if self.teacher_deltas is not None:
            teacher_delta, teacher_valid = self.teacher_deltas[record.pair.pair_id]
            teacher_delta = np.asarray(teacher_delta, dtype=np.float32)
            teacher_valid = np.asarray(teacher_valid, dtype=bool)
            if teacher_delta.shape != (
                record.pair.length,
                6,
            ) or teacher_valid.shape != (record.pair.length,):
                raise ValueError(f"teacher delta shape does not match pair {record.pair.pair_id}")
            if not np.isfinite(teacher_delta).all():
                raise ValueError(f"teacher delta for {record.pair.pair_id} contains non-finite values")
            scaled_teacher = teacher_delta.copy()
            scaled_teacher[..., :3] /= self.translation_scale
            scaled_teacher[..., 3:] /= self.rotation_scale
            if localization is not None:
                scaled_teacher *= localization[:, None]
            item["teacher_delta"] = scaled_teacher
            item["teacher_mask"] = (teacher_valid & valid).astype(np.float32)
        if self.include_spatial_graph:
            item["edge_features"], item["edge_mask"] = parent_spatial_graph(record.pair, spatial_neighbors=self.spatial_neighbors)
        return item


def iter_pair_batches(
    dataset: PairDataset,
    batch_size: int = 1,
    *,
    shuffle: bool = False,
    seed: int = 0,
):
    """Yield padded numpy batches with reproducible optional shuffling."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield collate_pair_records([dataset[int(index)] for index in indices[start : start + batch_size]])


def collate_pair_records(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    max_length = max(item["parent_features"].shape[0] for item in batch)
    parent_dim = batch[0]["parent_features"].shape[-1]
    edit_dim = batch[0]["edit_features"].shape[-1]
    parent = np.zeros((len(batch), max_length, parent_dim), dtype=np.float32)
    edit = np.zeros((len(batch), max_length, edit_dim), dtype=np.float32)
    target = np.zeros((len(batch), max_length, 6), dtype=np.float32)
    parent_frame_origins = np.zeros((len(batch), max_length, 3), dtype=np.float32)
    parent_frame_rotations = np.zeros(
        (len(batch), max_length, 3, 3), dtype=np.float32
    )
    input_mask = np.zeros((len(batch), max_length), dtype=np.float32)
    loss_mask = np.zeros((len(batch), max_length), dtype=np.float32)
    neighborhood_mask = np.zeros((len(batch), max_length), dtype=np.float32)
    sample_weight = np.ones(len(batch), dtype=np.float32)
    has_teacher = ["teacher_delta" in item for item in batch]
    if any(has_teacher) and not all(has_teacher):
        raise ValueError("all batch items must either contain teacher deltas or omit them")
    teacher = np.zeros((len(batch), max_length, 6), dtype=np.float32) if all(has_teacher) else None
    teacher_mask = np.zeros((len(batch), max_length), dtype=np.float32) if all(has_teacher) else None
    has_graph = ["edge_features" in item for item in batch]
    if any(has_graph) and not all(has_graph):
        raise ValueError("all batch items must either contain spatial graphs or omit them")
    edge_features = np.zeros((len(batch), max_length, max_length, 16), dtype=np.float32) if all(has_graph) else None
    edge_mask = np.zeros((len(batch), max_length, max_length), dtype=bool) if all(has_graph) else None
    lengths = []
    pair_ids = []
    for index, item in enumerate(batch):
        length = item["parent_features"].shape[0]
        lengths.append(length)
        pair_ids.append(item["pair_id"])
        parent[index, :length] = item["parent_features"]
        edit[index, :length] = item["edit_features"]
        target[index, :length] = item["target_delta"]
        parent_frame_origins[index, :length] = item["parent_frame_origins"]
        parent_frame_rotations[index, :length] = item["parent_frame_rotations"]
        input_mask[index, :length] = item["input_mask"]
        loss_mask[index, :length] = item["loss_mask"]
        neighborhood_mask[index, :length] = item["neighborhood_mask"]
        sample_weight[index] = item.get("sample_weight", 1.0)
        if teacher is not None and teacher_mask is not None:
            teacher[index, :length] = item["teacher_delta"]
            teacher_mask[index, :length] = item["teacher_mask"]
        if edge_features is not None and edge_mask is not None:
            edge_features[index, :length, :length] = item["edge_features"]
            edge_mask[index, :length, :length] = item["edge_mask"]
    result = {
        "pair_ids": pair_ids,
        "lengths": np.asarray(lengths, dtype=np.int64),
        "parent_features": parent,
        "edit_features": edit,
        "target_delta": target,
        "parent_frame_origins": parent_frame_origins,
        "parent_frame_rotations": parent_frame_rotations,
        "input_mask": input_mask,
        "loss_mask": loss_mask,
        "neighborhood_mask": neighborhood_mask,
        "sample_weight": sample_weight,
    }
    if teacher is not None and teacher_mask is not None:
        result["teacher_delta"] = teacher
        result["teacher_mask"] = teacher_mask
    if edge_features is not None and edge_mask is not None:
        result["edge_features"] = edge_features
        result["edge_mask"] = edge_mask
    return result
