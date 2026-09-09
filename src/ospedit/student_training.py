from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[assignment,misc]

from .student_data import PairDataset, iter_pair_batches
from .teacher_cache import TeacherCache


def masked_delta_loss(
    prediction: Tensor,
    target: Tensor,
    residue_mask: Tensor,
    *,
    kind: str = "mse",
    beta: float = 1.0,
    residue_weights: Tensor | None = None,
) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if residue_mask.shape != prediction.shape[:2]:
        raise ValueError("residue_mask must have shape (batch, length)")
    if residue_weights is not None and residue_weights.shape != residue_mask.shape:
        raise ValueError("residue_weights must have shape (batch, length)")
    mask = residue_mask.to(dtype=prediction.dtype)
    if residue_weights is not None:
        if (residue_weights < 0).any():
            raise ValueError("residue_weights must be non-negative")
        mask = mask * residue_weights.to(dtype=prediction.dtype)
    mask = mask[..., None]
    if kind not in {"mse", "smooth_l1"}:
        raise ValueError("kind must be 'mse' or 'smooth_l1'")
    if beta <= 0:
        raise ValueError("beta must be positive")
    error = prediction - target
    per_element = error.pow(2) if kind == "mse" else torch.nn.functional.smooth_l1_loss(
        prediction, target, reduction="none", beta=beta
    )
    squared = per_element * mask
    per_sample_denominator = mask.sum(dim=1).squeeze(-1) * prediction.shape[-1]
    valid_samples = per_sample_denominator > 0
    if not bool(valid_samples.any()):
        return prediction.sum() * 0.0
    per_sample = squared.sum(dim=(1, 2)) / per_sample_denominator.clamp_min(1.0)
    return per_sample[valid_samples].mean()


def masked_prediction_norm_loss(prediction: Tensor, residue_mask: Tensor) -> Tensor:
    """Penalize unbounded edits while ignoring padded/invalid residues."""
    if residue_mask.shape != prediction.shape[:2]:
        raise ValueError("residue_mask must have shape (batch, length)")
    mask = residue_mask.to(dtype=prediction.dtype)[..., None]
    squared = prediction.pow(2) * mask
    per_sample_denominator = mask.sum(dim=1).squeeze(-1) * prediction.shape[-1]
    valid_samples = per_sample_denominator > 0
    if not bool(valid_samples.any()):
        return prediction.sum() * 0.0
    per_sample = squared.sum(dim=(1, 2)) / per_sample_denominator.clamp_min(1.0)
    return per_sample[valid_samples].mean()


def train_student(
    model: Any,
    batches: Iterable[dict[str, Any]],
    optimizer: Any,
    epochs: int = 1,
    device: str = "cpu",
    gradient_clip_norm: float | None = 1.0,
    grad_accumulation_steps: int = 1,
    delta_norm_weight: float = 0.0,
    delta_loss_kind: str = "mse",
    delta_loss_beta: float = 1.0,
    mutation_loss_weight: float = 0.0,
    neighborhood_loss_weight: float = 0.0,
    distill_weight: float = 0.0,
) -> list[float]:
    """Minimal supervised loop over collated numpy batches."""
    if torch is None:
        raise ImportError("student training requires torch; install ospedit[torch]")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive or None")
    if grad_accumulation_steps <= 0:
        raise ValueError("grad_accumulation_steps must be positive")
    if delta_norm_weight < 0:
        raise ValueError("delta_norm_weight must be non-negative")
    if mutation_loss_weight < 0:
        raise ValueError("mutation_loss_weight must be non-negative")
    if neighborhood_loss_weight < 0:
        raise ValueError("neighborhood_loss_weight must be non-negative")
    if distill_weight < 0:
        raise ValueError("distill_weight must be non-negative")
    materialized_batches = list(batches)
    if not materialized_batches:
        raise ValueError("batches must not be empty")
    model.to(device)
    history: list[float] = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        count = 0
        optimizer.zero_grad(set_to_none=True)
        for batch in materialized_batches:
            parent = torch.as_tensor(batch["parent_features"], dtype=torch.float32, device=device)
            edit = torch.as_tensor(batch["edit_features"], dtype=torch.float32, device=device)
            target = torch.as_tensor(batch["target_delta"], dtype=torch.float32, device=device)
            input_mask = torch.as_tensor(batch["input_mask"], dtype=torch.float32, device=device)
            loss_mask = torch.as_tensor(batch["loss_mask"], dtype=torch.float32, device=device)
            neighborhood_values = batch.get("neighborhood_mask", np.zeros(loss_mask.shape, dtype=np.float32))
            neighborhood = torch.as_tensor(neighborhood_values, dtype=torch.float32, device=device)
            residue_weights = (1.0 + mutation_loss_weight * edit[..., -1]) * (1.0 + neighborhood_loss_weight * neighborhood)
            prediction = model(parent, edit, residue_mask=input_mask)
            loss = masked_delta_loss(prediction, target, loss_mask, kind=delta_loss_kind, beta=delta_loss_beta, residue_weights=residue_weights)
            if distill_weight and "teacher_delta" in batch:
                teacher = torch.as_tensor(batch["teacher_delta"], dtype=torch.float32, device=device)
                teacher_mask = torch.as_tensor(batch["teacher_mask"], dtype=torch.float32, device=device)
                loss = loss + distill_weight * masked_delta_loss(
                    prediction, teacher, teacher_mask, kind=delta_loss_kind, beta=delta_loss_beta
                )
            if delta_norm_weight:
                loss = loss + delta_norm_weight * masked_prediction_norm_loss(prediction, input_mask)
            window_start = (count // grad_accumulation_steps) * grad_accumulation_steps
            window_size = min(grad_accumulation_steps, len(materialized_batches) - window_start)
            (loss / window_size).backward()
            is_window_end = (count + 1) % grad_accumulation_steps == 0 or count + 1 == len(materialized_batches)
            if is_window_end:
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach().cpu())
            count += 1
        history.append(total / max(count, 1))
    return history


def train_records(
    model: Any,
    records: Iterable[Any],
    optimizer: Any,
    *,
    epochs: int = 1,
    batch_size: int = 1,
    shuffle: bool = True,
    seed: int = 0,
    device: str = "cpu",
    gradient_clip_norm: float | None = 1.0,
    grad_accumulation_steps: int = 1,
    allow_nonexperimental: bool = False,
    translation_scale: float = 1.0,
    rotation_scale: float = 1.0,
    delta_norm_weight: float = 0.0,
    include_geometry: bool = False,
    neighborhood_radius: float = 10.0,
    delta_loss_kind: str = "mse",
    delta_loss_beta: float = 1.0,
    mutation_loss_weight: float = 0.0,
    neighborhood_loss_weight: float = 0.0,
    distill_weight: float = 0.0,
    teacher_cache: TeacherCache | None = None,
    teacher_noise_level: float | None = None,
) -> list[float]:
    """Train directly from PairRecords using the experimental target path."""
    materialized_records = list(records)
    if not allow_nonexperimental:
        materialized_records = [record for record in materialized_records if record.label_source == "experimental"]
    if not materialized_records:
        raise ValueError("no experimental PairRecords available for training")
    teacher_deltas = None
    if teacher_cache is not None:
        if teacher_noise_level is None:
            raise ValueError("teacher_noise_level is required when teacher_cache is provided")
        teacher_deltas = teacher_cache.local_deltas(teacher_noise_level)
    elif teacher_noise_level is not None:
        raise ValueError("teacher_noise_level requires teacher_cache")
    if distill_weight and teacher_cache is None:
        raise ValueError("distill_weight requires teacher_cache")
    dataset = PairDataset(materialized_records, translation_scale=translation_scale, rotation_scale=rotation_scale, include_geometry=include_geometry, neighborhood_radius=neighborhood_radius, teacher_deltas=teacher_deltas)
    batches = list(iter_pair_batches(dataset, batch_size=batch_size, shuffle=shuffle, seed=seed))
    return train_student(model, batches, optimizer, epochs=epochs, device=device, gradient_clip_norm=gradient_clip_norm, grad_accumulation_steps=grad_accumulation_steps, delta_norm_weight=delta_norm_weight, delta_loss_kind=delta_loss_kind, delta_loss_beta=delta_loss_beta, mutation_loss_weight=mutation_loss_weight, neighborhood_loss_weight=neighborhood_loss_weight, distill_weight=distill_weight)


def save_student_checkpoint(
    model: Any,
    path: str | Path,
    *,
    optimizer: Any | None = None,
    epoch: int = 0,
    history: Iterable[float] = (),
    config: dict[str, Any] | None = None,
) -> None:
    """Save a versioned local checkpoint for reproducible student experiments."""
    if torch is None:
        raise ImportError("checkpointing requires torch; install ospedit[torch]")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    payload = {
        "format_version": 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "history": [float(value) for value in history],
        "config": dict(config or {}),
    }
    torch.save(payload, Path(path))


def load_student_checkpoint(
    model: Any,
    path: str | Path,
    *,
    optimizer: Any | None = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Restore a checkpoint and return its epoch/history/config metadata."""
    if torch is None:
        raise ImportError("checkpointing requires torch; install ospedit[torch]")
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported student checkpoint format: {payload.get('format_version')!r}")
    model.load_state_dict(payload["model_state"], strict=strict)
    if optimizer is not None:
        if payload.get("optimizer_state") is None:
            raise ValueError("checkpoint does not contain optimizer state")
        optimizer.load_state_dict(payload["optimizer_state"])
    return {
        "epoch": int(payload["epoch"]),
        "history": [float(value) for value in payload.get("history", [])],
        "config": dict(payload.get("config", {})),
    }


def validate_student_checkpoint_config(config: dict[str, Any], **expected: int) -> None:
    """Raise a focused error when a checkpoint architecture cannot be reused."""
    mismatches = []
    for key, value in expected.items():
        if key in config and int(config[key]) != int(value):
            mismatches.append(f"{key}: checkpoint={config[key]} requested={value}")
    if mismatches:
        raise ValueError("student checkpoint architecture mismatch: " + "; ".join(mismatches))
