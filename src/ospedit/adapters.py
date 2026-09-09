"""Optional adapters for external sequence-conditioned structure models.

The core package deliberately does not depend on FoldFlow's environment.  This
module provides a small bridge for an already-constructed FoldFlow-2 model;
callers supply the repository-specific batch builder and checkpoint loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from .noise import SharedNoise


def _quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert ``[..., 4]`` (w, x, y, z) unit quaternions to rotation matrices."""
    q = np.asarray(quaternion, dtype=float)
    if q.shape[-1] != 4:
        raise ValueError(f"quaternion must end in 4 values, got {q.shape}")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm <= 1e-8) or not np.isfinite(norm).all():
        raise ValueError("rigids contain a zero or non-finite quaternion")
    normalized = q / norm
    w, x, y, z = (normalized[..., index] for index in range(4))
    matrix = np.empty(q.shape[:-1] + (3, 3), dtype=float)
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - z * w)
    matrix[..., 0, 2] = 2 * (x * z + y * w)
    matrix[..., 1, 0] = 2 * (x * y + z * w)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - x * w)
    matrix[..., 2, 0] = 2 * (x * z - y * w)
    matrix[..., 2, 1] = 2 * (y * z + x * w)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def _to_numpy(value: Any) -> np.ndarray:
    detached = value.detach() if hasattr(value, "detach") else value
    detached = detached.cpu() if hasattr(detached, "cpu") else detached
    return np.asarray(detached)


@dataclass
class FoldFlow2EndpointAdapter:
    """Expose an existing FoldFlow-2 model through :class:`EndpointModel`.

    ``batch_builder`` is intentionally injected because FoldFlow's data
    transforms depend on its installation and checkpoint configuration.  It
    receives ``(coords, sequence, noise_level, noise_state)`` and must return a
    batch accepted by ``model``.  The model output must contain ``rigids`` with
    shape ``[B, L, 7]`` in FoldFlow's ``(w, x, y, z, tx, ty, tz)`` convention.
    """

    model: Any
    batch_builder: Callable[[np.ndarray, str, float, SharedNoise], Mapping[str, Any]]
    device: str = "cpu"
    output_key: str = "rigids"
    activate_conditioning: bool = True

    def __post_init__(self) -> None:
        if self.activate_conditioning:
            conditional = getattr(self.model, "conditional_generation", None)
            if conditional is None:
                raise TypeError("FoldFlow-2 model must expose conditional_generation()")
            # FoldFlow's train(False)/eval() resets this flag; do not call eval
            # after this point unless the caller re-enables conditioning.
            conditional()

    def endpoint(
        self,
        coords: np.ndarray,
        sequence: str,
        noise_level: float,
        *,
        noise_state: SharedNoise,
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("FoldFlow-2 adapter requires torch") from error
        batch = dict(self.batch_builder(coords, sequence, noise_level, noise_state))
        prepared = {}
        for key, value in batch.items():
            prepared[key] = value.to(self.device) if hasattr(value, "to") else value
        conditional = getattr(self.model, "conditional_generation", None)
        is_conditional = getattr(self.model, "is_conditional_generation", None)
        if conditional is not None and is_conditional is False:
            # FoldFlow's train(False)/eval() clears this flag. Restore it here
            # so an external caller cannot silently disable sequence conditioning.
            conditional()
        with torch.no_grad():
            output = self.model(prepared)
        if self.output_key not in output:
            raise KeyError(f"FoldFlow output is missing {self.output_key!r}")
        rigids = _to_numpy(output[self.output_key])
        if rigids.ndim == 3 and rigids.shape[0] == 1:
            rigids = rigids[0]
        if rigids.ndim != 2 or rigids.shape[1] != 7:
            raise ValueError(f"rigids must have shape [L, 7] after batching, got {rigids.shape}")
        return _quaternion_to_matrix(rigids[:, :4]), rigids[:, 4:]
