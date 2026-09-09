"""Repository-independent construction of the minimal FoldFlow-2 input batch."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Callable

import numpy as np

from .geometry import residue_frames_masked
from .noise import SharedNoise


# FoldFlow follows the AlphaFold residue index convention for the 20 standard
# amino acids. Unknown residues are mapped to the final ``X`` entry.
_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
_AA_TO_INDEX = {aa: index for index, aa in enumerate(_AA_ORDER)}
_AA_TO_INDEX["X"] = len(_AA_ORDER)


def sequence_to_aatype(sequence: str) -> np.ndarray:
    """Encode a single-chain sequence as integer residue types."""
    if not sequence:
        raise ValueError("sequence must not be empty")
    try:
        return np.asarray([_AA_TO_INDEX[aa] for aa in sequence], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"unsupported amino acid {error.args[0]!r}") from error


def rotation_matrix_to_quaternion(rotations: np.ndarray) -> np.ndarray:
    """Convert ``[..., 3, 3]`` rotations to scalar-first quaternions."""
    matrix = np.asarray(rotations, dtype=float)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotations must end in [3, 3], got {matrix.shape}")
    flat = matrix.reshape(-1, 3, 3)
    out = np.zeros((len(flat), 4), dtype=float)
    trace = np.trace(flat, axis1=1, axis2=2)
    positive = trace > 0.0
    root = np.sqrt(np.maximum(trace[positive] + 1.0, 1e-12)) * 2.0
    out[positive, 0] = 0.25 * root
    out[positive, 1] = (flat[positive, 2, 1] - flat[positive, 1, 2]) / root
    out[positive, 2] = (flat[positive, 0, 2] - flat[positive, 2, 0]) / root
    out[positive, 3] = (flat[positive, 1, 0] - flat[positive, 0, 1]) / root
    remaining = ~positive
    for axis, (a, b, c) in enumerate(((0, 1, 2), (1, 2, 0), (2, 0, 1)), start=1):
        selected = remaining & (flat[:, axis - 1, axis - 1] >= flat[:, (axis % 3), (axis % 3)])
        selected &= flat[:, axis - 1, axis - 1] >= flat[:, (axis + 1) % 3, (axis + 1) % 3]
        if not selected.any():
            continue
        root = np.sqrt(np.maximum(1.0 + flat[selected, a, a] - flat[selected, b, b] - flat[selected, c, c], 1e-12)) * 2.0
        out[selected, axis] = 0.25 * root
        out[selected, 0] = (flat[selected, c, b] - flat[selected, b, c]) / root
        out[selected, b + 1] = (flat[selected, b, a] + flat[selected, a, b]) / root
        out[selected, c + 1] = (flat[selected, c, a] + flat[selected, a, c]) / root
        remaining[selected] = False
    if remaining.any():
        raise ValueError("rotation matrix is not convertible to a quaternion")
    out /= np.linalg.norm(out, axis=-1, keepdims=True)
    return out.reshape(matrix.shape[:-2] + (4,))


def backbone_to_rigids(coords: np.ndarray, atom_names: tuple[str, ...] = ("N", "CA", "C", "O")) -> np.ndarray:
    """Convert backbone coordinates to clean scalar-first tensor-7 rigids.

    Missing or degenerate residues receive an identity rotation and zero
    translation; their validity must be carried separately by the model mask.
    """
    coordinates = np.asarray(coords, dtype=float)
    rotations, origins, valid = residue_frames_masked(coordinates, atom_names)
    rigids = np.zeros((len(origins), 7), dtype=float)
    rigids[:, :4] = np.array([1.0, 0.0, 0.0, 0.0])
    rigids[:, 4:] = np.nan_to_num(origins, nan=0.0)
    if valid.any():
        rigids[valid, :4] = rotation_matrix_to_quaternion(rotations[valid])
    return rigids


@dataclass
class FoldFlow2BatchBuilder:
    """Build the fields consumed by ``FF2Model.forward``.

    ``coords_to_rigids`` is deliberately injected: converting Cartesian
    backbone coordinates to FoldFlow's noisy tensor-7 representation requires
    the installed FoldFlow/OpenFold flow matcher and its scaling convention.
    It receives ``(coords, noise_state)`` and returns an ``[L, 7]`` array or
    tensor. The resulting batch has a leading batch dimension of one.
    """

    coords_to_rigids: Callable[[np.ndarray, SharedNoise], Any]
    device: str = "cpu"
    sequence_encoder: Callable[[str], Any] | None = None
    residue_mask_builder: Callable[[np.ndarray], Any] | None = None
    atom_names: tuple[str, ...] = ("N", "CA", "C", "O")

    def __call__(
        self,
        coords: np.ndarray,
        sequence: str,
        noise_level: float,
        noise_state: SharedNoise,
    ) -> dict[str, Any]:
        if len(sequence) != len(coords):
            raise ValueError("sequence and coordinate length must match")
        if not np.isclose(float(noise_level), float(noise_state.level), rtol=0.0, atol=1e-8):
            raise ValueError("noise_level must match noise_state.level")
        rigids = self.coords_to_rigids(np.asarray(coords, dtype=float), noise_state)
        if hasattr(rigids, "shape"):
            shape = tuple(rigids.shape)
        else:
            rigids = np.asarray(rigids)
            shape = rigids.shape
        if shape != (len(sequence), 7):
            raise ValueError(f"coords_to_rigids must return {(len(sequence), 7)}, got {shape}")

        import torch

        def tensor(value: Any, *, dtype: Any) -> Any:
            return torch.as_tensor(value, dtype=dtype, device=self.device).unsqueeze(0)

        if self.sequence_encoder is None:
            aatype = sequence_to_aatype(sequence)
        else:
            aatype = self.sequence_encoder(sequence)
            shape = tuple(aatype.shape) if hasattr(aatype, "shape") else np.asarray(aatype).shape
            if shape != (len(sequence),):
                raise ValueError(f"sequence_encoder must return shape {(len(sequence),)}, got {shape}")
        if self.residue_mask_builder is None:
            try:
                residue_mask = residue_frames_masked(np.asarray(coords), self.atom_names)[2]
            except (ValueError, IndexError):
                # Non-standard injected coordinate layouts must provide their
                # own mask; preserve the historical all-valid fallback.
                residue_mask = np.ones(len(sequence), dtype=bool)
        else:
            residue_mask = self.residue_mask_builder(np.asarray(coords))
            shape = tuple(residue_mask.shape) if hasattr(residue_mask, "shape") else np.asarray(residue_mask).shape
            if shape != (len(sequence),):
                raise ValueError(f"residue_mask_builder must return shape {(len(sequence),)}, got {shape}")
            residue_mask = np.asarray(residue_mask, dtype=bool)
        return {
            "aatype": tensor(aatype, dtype=torch.long),
            "chain_idx": tensor(np.zeros(len(sequence), dtype=np.int64), dtype=torch.long),
            "seq_idx": tensor(np.arange(1, len(sequence) + 1, dtype=np.int64), dtype=torch.long),
            "res_mask": tensor(residue_mask, dtype=torch.bool),
            "fixed_mask": tensor(np.zeros(len(sequence), dtype=bool), dtype=torch.bool),
            "sc_ca_t": tensor(np.zeros((len(sequence), 3), dtype=np.float32), dtype=torch.float32),
            "t": torch.tensor([float(noise_level)], dtype=torch.float32, device=self.device),
            "rigids_t": tensor(rigids, dtype=torch.float32),
        }


@dataclass
class FoldFlow2MarginalConverter:
    """Bridge clean backbone coordinates to a noisy FoldFlow tensor-7 state.

    ``clean_rigid_builder`` wraps the output of :func:`backbone_to_rigids` in
    the installed OpenFold ``Rigid`` type. ``marginal_sampler`` is a thin
    project-local wrapper around FoldFlow's ``SE3FlowMatcher.forward_marginal``
    (or an equivalent implementation) and must consume the shared noise state.
    Keeping that function injected makes the stochastic convention explicit.
    """

    clean_rigid_builder: Callable[[np.ndarray], Any]
    marginal_sampler: Callable[[Any, float, SharedNoise], Any]
    atom_names: tuple[str, ...] = ("N", "CA", "C", "O")

    def __call__(self, coords: np.ndarray, noise_state: SharedNoise) -> np.ndarray:
        clean_tensor_7 = backbone_to_rigids(np.asarray(coords, dtype=float), self.atom_names)
        clean_rigid = self.clean_rigid_builder(clean_tensor_7)
        noisy = self.marginal_sampler(clean_rigid, float(noise_state.level), noise_state)
        if isinstance(noisy, dict):
            if "rigids_t" not in noisy:
                raise KeyError("marginal_sampler output is missing 'rigids_t'")
            noisy = noisy["rigids_t"]
        noisy = _to_numpy_value(noisy)
        if noisy.ndim == 3 and noisy.shape[0] == 1:
            noisy = noisy[0]
        expected = (len(coords), 7)
        if noisy.shape != expected:
            raise ValueError(f"marginal_sampler must return {expected}, got {noisy.shape}")
        if not np.isfinite(noisy).all():
            raise ValueError("marginal_sampler returned non-finite rigids")
        return noisy


def _to_numpy_value(value: Any) -> np.ndarray:
    detached = value.detach() if hasattr(value, "detach") else value
    detached = detached.cpu() if hasattr(detached, "cpu") else detached
    return np.asarray(detached, dtype=float)


def make_forward_marginal_sampler(
    flow_matcher: Any,
    target_rigid_sampler: Callable[[Any, SharedNoise], Any],
) -> Callable[[Any, float, SharedNoise], Any]:
    """Create a callback matching FoldFlow's ``forward_marginal`` contract.

    The target rigid sampler is injected because FoldFlow's SO(3)/R(3) noise
    implementations own the stochastic path and are not interchangeable with
    Cartesian Gaussian noise. It must consume the supplied ``SharedNoise``.
    """

    def sample(clean_rigid: Any, level: float, noise_state: SharedNoise) -> Any:
        target_rigid = target_rigid_sampler(clean_rigid, noise_state)
        with _openfold_numpy_rigid_compat():
            return flow_matcher.forward_marginal(
                clean_rigid,
                t=float(level),
                rigids_1=target_rigid,
                as_tensor_7=True,
            )

    return sample


@contextmanager
def _openfold_numpy_rigid_compat():
    """Coerce legacy FoldFlow NumPy outputs at the OpenFold boundary.

    FoldFlow 0.2's flow matchers return NumPy arrays, while the paired
    OpenFold rigid implementation expects Torch tensors. The patch is scoped
    to one call and is a no-op when OpenFold is unavailable or already accepts
    the values, keeping the core package dependency-free.
    """
    try:
        import torch
        from openfold.utils import rigid_utils as rigid_utils
    except ImportError:
        yield
        return

    rotation_init = rigid_utils.Rotation.__init__
    rigid_init = rigid_utils.Rigid.__init__

    def rotation_compat(self: Any, rot_mats: Any = None, quats: Any = None, normalize_quats: bool = True) -> None:
        if rot_mats is not None and not hasattr(rot_mats, "type"):
            rot_mats = torch.as_tensor(rot_mats, dtype=torch.float32)
        if quats is not None and not hasattr(quats, "type"):
            quats = torch.as_tensor(quats, dtype=torch.float32)
        rotation_init(self, rot_mats, quats, normalize_quats)

    def rigid_compat(self: Any, rots: Any, trans: Any) -> None:
        if not hasattr(trans, "type"):
            trans = torch.as_tensor(trans, dtype=torch.float32)
        rigid_init(self, rots, trans)

    rigid_utils.Rotation.__init__ = rotation_compat
    rigid_utils.Rigid.__init__ = rigid_compat
    try:
        yield
    finally:
        rigid_utils.Rotation.__init__ = rotation_init
        rigid_utils.Rigid.__init__ = rigid_init


@contextmanager
def _foldflow_seed(seed: int):
    """Seed upstream NumPy/Torch samplers without changing caller RNG state."""
    import torch

    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def make_foldflow_reference_rigid_sampler(flow_matcher: Any) -> Callable[[Any, SharedNoise], Any]:
    """Adapt ``SE3FlowMatcher.sample_ref`` to the shared-noise callback.

    FoldFlow's reference sampler draws from both NumPy (translations) and
    Torch (rotations) without accepting a generator. The temporary seeded
    context makes source/target calls reproducible while restoring the caller's
    RNG streams after each call.
    """

    def sample(clean_rigid: Any, noise_state: SharedNoise) -> Any:
        if not hasattr(clean_rigid, "shape"):
            raise ValueError("clean_rigid must expose a batch shape")
        shape = tuple(clean_rigid.shape)
        if not shape:
            raise ValueError("clean_rigid must contain at least one residue")
        # ``forward_marginal`` consumes an unbatched Rigid for FoldFlow's
        # SO(3) implementation; callers may still keep a leading batch axis
        # while constructing the clean input.
        n_residues = int(shape[-1])
        with _foldflow_seed(noise_state.seed):
            sampled = flow_matcher.sample_ref(
                n_samples=n_residues, as_tensor_7=False
            )
        # FoldFlow 0.2 returns a mapping even when ``as_tensor_7=False``;
        # older wrappers may return the Rigid object directly.
        if isinstance(sampled, dict) and "rigids_t" in sampled:
            sampled = sampled["rigids_t"]
        # ``sample_ref`` returns an unbatched [L] Rigid, while the model batch
        # and ``forward_marginal`` use [B, L]. Add only the missing singleton
        # batch axis; do not reshape arbitrary injected test doubles.
        sampled_shape = getattr(sampled, "shape", None)
        if sampled_shape is not None and tuple(sampled_shape) == shape[1:] and len(shape) == 2:
            sampled = sampled.unsqueeze(0)
        return sampled

    return sample


def openfold_rigid_from_tensor7(clean_tensor_7: np.ndarray, device: str = "cpu") -> Any:
    """Lazily construct an OpenFold ``Rigid`` from an ``[L, 7]`` tensor-7.

    OpenFold is intentionally optional. This helper keeps the project usable
    without the FoldFlow environment while providing the exact constructor
    needed by :class:`FoldFlow2MarginalConverter` when it is installed.
    """
    try:
        import torch
        from openfold.utils import rigid_utils as ru
    except ImportError as error:  # pragma: no cover - exercised in external env
        raise RuntimeError("openfold_rigid_from_tensor7 requires FoldFlow/OpenFold dependencies") from error
    tensor = torch.as_tensor(clean_tensor_7, dtype=torch.float32, device=device)
    if tensor.ndim != 2 or tensor.shape[-1] != 7:
        raise ValueError(f"clean tensor-7 must have shape [L, 7], got {tuple(tensor.shape)}")
    return ru.Rigid.from_tensor_7(tensor.unsqueeze(0))
