from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - exercised only in environments without torch.
    torch = None
    Tensor = object  # type: ignore[assignment,misc]
    nn = None  # type: ignore[assignment]


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
TOKEN_INDEX = {letter: index for index, letter in enumerate(AMINO_ACIDS)}


def _sinusoidal_positions(length: int, dimension: int, *, device: object, dtype: object) -> "Tensor":
    positions = torch.arange(length, device=device, dtype=dtype)[:, None]
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=dtype)
        * (-np.log(10000.0) / dimension)
    )
    encoding = torch.zeros((length, dimension), device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
    return encoding


def encode_edit_features(parent_sequences: Sequence[str], mutant_sequences: Sequence[str]) -> "Tensor":
    """Encode aligned source/target residues as source+target one-hot plus mask."""
    if torch is None:
        raise ImportError("ParentEditStudent requires torch; install ospedit[torch]")
    if len(parent_sequences) != len(mutant_sequences):
        raise ValueError("parent and mutant batch sizes must match")
    if not parent_sequences:
        raise ValueError("sequence batch must not be empty")
    length = len(parent_sequences[0])
    if any(len(source) != length or len(target) != length for source, target in zip(parent_sequences, mutant_sequences, strict=True)):
        raise ValueError("all sequences in a batch must have equal length")
    features = torch.zeros((len(parent_sequences), length, 2 * len(AMINO_ACIDS) + 1), dtype=torch.float32)
    for batch, (source, target) in enumerate(zip(parent_sequences, mutant_sequences, strict=True)):
        for residue, (source_letter, target_letter) in enumerate(zip(source, target, strict=True)):
            if source_letter not in TOKEN_INDEX or target_letter not in TOKEN_INDEX:
                raise ValueError(f"unsupported amino-acid token at batch={batch}, residue={residue}")
            features[batch, residue, TOKEN_INDEX[source_letter]] = 1.0
            features[batch, residue, len(AMINO_ACIDS) + TOKEN_INDEX[target_letter]] = 1.0
            features[batch, residue, -1] = float(source_letter != target_letter)
    return features


if nn is not None:

    class _SpatialMessageBlock(nn.Module):
        def __init__(self, hidden_dim: int, edge_dim: int):
            super().__init__()
            self.message = nn.Sequential(
                nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.message_norm = nn.LayerNorm(hidden_dim)
            self.feed_forward = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.SiLU(),
                nn.Linear(4 * hidden_dim, hidden_dim),
            )
            self.output_norm = nn.LayerNorm(hidden_dim)

        def forward(self, hidden: Tensor, edge: Tensor, edge_mask: Tensor) -> Tensor:
            length = hidden.shape[1]
            source = hidden[:, :, None, :].expand(-1, -1, length, -1)
            target = hidden[:, None, :, :].expand(-1, length, -1, -1)
            messages = self.message(torch.cat((source, target, edge), dim=-1))
            mask = edge_mask[..., None].to(messages.dtype)
            aggregate = (messages * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
            hidden = self.message_norm(hidden + aggregate)
            return self.output_norm(hidden + self.feed_forward(hidden))

    class ParentEditStudent(nn.Module):
        """One-pass parent-context editor producing local-frame 6D deltas.

        ``parent_features`` must already be expressed in an SE(3)-appropriate
        parent-local representation. The network itself does not consume global
        Cartesian coordinates, which keeps the frame application boundary
        explicit and testable.
        """

        def __init__(self, parent_dim: int, edit_dim: int = 41, hidden_dim: int = 256, blocks: int = 4, heads: int = 8, max_normalized_delta: float | None = None, use_positional_encoding: bool = True):
            super().__init__()
            if parent_dim <= 0 or edit_dim <= 0:
                raise ValueError("parent_dim and edit_dim must be positive")
            if hidden_dim % heads:
                raise ValueError("hidden_dim must be divisible by heads")
            if max_normalized_delta is not None and max_normalized_delta <= 0:
                raise ValueError("max_normalized_delta must be positive or None")
            self.max_normalized_delta = max_normalized_delta
            self.use_positional_encoding = bool(use_positional_encoding)
            self.input_projection = nn.Linear(parent_dim + edit_dim, hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=4 * hidden_dim,
                batch_first=True,
                norm_first=True,
            )
            self.context = nn.TransformerEncoder(layer, num_layers=blocks)
            self.output_projection = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 6))
            # Begin as an exact reference-preserving editor; training must
            # earn each non-zero structural update from the mutation signal.
            nn.init.zeros_(self.output_projection[-1].weight)
            nn.init.zeros_(self.output_projection[-1].bias)

        def forward(self, parent_features: Tensor, edit_features: Tensor, residue_mask: Tensor | None = None) -> Tensor:
            if parent_features.ndim != 3 or edit_features.ndim != 3:
                raise ValueError("student inputs must have shape (batch, length, features)")
            if parent_features.shape[:2] != edit_features.shape[:2]:
                raise ValueError("parent and edit feature batch/length dimensions must match")
            if residue_mask is not None and residue_mask.shape != parent_features.shape[:2]:
                raise ValueError("residue_mask must have shape (batch, length)")
            hidden = self.input_projection(torch.cat((parent_features, edit_features), dim=-1))
            if self.use_positional_encoding:
                hidden = hidden + _sinusoidal_positions(
                    hidden.shape[1], hidden.shape[2], device=hidden.device, dtype=hidden.dtype
                )[None]
            padding_mask = None if residue_mask is None else ~residue_mask.to(dtype=torch.bool)
            hidden = self.context(hidden, src_key_padding_mask=padding_mask)
            delta = self.output_projection(hidden)
            if self.max_normalized_delta is not None:
                delta = torch.tanh(delta) * self.max_normalized_delta
            edit_mask = edit_features[..., -1].abs().sum(dim=1) > 0
            output = delta * edit_mask[:, None, None].to(delta.dtype)
            if residue_mask is not None:
                output = output * residue_mask[:, :, None].to(output.dtype)
            return output


    class SpatialGraphStudent(nn.Module):
        """One-pass editor with explicit invariant parent spatial relations."""

        def __init__(
            self,
            parent_dim: int,
            edit_dim: int = 41,
            edge_dim: int = 16,
            hidden_dim: int = 128,
            blocks: int = 4,
            max_normalized_delta: float | None = None,
        ):
            super().__init__()
            if min(parent_dim, edit_dim, edge_dim, hidden_dim, blocks) <= 0:
                raise ValueError("graph student dimensions and blocks must be positive")
            if max_normalized_delta is not None and max_normalized_delta <= 0:
                raise ValueError("max_normalized_delta must be positive or None")
            self.max_normalized_delta = max_normalized_delta
            self.node_projection = nn.Linear(parent_dim + edit_dim, hidden_dim)
            self.blocks = nn.ModuleList(
                _SpatialMessageBlock(hidden_dim, edge_dim) for _ in range(blocks)
            )
            self.output_projection = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 6))
            nn.init.zeros_(self.output_projection[-1].weight)
            nn.init.zeros_(self.output_projection[-1].bias)

        def forward(
            self,
            parent_features: Tensor,
            edit_features: Tensor,
            residue_mask: Tensor | None = None,
            edge_features: Tensor | None = None,
            edge_mask: Tensor | None = None,
        ) -> Tensor:
            if edge_features is None or edge_mask is None:
                raise ValueError("SpatialGraphStudent requires edge_features and edge_mask")
            if edge_features.shape[:3] != (
                parent_features.shape[0], parent_features.shape[1], parent_features.shape[1]
            ) or edge_mask.shape != edge_features.shape[:3]:
                raise ValueError("spatial graph dimensions must match the node batch")
            hidden = self.node_projection(torch.cat((parent_features, edit_features), dim=-1))
            for block in self.blocks:
                hidden = block(hidden, edge_features, edge_mask)
            delta = self.output_projection(hidden)
            if self.max_normalized_delta is not None:
                delta = torch.tanh(delta) * self.max_normalized_delta
            has_edit = edit_features[..., -1].abs().sum(dim=1) > 0
            output = delta * has_edit[:, None, None].to(delta.dtype)
            if residue_mask is not None:
                output = output * residue_mask[..., None].to(output.dtype)
            return output

else:

    class ParentEditStudent:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("ParentEditStudent requires torch; install ospedit[torch]")

    class SpatialGraphStudent:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("SpatialGraphStudent requires torch; install ospedit[torch]")
