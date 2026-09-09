"""Load an official FoldFlow-2 checkpoint as an ospedit endpoint factory.

This module is intended to run inside the isolated FoldFlow Python 3.8--3.10
environment.  Configure ``OSPEDIT_FOLDFLOW_ROOT`` and
``OSPEDIT_FOLDFLOW_CHECKPOINT`` before passing
``examples.foldflow_endpoint_factory:build_endpoint`` to the CLI scripts.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"set {name} before loading FoldFlow-2")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def build_endpoint() -> Any:
    """Construct a deterministic, sequence-conditioned FoldFlow-2 endpoint."""
    root = _required_path("OSPEDIT_FOLDFLOW_ROOT")
    checkpoint = _required_path("OSPEDIT_FOLDFLOW_CHECKPOINT")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        import torch
        from hydra import compose, initialize_config_dir
        from foldflow.data import utils as data_utils
        from foldflow.models.ff2flow.ff2_dependencies import FF2Dependencies
        from foldflow.models.ff2flow.flow_model import FF2Model
    except ImportError as error:  # pragma: no cover - optional external environment
        raise RuntimeError("FoldFlow-2 dependencies are not installed in this environment") from error

    config_dir = root / "runner" / "config"
    if not config_dir.is_dir():
        raise FileNotFoundError(f"FoldFlow config directory not found: {config_dir}")
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="inference")
    checkpoint_payload = data_utils.read_pkl(
        str(checkpoint), use_torch=True, map_location="cpu"
    )
    dependencies = FF2Dependencies(config)
    model = FF2Model.from_ckpt(checkpoint_payload, dependencies)
    device = os.environ.get("OSPEDIT_FOLDFLOW_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(
            f"OSPEDIT_FOLDFLOW_DEVICE={device!r} requested, but the FoldFlow environment "
            "Torch build has no CUDA support; use cpu or install a CUDA-enabled build"
        )
    if device == "cpu":
        # FF2Dependencies follows the upstream GPU path and half-casts ESM;
        # PyTorch 1.13 has no CPU LayerNorm kernel for float16.
        model.seq_encoder.esm.float()
    model = model.to(device)
    model.eval()

    from ospedit.adapters import FoldFlow2EndpointAdapter
    from ospedit.foldflow_batch import (
        FoldFlow2BatchBuilder,
        FoldFlow2MarginalConverter,
        make_foldflow_reference_rigid_sampler,
        make_forward_marginal_sampler,
        openfold_rigid_from_tensor7,
    )

    matcher = dependencies.flow_matcher
    target_sampler = make_foldflow_reference_rigid_sampler(matcher)
    marginal_sampler = make_forward_marginal_sampler(matcher, target_sampler)

    def clean_builder(tensor_7):
        # FoldFlow's SO(3) conditional marginal expects an unbatched [L]
        # Rigid; FoldFlow2BatchBuilder adds the singleton batch axis later.
        return openfold_rigid_from_tensor7(tensor_7, device=device)[0]

    converter = FoldFlow2MarginalConverter(clean_builder, marginal_sampler)
    batch_builder = FoldFlow2BatchBuilder(converter, device=device)
    # Force the conditional flag after the final eval() call in model setup.
    return FoldFlow2EndpointAdapter(model, batch_builder, device=device)


__all__ = ["build_endpoint"]
