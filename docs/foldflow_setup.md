# FoldFlow-2 Runtime Setup

This project keeps FoldFlow optional because the upstream environment currently
targets Python `<3.11` and PyTorch `<2`. Do not install those requirements into
the main `ospedit` environment.

## Isolated environment

Use the upstream environment file in a separate conda environment:

```bash
git clone https://github.com/DreamFold/FoldFlow.git
cd FoldFlow
conda env create -f environment.yaml
conda activate foldflow-env
pip install -e .
```

The upstream repository documents the FF-2 checkpoint names `ff2_base.pth` and
`ff2_reft.pth`. Download the required checkpoint from its release assets and
keep it outside this repository, for example:

```text
/data/checkpoints/foldflow/ff2_base.pth
```

Keep NumPy below version 2 in this environment (`numpy<2`); Torch 1.13 and
the upstream OpenFold stack rely on the NumPy 1.x ABI. The `ospedit` package
declares this upper bound so an editable install cannot silently upgrade it.

Checkpoint loading remains caller-owned because it depends on the upstream
Hydra configuration and ESM assets.

## Verify before running

Install this package in the same isolated environment, then run:

```bash
cd /path/to/ospedit
pip install -e .
ospedit-foldflow-check --strict
```

The audit also honors `OSPEDIT_FOLDFLOW_ROOT` when FoldFlow is being used from
an editable source checkout. Set it to the upstream repository root before
running the P1 preflight so the source tree is included in the dependency
check:

```bash
export OSPEDIT_FOLDFLOW_ROOT=/data/src/FoldFlow
ospedit-p1-preflight --manifest data/manifest/pairs.jsonl
```

The equivalent command-local form is
`ospedit-foldflow-check --strict --foldflow-root /data/src/FoldFlow`; the P1
preflight accepts the same path with its own `--foldflow-root` option.

The command must exit with status `0` and report `ready_for_foldflow_import:
true`. If it fails, do not start an experiment; inspect the JSON fields for the
missing package, version mismatch, or compatibility warning such as a missing
`numpy.trapz` API required by legacy geomstats.

On ROCm workstations, an existing protein environment may already provide
OpenFold and ESM, but it is not automatically a supported FoldFlow runtime.
For example, importing FF-2 under Python 3.12 currently fails in upstream
`pydantic.dataclasses` on mutable configuration defaults. Use the isolated
Python 3.8--3.10 environment above rather than patching the upstream model in
place. The project environment audit reports this incompatibility before any
checkpoint is loaded.

## Adapter wiring

The integration boundary is intentionally explicit:

1. Convert parent N/CA/C coordinates with `backbone_to_rigids`.
2. Use `openfold_rigid_from_tensor7` to construct the upstream `Rigid` object.
3. Supply a `target_rigid_sampler` that uses FoldFlow's SO(3)/R(3) stochastic
   path and consumes `SharedNoise`.
4. Wrap it with `make_forward_marginal_sampler` and
   `FoldFlow2MarginalConverter`.
5. Pass the resulting batch builder to `FoldFlow2EndpointAdapter`.

The repository provides `examples/foldflow_endpoint_factory.py`, which wires
these pieces to the official `inference` Hydra configuration and checkpoint
loader. In the isolated environment, set:

```bash
export OSPEDIT_FOLDFLOW_ROOT=/data/src/FoldFlow
export OSPEDIT_FOLDFLOW_CHECKPOINT=/data/checkpoints/foldflow/ff2_base.pth
export OSPEDIT_FOLDFLOW_DEVICE=cuda
```

Then use `examples.foldflow_endpoint_factory:build_endpoint` with
`run_teacher_admission.py`, `build_teacher_cache.py`, or
`run_mechanism_grid.py`. The factory is intentionally outside the core package
because it imports FoldFlow, Hydra, OpenFold, and the checkpoint-specific ESM
dependencies.

Do not replace the target rigid sampler with Cartesian Gaussian noise: that
would invalidate the shared-noise comparison.

When source structures contain missing or unusable residues, pass a
`residue_mask_builder` to `FoldFlow2BatchBuilder`; the mask is propagated to
`res_mask` instead of treating identity fallback frames as observed data.

For FoldFlow's built-in `SE3FlowMatcher.sample_ref`, use
`make_foldflow_reference_rigid_sampler`. It temporarily seeds NumPy and Torch
from `SharedNoise.seed`, then restores both RNG states after each call.
