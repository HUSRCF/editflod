from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SharedNoise:
    """Deterministic noise realization shared by source/target condition calls."""

    level: float
    seed: int
    values: np.ndarray


def make_shared_noise(shape: tuple[int, ...], level: float, seed: int) -> SharedNoise:
    if level < 0:
        raise ValueError("noise level must be non-negative")
    base = np.random.default_rng(seed).standard_normal(shape)
    return SharedNoise(float(level), int(seed), base * float(level))
