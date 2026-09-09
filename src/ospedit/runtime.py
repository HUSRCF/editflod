from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable


@dataclass
class RuntimeStats:
    """Cost accounting kept separate from structural quality metrics."""

    network_calls: int = 0
    sequence_encoder_calls: int = 0
    structure_updates: int = 0
    condition_branches: int = 0
    postprocessing_seconds: float = 0.0
    total_seconds: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def timed_call(stats: RuntimeStats, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    start = perf_counter()
    try:
        return fn(*args, **kwargs)
    finally:
        stats.total_seconds += perf_counter() - start
