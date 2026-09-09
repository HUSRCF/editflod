"""Synthetic endpoint factory used by admission-gate smoke checks."""

from .run_mechanism_smoke import SyntheticEndpoint


def build_endpoint() -> SyntheticEndpoint:
    return SyntheticEndpoint()
