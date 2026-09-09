"""Stable grouping for records that share the same physical endpoint pair."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ospedit.data import PairRecord


def endpoint_identity(record: PairRecord, *, source: bool) -> str:
    checksum = record.source_checksum if source else record.target_checksum
    chain = record.source_chain if source else record.target_chain
    if checksum:
        return f"sha256:{checksum.lower()}|chain:{chain}"
    path = record.source_file if source else record.target_file
    if path and chain:
        return f"path:{Path(path).resolve()}|chain:{chain}"
    sequence = record.pair.parent_sequence if source else record.pair.mutant_sequence
    coords = record.pair.parent_coords if source else record.pair.mutant_coords
    digest = hashlib.sha256()
    digest.update(sequence.encode())
    digest.update(coords.tobytes())
    return f"content:{digest.hexdigest()}|chain:{chain}"


def endpoint_group_key(record: PairRecord) -> tuple[str, str]:
    """Return an unordered identity pair for one physical endpoint comparison."""
    left, right = sorted((
        endpoint_identity(record, source=True),
        endpoint_identity(record, source=False),
    ))
    return left, right


def endpoint_group_id(record: PairRecord) -> str:
    digest = hashlib.sha256("\n".join(endpoint_group_key(record)).encode()).hexdigest()
    return "endpoint_group_" + digest[:16]
