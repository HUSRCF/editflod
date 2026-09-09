"""Strict reader for auditable endpoint-response teacher caches."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .data import PairRecord, file_sha256, manifest_fingerprint


@dataclass(frozen=True)
class TeacherCache:
    """Read-only view over a ``build_teacher_cache.py`` output directory."""

    index_path: Path
    metadata: dict[str, Any]
    entries: dict[str, dict[str, Any]]

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        *,
        records: Iterable[PairRecord] | None = None,
        split: str | None = None,
    ) -> "TeacherCache":
        path = Path(index_path).resolve()
        payload = json.loads(path.read_text())
        if payload.get("format") != "ospedit.teacher_cache.v1":
            raise ValueError(f"unsupported teacher cache format: {payload.get('format')!r}")
        if split is not None and payload.get("split") != split:
            raise ValueError(f"teacher cache split={payload.get('split')!r} does not match requested {split!r}")
        rows = payload.get("entries")
        if not isinstance(rows, list) or not rows:
            raise ValueError("teacher cache index has no entries")
        entries: dict[str, dict[str, Any]] = {}
        for row in rows:
            pair_id = str(row.get("pair_id", ""))
            filename = row.get("file")
            if not pair_id or not isinstance(filename, str):
                raise ValueError("teacher cache entry must contain pair_id and file")
            if pair_id in entries:
                raise ValueError(f"duplicate teacher cache pair_id: {pair_id}")
            cache_file = path.parent / filename
            if not cache_file.is_file():
                raise FileNotFoundError(f"teacher cache file is missing: {cache_file}")
            expected_checksum = row.get("file_sha256")
            if expected_checksum is not None and file_sha256(cache_file) != expected_checksum:
                raise ValueError(f"teacher cache checksum mismatch: {cache_file}")
            entries[pair_id] = dict(row)
        if records is not None:
            materialized = list(records)
            expected = str(payload.get("manifest_fingerprint", ""))
            if expected and manifest_fingerprint(materialized) != expected:
                raise ValueError("teacher cache manifest fingerprint does not match records")
            selected = [record.pair.pair_id for record in materialized if split is None or record.split == split]
            missing = [pair_id for pair_id in selected if pair_id not in entries]
            if missing:
                raise ValueError(f"teacher cache is missing pair ids: {missing}")
        return cls(path, dict(payload), entries)

    @property
    def noise_levels(self) -> tuple[float, ...]:
        return tuple(float(level) for level in self.metadata.get("noise_levels", ()))

    def _level_key(self, noise_level: float) -> str:
        matches = [level for level in self.noise_levels if np.isclose(level, noise_level, rtol=0.0, atol=1e-8)]
        if len(matches) != 1:
            raise KeyError(f"noise level {noise_level} is not present in teacher cache {self.noise_levels}")
        return f"level_{matches[0]:g}"

    def load_pair(self, pair_id: str, noise_level: float) -> dict[str, np.ndarray]:
        """Load one pair's arrays and validate their shape and finiteness."""
        if pair_id not in self.entries:
            raise KeyError(f"pair id is not present in teacher cache: {pair_id}")
        entry = self.entries[pair_id]
        expected_length = int(entry["length"])
        key = self._level_key(noise_level)
        with np.load(self.index_path.parent / str(entry["file"])) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        names = {
            "source_rotations": (expected_length, 3, 3),
            "source_origins": (expected_length, 3),
            "target_rotations": (expected_length, 3, 3),
            "target_origins": (expected_length, 3),
            "local_delta": (expected_length, 6),
            "valid": (expected_length,),
        }
        result: dict[str, np.ndarray] = {}
        for suffix, shape in names.items():
            name = f"{key}_{suffix}"
            if name not in arrays:
                raise ValueError(f"teacher cache file is missing array {name!r}")
            value = arrays[name]
            if value.shape != shape:
                raise ValueError(f"teacher cache array {name!r} has shape {value.shape}, expected {shape}")
            if suffix != "valid" and not np.isfinite(value).all():
                raise ValueError(f"teacher cache array {name!r} contains non-finite values")
            result[suffix] = value
        result["valid"] = result["valid"].astype(bool)
        return result

    def local_deltas(self, noise_level: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return ``pair_id -> (local_delta, valid_mask)`` for one noise level."""
        return {
            pair_id: (arrays["local_delta"], arrays["valid"])
            for pair_id in self.entries
            for arrays in (self.load_pair(pair_id, noise_level),)
        }


__all__ = ["TeacherCache"]
