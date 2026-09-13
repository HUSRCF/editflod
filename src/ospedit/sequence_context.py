from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


SEQUENCE_CONTEXT_FORMAT = "ospedit.sequence_context.v1"


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _metadata_fingerprint(metadata: Mapping[str, object]) -> str:
    payload = {key: value for key, value in metadata.items() if key != "fingerprint"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SequenceContextCache:
    """Validated frozen per-residue sequence embeddings loaded from one NPZ."""

    metadata: dict[str, object]
    embeddings: Mapping[str, np.ndarray]

    @property
    def embedding_dim(self) -> int:
        value = self.metadata["embedding_dim"]
        if not isinstance(value, int):
            raise ValueError("sequence context embedding_dim must be an integer")
        return value

    @property
    def model_id(self) -> str:
        return str(self.metadata["model_id"])

    @property
    def fingerprint(self) -> str:
        return str(self.metadata["fingerprint"])

    def get(self, sequence: str) -> np.ndarray:
        key = sequence_sha256(sequence)
        try:
            embedding = self.embeddings[key]
        except KeyError as error:
            raise KeyError(f"sequence context missing sequence sha256={key}") from error
        if embedding.shape != (len(sequence), self.embedding_dim):
            raise ValueError(
                f"sequence context shape {embedding.shape} does not match "
                f"{(len(sequence), self.embedding_dim)}"
            )
        return embedding

    @classmethod
    def load(cls, path: str | Path) -> "SequenceContextCache":
        source = Path(path)
        with np.load(source, allow_pickle=False) as payload:
            if "metadata_json" not in payload:
                raise ValueError("sequence context cache is missing metadata_json")
            metadata = json.loads(str(payload["metadata_json"].item()))
            if metadata.get("format") != SEQUENCE_CONTEXT_FORMAT:
                raise ValueError(
                    f"unsupported sequence context format: {metadata.get('format')!r}"
                )
            if metadata.get("fingerprint") != _metadata_fingerprint(metadata):
                raise ValueError("sequence context metadata fingerprint mismatch")
            embedding_dim = int(metadata.get("embedding_dim", 0))
            if embedding_dim <= 0:
                raise ValueError("sequence context embedding_dim must be positive")
            embeddings: dict[str, np.ndarray] = {}
            for entry in metadata.get("entries", []):
                key = str(entry["sequence_sha256"])
                array_key = str(entry["array_key"])
                if key in embeddings:
                    raise ValueError(f"duplicate sequence context entry: {key}")
                if array_key not in payload:
                    raise ValueError(f"sequence context array is missing: {array_key}")
                embedding = np.asarray(payload[array_key], dtype=np.float32)
                expected_shape = (int(entry["length"]), embedding_dim)
                if embedding.shape != expected_shape:
                    raise ValueError(
                        f"sequence context array {array_key} has shape {embedding.shape}, "
                        f"expected {expected_shape}"
                    )
                if not np.isfinite(embedding).all():
                    raise ValueError(f"sequence context array {array_key} contains non-finite values")
                if _array_sha256(embedding) != entry.get("array_sha256"):
                    raise ValueError(f"sequence context checksum mismatch: {array_key}")
                embedding.setflags(write=False)
                embeddings[key] = embedding
        if len(embeddings) != int(metadata.get("sequence_count", -1)):
            raise ValueError("sequence context sequence_count does not match entries")
        return cls(metadata=dict(metadata), embeddings=embeddings)


def write_sequence_context_cache(
    path: str | Path,
    embeddings: Mapping[str, np.ndarray],
    *,
    model_id: str,
    model_revision: str | None = None,
) -> dict[str, object]:
    """Write deterministic metadata and float32 embeddings to a compressed NPZ."""
    if not embeddings:
        raise ValueError("sequence context embeddings must not be empty")
    normalized: dict[str, np.ndarray] = {}
    dimensions: set[int] = set()
    for sequence, values in sorted(embeddings.items()):
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != len(sequence) or array.shape[1] <= 0:
            raise ValueError(f"invalid embedding shape for sequence length {len(sequence)}")
        if not np.isfinite(array).all():
            raise ValueError("sequence context embeddings must be finite")
        normalized[sequence] = np.ascontiguousarray(array)
        dimensions.add(array.shape[1])
    if len(dimensions) != 1:
        raise ValueError("all sequence context embeddings must share one dimension")

    arrays: dict[str, np.ndarray] = {}
    entries: list[dict[str, object]] = []
    for index, (sequence, array) in enumerate(normalized.items()):
        array_key = f"embedding_{index:06d}"
        key = sequence_sha256(sequence)
        arrays[array_key] = array
        entries.append(
            {
                "sequence_sha256": key,
                "length": len(sequence),
                "array_key": array_key,
                "array_sha256": _array_sha256(array),
            }
        )
    metadata: dict[str, object] = {
        "format": SEQUENCE_CONTEXT_FORMAT,
        "model_id": model_id,
        "model_revision": model_revision,
        "embedding_dim": dimensions.pop(),
        "sequence_count": len(entries),
        "entries": entries,
    }
    metadata["fingerprint"] = _metadata_fingerprint(metadata)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **arrays,
    )
    return metadata
