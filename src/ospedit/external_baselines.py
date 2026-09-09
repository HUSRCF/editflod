from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import subprocess
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from .data import PairRecord, StructurePair, file_sha256
from .data import align_coordinates_to_reference


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


@dataclass
class PreMutEditor:
    """Run the published PreMut checkpoint through an isolated subprocess."""

    records: Iterable[PairRecord]
    upstream_root: str | Path
    checkpoint: str | Path
    python_executable: str | Path
    prediction_dir: str | Path
    runner: str | Path
    cache_runner_checksum: str | None = None
    device: str = "cpu"
    seed: int = 0
    reuse_cache: bool = True
    replay_generation_cost: bool = False
    input_tolerance: float = 2e-3
    condition_branches_per_edit: int = field(default=1, init=False)
    sequence_encoder_calls_per_edit: int = field(default=0, init=False)
    network_calls_per_edit: int = field(default=1, init=False)
    use_reported_runtime: bool = field(default=False, init=False)
    last_runtime: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.upstream_root = Path(self.upstream_root).expanduser().resolve()
        self.checkpoint = Path(self.checkpoint).expanduser().resolve()
        self.python_executable = Path(self.python_executable).expanduser().resolve()
        self.prediction_dir = Path(self.prediction_dir).expanduser().resolve()
        self.runner = Path(self.runner).expanduser().resolve()
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        for label, path in (
            ("upstream_root", self.upstream_root),
            ("checkpoint", self.checkpoint),
            ("python_executable", self.python_executable),
            ("runner", self.runner),
        ):
            if not path.exists():
                raise ValueError(f"PreMut {label} does not exist: {path}")
        self._records = {record.pair.pair_id: record for record in self.records}
        if not self._records:
            raise ValueError("PreMutEditor requires at least one manifest record")
        self._checkpoint_checksum = file_sha256(self.checkpoint)
        self._runner_checksum = self.cache_runner_checksum or file_sha256(self.runner)
        self.use_reported_runtime = self.replay_generation_cost
        revision = subprocess.run(
            ["git", "-C", str(self.upstream_root), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
        self._upstream_revision = revision.stdout.strip() if revision.returncode == 0 else None

    def _cache_path(self, record: PairRecord) -> Path:
        return Path(self.prediction_dir) / f"{_safe_name(record.pair.pair_id)}.npz"

    def cache_path(self, record: PairRecord) -> Path:
        return self._cache_path(record)

    @staticmethod
    def _validate_record(record: PairRecord) -> None:
        pair = record.pair
        if len(pair.mutation_indices) != 1:
            raise ValueError("PreMut supports exactly one substitution per record")
        if not record.source_file or not record.source_chain:
            raise ValueError("PreMut requires source_file and source_chain in the manifest")
        if record.residue_map:
            residue_ids = [tuple(item) for item in record.residue_map]
            chains = {str(item[0]) for item in residue_ids}
            insertions = {str(item[2]).strip() for item in residue_ids}
            numbers = [int(item[1]) for item in residue_ids]
            expected_numbers = list(range(numbers[0], numbers[0] + len(numbers)))
            if chains != {record.source_chain} or insertions != {""} or numbers != expected_numbers:
                raise ValueError(
                    "PreMut requires one chain with consecutive author residue numbers and no insertion codes"
                )

    def _expected_metadata(self, record: PairRecord) -> dict[str, Any]:
        source_checksum = file_sha256(record.source_file)
        if record.source_checksum is not None and record.source_checksum != source_checksum:
            raise ValueError(f"PreMut source checksum mismatch for {record.pair.pair_id}")
        pair = record.pair
        index = pair.mutation_indices[0]
        return {
            "format": "ospedit.premut_prediction.v1",
            "pair_id": record.pair.pair_id,
            "source_checksum": source_checksum,
            "checkpoint_checksum": self._checkpoint_checksum,
            "runner_checksum": self._runner_checksum,
            "upstream_revision": self._upstream_revision,
            "device": self.device,
            "seed": self.seed,
            "mutation": f"{pair.parent_sequence[index]}_{index}_{pair.mutant_sequence[index]}",
            "chain": record.source_chain,
        }

    def expected_metadata(self, record: PairRecord) -> dict[str, Any]:
        self._validate_record(record)
        return self._expected_metadata(record)

    def cache_is_valid(self, record: PairRecord) -> bool:
        self._validate_record(record)
        output = self._cache_path(record)
        if not output.exists():
            return False
        try:
            _, _, metadata = self._read_prediction(output)
        except ValueError:
            return False
        expected = self._expected_metadata(record)
        return all(metadata.get(key) == value for key, value in expected.items())

    def job(self, record: PairRecord) -> dict[str, Any]:
        pair = record.pair
        index = pair.mutation_indices[0]
        return {
            "pair_id": pair.pair_id,
            "parent_structure": str(Path(record.source_file).expanduser().resolve()),
            "chain": record.source_chain,
            "mutation": f"{pair.parent_sequence[index]}_{index}_{pair.mutant_sequence[index]}",
            "seed": self.seed,
            "output": str(self._cache_path(record)),
            "metadata": self.expected_metadata(record),
        }

    @staticmethod
    def _read_prediction(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        try:
            with np.load(path, allow_pickle=False) as payload:
                prediction = np.asarray(payload["prediction"], dtype=float)
                input_backbone = np.asarray(payload["input_backbone"], dtype=float)
                metadata = json.loads(str(payload["metadata_json"].item()))
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid PreMut prediction cache {path}: {error}") from error
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid PreMut prediction metadata in {path}")
        return prediction, input_backbone, metadata

    def _run(self, record: PairRecord, output: Path) -> None:
        pair = record.pair
        index = pair.mutation_indices[0]
        mutation = f"{pair.parent_sequence[index]}_{index}_{pair.mutant_sequence[index]}"
        command = [
            str(self.python_executable),
            str(self.runner),
            "--upstream-root",
            str(self.upstream_root),
            "--checkpoint",
            str(self.checkpoint),
            "--parent-structure",
            str(Path(record.source_file).expanduser().resolve()),
            "--chain",
            record.source_chain,
            "--mutation",
            mutation,
            "--pair-id",
            pair.pair_id,
            "--source-checksum",
            file_sha256(record.source_file),
            "--runner-checksum",
            self._runner_checksum,
            "--upstream-revision",
            self._upstream_revision or "unknown",
            "--device",
            self.device,
            "--seed",
            str(self.seed),
            "--output",
            str(output),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no subprocess output"
            raise RuntimeError(f"PreMut inference failed for {pair.pair_id}: {detail}")

    def predict(self, pair: StructurePair) -> np.ndarray:
        try:
            record = self._records[pair.pair_id]
        except KeyError as error:
            raise ValueError(f"pair_id is absent from PreMutEditor records: {pair.pair_id}") from error
        self._validate_record(record)
        output = self._cache_path(record)
        expected = self._expected_metadata(record)
        use_cached = False
        if self.reuse_cache and output.exists():
            try:
                _, _, cached_metadata = self._read_prediction(output)
                use_cached = all(cached_metadata.get(key) == value for key, value in expected.items())
            except ValueError:
                use_cached = False
        if not use_cached:
            self._run(record, output)
        prediction, input_backbone, metadata = self._read_prediction(output)
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"PreMut cache metadata mismatch for {key}: {metadata.get(key)!r} != {value!r}")
        expected_shape = pair.parent_coords.shape
        if prediction.shape != expected_shape or input_backbone.shape != expected_shape:
            raise ValueError(
                f"PreMut backbone shape mismatch: prediction={prediction.shape}, "
                f"input={input_backbone.shape}, expected={expected_shape}"
            )
        valid = np.isfinite(pair.parent_coords).all(axis=-1) & np.isfinite(input_backbone).all(axis=-1)
        if not valid.any():
            raise ValueError("PreMut input validation found no shared finite backbone atoms")
        input_error = float(np.sqrt(np.mean((pair.parent_coords[valid] - input_backbone[valid]) ** 2)))
        if input_error > self.input_tolerance:
            raise ValueError(
                f"PreMut reconstructed input differs from manifest parent by {input_error:.6g} Angstrom"
            )
        if not np.isfinite(prediction[valid]).all():
            raise ValueError("PreMut prediction contains non-finite values at valid backbone atoms")
        runtime = metadata.get("runtime", {})
        self.last_runtime = {
            key: float(value)
            for key, value in runtime.items()
            if isinstance(value, (int, float)) and np.isfinite(value)
        }
        self.last_runtime["cache_hit"] = float(use_cached)
        self.last_runtime["input_reconstruction_rmsd"] = input_error
        load_seconds = self.last_runtime.get(
            "model_load_seconds_amortized", self.last_runtime.get("model_load_seconds", 0.0)
        )
        self.last_runtime["end_to_end_seconds"] = (
            load_seconds
            + self.last_runtime.get("preprocess_seconds", 0.0)
            + self.last_runtime.get("inference_seconds", 0.0)
            + self.last_runtime.get("postprocess_seconds", 0.0)
        )
        return prediction


@dataclass
class ESMFoldEditor:
    """Read identity-checked ESMFold predictions and align them to the parent frame."""

    prediction_dir: str | Path
    mode: str
    model_identity: dict[str, str]
    runner_checksum: str
    chunk_size: int = 64
    esm_precision: str = "bf16"
    condition_branches_per_edit: int = field(default=1, init=False)
    sequence_encoder_calls_per_edit: int = field(default=1, init=False)
    network_calls_per_edit: int = field(default=1, init=False)
    use_reported_runtime: bool = field(default=True, init=False)
    last_runtime: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.prediction_dir = Path(self.prediction_dir).expanduser().resolve()
        if self.mode not in {"default", "zero_extra_recycles"}:
            raise ValueError("ESMFold mode must be 'default' or 'zero_extra_recycles'")
        if self.chunk_size <= 0:
            raise ValueError("ESMFold chunk_size must be positive")
        if self.esm_precision not in {"bf16", "fp16", "fp32"}:
            raise ValueError("unsupported ESMFold ESM precision")

    def cache_path(self, pair: StructurePair) -> Path:
        return Path(self.prediction_dir) / f"{_safe_name(pair.pair_id)}.{self.mode}.npz"

    def expected_metadata(self, pair: StructurePair) -> dict[str, Any]:
        return {
            "format": "ospedit.esmfold_prediction.v1",
            "pair_id": pair.pair_id,
            "sequence_sha256": hashlib.sha256(pair.mutant_sequence.encode()).hexdigest(),
            "model_identity": self.model_identity,
            "runner_checksum": self.runner_checksum,
            "mode": self.mode,
            "num_recycles": None if self.mode == "default" else 0,
            "chunk_size": self.chunk_size,
            "esm_precision": self.esm_precision,
        }

    def cache_is_valid(self, pair: StructurePair) -> bool:
        path = self.cache_path(pair)
        if not path.exists():
            return False
        try:
            _, metadata = self._read_prediction(path)
        except ValueError:
            return False
        expected = self.expected_metadata(pair)
        return all(metadata.get(key) == value for key, value in expected.items())

    @staticmethod
    def _read_prediction(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
        try:
            with np.load(path, allow_pickle=False) as payload:
                prediction = np.asarray(payload["prediction"], dtype=float)
                metadata = json.loads(str(payload["metadata_json"].item()))
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid ESMFold prediction cache {path}: {error}") from error
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid ESMFold prediction metadata in {path}")
        return prediction, metadata

    def job(self, pair: StructurePair) -> dict[str, Any]:
        return {
            "pair_id": pair.pair_id,
            "sequence": pair.mutant_sequence,
            "output": str(self.cache_path(pair)),
            "metadata": self.expected_metadata(pair),
        }

    def predict(self, pair: StructurePair) -> np.ndarray:
        path = self.cache_path(pair)
        prediction, metadata = self._read_prediction(path)
        expected = self.expected_metadata(pair)
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"ESMFold cache metadata mismatch for {key}")
        if prediction.shape != pair.parent_coords.shape:
            raise ValueError(
                f"ESMFold backbone shape mismatch: {prediction.shape} != {pair.parent_coords.shape}"
            )
        if not np.isfinite(prediction).all():
            raise ValueError("ESMFold prediction contains non-finite backbone coordinates")
        started = perf_counter()
        aligned = align_coordinates_to_reference(pair.parent_coords, prediction)
        alignment_seconds = perf_counter() - started
        runtime = metadata.get("runtime", {})
        self.last_runtime = {
            key: float(value)
            for key, value in runtime.items()
            if isinstance(value, (int, float)) and np.isfinite(value)
        }
        self.last_runtime["parent_alignment_seconds"] = alignment_seconds
        self.last_runtime["end_to_end_seconds"] = (
            self.last_runtime.get("model_load_seconds_amortized", 0.0)
            + self.last_runtime.get("inference_seconds", 0.0)
            + self.last_runtime.get("pdb_conversion_seconds", 0.0)
            + alignment_seconds
        )
        return aligned
