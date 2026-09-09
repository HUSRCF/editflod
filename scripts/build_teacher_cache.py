"""Build auditable endpoint-response labels after teacher admission."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

from ospedit.data import file_sha256, json_safe, load_manifest, manifest_fingerprint, validate_manifest, verify_record_checksums
from ospedit.diagnostics import condition_response_diagnostic
from ospedit.geometry import local_frame_difference, residue_frames_masked
from ospedit.models import _endpoint
from ospedit.noise import make_shared_noise


def _load_factory(specification: str) -> Callable[[], Any]:
    if ":" not in specification:
        raise ValueError("endpoint factory must use module:callable syntax")
    module, attribute = specification.split(":", 1)
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    factory = getattr(importlib.import_module(module), attribute, None)
    if not callable(factory):
        raise TypeError(f"endpoint factory {specification!r} is not callable")
    return factory


def _safe_name(value: str) -> str:
    digest = hashlib.sha256(value.encode()).hexdigest()[:12]
    stem = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    return f"{stem[:80]}-{digest}"


def _admission_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise ValueError("admission report is missing records")
    return {str(row["pair_id"]): row for row in rows}


def build_cache(
    manifest_path: str | Path,
    admission_path: str | Path,
    endpoint_factory: str,
    output_dir: str | Path,
    *,
    split: str,
    noise_levels: tuple[float, ...],
    verify_checksums: bool = False,
    direction_report: str | Path | None = None,
) -> dict[str, Any]:
    records = load_manifest(manifest_path)
    errors = validate_manifest(records)
    if verify_checksums:
        errors.extend(error for record in records for error in verify_record_checksums(record))
    if errors:
        raise ValueError("manifest audit failed: " + "; ".join(errors))
    admission_file = Path(admission_path)
    admission = json.loads(admission_file.read_text())
    if not admission.get("ready_for_distillation", False):
        raise ValueError("admission report is not ready_for_distillation")
    direction = None
    if direction_report is not None:
        direction_path = Path(direction_report)
        direction = json.loads(direction_path.read_text())
        if direction.get("format") != "ospedit.teacher_evaluation.v1":
            raise ValueError("direction report must be ospedit.teacher_evaluation.v1")
        gate = direction.get("direction_gate", {})
        if gate.get("accepted") is not True:
            raise ValueError("direction report does not pass direction_gate")
    expected_fingerprint = manifest_fingerprint(records)
    if admission.get("manifest_fingerprint") != expected_fingerprint:
        raise ValueError("admission report manifest fingerprint does not match input manifest")
    if tuple(float(level) for level in admission.get("noise_levels", ())) != noise_levels:
        raise ValueError("cache noise levels must exactly match the admission report")
    if direction is not None:
        if direction.get("manifest_fingerprint") != expected_fingerprint:
            raise ValueError("direction report manifest fingerprint does not match input manifest")
        if direction.get("split") != split:
            raise ValueError("direction report split does not match requested cache split")
        if not np.isclose(float(direction.get("noise_level", float("nan"))), noise_levels[0], rtol=0.0, atol=1e-8):
            raise ValueError("direction report noise level does not match cache noise grid")
    admitted = _admission_rows(admission)
    selected = [record for record in records if record.split == split]
    if not selected:
        raise ValueError(f"manifest contains no records for split={split!r}")
    missing = [record.pair.pair_id for record in selected if record.pair.pair_id not in admitted]
    rejected = [record.pair.pair_id for record in selected if not admitted.get(record.pair.pair_id, {}).get("admission", {}).get("accepted", False)]
    if missing or rejected:
        raise ValueError(f"selected records are not admitted (missing={missing}, rejected={rejected})")

    endpoint = _load_factory(endpoint_factory)()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for record in selected:
        pair = record.pair
        diagnostic = condition_response_diagnostic(endpoint, pair, noise_levels)
        parent_rotations, parent_origins, parent_valid = residue_frames_masked(pair.parent_coords, pair.atom_names)
        level_arrays: dict[str, np.ndarray] = {}
        for level in noise_levels:
            noise = make_shared_noise(pair.parent_coords.shape, level, seed=0)
            target_rotations, target_origins = _endpoint(endpoint, pair.parent_coords, pair.mutant_sequence, level, noise)
            source_rotations, source_origins = _endpoint(endpoint, pair.parent_coords, pair.parent_sequence, level, noise)
            translation, rotation = local_frame_difference(
                parent_rotations, parent_origins, target_rotations, target_origins, source_rotations, source_origins
            )
            delta = np.concatenate((translation, rotation), axis=-1)
            valid = parent_valid & np.isfinite(delta).all(axis=-1)
            delta[~valid] = 0.0
            key = f"level_{level:g}"
            level_arrays[f"{key}_source_rotations"] = np.asarray(source_rotations, dtype=np.float32)
            level_arrays[f"{key}_source_origins"] = np.asarray(source_origins, dtype=np.float32)
            level_arrays[f"{key}_target_rotations"] = np.asarray(target_rotations, dtype=np.float32)
            level_arrays[f"{key}_target_origins"] = np.asarray(target_origins, dtype=np.float32)
            level_arrays[f"{key}_local_delta"] = np.asarray(delta, dtype=np.float32)
            level_arrays[f"{key}_valid"] = valid.astype(np.uint8)
        filename = _safe_name(pair.pair_id) + ".npz"
        np.savez_compressed(destination / filename, **level_arrays)
        entries.append({
            "pair_id": pair.pair_id,
            "parent_id": record.parent_id,
            "family_id": record.family_id,
            "split": record.split,
            "file": filename,
            "file_sha256": file_sha256(destination / filename),
            "length": pair.length,
            "mutation_indices": list(pair.mutation_indices),
            "diagnostic": diagnostic,
        })
    index = {
        "format": "ospedit.teacher_cache.v1",
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_fingerprint": expected_fingerprint,
        "admission_report": str(admission_file.resolve()),
        "endpoint_factory": endpoint_factory,
        "split": split,
        "noise_levels": list(noise_levels),
        "noise_seed": 0,
        "direction_report": str(Path(direction_report).resolve()) if direction_report is not None else None,
        "direction_gate": direction.get("direction_gate") if direction is not None else None,
        "entries": entries,
    }
    (destination / "index.json").write_text(json.dumps(json_safe(index), indent=2, sort_keys=True, allow_nan=False) + "\n")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build admitted teacher endpoint-response cache")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--admission-report", required=True)
    parser.add_argument("--endpoint-factory", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--noise-levels", type=float, nargs="+", default=(0.25, 0.5, 0.75))
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--direction-report", help="Optional direction audit JSON required to pass before cache generation")
    args = parser.parse_args()
    index = build_cache(
        args.manifest,
        args.admission_report,
        args.endpoint_factory,
        args.output_dir,
        split=args.split,
        noise_levels=tuple(args.noise_levels),
        verify_checksums=args.verify_checksums,
        direction_report=args.direction_report,
    )
    print(f"teacher cache written: {Path(args.output_dir) / 'index.json'} ({len(index['entries'])} records)")


if __name__ == "__main__":
    main()
