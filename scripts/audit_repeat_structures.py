"""Estimate experimental background variation from same-sequence structures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ospedit.data import (
    align_coordinates_to_reference,
    load_manifest,
    parse_structure,
    validate_manifest,
    write_manifest,
)
from ospedit.metrics import evaluate_pair
from ospedit.geometry import local_frame_difference, residue_frames_masked


REQUIRED_COLUMNS = {
    "pair_id",
    "parent_structure",
    "parent_chain",
    "repeat_structure",
    "repeat_chain",
    "mutation_index",
}


def _rmsd(reference: np.ndarray, target: np.ndarray, residue_mask: np.ndarray | None = None) -> float:
    valid = np.isfinite(reference).all(axis=-1) & np.isfinite(target).all(axis=-1)
    if residue_mask is not None:
        valid &= residue_mask[:, None]
    difference = reference[valid] - target[valid]
    return float(np.sqrt(np.mean(np.sum(difference * difference, axis=-1)))) if len(difference) else float("nan")


def _distance_change_rms(reference: np.ndarray, target: np.ndarray, ca_index: int) -> float:
    ref_ca, target_ca = reference[:, ca_index], target[:, ca_index]
    valid = np.isfinite(ref_ca).all(axis=-1) & np.isfinite(target_ca).all(axis=-1)
    ref_ca, target_ca = ref_ca[valid], target_ca[valid]
    if len(ref_ca) < 2:
        return float("nan")
    ref_distances = np.linalg.norm(ref_ca[:, None] - ref_ca[None, :], axis=-1)
    target_distances = np.linalg.norm(target_ca[:, None] - target_ca[None, :], axis=-1)
    upper = np.triu_indices(len(ref_ca), k=1)
    return float(np.sqrt(np.mean((target_distances[upper] - ref_distances[upper]) ** 2)))


def _summary(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not len(finite):
        return {"count": 0, "mean": None, "median": None, "max": None}
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
    }


def _ratio(signal: float, background: float) -> float | None:
    return float(signal / background) if math.isfinite(signal) and math.isfinite(background) and background > 1e-12 else None


def _aggregate(values: list[float], method: str) -> float:
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("background metrics must be finite")
    return float(np.max(array) if method == "max" else np.median(array))


def audit_repeat_pairs(
    csv_path: str | Path,
    *,
    neighborhood_radius: float = 10.0,
    translation_scale: float = 1.0,
    rotation_scale: float = 0.25,
    mutation_manifest: str | Path | None = None,
    min_site_signal_to_background: float | None = None,
    min_neighborhood_signal_to_background: float | None = None,
    min_distance_signal_to_background: float | None = None,
    background_aggregation: str = "max",
    min_repeat_structures: int = 1,
    min_length: int | None = None,
    max_length: int | None = None,
    skip_unlisted_pairs: bool = False,
) -> dict[str, Any]:
    """Audit same-sequence pairs listed in a CSV without creating mutation labels."""
    if neighborhood_radius <= 0 or translation_scale <= 0 or rotation_scale <= 0:
        raise ValueError("radius and frame scales must be positive")
    if background_aggregation not in {"max", "median"}:
        raise ValueError("background_aggregation must be 'max' or 'median'")
    if min_repeat_structures <= 0:
        raise ValueError("min_repeat_structures must be positive")
    if min_length is not None and min_length <= 0:
        raise ValueError("min_length must be positive")
    if max_length is not None and max_length <= 0:
        raise ValueError("max_length must be positive")
    if min_length is not None and max_length is not None and min_length > max_length:
        raise ValueError("min_length must not exceed max_length")
    if skip_unlisted_pairs and mutation_manifest is None:
        raise ValueError("skip_unlisted_pairs requires mutation_manifest")
    thresholds = {
        "site_signal_to_background": min_site_signal_to_background,
        "neighborhood_signal_to_background": min_neighborhood_signal_to_background,
        "distance_signal_to_background": min_distance_signal_to_background,
    }
    if any(value is not None and (not math.isfinite(value) or value < 0) for value in thresholds.values()):
        raise ValueError("signal-to-background thresholds must be finite and non-negative")
    if any(value is not None for value in thresholds.values()) and mutation_manifest is None:
        raise ValueError("signal-to-background thresholds require mutation_manifest")
    source = Path(csv_path)
    mutation_records = {}
    if mutation_manifest is not None:
        manifest_records = load_manifest(mutation_manifest)
        errors = validate_manifest(manifest_records)
        if errors:
            raise ValueError("mutation manifest audit failed: " + "; ".join(errors))
        mutation_records = {record.pair.pair_id: record for record in manifest_records}
    rows: list[dict[str, Any]] = []
    repeat_keys: set[tuple[str, str, str]] = set()
    skipped_unlisted_rows = 0
    skipped_out_of_scope_rows = 0
    skipped_out_of_scope_pairs: set[str] = set()
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or ()):
            raise ValueError("repeat CSV is missing required columns")
        for line_number, raw in enumerate(reader, start=2):
            if mutation_manifest is not None and raw["pair_id"] not in mutation_records:
                if skip_unlisted_pairs:
                    skipped_unlisted_rows += 1
                    continue
                raise ValueError(f"line {line_number}: pair_id is absent from mutation manifest")
            if mutation_manifest is not None:
                pair_length = mutation_records[raw["pair_id"]].pair.length
                if (min_length is not None and pair_length < min_length) or (
                    max_length is not None and pair_length > max_length
                ):
                    skipped_out_of_scope_rows += 1
                    skipped_out_of_scope_pairs.add(raw["pair_id"])
                    continue
            parent_path = (source.parent / raw["parent_structure"]).resolve()
            repeat_path = (source.parent / raw["repeat_structure"]).resolve()
            repeat_key = (raw["pair_id"], str(repeat_path), raw["repeat_chain"].strip())
            if repeat_key in repeat_keys:
                raise ValueError(f"line {line_number}: duplicate repeat structure for pair_id")
            repeat_keys.add(repeat_key)
            parent = parse_structure(parent_path, raw["parent_chain"].strip())
            repeat = parse_structure(repeat_path, raw["repeat_chain"].strip())
            if parent.sequence != repeat.sequence:
                raise ValueError(f"line {line_number}: repeat structures do not have identical sequences")
            mutation_index = int(raw["mutation_index"])
            if not 0 <= mutation_index < len(parent.sequence):
                raise ValueError(f"line {line_number}: mutation_index is out of bounds")
            aligned = align_coordinates_to_reference(parent.coords, repeat.coords)
            parent_rot, parent_origin, parent_valid = residue_frames_masked(parent.coords, parent.atom_names)
            repeat_rot, repeat_origin, repeat_valid = residue_frames_masked(aligned, repeat.atom_names)
            translation, rotation = local_frame_difference(
                parent_rot,
                parent_origin,
                repeat_rot,
                repeat_origin,
                parent_rot,
                parent_origin,
            )
            valid = parent_valid & repeat_valid
            if not valid.any():
                raise ValueError(f"line {line_number}: no shared valid residue frames")
            normalized = np.concatenate(
                (translation / translation_scale, rotation / rotation_scale), axis=-1
            )
            ca_index = parent.atom_names.index("CA")
            ca = parent.coords[:, ca_index]
            mutation_ca = ca[mutation_index]
            local = np.isfinite(ca).all(axis=-1)
            if np.isfinite(mutation_ca).all():
                local &= np.linalg.norm(ca - mutation_ca, axis=-1) <= neighborhood_radius
            else:
                local[:] = False
            row: dict[str, Any] = {
                "pair_id": raw["pair_id"],
                "parent_structure": str(parent_path),
                "repeat_structure": str(repeat_path),
                "length": len(parent.sequence),
                "mutation_index": mutation_index,
                "mapping": "residue_id" if parent.residue_ids == repeat.residue_ids else "sequence_index",
                "backbone_rmsd_angstrom": _rmsd(parent.coords, aligned),
                "mutation_site_rmsd_angstrom": _rmsd(
                    parent.coords,
                    aligned,
                    np.arange(len(parent.sequence)) == mutation_index,
                ),
                "neighborhood_rmsd_angstrom": _rmsd(parent.coords, aligned, local),
                "distance_change_rms_angstrom": _distance_change_rms(parent.coords, aligned, ca_index),
                "max_translation_angstrom": float(np.max(np.linalg.norm(translation[valid], axis=-1))),
                "max_rotation_radian": float(np.max(np.linalg.norm(rotation[valid], axis=-1))),
                "max_normalized_delta_norm": float(np.max(np.linalg.norm(normalized[valid], axis=-1))),
            }
            if mutation_manifest is not None:
                mutation_pair = mutation_records[raw["pair_id"]].pair
                if mutation_pair.parent_sequence != parent.sequence:
                    raise ValueError(f"line {line_number}: manifest parent sequence does not match repeat audit")
                mutation_metrics = evaluate_pair(mutation_pair, mutation_pair.parent_coords)
                mutant_backbone = _rmsd(mutation_pair.parent_coords, mutation_pair.mutant_coords)
                mutant_site = float(mutation_metrics["mutation_site_backbone_error"])
                mutant_local = float(mutation_metrics["local_backbone_error"])
                mutant_distance = float(mutation_metrics["distance_change_error"])
                row.update({
                    "mutant_backbone_rmsd_angstrom": mutant_backbone,
                    "mutant_site_rmsd_angstrom": mutant_site,
                    "mutant_neighborhood_rmsd_angstrom": mutant_local,
                    "mutant_distance_change_rms_angstrom": mutant_distance,
                    "site_signal_to_background": _ratio(mutant_site, float(row["mutation_site_rmsd_angstrom"])),
                    "neighborhood_signal_to_background": _ratio(mutant_local, float(row["neighborhood_rmsd_angstrom"])),
                    "distance_signal_to_background": _ratio(mutant_distance, float(row["distance_change_rms_angstrom"])),
                })
            rows.append(row)
    background_metric_names: tuple[str, ...] = (
        "backbone_rmsd_angstrom",
        "mutation_site_rmsd_angstrom",
        "neighborhood_rmsd_angstrom",
        "distance_change_rms_angstrom",
        "max_translation_angstrom",
        "max_rotation_radian",
        "max_normalized_delta_norm",
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["pair_id"]), []).append(row)
    pair_rows: list[dict[str, Any]] = []
    for pair_id, repeats in grouped.items():
        parent_structures = {str(row["parent_structure"]) for row in repeats}
        mutation_indices = {int(row["mutation_index"]) for row in repeats}
        if len(parent_structures) != 1 or len(mutation_indices) != 1:
            raise ValueError(f"pair_id {pair_id!r} has inconsistent parent or mutation index")
        pair_row: dict[str, Any] = {
            "pair_id": pair_id,
            "parent_structure": repeats[0]["parent_structure"],
            "length": repeats[0]["length"],
            "mutation_index": repeats[0]["mutation_index"],
            "repeat_structures": len(repeats),
            "mapping_modes": sorted({str(row["mapping"]) for row in repeats}),
        }
        pair_row.update({
            name: _aggregate([float(row[name]) for row in repeats], background_aggregation)
            for name in background_metric_names
        })
        if mutation_manifest is not None:
            signal_names = (
                "mutant_backbone_rmsd_angstrom",
                "mutant_site_rmsd_angstrom",
                "mutant_neighborhood_rmsd_angstrom",
                "mutant_distance_change_rms_angstrom",
            )
            pair_row.update({name: float(repeats[0][name]) for name in signal_names})
            pair_row.update({
                "site_signal_to_background": _ratio(
                    float(pair_row["mutant_site_rmsd_angstrom"]),
                    float(pair_row["mutation_site_rmsd_angstrom"]),
                ),
                "neighborhood_signal_to_background": _ratio(
                    float(pair_row["mutant_neighborhood_rmsd_angstrom"]),
                    float(pair_row["neighborhood_rmsd_angstrom"]),
                ),
                "distance_signal_to_background": _ratio(
                    float(pair_row["mutant_distance_change_rms_angstrom"]),
                    float(pair_row["distance_change_rms_angstrom"]),
                ),
            })
            length = int(pair_row["length"])
            pair_row["selected"] = (
                len(repeats) >= min_repeat_structures
                and (min_length is None or length >= min_length)
                and (max_length is None or length <= max_length)
                and all(
                    threshold is None
                    or (pair_row.get(name) is not None and float(pair_row[name]) >= threshold)
                    for name, threshold in thresholds.items()
                )
            )
        pair_rows.append(pair_row)
    metric_names = background_metric_names
    if mutation_manifest is not None:
        metric_names += (
            "mutant_backbone_rmsd_angstrom",
            "mutant_site_rmsd_angstrom",
            "mutant_neighborhood_rmsd_angstrom",
            "mutant_distance_change_rms_angstrom",
            "site_signal_to_background",
            "neighborhood_signal_to_background",
            "distance_signal_to_background",
        )
    return {
        "format": "ospedit.repeat_structure_audit.v3",
        "coordinate_rmsd_definition": "sqrt(mean(sum((atom_xyz_error)**2, axis=-1)))",
        "csv": str(source.resolve()),
        "neighborhood_radius": neighborhood_radius,
        "translation_scale": translation_scale,
        "rotation_scale": rotation_scale,
        "mutation_manifest": str(Path(mutation_manifest).resolve()) if mutation_manifest is not None else None,
        "selection_thresholds": thresholds,
        "background_aggregation": background_aggregation,
        "min_repeat_structures": min_repeat_structures,
        "min_length": min_length,
        "max_length": max_length,
        "skipped_unlisted_rows": skipped_unlisted_rows,
        "skipped_out_of_scope_rows": skipped_out_of_scope_rows,
        "skipped_out_of_scope_pairs": sorted(skipped_out_of_scope_pairs),
        "selected_records": sum(bool(row.get("selected")) for row in pair_rows) if mutation_manifest is not None else None,
        "records": rows,
        "pairs": pair_rows,
        "repeat_summary": {
            name: _summary([float(row[name]) for row in rows if row.get(name) is not None])
            for name in background_metric_names
        },
        "summary": {
            name: _summary([float(row[name]) for row in pair_rows if row.get(name) is not None])
            for name in metric_names
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit same-sequence experimental structure variation")
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--neighborhood-radius", type=float, default=10.0)
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument("--rotation-scale", type=float, default=0.25)
    parser.add_argument("--mutation-manifest", help="Optional mutant-pair manifest keyed by pair_id")
    parser.add_argument("--min-site-signal-to-background", type=float)
    parser.add_argument("--min-neighborhood-signal-to-background", type=float)
    parser.add_argument("--min-distance-signal-to-background", type=float)
    parser.add_argument("--filtered-output", help="Optional mutation manifest containing selected records")
    parser.add_argument("--background-aggregation", choices=("max", "median"), default="max")
    parser.add_argument("--min-repeat-structures", type=int, default=1)
    parser.add_argument("--min-length", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--skip-unlisted-pairs", action="store_true")
    args = parser.parse_args()
    try:
        report = audit_repeat_pairs(
            args.pairs_csv,
            neighborhood_radius=args.neighborhood_radius,
            translation_scale=args.translation_scale,
            rotation_scale=args.rotation_scale,
            mutation_manifest=args.mutation_manifest,
            min_site_signal_to_background=args.min_site_signal_to_background,
            min_neighborhood_signal_to_background=args.min_neighborhood_signal_to_background,
            min_distance_signal_to_background=args.min_distance_signal_to_background,
            background_aggregation=args.background_aggregation,
            min_repeat_structures=args.min_repeat_structures,
            min_length=args.min_length,
            max_length=args.max_length,
            skip_unlisted_pairs=args.skip_unlisted_pairs,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if args.filtered_output:
        if not args.mutation_manifest:
            raise SystemExit("--filtered-output requires --mutation-manifest")
        selected_ids = {row["pair_id"] for row in report["pairs"] if row.get("selected")}
        if not selected_ids:
            raise SystemExit("signal-to-background filter selected no records")
        records = [
            record
            for record in load_manifest(args.mutation_manifest)
            if record.pair.pair_id in selected_ids
        ]
        errors = validate_manifest(records)
        if errors:
            raise SystemExit("filtered manifest failed validation: " + "; ".join(errors))
        write_manifest(records, args.filtered_output)
    print(f"repeat-structure audit written: {output} ({len(report['records'])} records)")


if __name__ == "__main__":
    main()
