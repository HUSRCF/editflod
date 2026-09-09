"""Audit response scope and representation recoverability by family."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ospedit.data import PairRecord, json_safe, load_manifest, manifest_fingerprint, validate_manifest
from ospedit.metrics import METRIC_SCHEMA_VERSION, evaluate_pair, region_masks
from ospedit.oracle import oracle_local_delta, oracle_prediction


REPORT_FORMAT = "ospedit.response_learnability_audit.v1"
ERROR_METRICS = (
    "local_backbone_error",
    "mutation_site_backbone_error",
    "remote_target_error",
    "distance_change_error",
    "local_distance_change_error",
    "remote_distance_change_error",
)


def _finite_mean(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if len(finite) else None


def _finite_quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {"count": 0, "mean": None, "q25": None, "median": None, "q75": None}
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "median": float(np.median(finite)),
        "q75": float(np.quantile(finite, 0.75)),
    }


def _recoverable_fraction(copy_error: float, oracle_error: float) -> float:
    if not math.isfinite(copy_error) or not math.isfinite(oracle_error) or copy_error <= 1e-12:
        return float("nan")
    return float((copy_error - oracle_error) / copy_error)


def _rms_norm(values: np.ndarray) -> float:
    finite = values[np.isfinite(values).all(axis=-1)]
    return float(np.sqrt(np.mean(np.sum(finite * finite, axis=-1)))) if len(finite) else float("nan")


def _energy_fraction(values: np.ndarray, mask: np.ndarray) -> float:
    finite = np.isfinite(values).all(axis=-1)
    energy = np.sum(values[finite] * values[finite])
    selected = np.sum(values[finite & mask] * values[finite & mask])
    return float(selected / energy) if energy > 1e-12 else float("nan")


def _record_row(record: PairRecord, translation_scale: float, rotation_scale: float) -> dict[str, Any]:
    pair = record.pair
    copy_metrics = evaluate_pair(pair, pair.parent_coords)
    oracle_metrics = evaluate_pair(
        pair,
        oracle_prediction(
            pair,
            translation_scale=translation_scale,
            rotation_scale=rotation_scale,
        ),
    )
    delta, valid = oracle_local_delta(
        pair,
        translation_scale=translation_scale,
        rotation_scale=rotation_scale,
    )
    regions = region_masks(pair)
    translation = delta[:, :3] * translation_scale
    rotation = delta[:, 3:] * rotation_scale
    local = valid & regions["local"]
    remote = valid & regions["remote"]
    recoverability = {
        metric: _recoverable_fraction(float(copy_metrics[metric]), float(oracle_metrics[metric]))
        for metric in ERROR_METRICS
    }
    return {
        "pair_id": pair.pair_id,
        "parent_id": record.parent_id,
        "family_id": record.family_id,
        "split": record.split,
        "length": pair.length,
        "mutation_count": len(pair.mutation_indices),
        "valid_residues": int(valid.sum()),
        "local_residues": int(local.sum()),
        "remote_residues": int(remote.sum()),
        "copy_error": {metric: copy_metrics[metric] for metric in ERROR_METRICS},
        "oracle_error": {metric: oracle_metrics[metric] for metric in ERROR_METRICS},
        "recoverable_fraction": recoverability,
        "response_scope": {
            "translation_rms_angstrom": _rms_norm(translation[valid]),
            "rotation_rms_radian": _rms_norm(rotation[valid]),
            "local_translation_rms_angstrom": _rms_norm(translation[local]),
            "remote_translation_rms_angstrom": _rms_norm(translation[remote]),
            "local_rotation_rms_radian": _rms_norm(rotation[local]),
            "remote_rotation_rms_radian": _rms_norm(rotation[remote]),
            "local_translation_energy_fraction": _energy_fraction(translation, local),
            "remote_translation_energy_fraction": _energy_fraction(translation, remote),
            "local_rotation_energy_fraction": _energy_fraction(rotation, local),
            "remote_rotation_energy_fraction": _energy_fraction(rotation, remote),
        },
    }


def _flat_numeric(row: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for section in ("copy_error", "oracle_error", "recoverable_fraction", "response_scope"):
        for name, value in row[section].items():
            values[f"{section}.{name}"] = float(value)
    return values


def _group_mean(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    flattened = [_flat_numeric(row) for row in rows]
    names = sorted({name for row in flattened for name in row})
    return {name: _finite_mean(row[name] for row in flattened) for name in names}


def response_learnability_report(
    manifest: str | Path,
    *,
    translation_scale: float = 1.0,
    rotation_scale: float = 1.0,
) -> dict[str, Any]:
    records = load_manifest(manifest)
    errors = validate_manifest(records)
    if errors:
        raise ValueError("manifest audit failed: " + "; ".join(errors))
    if translation_scale <= 0 or rotation_scale <= 0:
        raise ValueError("translation_scale and rotation_scale must be positive")
    rows = [_record_row(record, translation_scale, rotation_scale) for record in records]
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family_rows[row["family_id"]].append(row)
    families = {
        family_id: {
            "split": grouped[0]["split"],
            "records": len(grouped),
            "metrics": _group_mean(grouped),
        }
        for family_id, grouped in sorted(family_rows.items())
    }
    split_summary: dict[str, Any] = {}
    for split in ("train", "dev", "test"):
        split_records = [row for row in rows if row["split"] == split]
        split_families = [payload for payload in families.values() if payload["split"] == split]
        names = sorted({name for row in split_records for name in _flat_numeric(row)})
        split_summary[split] = {
            "records": len(split_records),
            "families": len(split_families),
            "record_macro": {
                name: _finite_quantiles(_flat_numeric(row)[name] for row in split_records)
                for name in names
            },
            "family_macro": {
                name: _finite_quantiles(float(family["metrics"][name]) for family in split_families if family["metrics"][name] is not None)
                for name in names
            },
        }
    return json_safe({
        "format": REPORT_FORMAT,
        "metric_schema": METRIC_SCHEMA_VERSION,
        "manifest": str(Path(manifest).resolve()),
        "manifest_fingerprint": manifest_fingerprint(records),
        "translation_scale": float(translation_scale),
        "rotation_scale": float(rotation_scale),
        "records": rows,
        "families": families,
        "summary": split_summary,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument("--rotation-scale", type=float, default=1.0)
    args = parser.parse_args()
    try:
        report = response_learnability_report(
            args.manifest,
            translation_scale=args.translation_scale,
            rotation_scale=args.rotation_scale,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"response learnability audit written: {destination} ({len(report['records'])} records)")


if __name__ == "__main__":
    main()
