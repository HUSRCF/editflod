"""Audit experimental parent-to-mutant response magnitudes in a manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ospedit.data import load_manifest, manifest_fingerprint, validate_manifest, write_manifest
from ospedit.student_data import target_local_delta


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def audit_response_scale(
    manifest: str | Path,
    *,
    translation_scale: float = 1.0,
    rotation_scale: float = 0.25,
    max_normalized_norm: float | None = None,
) -> tuple[dict[str, Any], list[Any]]:
    records = load_manifest(manifest)
    errors = validate_manifest(records)
    if errors:
        raise ValueError("manifest audit failed: " + "; ".join(errors))
    if translation_scale <= 0 or rotation_scale <= 0:
        raise ValueError("translation_scale and rotation_scale must be positive")
    if max_normalized_norm is not None and (not math.isfinite(max_normalized_norm) or max_normalized_norm <= 0):
        raise ValueError("max_normalized_norm must be finite and positive")
    rows: list[dict[str, Any]] = []
    by_split: dict[str, list[float]] = {split: [] for split in ("train", "dev", "test")}
    selected: list[Any] = []
    for record in records:
        delta, valid = target_local_delta(
            record.pair,
            translation_scale=translation_scale,
            rotation_scale=rotation_scale,
        )
        values = delta[valid]
        norm = float(np.max(np.linalg.norm(values, axis=-1))) if len(values) else float("nan")
        translation = values[..., :3] * translation_scale
        rotation = values[..., 3:] * rotation_scale
        max_translation = (
            float(np.max(np.linalg.norm(translation, axis=-1))) if len(translation) else float("nan")
        )
        max_rotation = float(np.max(np.linalg.norm(rotation, axis=-1))) if len(rotation) else float("nan")
        finite = math.isfinite(norm)
        keep = finite and (max_normalized_norm is None or norm <= max_normalized_norm)
        if keep:
            selected.append(record)
        if finite:
            by_split.setdefault(record.split, []).append(norm)
        rows.append({
            "pair_id": record.pair.pair_id,
            "parent_id": record.parent_id,
            "family_id": record.family_id,
            "split": record.split,
            "max_normalized_delta_norm": norm,
            "max_translation_angstrom": max_translation,
            "max_rotation_radian": max_rotation,
            "selected": keep,
        })
    report = {
        "format": "ospedit.response_scale_audit.v1",
        "manifest": str(Path(manifest).resolve()),
        "manifest_fingerprint": manifest_fingerprint(records),
        "translation_scale": float(translation_scale),
        "rotation_scale": float(rotation_scale),
        "max_normalized_norm": max_normalized_norm,
        "records": rows,
        "summary": {split: _summary(values) for split, values in by_split.items()},
        "selected_records": len(selected),
    }
    return report, selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit parent-to-mutant response magnitudes")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--filtered-output", help="Optional manifest containing records under the response threshold")
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument("--rotation-scale", type=float, default=0.25)
    parser.add_argument("--max-normalized-norm", type=float)
    args = parser.parse_args()
    try:
        report, selected = audit_response_scale(
            args.manifest,
            translation_scale=args.translation_scale,
            rotation_scale=args.rotation_scale,
            max_normalized_norm=args.max_normalized_norm,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if args.filtered_output:
        if not selected:
            raise SystemExit("response filter selected no records")
        filtered_errors = validate_manifest(selected)
        if filtered_errors:
            raise SystemExit("filtered manifest failed validation: " + "; ".join(filtered_errors))
        write_manifest(selected, args.filtered_output)
    print(f"response audit written: {destination} ({len(report['records'])} records)")


if __name__ == "__main__":
    main()
