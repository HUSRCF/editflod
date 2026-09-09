"""Measure the representation floor of local-frame student outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ospedit.data import json_safe, load_manifest, manifest_fingerprint, validate_manifest
from ospedit.metrics import METRIC_SCHEMA_VERSION, evaluate_pair
from ospedit.oracle import oracle_local_delta, oracle_prediction


def oracle_report(
    manifest: str | Path,
    *,
    split: str | None = None,
    bounds: tuple[float | None, ...] = (None, 0.1),
    translation_scale: float = 1.0,
    rotation_scale: float = 1.0,
    localization_radius: float | None = None,
    localization_transition: float = 5.0,
) -> dict[str, Any]:
    records = load_manifest(manifest)
    errors = validate_manifest(records)
    if errors:
        raise ValueError("manifest audit failed: " + "; ".join(errors))
    selected = [record for record in records if split is None or record.split == split]
    if not selected:
        raise ValueError(f"manifest contains no records for split={split!r}")
    if not bounds or any(bound is not None and bound <= 0 for bound in bounds):
        raise ValueError("bounds must contain None or positive values")
    rows: list[dict[str, Any]] = []
    for record in selected:
        for bound in bounds:
            delta, valid = oracle_local_delta(
                record.pair,
                translation_scale=translation_scale,
                rotation_scale=rotation_scale,
                max_normalized_delta=bound,
                localization_radius=localization_radius,
                localization_transition=localization_transition,
            )
            prediction = oracle_prediction(
                record.pair,
                translation_scale=translation_scale,
                rotation_scale=rotation_scale,
                max_normalized_delta=bound,
                localization_radius=localization_radius,
                localization_transition=localization_transition,
            )
            rows.append({
                "pair_id": record.pair.pair_id,
                "parent_id": record.parent_id,
                "family_id": record.family_id,
                "split": record.split,
                "bound": bound,
                "valid_residues": int(valid.sum()),
                "max_normalized_delta": float(np.max(np.abs(delta[valid]))) if valid.any() else None,
                "metrics": evaluate_pair(record.pair, prediction),
            })
    summaries: dict[str, dict[str, float]] = {}
    for bound in bounds:
        key = "unbounded" if bound is None else str(bound)
        matching = [row for row in rows if row["bound"] == bound]
        metric_names = [name for name, value in matching[0]["metrics"].items() if isinstance(value, (int, float))]
        summaries[key] = {
            name: float(np.nanmean([float(row["metrics"][name]) for row in matching]))
            for name in metric_names
            if np.isfinite([float(row["metrics"][name]) for row in matching]).any()
        }
    return json_safe({
        "format": "ospedit.oracle_reconstruction.v1",
        "metric_schema": METRIC_SCHEMA_VERSION,
        "manifest": str(Path(manifest).resolve()),
        "manifest_fingerprint": manifest_fingerprint(selected),
        "split": split,
        "translation_scale": translation_scale,
        "rotation_scale": rotation_scale,
        "localization_radius": localization_radius,
        "localization_transition": localization_transition,
        "bounds": list(bounds),
        "records": rows,
        "summary": summaries,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--split")
    parser.add_argument("--bound", type=float, action="append", default=[])
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument("--rotation-scale", type=float, default=1.0)
    parser.add_argument("--localization-radius", type=float)
    parser.add_argument("--localization-transition", type=float, default=5.0)
    args = parser.parse_args()
    bounds: tuple[float | None, ...] = (None, *(args.bound or [0.1]))
    report = oracle_report(
        args.manifest,
        split=args.split,
        bounds=bounds,
        translation_scale=args.translation_scale,
        rotation_scale=args.rotation_scale,
        localization_radius=args.localization_radius,
        localization_transition=args.localization_transition,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
