"""Consolidate same-sequence structure controls without using mutant labels."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ospedit.data import load_manifest, manifest_fingerprint, validate_manifest


REPORT_FORMAT = "ospedit.background_control_coverage.v1"
REPEAT_FORMAT = "ospedit.repeat_structure_audit.v3"
RMSD_DEFINITION = "sqrt(mean(sum((atom_xyz_error)**2, axis=-1)))"
BACKGROUND_METRICS = (
    "backbone_rmsd_angstrom",
    "mutation_site_rmsd_angstrom",
    "neighborhood_rmsd_angstrom",
    "distance_change_rms_angstrom",
    "max_translation_angstrom",
    "max_rotation_radian",
    "max_normalized_delta_norm",
)


def _quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {"count": 0, "mean": None, "q25": None, "median": None, "q75": None, "max": None}
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "median": float(np.median(finite)),
        "q75": float(np.quantile(finite, 0.75)),
        "max": float(np.max(finite)),
    }


def _coverage(records: list[Any], covered_ids: set[str]) -> dict[str, int | float]:
    covered = [record for record in records if record.pair.pair_id in covered_ids]
    total_parents = {record.parent_id for record in records}
    total_families = {record.family_id for record in records}
    covered_parents = {record.parent_id for record in covered}
    covered_families = {record.family_id for record in covered}
    return {
        "pairs": len(records),
        "covered_pairs": len(covered),
        "pair_fraction": len(covered) / len(records) if records else 0.0,
        "parents": len(total_parents),
        "covered_parents": len(covered_parents),
        "families": len(total_families),
        "covered_families": len(covered_families),
    }


def background_coverage_report(
    manifest: str | Path,
    repeat_audits: Iterable[str | Path],
) -> dict[str, Any]:
    """Build a background-only index from corrected repeat-structure audits."""
    manifest_records = load_manifest(manifest)
    errors = validate_manifest(manifest_records)
    if errors:
        raise ValueError("manifest audit failed: " + "; ".join(errors))
    by_pair = {record.pair.pair_id: record for record in manifest_records}
    audit_paths = [Path(path).resolve() for path in repeat_audits]
    if not audit_paths:
        raise ValueError("at least one repeat audit is required")

    # A repeat may occur in multiple discovery batches. It contributes once.
    repeats: dict[tuple[str, str, str], dict[str, Any]] = {}
    sources: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for audit_path in audit_paths:
        payload = json.loads(audit_path.read_text())
        if payload.get("format") != REPEAT_FORMAT:
            raise ValueError(f"{audit_path}: expected {REPEAT_FORMAT}")
        if payload.get("coordinate_rmsd_definition") != RMSD_DEFINITION:
            raise ValueError(f"{audit_path}: incompatible coordinate RMSD definition")
        for row in payload.get("records", []):
            pair_id = str(row["pair_id"])
            if pair_id not in by_pair:
                continue
            record = by_pair[pair_id]
            mutation_indices = tuple(record.pair.mutation_indices)
            if mutation_indices != (int(row["mutation_index"]),):
                raise ValueError(f"{audit_path}: mutation index mismatch for {pair_id}")
            repeat_structure = str(Path(row["repeat_structure"]).resolve())
            repeat_chain = str(row["repeat_chain"])
            key = pair_id, repeat_structure, repeat_chain
            metrics = {name: float(row[name]) for name in BACKGROUND_METRICS}
            if not all(math.isfinite(value) and value >= 0 for value in metrics.values()):
                raise ValueError(f"{audit_path}: invalid background metric for {pair_id}")
            if key in repeats and any(
                not math.isclose(metrics[name], repeats[key][name], rel_tol=1e-9, abs_tol=1e-9)
                for name in BACKGROUND_METRICS
            ):
                raise ValueError(f"{audit_path}: inconsistent duplicate repeat for {pair_id}")
            repeats[key] = {
                "repeat_structure": repeat_structure,
                "repeat_chain": repeat_chain,
                "parent_structure": str(Path(row["parent_structure"]).resolve()),
                "parent_chain": str(row["parent_chain"]),
                **metrics,
            }
            sources[key].add(str(audit_path))

    grouped: dict[str, list[tuple[tuple[str, str, str], dict[str, Any]]]] = defaultdict(list)
    for key, row in repeats.items():
        grouped[key[0]].append((key, row))
    rows: list[dict[str, Any]] = []
    for pair_id, controls in sorted(grouped.items()):
        record = by_pair[pair_id]
        if not record.source_file or not record.source_chain:
            raise ValueError(f"manifest is missing parent structure provenance for {pair_id}")
        rows.append({
            "pair_id": pair_id,
            "parent_id": record.parent_id,
            "family_id": record.family_id,
            "split": record.split,
            "mutation_index": record.pair.mutation_indices[0],
            "parent_structure": str(Path(record.source_file).resolve()),
            "parent_chain": record.source_chain,
            "repeat_structures": len(controls),
            "background_max": {
                name: max(row[name] for _, row in controls)
                for name in BACKGROUND_METRICS
            },
            "background_median": {
                name: float(np.median([row[name] for _, row in controls]))
                for name in BACKGROUND_METRICS
            },
            "controls": [
                {
                    "repeat_structure": row["repeat_structure"],
                    "repeat_chain": row["repeat_chain"],
                    "audit_parent_structure": row["parent_structure"],
                    "audit_parent_chain": row["parent_chain"],
                    "background": {name: row[name] for name in BACKGROUND_METRICS},
                    "sources": sorted(sources[key]),
                }
                for key, row in sorted(controls, key=lambda item: item[1]["repeat_structure"])
            ],
        })

    covered_ids = set(grouped)
    split_summary = {
        split: _coverage([record for record in manifest_records if record.split == split], covered_ids)
        for split in ("train", "dev", "test")
    }
    return {
        "format": REPORT_FORMAT,
        "metric_schema": "ospedit.structure_metrics.v2",
        "coordinate_rmsd_definition": RMSD_DEFINITION,
        "label_policy": "parent_same_sequence_controls_only",
        "environment_matching": "not_established_by_repeat_structure_identity_alone",
        "manifest": str(Path(manifest).resolve()),
        "manifest_fingerprint": manifest_fingerprint(manifest_records),
        "repeat_audits": [str(path) for path in audit_paths],
        "deduplication_key": ["pair_id", "repeat_structure", "repeat_chain"],
        "summary": {
            "overall": _coverage(manifest_records, covered_ids),
            "splits": split_summary,
            "unique_controls": len(repeats),
            "background_max": {
                name: _quantiles(row["background_max"][name] for row in rows)
                for name in BACKGROUND_METRICS
            },
            "background_median": {
                name: _quantiles(row["background_median"][name] for row in rows)
                for name in BACKGROUND_METRICS
            },
        },
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--repeat-audit", action="append", required=True)
    args = parser.parse_args()
    try:
        report = background_coverage_report(args.manifest, args.repeat_audit)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        f"background coverage written: {destination} "
        f"({report['summary']['overall']['covered_pairs']}/{report['summary']['overall']['pairs']} pairs)"
    )


if __name__ == "__main__":
    main()
