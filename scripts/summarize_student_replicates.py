"""Summarize comparable student checkpoints across random seeds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from scripts.export_student_checkpoint_report import checkpoint_report
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from export_student_checkpoint_report import checkpoint_report  # type: ignore[no-redef]


REPORT_FORMAT = "ospedit.student_replicates.v1"
COMPARABILITY_KEYS = (
    "manifest_fingerprint",
    "student_architecture",
    "geometry_features",
    "biochemical_edit_features",
    "ablate_target_residue",
    "loss_schema",
    "parent_dim",
    "edit_dim",
    "hidden_dim",
    "blocks",
    "heads",
    "spatial_neighbors",
    "epochs",
    "translation_scale",
    "rotation_scale",
    "delta_loss",
    "delta_loss_beta",
    "mutation_loss_weight",
    "neighborhood_loss_weight",
    "family_balanced_loss",
    "target_localization_radius",
    "target_localization_transition",
)
CONFIG_DEFAULTS: dict[str, Any] = {
    "biochemical_edit_features": False,
    "ablate_target_residue": False,
    "edit_dim": 41,
    "target_localization_radius": None,
    "target_localization_transition": 5.0,
}


def _config_value(config: dict[str, Any], key: str) -> Any:
    return config.get(key, CONFIG_DEFAULTS.get(key))


def _finite_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def summarize_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one checkpoint report is required")
    reference = reports[0]["configuration"]
    mismatches: list[str] = []
    for index, report in enumerate(reports[1:], start=1):
        config = report["configuration"]
        for key in COMPARABILITY_KEYS:
            actual = _config_value(config, key)
            expected = _config_value(reference, key)
            if actual != expected:
                mismatches.append(f"run {index} {key}: {actual!r} != {expected!r}")
    if mismatches:
        raise ValueError("incomparable checkpoints: " + "; ".join(mismatches))
    rows = []
    for report in reports:
        comparison = report.get("comparison", {})
        metrics = comparison.get("student_parent_family_macro", {})
        history = report.get("loss_history", [])
        rows.append({
            "checkpoint": report["checkpoint"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "seed": report["configuration"].get("seed"),
            "epoch": report.get("epoch"),
            "optimizer_steps": report.get("optimizer_steps"),
            "final_loss": history[-1] if history else None,
            "metrics": metrics,
        })
    metric_names = sorted({name for row in rows for name, value in row["metrics"].items() if isinstance(value, (int, float))})
    aggregate = {
        name: _finite_summary(float(row["metrics"].get(name, math.nan)) for row in rows)
        for name in metric_names
    }
    return {
        "format": REPORT_FORMAT,
        "runs": rows,
        "configuration": {key: _config_value(reference, key) for key in COMPARABILITY_KEYS},
        "aggregate": aggregate,
    }


def summarize_checkpoints(checkpoints: list[str | Path]) -> dict[str, Any]:
    return summarize_reports([checkpoint_report(path) for path in checkpoints])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        report = summarize_checkpoints(args.checkpoints)
    except (ImportError, ValueError) as error:
        raise SystemExit(str(error)) from error
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"student replicate summary written: {destination} ({len(report['runs'])} runs)")


if __name__ == "__main__":
    main()
