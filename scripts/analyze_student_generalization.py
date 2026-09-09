"""Decompose student-versus-copy performance across records and families."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ospedit.data import file_sha256, json_safe


REPORT_FORMAT = "ospedit.student_generalization_diagnostic.v1"
METRICS = (
    "local_backbone_error",
    "mutation_site_backbone_error",
    "distance_change_error",
    "distance_change_cosine",
    "remote_target_error",
    "remote_scaffold_drift",
    "predicted_distance_change_norm",
    "true_distance_change_norm",
)
ERROR_METRICS = (
    "local_backbone_error",
    "mutation_site_backbone_error",
    "distance_change_error",
    "remote_target_error",
    "remote_scaffold_drift",
)


def _load(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _index_records(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("evaluation report must contain a records list")
    indexed = {str(row["pair_id"]): row for row in records}
    if len(indexed) != len(records):
        raise ValueError("evaluation report contains duplicate pair_id values")
    return indexed


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)) if len(finite) else None,
        "std": float(np.std(finite)) if len(finite) else None,
        "min": float(np.min(finite)) if len(finite) else None,
        "max": float(np.max(finite)) if len(finite) else None,
    }


def _parse_spec(value: str) -> tuple[str, int, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3 or not parts[0]:
        raise argparse.ArgumentTypeError("expected ARCHITECTURE:SEED:PATH")
    try:
        seed = int(parts[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError("student report seed must be an integer") from error
    return parts[0], seed, Path(parts[2])


def analyze_evaluations(
    copy_payload: dict[str, Any],
    student_payloads: Iterable[tuple[str, int, dict[str, Any]]],
) -> dict[str, Any]:
    copy_records = _index_records(copy_payload)
    grouped: dict[str, list[tuple[int, dict[str, dict[str, Any]]]]] = defaultdict(list)
    for architecture, seed, payload in student_payloads:
        if payload.get("split") != copy_payload.get("split"):
            raise ValueError("student and copy reports must use the same split")
        if payload.get("metric_schema") != copy_payload.get("metric_schema"):
            raise ValueError("student and copy reports must use the same metric schema")
        records = _index_records(payload)
        if set(records) != set(copy_records):
            raise ValueError("student and copy reports must contain identical pair_id sets")
        for pair_id, row in records.items():
            if row.get("family_id") != copy_records[pair_id].get("family_id"):
                raise ValueError(f"family mismatch for {pair_id}")
        grouped[architecture].append((seed, records))
    if not grouped:
        raise ValueError("at least one student report is required")

    architectures: dict[str, Any] = {}
    for architecture, runs in sorted(grouped.items()):
        seeds = [seed for seed, _ in sorted(runs)]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seed for architecture {architecture}")
        comparisons: list[dict[str, Any]] = []
        by_pair: dict[str, Any] = {}
        by_family_values: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for pair_id, copy_row in sorted(copy_records.items()):
            family_id = str(copy_row["family_id"])
            copy_metrics = copy_row["metrics"]
            pair_deltas: dict[str, list[float]] = defaultdict(list)
            pair_students: dict[str, list[float]] = defaultdict(list)
            for seed, records in sorted(runs):
                student_metrics = records[pair_id]["metrics"]
                row = {"pair_id": pair_id, "family_id": family_id, "seed": seed}
                for metric in METRICS:
                    student_value = student_metrics.get(metric)
                    copy_value = copy_metrics.get(metric)
                    row[f"student_{metric}"] = student_value
                    if student_value is not None:
                        pair_students[metric].append(float(student_value))
                    if student_value is not None and copy_value is not None:
                        delta = float(student_value) - float(copy_value)
                        row[f"student_minus_copy_{metric}"] = delta
                        pair_deltas[metric].append(delta)
                        by_family_values[family_id][metric].append(delta)
                comparisons.append(row)
            by_pair[pair_id] = {
                "family_id": family_id,
                "copy": {metric: copy_metrics.get(metric) for metric in METRICS},
                "student": {
                    metric: _summary(pair_students[metric]) for metric in METRICS
                },
                "student_minus_copy": {
                    metric: _summary(pair_deltas[metric]) for metric in ERROR_METRICS
                },
                "improving_seeds": {
                    metric: sum(value < 0 for value in pair_deltas[metric])
                    for metric in ERROR_METRICS
                },
            }
        by_family = {
            family_id: {
                metric: _summary(values) for metric, values in sorted(metrics.items())
            }
            for family_id, metrics in sorted(by_family_values.items())
        }
        overall = {
            metric: {
                **_summary(
                    row[f"student_minus_copy_{metric}"]
                    for row in comparisons
                    if f"student_minus_copy_{metric}" in row
                ),
                "improved": sum(
                    row.get(f"student_minus_copy_{metric}", float("inf")) < 0
                    for row in comparisons
                ),
            }
            for metric in ERROR_METRICS
        }
        architectures[architecture] = {
            "seeds": sorted(seeds),
            "comparisons": len(comparisons),
            "overall_student_minus_copy": overall,
            "all_record_seed_local_errors_worse_than_copy": overall[
                "local_backbone_error"
            ]["improved"]
            == 0,
            "by_family_student_minus_copy": by_family,
            "by_pair": by_pair,
        }
    return {
        "format": REPORT_FORMAT,
        "split": copy_payload.get("split"),
        "records": len(copy_records),
        "families": len({row["family_id"] for row in copy_records.values()}),
        "selection_used_observed_response": False,
        "analysis_changes_dataset_selection": False,
        "architectures": architectures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copy-report", required=True)
    parser.add_argument(
        "--student-report",
        action="append",
        required=True,
        type=_parse_spec,
        metavar="ARCHITECTURE:SEED:PATH",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    copy_path = Path(args.copy_report)
    student_specs: list[tuple[str, int, Path]] = args.student_report
    report = analyze_evaluations(
        _load(copy_path),
        [(architecture, seed, _load(path)) for architecture, seed, path in student_specs],
    )
    report["inputs"] = {
        "copy_report": {"path": str(copy_path.resolve()), "sha256": file_sha256(copy_path)},
        "student_reports": [
            {
                "architecture": architecture,
                "seed": seed,
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
            for architecture, seed, path in student_specs
        ],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(f"generalization diagnostic written: {destination}")


if __name__ == "__main__":
    main()
