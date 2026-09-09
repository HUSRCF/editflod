"""Compare endpoint response with same-sequence experimental background."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPORT_FORMAT = "ospedit.background_signal_diagnostic.v1"
BACKGROUND_REPORT_FORMAT = "ospedit.background_control_coverage.v1"
RESPONSE_REPORT_FORMAT = "ospedit.response_learnability_audit.v1"
CONTEXT_REPORT_FORMAT = "ospedit.repeat_control_context_audit.v1"
RCSB_REPORT_FORMAT = "ospedit.rcsb_environment_audit.v1"
METRICS = {
    "local": ("neighborhood_rmsd_angstrom", "local_backbone_error"),
    "site": ("mutation_site_rmsd_angstrom", "mutation_site_backbone_error"),
    "distance": ("distance_change_rms_angstrom", "distance_change_error"),
}


def _quantiles(values: Iterable[float | None]) -> dict[str, float | int | None]:
    array = np.asarray([value for value in values if value is not None], dtype=float)
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


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _correlation(first: Iterable[float], second: Iterable[float], *, rank: bool = False) -> float | None:
    x = np.asarray(list(first), dtype=float)
    y = np.asarray(list(second), dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    if rank:
        x, y = _ranks(x), _ranks(y)
    return float(np.corrcoef(x, y)[0, 1])


def _ratio(signal: float, background: float) -> float | None:
    return signal / background if background > 1e-12 else None


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"records": len(rows)}
    for region in METRICS:
        result[region] = {}
        for aggregation in ("max", "median"):
            backgrounds = [row[region][f"background_{aggregation}"] for row in rows]
            signals = [row[region]["endpoint_signal"] for row in rows]
            result[region][aggregation] = {
                "signal_to_background": _quantiles(
                    row[region][f"signal_to_background_{aggregation}"] for row in rows
                ),
                "background_signal_pearson": _correlation(backgrounds, signals),
                "background_signal_spearman": _correlation(backgrounds, signals, rank=True),
            }
    return result


def background_signal_report(
    background_report: str | Path,
    response_report: str | Path,
    control_context_report: str | Path | None = None,
    rcsb_environment_report: str | Path | None = None,
) -> dict[str, Any]:
    background_path = Path(background_report).resolve()
    response_path = Path(response_report).resolve()
    background = json.loads(background_path.read_text())
    response = json.loads(response_path.read_text())
    if background.get("format") != BACKGROUND_REPORT_FORMAT:
        raise ValueError(f"expected {BACKGROUND_REPORT_FORMAT}")
    if response.get("format") != RESPONSE_REPORT_FORMAT:
        raise ValueError(f"expected {RESPONSE_REPORT_FORMAT}")
    if background.get("manifest_fingerprint") != response.get("manifest_fingerprint"):
        raise ValueError("background and response reports use different manifests")
    response_rows = {row["pair_id"]: row for row in response.get("records", [])}
    def diagnostic_row(control: dict[str, Any]) -> dict[str, Any]:
        pair_id = control["pair_id"]
        if pair_id not in response_rows:
            raise ValueError(f"response report is missing {pair_id}")
        response_row = response_rows[pair_id]
        row: dict[str, Any] = {
            "pair_id": pair_id,
            "parent_id": control["parent_id"],
            "family_id": control["family_id"],
            "split": control["split"],
            "repeat_structures": control["repeat_structures"],
        }
        for region, (background_name, signal_name) in METRICS.items():
            signal = float(response_row["copy_error"][signal_name])
            maximum = float(control["background_max"][background_name])
            median = float(control["background_median"][background_name])
            if not all(math.isfinite(value) and value >= 0 for value in (signal, maximum, median)):
                raise ValueError(f"invalid signal/background value for {pair_id}")
            row[region] = {
                "endpoint_signal": signal,
                "background_max": maximum,
                "background_median": median,
                "signal_to_background_max": _ratio(signal, maximum),
                "signal_to_background_median": _ratio(signal, median),
            }
        return row

    rows = [diagnostic_row(control) for control in background.get("records", [])]
    background_rows = {row["pair_id"]: row for row in background.get("records", [])}

    def aggregate_selected(selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected: dict[str, list[dict[str, Any]]] = {}
        for row in selected_rows:
            selected.setdefault(row["pair_id"], []).append(row)
        output = []
        for pair_id, controls in sorted(selected.items()):
            source_row = background_rows[pair_id]
            aggregated = {
                "pair_id": pair_id,
                "parent_id": source_row["parent_id"],
                "family_id": source_row["family_id"],
                "split": source_row["split"],
                "repeat_structures": len(controls),
                "background_max": {
                    name: max(float(row["background"][name]) for row in controls)
                    for name, _ in METRICS.values()
                },
                "background_median": {
                    name: float(np.median([float(row["background"][name]) for row in controls]))
                    for name, _ in METRICS.values()
                },
            }
            output.append(diagnostic_row(aggregated))
        return output

    context_path = Path(control_context_report).resolve() if control_context_report else None
    context_rows: list[dict[str, Any]] = []
    if context_path is not None:
        context = json.loads(context_path.read_text())
        if context.get("format") != CONTEXT_REPORT_FORMAT:
            raise ValueError(f"expected {CONTEXT_REPORT_FORMAT}")
        if context.get("manifest_fingerprint") != background.get("manifest_fingerprint"):
            raise ValueError("context and background reports use different manifests")
        context_rows = aggregate_selected([
            row for row in context.get("records", []) if row.get("selected")
        ])

    rcsb_path = Path(rcsb_environment_report).resolve() if rcsb_environment_report else None
    assembly_rows: list[dict[str, Any]] = []
    crystal_rows: list[dict[str, Any]] = []
    if rcsb_path is not None:
        rcsb = json.loads(rcsb_path.read_text())
        if rcsb.get("format") != RCSB_REPORT_FORMAT:
            raise ValueError(f"expected {RCSB_REPORT_FORMAT}")
        if rcsb.get("manifest_fingerprint") != background.get("manifest_fingerprint"):
            raise ValueError("RCSB and background reports use different manifests")
        assembly_rows = aggregate_selected([
            row for row in rcsb.get("records", []) if row.get("assembly_compatible")
        ])
        crystal_rows = aggregate_selected([
            row for row in rcsb.get("records", []) if row.get("crystal_form_compatible")
        ])

    cohorts = {
        "all_controls": rows,
        "at_least_two_controls": [row for row in rows if row["repeat_structures"] >= 2],
    }
    if context_path is not None:
        cohorts.update({
            "context_prescreened_controls": context_rows,
            "context_prescreened_at_least_two_controls": [
                row for row in context_rows if row["repeat_structures"] >= 2
            ],
        })
    if rcsb_path is not None:
        cohorts.update({
            "rcsb_assembly_controls": assembly_rows,
            "rcsb_assembly_at_least_two_controls": [
                row for row in assembly_rows if row["repeat_structures"] >= 2
            ],
            "rcsb_crystal_form_controls": crystal_rows,
            "rcsb_crystal_form_at_least_two_controls": [
                row for row in crystal_rows if row["repeat_structures"] >= 2
            ],
        })
    summary = {
        name: {
            "overall": _summary(cohort),
            "splits": {
                split: _summary([row for row in cohort if row["split"] == split])
                for split in ("train", "dev", "test")
            },
        }
        for name, cohort in cohorts.items()
    }
    return {
        "format": REPORT_FORMAT,
        "metric_schema": background["metric_schema"],
        "manifest_fingerprint": background["manifest_fingerprint"],
        "background_report": str(background_path),
        "response_report": str(response_path),
        "control_context_report": str(context_path) if context_path is not None else None,
        "rcsb_environment_report": str(rcsb_path) if rcsb_path is not None else None,
        "usage": "diagnostic_only_not_a_training_weight",
        "limitations": [
            "same_sequence_identity_does_not_establish_matched_experimental_environment",
            "maximum_background_increases_with_number_of_available_controls",
            "endpoint_difference_is_not_assumed_to_be_caused_only_by_mutation",
        ],
        "summary": summary,
        "records": rows,
        "context_prescreened_records": context_rows,
        "rcsb_assembly_records": assembly_rows,
        "rcsb_crystal_form_records": crystal_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("background_report")
    parser.add_argument("response_report")
    parser.add_argument("output")
    parser.add_argument("--control-context-report")
    parser.add_argument("--rcsb-environment-report")
    args = parser.parse_args()
    try:
        report = background_signal_report(
            args.background_report,
            args.response_report,
            args.control_context_report,
            args.rcsb_environment_report,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"background signal diagnostic written: {destination} ({len(report['records'])} records)")


if __name__ == "__main__":
    main()
