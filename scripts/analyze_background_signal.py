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
    rows: list[dict[str, Any]] = []
    for control in background.get("records", []):
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
        rows.append(row)

    cohorts = {
        "all_controls": rows,
        "at_least_two_controls": [row for row in rows if row["repeat_structures"] >= 2],
    }
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
        "usage": "diagnostic_only_not_a_training_weight",
        "limitations": [
            "same_sequence_identity_does_not_establish_matched_experimental_environment",
            "maximum_background_increases_with_number_of_available_controls",
            "endpoint_difference_is_not_assumed_to_be_caused_only_by_mutation",
        ],
        "summary": summary,
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("background_report")
    parser.add_argument("response_report")
    parser.add_argument("output")
    args = parser.parse_args()
    try:
        report = background_signal_report(args.background_report, args.response_report)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"background signal diagnostic written: {destination} ({len(report['records'])} records)")


if __name__ == "__main__":
    main()
