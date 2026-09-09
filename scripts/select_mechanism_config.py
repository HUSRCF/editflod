"""Freeze one mechanism-sweep configuration using an explicit dev-set policy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _finite(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return float("inf")
    return float(value)


def select_configuration(
    payload: dict[str, Any],
    *,
    method: str,
    max_remote_drift: float,
    max_local_error: float | None = None,
    max_local_regression: float | None = None,
    max_network_calls: float | None = None,
    allow_non_dev: bool = False,
) -> dict[str, Any]:
    """Select the best finite row under a declared development-set policy."""
    if payload.get("format") != "ospedit.mechanism_sweep.v1":
        raise ValueError("unsupported mechanism sweep format")
    if payload.get("status") != "completed":
        raise ValueError("cannot select from an incomplete sweep; status must be completed")
    expected = payload.get("expected_configurations")
    completed = payload.get("completed_configurations")
    if not isinstance(expected, int) or not isinstance(completed, int) or expected != completed:
        raise ValueError("sweep configuration counts are incomplete or malformed")
    if payload.get("split") != "dev" and not allow_non_dev:
        raise ValueError("configuration selection requires a dev split; pass allow_non_dev only for diagnostics")
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("sweep report contains no runs")
    fingerprints = {
        run.get("manifest_fingerprint")
        for run in runs
        if isinstance(run, dict) and run.get("manifest_fingerprint") is not None
    }
    if len(fingerprints) > 1:
        raise ValueError("sweep runs contain multiple manifest fingerprints")
    candidates: list[tuple[tuple[float, ...], dict[str, Any], dict[str, Any]]] = []
    for run in runs:
        if not isinstance(run, dict) or not isinstance(run.get("rows"), list):
            raise ValueError("malformed sweep run")
        row = next((item for item in run["rows"] if isinstance(item, dict) and item.get("method") == method), None)
        if row is None:
            continue
        remote = _finite(row, "mean_remote_scaffold_frame_drift")
        local = _finite(row, "mean_local_backbone_error")
        calls = _finite(row, "network_calls")
        if (
            remote > max_remote_drift
            or (max_local_error is not None and local > max_local_error)
            or (max_network_calls is not None and calls > max_network_calls)
        ):
            continue
        if max_local_regression is not None:
            baseline = next(
                (item for item in run["rows"] if isinstance(item, dict) and item.get("method") == "C0_copy_parent"),
                None,
            )
            if baseline is None:
                continue
            baseline_local = _finite(baseline, "mean_local_backbone_error")
            if not math.isfinite(local) or not math.isfinite(baseline_local) or local - baseline_local > max_local_regression:
                continue
        cost = _finite(row, "mean_seconds")
        change = _finite(row, "mean_distance_change_error")
        candidates.append(((local, change, cost, float(run.get("run_id", math.inf))), run, row))
    if not candidates:
        raise ValueError(f"no {method!r} run satisfies the declared development constraints")
    _, run, row = min(candidates, key=lambda item: item[0])
    return {
        "format": "ospedit.mechanism_config.v1",
        "source_format": payload["format"],
        "source_manifest": payload.get("manifest"),
        "source_manifest_fingerprint": run.get("manifest_fingerprint"),
        "split": payload.get("split"),
        "method": method,
        "run_id": run.get("run_id"),
        "config": run.get("config"),
        "selected_metrics": row,
        "selection_policy": {
            "max_remote_scaffold_frame_drift": max_remote_drift,
            "max_local_backbone_error": max_local_error,
            "max_local_regression_vs_copy_parent": max_local_regression,
            "max_network_calls": max_network_calls,
            "sort": ["mean_local_backbone_error", "mean_distance_change_error", "mean_seconds", "run_id"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze one mechanism sweep configuration")
    parser.add_argument("--input", required=True, help="ospedit.mechanism_sweep.v1 JSON")
    parser.add_argument("--output", required=True, help="frozen mechanism config JSON")
    parser.add_argument("--method", default="C3_two_noise_shared_difference")
    parser.add_argument("--max-remote-drift", type=float, required=True)
    parser.add_argument("--max-local-error", type=float)
    parser.add_argument("--max-local-regression", type=float, help="maximum candidate local error minus C0 local error")
    parser.add_argument("--max-network-calls", type=float, help="maximum reported network calls per evaluated manifest")
    parser.add_argument("--allow-non-dev", action="store_true", help="diagnostic override; do not use for parameter selection")
    args = parser.parse_args()
    if args.max_remote_drift < 0 or any(value is not None and value < 0 for value in (args.max_local_error, args.max_local_regression, args.max_network_calls)):
        raise SystemExit("selection thresholds must be non-negative")
    payload = json.loads(Path(args.input).read_text())
    selected = select_configuration(
        payload,
        method=args.method,
        max_remote_drift=args.max_remote_drift,
        max_local_error=args.max_local_error,
        max_local_regression=args.max_local_regression,
        max_network_calls=args.max_network_calls,
        allow_non_dev=args.allow_non_dev,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(selected, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"mechanism config written: {destination} (run_id={selected['run_id']})")


if __name__ == "__main__":
    main()
