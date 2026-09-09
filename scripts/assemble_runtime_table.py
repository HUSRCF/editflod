"""Assemble resident-model setup and per-candidate costs under one contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from ospedit.data import json_safe


def _nested(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"runtime setup key is absent: {path}")
        value = value[key]
    return value


def _parse_entry(value: str) -> tuple[str, Path, str, str, str]:
    if "=" not in value:
        raise ValueError("entry must use NAME=SUITE::METHOD::SETUP::DEVICE")
    name, remainder = value.split("=", 1)
    parts = remainder.split("::")
    if not name or len(parts) != 4 or not all(parts):
        raise ValueError("entry must use NAME=SUITE::METHOD::SETUP::DEVICE")
    return name, Path(parts[0]).expanduser().resolve(), parts[1], parts[2], parts[3]


def _setup_seconds(suite: dict[str, Any], specification: str) -> tuple[float, str]:
    if specification.startswith("metadata."):
        value = _nested(suite, f"run_metadata.{specification.removeprefix('metadata.')}")
        source = "suite:" + specification
    else:
        if "#" not in specification:
            raise ValueError("external setup must use JSON_PATH#DOTTED_KEY")
        path_text, key = specification.rsplit("#", 1)
        path = Path(path_text).expanduser().resolve()
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"unable to read runtime setup report {path}: {error}") from error
        value = _nested(payload, key)
        source = f"{path}#{key}"
    if not isinstance(value, (int, float)) or float(value) < 0:
        raise ValueError(f"runtime setup value must be non-negative: {value!r}")
    return float(value), source


def assemble(entries: Iterable[str], candidate_counts: Iterable[int]) -> dict[str, Any]:
    counts = tuple(int(count) for count in candidate_counts)
    if not counts or any(count <= 0 for count in counts) or len(set(counts)) != len(counts):
        raise ValueError("candidate counts must be unique positive integers")
    rows = []
    expected_identity: tuple[str, tuple[str, ...]] | None = None
    names: set[str] = set()
    for specification in entries:
        name, suite_path, method_name, setup_spec, device = _parse_entry(specification)
        if name in names:
            raise ValueError(f"duplicate runtime method name: {name}")
        names.add(name)
        try:
            suite = json.loads(suite_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"unable to read suite report {suite_path}: {error}") from error
        methods = suite.get("methods")
        if not isinstance(methods, dict) or method_name not in methods:
            raise ValueError(f"suite report has no method {method_name!r}: {suite_path}")
        method = methods[method_name]
        records = method.get("records") if isinstance(method, dict) else None
        runtime = method.get("runtime_summary") if isinstance(method, dict) else None
        fingerprint = suite.get("manifest_fingerprint")
        if not isinstance(records, list) or not records or not isinstance(runtime, dict):
            raise ValueError(f"suite method lacks records/runtime: {suite_path}::{method_name}")
        if not isinstance(fingerprint, str):
            raise ValueError(f"suite report lacks manifest fingerprint: {suite_path}")
        pair_ids = tuple(str(record.get("pair_id")) for record in records)
        identity = (fingerprint, pair_ids)
        if expected_identity is not None and expected_identity != identity:
            raise ValueError(f"runtime suite identity mismatch: {expected_identity} != {identity}")
        expected_identity = identity
        setup_seconds, setup_source = _setup_seconds(suite, setup_spec)
        total_seconds = float(runtime.get("total_seconds", 0.0))
        allocated_setup = float(runtime.get("model_load_seconds_amortized", 0.0))
        candidate_total = total_seconds - allocated_setup
        if candidate_total < -1e-8:
            raise ValueError(f"allocated setup exceeds total runtime for {name}")
        candidate_total = max(candidate_total, 0.0)
        mean_candidate = candidate_total / len(records)
        row: dict[str, Any] = {
            "method": name,
            "source_method": method_name,
            "suite_report": str(suite_path),
            "device": device,
            "records": len(records),
            "observed_candidate_count": len(records),
            "setup_seconds": setup_seconds,
            "setup_source": setup_source,
            "observed_candidate_seconds": candidate_total,
            "mean_warm_candidate_seconds": mean_candidate,
            "observed_resident_total_seconds": setup_seconds + candidate_total,
            "network_calls_per_candidate": float(runtime.get("network_calls", 0.0)) / len(records),
            "sequence_encoder_calls_per_candidate": float(
                runtime.get("sequence_encoder_calls", 0.0)
            ) / len(records),
            "peak_vram_bytes": runtime.get("peak_vram_bytes"),
        }
        for count in counts:
            row[f"resident_total_seconds_n{count}"] = setup_seconds + count * mean_candidate
            row[f"warm_total_seconds_n{count}"] = count * mean_candidate
        rows.append(row)
    if expected_identity is None:
        raise ValueError("at least one runtime entry is required")
    rows.sort(key=lambda row: str(row["method"]))
    return {
        "format": "ospedit.resident_runtime_table.v1",
        "manifest_fingerprint": expected_identity[0],
        "pair_ids": list(expected_identity[1]),
        "candidate_counts": list(counts),
        "projection_model": "linear_from_observed_candidate_mean",
        "only_observed_resident_total_is_measured": True,
        "cross_device_ranking_allowed": False,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble a resident-model runtime table")
    parser.add_argument("--entry", action="append", required=True)
    parser.add_argument("--candidate-counts", type=int, nargs="+", default=(1, 3, 32, 128))
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", required=True)
    args = parser.parse_args()
    try:
        payload = assemble(args.entry, args.candidate_counts)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    rows = payload["rows"]
    fieldnames = sorted({key for row in rows for key in row})
    csv_output = Path(args.csv_output)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(json_safe(rows))
    print(f"resident runtime table written: {output} ({len(rows)} methods)")


if __name__ == "__main__":
    main()
