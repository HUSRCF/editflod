"""Sweep mechanism-editor scales on one frozen development split.

The endpoint is constructed once and reused across runs.  This command is a
development calibration tool: it never selects a configuration from a test
split and records every evaluated configuration in the output report.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable

from ospedit.data import json_safe, load_manifest, validate_manifest, verify_record_checksums
from ospedit.experiment import (
    build_mechanism_editors,
    evaluate_editor_suite,
    flatten_suite_reports,
)


def _load_factory(specification: str) -> Callable[[], Any]:
    if ":" not in specification:
        raise ValueError("endpoint factory must use module:callable syntax")
    module_name, attribute = specification.split(":", 1)
    if not module_name or not attribute:
        raise ValueError("endpoint factory must use module:callable syntax")
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    factory = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(factory):
        raise TypeError(f"endpoint factory {specification!r} is not callable")
    return factory


def _floats(values: str, name: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(value.strip()) for value in values.split(",") if value.strip())
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated list of numbers") from error
    if not parsed or any(value < 0.0 for value in parsed):
        raise ValueError(f"{name} must contain at least one non-negative value")
    return parsed


def _pair(values: str, name: str) -> tuple[float, float]:
    parsed = _floats(values, name)
    if len(parsed) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return parsed[0], parsed[1]


def _pair_weights(values: str, name: str) -> tuple[float, float]:
    try:
        parsed = tuple(float(value.strip()) for value in values.split(",") if value.strip())
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated list of numbers") from error
    if len(parsed) != 2 or not all(math.isfinite(value) for value in parsed):
        raise ValueError(f"{name} must contain exactly two finite values")
    return parsed[0], parsed[1]


def _method_names(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names:
        raise ValueError("--methods must contain at least one method name")
    return names


def _select_records(records: list[Any], limit: int | None, strategy: str) -> list[Any]:
    if limit is None:
        return records
    if limit <= 0:
        raise ValueError("max_records must be positive")
    if strategy == "prefix":
        return records[:limit]
    if strategy != "family_round_robin":
        raise ValueError(f"unknown record selection strategy: {strategy!r}")
    families: dict[str, list[Any]] = {}
    for record in records:
        families.setdefault(str(record.family_id), []).append(record)
    selected: list[Any] = []
    while len(selected) < limit:
        progressed = False
        for family_records in families.values():
            if len(selected) >= limit:
                break
            if family_records:
                selected.append(family_records.pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def _write_report(path: Path, payload: dict[str, object]) -> None:
    """Atomically persist progress so interrupted sweeps retain completed runs."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _config_key(config: dict[str, object]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate C0-C5 mechanism scales on a development split")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--endpoint-factory", required=True)
    parser.add_argument("--output", required=True, help="strict JSON sweep report")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--noise-levels", default="0.25,0.5,0.75")
    parser.add_argument("--two-noise-levels", default="0.25,0.75", help="low,high levels used by C3/C6")
    parser.add_argument("--two-noise-weights", default="1.0,1.0", help="low,high combination weights used by C3/C6")
    parser.add_argument("--methods", default=None, help="comma-separated editor names to evaluate (default: all)")
    parser.add_argument("--max-records", type=int, help="deterministic development subset limit for quick diagnostics")
    parser.add_argument("--record-selection", choices=("prefix", "family_round_robin"), default="prefix")
    parser.add_argument("--resume", action="store_true", help="resume an incomplete report at --output")
    parser.add_argument("--difference-step-sizes", default="0.05,0.1,0.2")
    parser.add_argument("--translation-scales", default="1.0")
    parser.add_argument("--rotation-scales", default="1.0")
    parser.add_argument("--verify-checksums", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    records = load_manifest(args.manifest)
    errors = validate_manifest(records)
    if args.verify_checksums:
        errors.extend(error for record in records for error in verify_record_checksums(record))
    selected = [record for record in records if record.split == args.split]
    if args.max_records is not None:
        try:
            selected = _select_records(selected, args.max_records, args.record_selection)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    if not selected:
        errors.append(f"manifest contains no records for split={args.split!r}")
    if errors:
        raise SystemExit("manifest audit failed: " + "; ".join(errors))

    noise_levels = _floats(args.noise_levels, "--noise-levels")
    two_noise_levels = _pair(args.two_noise_levels, "--two-noise-levels")
    try:
        two_noise_weights = _pair_weights(args.two_noise_weights, "--two-noise-weights")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    difference_steps = _floats(args.difference_step_sizes, "--difference-step-sizes")
    translation_scales = _floats(args.translation_scales, "--translation-scales")
    rotation_scales = _floats(args.rotation_scales, "--rotation-scales")
    requested_methods = _method_names(args.methods) if args.methods is not None else None
    methods_filter = list(requested_methods) if requested_methods is not None else None
    endpoint = _load_factory(args.endpoint_factory)()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    expected_configurations = len(noise_levels) * len(difference_steps) * len(translation_scales) * len(rotation_scales)
    if args.resume and destination.exists():
        try:
            previous = json.loads(destination.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"unable to resume --output: {error}") from error
        if previous.get("format") != "ospedit.mechanism_sweep.v1" or previous.get("status") != "running":
            raise SystemExit("--resume requires an existing mechanism sweep with status=running")
        if previous.get("manifest") != str(Path(args.manifest).resolve()) or previous.get("split") != args.split:
            raise SystemExit("cannot resume: manifest or split does not match the existing report")
        if previous.get("expected_configurations") != expected_configurations:
            raise SystemExit("cannot resume: configuration grid size does not match the existing report")
        previous_methods = previous.get("methods_filter")
        if previous_methods != methods_filter:
            raise SystemExit("cannot resume: method filter does not match the existing report")
        if previous.get("two_noise_levels", [0.25, 0.75]) != list(two_noise_levels) or previous.get("two_noise_weights", [1.0, 1.0]) != list(two_noise_weights):
            raise SystemExit("cannot resume: two-noise levels or weights do not match the existing report")
        if (
            previous.get("batch_size") != args.batch_size
            or previous.get("max_records") != args.max_records
            or previous.get("record_selection", "prefix") != args.record_selection
        ):
            raise SystemExit("cannot resume: batch size or record limit does not match the existing report")
        if not isinstance(previous.get("runs"), list):
            raise SystemExit("cannot resume: existing report has malformed runs")
        runs = list(previous["runs"])
    completed_keys: set[str] = set()
    for run in runs:
        if isinstance(run, dict):
            config_value = run.get("config")
            if isinstance(config_value, dict):
                completed_keys.add(_config_key(config_value))
    base_payload: dict[str, object] = {
        "format": "ospedit.mechanism_sweep.v1",
        "status": "running",
        "manifest": str(Path(args.manifest).resolve()),
        "split": args.split,
        "batch_size": args.batch_size,
        "endpoint_factory": args.endpoint_factory,
        "max_records": args.max_records,
        "record_selection": args.record_selection,
        "methods_filter": methods_filter,
        "two_noise_levels": list(two_noise_levels),
        "two_noise_weights": list(two_noise_weights),
        "expected_configurations": expected_configurations,
        "completed_configurations": 0,
        "runs": runs,
    }
    _write_report(destination, base_payload)
    existing_ids: list[int] = []
    for run in runs:
        if isinstance(run, dict):
            run_value = run.get("run_id")
            if isinstance(run_value, int):
                existing_ids.append(run_value)
    run_id = max(existing_ids, default=-1) + 1
    for noise_level in noise_levels:
        for difference_step_size in difference_steps:
            for translation_scale in translation_scales:
                for rotation_scale in rotation_scales:
                    editors = build_mechanism_editors(
                        endpoint_model=endpoint,
                        noise_level=noise_level,
                        noise_levels=two_noise_levels,
                        multi_noise_weights=two_noise_weights,
                        difference_step_size=difference_step_size,
                        translation_scale=translation_scale,
                        rotation_scale=rotation_scale,
                    )
                    if requested_methods is not None:
                        unknown = sorted(set(requested_methods) - set(editors))
                        if unknown:
                            raise SystemExit(f"unknown mechanism method(s): {', '.join(unknown)}")
                        editors = {name: editors[name] for name in requested_methods}
                    config = {
                        "noise_level": noise_level,
                        "noise_levels": list(two_noise_levels),
                        "noise_weights": list(two_noise_weights),
                        "difference_step_size": difference_step_size,
                        "translation_scale": translation_scale,
                        "rotation_scale": rotation_scale,
                        "methods": list(editors),
                    }
                    if _config_key(config) in completed_keys:
                        continue
                    suite = evaluate_editor_suite(
                        selected,
                        editors,
                        split=args.split,
                        batch_size=args.batch_size,
                        run_metadata={
                            "command": "run_mechanism_sweep",
                            "run_id": run_id,
                            "noise_level": noise_level,
                            "noise_levels": list(two_noise_levels),
                            "noise_weights": list(two_noise_weights),
                            "difference_step_size": difference_step_size,
                            "translation_scale": translation_scale,
                            "rotation_scale": rotation_scale,
                            "methods": list(editors),
                            "max_records": args.max_records,
                            "record_selection": args.record_selection,
                        },
                    )
                    runs.append({
                        "run_id": run_id,
                        "config": config,
                        "rows": flatten_suite_reports(suite),
                        "manifest_fingerprint": suite.manifest_fingerprint,
                    })
                    base_payload["completed_configurations"] = len(runs)
                    _write_report(destination, base_payload)
                    run_id += 1

    base_payload["completed_configurations"] = len(runs)
    base_payload["status"] = "completed"
    _write_report(destination, base_payload)
    print(f"mechanism sweep written: {destination} ({len(runs)} configurations)")


if __name__ == "__main__":
    main()
