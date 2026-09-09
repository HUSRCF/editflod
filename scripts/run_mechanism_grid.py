"""Run the C0-C5 mechanism grid with an injected endpoint model.

The endpoint factory is kept external because FoldFlow checkpoint loading is
repository/configuration specific.  It must be importable as ``module:name``
and return an object implementing ``EndpointModel.endpoint``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from ospedit.data import load_manifest, validate_manifest, verify_record_checksums
from ospedit.experiment import build_mechanism_editors, evaluate_editor_suite, write_suite_csv, write_suite_report


def _load_factory(specification: str) -> Callable[[], Any]:
    if ":" not in specification:
        raise ValueError("endpoint factory must use module:callable syntax")
    module_name, attribute = specification.split(":", 1)
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    if not module_name or not attribute:
        raise ValueError("endpoint factory must use module:callable syntax")
    factory = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(factory):
        raise TypeError(f"endpoint factory {specification!r} is not callable")
    return factory


def _method_names(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names:
        raise ValueError("--methods must contain at least one method name")
    return names


def _apply_frozen_config(args: argparse.Namespace) -> dict[str, object]:
    """Load a selector output and override only its calibrated parameters."""
    if args.frozen_config is None:
        return {}
    try:
        payload = json.loads(Path(args.frozen_config).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"unable to read --frozen-config: {error}") from error
    if payload.get("format") != "ospedit.mechanism_config.v1":
        raise SystemExit("--frozen-config must be ospedit.mechanism_config.v1")
    frozen_method = payload.get("method")
    if frozen_method is not None and not isinstance(frozen_method, str):
        raise SystemExit("frozen config method must be a string")
    args.frozen_method = frozen_method
    source_manifest = payload.get("source_manifest")
    if source_manifest is not None:
        expected = Path(str(source_manifest)).expanduser().resolve()
        actual = Path(args.manifest).expanduser().resolve()
        if expected != actual:
            raise SystemExit(
                "frozen config source_manifest does not match --manifest; "
                "re-run selection for this manifest"
            )
    config = payload.get("config")
    if not isinstance(config, dict):
        raise SystemExit("frozen config is missing its config object")
    required = ("noise_level", "difference_step_size", "translation_scale", "rotation_scale")
    if any(key not in config for key in required):
        raise SystemExit("frozen config must contain noise_level, difference_step_size, translation_scale, and rotation_scale")
    for key in required:
        setattr(args, key, float(config[key]))
    if "noise_levels" in config:
        levels = config["noise_levels"]
        if not isinstance(levels, list) or len(levels) != 2:
            raise SystemExit("frozen config noise_levels must contain exactly two values")
        args.noise_levels = (float(levels[0]), float(levels[1]))
    if "noise_weights" in config:
        weights = config["noise_weights"]
        if not isinstance(weights, list) or len(weights) != 2:
            raise SystemExit("frozen config noise_weights must contain exactly two values")
        args.noise_weights = (float(weights[0]), float(weights[1]))
    return {
        "frozen_config": str(Path(args.frozen_config).resolve()),
        "frozen_config_run_id": payload.get("run_id"),
        "frozen_config_source_manifest_fingerprint": payload.get("source_manifest_fingerprint"),
        "frozen_config_source_manifest": source_manifest,
        "frozen_config_method": frozen_method,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reference-conditioned C0-C5 mechanism grid")
    parser.add_argument("--manifest", required=True, help="JSON/JSONL pair manifest")
    parser.add_argument("--endpoint-factory", required=True, help="module:callable returning an endpoint model")
    parser.add_argument("--output", required=True, help="strict JSON suite report path")
    parser.add_argument("--csv-output", help="Optional one-row-per-method summary CSV path")
    parser.add_argument("--frozen-config", help="selector output from select_mechanism_config.py")
    parser.add_argument("--methods", help="comma-separated editor names to evaluate (default: all)")
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--max-mutations", type=int, default=None)
    parser.add_argument("--split", choices=("train", "dev", "test", "all"), default="dev")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--noise-level", type=float, default=0.5)
    parser.add_argument("--noise-levels", type=float, nargs=2, default=(0.25, 0.75), metavar=("LOW", "HIGH"))
    parser.add_argument("--noise-weights", type=float, nargs=2, default=(1.0, 1.0), metavar=("LOW", "HIGH"))
    parser.add_argument("--target-step-size", type=float, default=1.0)
    parser.add_argument("--difference-step-size", type=float, default=1.0)
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument("--rotation-scale", type=float, default=1.0)
    parser.add_argument("--neighborhood-radius", type=float, default=None, help="Add C6 mutation-neighborhood ablation at this Angstrom radius")
    args = parser.parse_args()
    frozen_metadata = _apply_frozen_config(args)
    if args.methods is not None and getattr(args, "frozen_method", None) is not None:
        try:
            requested_methods = _method_names(args.methods)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        if args.frozen_method not in requested_methods:
            raise SystemExit("--methods must include the method named by --frozen-config")

    records = load_manifest(args.manifest)
    errors = validate_manifest(records, max_mutations=args.max_mutations)
    if args.verify_checksums:
        errors.extend(error for record in records for error in verify_record_checksums(record))
    if errors:
        raise SystemExit("manifest audit failed: " + "; ".join(errors))
    factory = _load_factory(args.endpoint_factory)
    endpoint_setup_started = perf_counter()
    endpoint = factory()
    endpoint_setup_seconds = perf_counter() - endpoint_setup_started
    editors = build_mechanism_editors(
        endpoint_model=endpoint,
        noise_level=args.noise_level,
        noise_levels=tuple(args.noise_levels),
        multi_noise_weights=tuple(args.noise_weights),
        target_step_size=args.target_step_size,
        difference_step_size=args.difference_step_size,
        translation_scale=args.translation_scale,
        rotation_scale=args.rotation_scale,
        neighborhood_radius=args.neighborhood_radius,
    )
    if args.methods is not None:
        try:
            requested_methods = _method_names(args.methods)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        unknown = sorted(set(requested_methods) - set(editors))
        if unknown:
            raise SystemExit(f"unknown mechanism method(s): {', '.join(unknown)}")
        if getattr(args, "frozen_method", None) is not None and args.frozen_method not in requested_methods:
            raise SystemExit("--methods must include the method named by --frozen-config")
        editors = {name: editors[name] for name in requested_methods}
    suite = evaluate_editor_suite(
        records,
        editors,
        split=None if args.split == "all" else args.split,
        batch_size=args.batch_size,
        run_metadata={
            "command": "run_mechanism_grid",
            "endpoint_factory": args.endpoint_factory,
            "manifest": str(Path(args.manifest).resolve()),
            "split": args.split,
            "batch_size": args.batch_size,
            "noise_level": args.noise_level,
            "noise_levels": list(args.noise_levels),
            "noise_weights": list(args.noise_weights),
            "target_step_size": args.target_step_size,
            "difference_step_size": args.difference_step_size,
            "translation_scale": args.translation_scale,
            "rotation_scale": args.rotation_scale,
            "neighborhood_radius": args.neighborhood_radius,
            "methods": list(editors),
            "endpoint_setup_seconds": endpoint_setup_seconds,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            **frozen_metadata,
        },
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_suite_report(suite, output)
    if args.csv_output:
        csv_output = Path(args.csv_output)
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        write_suite_csv(suite, csv_output)
    print(f"mechanism grid written: {output} ({len(suite.methods)} methods)")


if __name__ == "__main__":
    main()
