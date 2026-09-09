"""Run the endpoint condition-response admission gate on an audited manifest."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

from ospedit.data import json_safe, load_manifest, manifest_fingerprint, validate_manifest, verify_record_checksums
from ospedit.diagnostics import (
    assess_teacher_admission,
    condition_response_diagnostic,
    condition_response_repeat_error,
)


def _load_factory(specification: str) -> Callable[[], Any]:
    if ":" not in specification:
        raise ValueError("endpoint factory must use module:callable syntax")
    module, attribute = specification.split(":", 1)
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    factory = getattr(importlib.import_module(module), attribute, None)
    if not callable(factory):
        raise TypeError(f"endpoint factory {specification!r} is not callable")
    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description="Admit a sequence-conditioned endpoint for teacher labels")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--endpoint-factory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--noise-levels", type=float, nargs="+", default=(0.25, 0.5, 0.75))
    parser.add_argument("--min-mutation-response", type=float, default=1e-5)
    parser.add_argument("--repeat-tolerance", type=float, default=1e-6)
    parser.add_argument("--verify-checksums", action="store_true")
    args = parser.parse_args()
    records = load_manifest(args.manifest)
    errors = validate_manifest(records)
    if args.verify_checksums:
        errors.extend(error for record in records for error in verify_record_checksums(record))
    if errors:
        raise SystemExit("manifest audit failed: " + "; ".join(errors))
    endpoint = _load_factory(args.endpoint_factory)()
    selected = [record for record in records if record.split == args.split]
    if not selected:
        raise SystemExit(f"manifest contains no records for split={args.split!r}")
    rows = []
    for record in selected:
        diagnostic = condition_response_diagnostic(endpoint, record.pair, args.noise_levels)
        repeat_diagnostic = condition_response_diagnostic(endpoint, record.pair, args.noise_levels)
        repeat_error = condition_response_repeat_error(diagnostic, repeat_diagnostic)
        rows.append({
            "pair_id": record.pair.pair_id,
            "family_id": record.family_id,
            "diagnostic": diagnostic,
            "repeat_error": repeat_error,
            "admission": assess_teacher_admission(
                diagnostic,
                min_mutation_response=args.min_mutation_response,
                repeat_error=repeat_error,
                max_repeat_error=args.repeat_tolerance,
            ),
        })
    payload = {
        "manifest": str(Path(args.manifest).resolve()),
        "endpoint_factory": args.endpoint_factory,
        "manifest_fingerprint": manifest_fingerprint(records),
        "split": args.split,
        "noise_levels": list(args.noise_levels),
        "min_mutation_response": args.min_mutation_response,
        "repeat_tolerance": args.repeat_tolerance,
        "records": rows,
        "ready_for_distillation": all(row["admission"]["accepted"] for row in rows),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"teacher admission report written: {output} ({len(rows)} records)")


if __name__ == "__main__":
    main()
